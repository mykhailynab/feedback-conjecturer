from __future__ import annotations

import re

from conjecturing_agents.tool_calling_backends.lean4_compiler import (
    Lean4CompilerBackend,
    LeanCompileResult,
)

from .result import CheckResult


# Matches the single-line abbrev placeholder in the scaffold.
_ABBREV_LINE_RE = re.compile(
    r"^\s*(?:(?:noncomputable|unsafe|protected|private)\s+)*abbrev\s+[\w']+"
)

# Tactic block tried in order for each attempt.
_TACTIC_BLOCKS = [
    # norm_num handles most numeric and algebraic equalities
    "simp only [{name}]\n  norm_num",
    # ring works for ring expressions that norm_num misses
    "simp only [{name}]\n  ring",
    # decide works for decidable propositions (ℕ, ℤ, ℚ concrete values)
    "simp only [{name}]\n  decide",
    # combined: unfold + simp lemmas + norm_num
    "simp only [{name}]\n  simp\n  norm_num",
    # native_decide for computationally decidable goals
    "simp only [{name}]\n  native_decide",
]


def extract_preamble(lean_statement: str) -> str:
    """Return the import/open lines that precede the abbrev placeholder."""
    lines = lean_statement.splitlines()
    preamble: list[str] = []
    for line in lines:
        if _ABBREV_LINE_RE.match(line):
            break
        preamble.append(line)
    return "\n".join(preamble).rstrip()


def build_equiv_check_file(
    lean_statement: str,
    proposed_abbrev_decl: str,
    gt_rhs: str,
    abbrev_name: str,
    *,
    tactic_block: str,
) -> str:
    """
    Build a self-contained Lean file that attempts to prove
    ``abbrev_name = gt_rhs`` using the given tactic block.

    The file contains:
      - preamble (imports + opens) from the scaffold
      - proposed abbrev declaration (sorry replaced with proposed_rhs)
      - an ``example`` that asserts equivalence with the GT answer
    """
    preamble = extract_preamble(lean_statement)
    tactic = tactic_block.format(name=abbrev_name)

    return (
        f"{preamble}\n\n"
        f"{proposed_abbrev_decl.strip()}\n\n"
        f"example : {abbrev_name} = ({gt_rhs}) := by\n"
        f"  {tactic}\n"
    )


def check_lean_equiv(
    lean_statement: str,
    proposed_abbrev_decl: str,
    gt_rhs: str,
    abbrev_name: str,
    compiler: Lean4CompilerBackend,
) -> CheckResult:
    """
    Heuristic 2: try to prove ``proposed = gt`` in Lean using a set of
    pre-canned tactic blocks.  Returns equivalent=True on the first success,
    equivalent=None if all tactics fail (never returns False).

    Parameters
    ----------
    lean_statement:
        The scaffold text (``lean_statement_without_comment`` from the log).
    proposed_abbrev_decl:
        Full abbrev declaration with the proposed RHS (e.g.
        ``abbrev foo_solution : ℝ := -1``).
    gt_rhs:
        The ground-truth RHS string extracted from the original scaffold
        comment (e.g. ``-1``).
    abbrev_name:
        The abbrev identifier (e.g. ``foo_solution``).
    compiler:
        A ready-to-use ``Lean4CompilerBackend`` instance.
    """
    attempts: list[dict] = []

    for tactic_block in _TACTIC_BLOCKS:
        lean_file = build_equiv_check_file(
            lean_statement,
            proposed_abbrev_decl,
            gt_rhs,
            abbrev_name,
            tactic_block=tactic_block,
        )
        result: LeanCompileResult = compiler.compile_code(lean_file)

        attempt_record = {
            "tactic_block": tactic_block,
            "ok": result.ok,
            "timed_out": result.timed_out,
            "oom": result.oom,
            "elapsed_ms": result.elapsed_ms,
            "json_error_count": len(result.json_errors),
            "sorry_warning_count": len(result.sorry_warnings),
            "formatted_diagnostics": result.formatted_diagnostics,
            "relative_path": result.relative_path,
        }
        attempts.append(attempt_record)

        if result.ok:
            return CheckResult(
                equivalent=True,
                method="lean_equiv",
                details={"winning_tactic": tactic_block, "attempts": attempts},
            )

    return CheckResult(
        equivalent=None,
        method="lean_equiv",
        details={"attempts": attempts},
    )


__all__ = [
    "extract_preamble",
    "build_equiv_check_file",
    "check_lean_equiv",
]
