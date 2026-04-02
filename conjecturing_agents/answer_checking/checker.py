from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from conjecturing_agents.agents.conjecture_formalizer import (
    extract_rhs_from_abbrev_declaration,
)
from conjecturing_agents.tool_calling_backends.lean4_compiler import (
    Lean4CompilerBackend,
    LeanCompilerConfig,
)

from .result import CheckResult
from .string_match import check_string_match
from .lean_equiv import check_lean_equiv


@dataclass
class AnswerCheckerConfig:
    # Lean project settings (required for heuristic 2)
    lean_project_dir: str = "."
    lean_workspace_subdir: str = ".conjecturing_agents/answer_checking"
    lean_timeout_seconds: int = 120
    lean_jobs: int = 4

    # Memory limit for the lake/lean subprocess (bytes); 0 = no limit
    lean_max_memory_bytes: int = 17179869184  # 16 GiB

    # Which heuristics to run
    use_string_match: bool = True
    use_lean_equiv: bool = True
    # use_goedel_prover: bool = False  # heuristic 3 — not yet implemented


class AnswerChecker:
    """
    Orchestrates answer-checking heuristics in order:

      1. String match (fast, no Lean needed)
      2. Lean equivalence proof via canned tactics
      3. [future] Goedel-prover for a generated proof / disproof

    Stops as soon as any heuristic returns a conclusive result (equivalent != None).
    """

    def __init__(self, cfg: Optional[AnswerCheckerConfig] = None):
        self.cfg = cfg or AnswerCheckerConfig()
        self._compiler: Optional[Lean4CompilerBackend] = None

    # ------------------------------------------------------------------
    # Lean compiler (lazy init so we don't touch the filesystem unless
    # heuristic 2 is actually needed)
    # ------------------------------------------------------------------

    def _get_compiler(self) -> Lean4CompilerBackend:
        if self._compiler is None:
            lean_cfg = LeanCompilerConfig(
                project_dir=self.cfg.lean_project_dir,
                workspace_subdir=self.cfg.lean_workspace_subdir,
                timeout_seconds=self.cfg.lean_timeout_seconds,
                lean_jobs=self.cfg.lean_jobs,
                max_memory_bytes=self.cfg.lean_max_memory_bytes,
                treat_sorry_warning_as_failure=True,
                treat_any_warning_as_failure=False,
            )
            self._compiler = Lean4CompilerBackend(lean_cfg)
        return self._compiler

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check a single formalization record from formalizations.jsonl.

        Returns a dict with keys:
          - ``check_result``: final CheckResult (as dict)
          - ``all_results``: list of CheckResult dicts, one per heuristic tried
          - ``equivalent``: bool or None
          - ``method``: method that gave the conclusive result (or "inconclusive")
        """
        status = record.get("status")
        if status != "success":
            result = CheckResult(
                equivalent=None,
                method="skipped",
                details={"reason": f"record_status={status}"},
            )
            return self._format_output(result, [result])

        proposed_decl = record.get("final_abbrev_declaration") or ""
        gt_rhs = record.get("ground_truth_extracted_answer") or ""
        lean_statement = record.get("lean_statement_without_comment") or ""
        abbrev_name = record.get("required_abbrev_name") or ""

        if not proposed_decl or not gt_rhs or not abbrev_name:
            result = CheckResult(
                equivalent=None,
                method="skipped",
                details={"reason": "missing_fields"},
            )
            return self._format_output(result, [result])

        proposed_rhs = extract_rhs_from_abbrev_declaration(proposed_decl)
        all_results: list[CheckResult] = []

        # --- Heuristic 1: string match ---
        if self.cfg.use_string_match:
            r1 = check_string_match(proposed_rhs, gt_rhs)
            all_results.append(r1)
            if r1.equivalent is not None:
                return self._format_output(r1, all_results)

        # --- Heuristic 2: Lean equivalence proof ---
        if self.cfg.use_lean_equiv and lean_statement:
            compiler = self._get_compiler()
            r2 = check_lean_equiv(
                lean_statement,
                proposed_decl,
                gt_rhs,
                abbrev_name,
                compiler,
            )
            all_results.append(r2)
            if r2.equivalent is not None:
                return self._format_output(r2, all_results)

        # --- Heuristic 3 (placeholder): Goedel-prover ---
        # When implemented, insert here before the inconclusive fallback.

        inconclusive = CheckResult(equivalent=None, method="inconclusive", details={})
        return self._format_output(inconclusive, all_results)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_output(
        final: CheckResult,
        all_results: list[CheckResult],
    ) -> Dict[str, Any]:
        return {
            "equivalent": final.equivalent,
            "method": final.method,
            "check_result": vars(final),
            "all_results": [vars(r) for r in all_results],
        }

    def close(self) -> None:
        pass  # Lean4CompilerBackend has no teardown; reserved for future use

    def __enter__(self) -> "AnswerChecker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = ["AnswerCheckerConfig", "AnswerChecker"]
