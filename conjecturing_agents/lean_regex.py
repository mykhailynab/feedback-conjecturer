"""
Lean 4 syntax patterns and text-extraction utilities.

All regex patterns and pure-text helpers for parsing Lean 4 source are
defined here so they are compiled once and shared across the codebase.
No domain imports — only ``re`` and ``typing``.
"""
from __future__ import annotations

import re
from typing import List, Optional

# ---------------------------------------------------------------------------
# Internal building block
# ---------------------------------------------------------------------------

# Modifier keywords that may precede any top-level Lean declaration.
_MODIFIERS = r"(?:(?:noncomputable|unsafe|protected|private)\s+)*"

# ---------------------------------------------------------------------------
# Abbrev patterns
# ---------------------------------------------------------------------------

# Matches any abbrev declaration line; captures the abbrev identifier.
ABBREV_NAME_RE = re.compile(
    r"^\s*" + _MODIFIERS + r"abbrev\s+([\w']+)",
    re.MULTILINE,
)

# Matches a complete single-line abbrev (has ``:=``); captures the name.
ABBREV_SINGLE_LINE_RE = re.compile(
    r"^\s*" + _MODIFIERS + r"abbrev\s+([\w']+)[^\n]*:=.*$",
    re.MULTILINE,
)

# Matches an abbrev declaration line followed immediately by a ``--`` comment.
# Named groups: ``abbrev_line``, ``comment_line``.
# Used to strip the ground-truth answer comment embedded in scaffolds.
ABBREV_WITH_COMMENT_RE = re.compile(
    r"(?P<abbrev_line>^[ \t]*" + _MODIFIERS + r"abbrev[^\n]*:=.*\r?\n)"
    r"(?P<comment_line>^[ \t]*--[^\n]*\r?\n?)",
    re.MULTILINE,
)

# Matches an abbrev line without capturing the name (used as a stop-marker).
ABBREV_LINE_RE = re.compile(
    r"^\s*" + _MODIFIERS + r"abbrev\s+[\w']+",
)

# Matches an abbrev declaration and captures everything after ``:=``.
ABBREV_RHS_RE = re.compile(
    r"^\s*" + _MODIFIERS + r"abbrev\s+[\w']+\b[^\n]*:=\s*(.*)$",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Top-level declaration boundary
# ---------------------------------------------------------------------------

# Matches the start of any top-level Lean declaration / command.
# Used as an extraction boundary when collecting an abbrev block.
TOP_LEVEL_DECL_RE = re.compile(
    r"^\s*" + _MODIFIERS +
    r"(?:abbrev|theorem|lemma|def|example|structure|class|inductive|instance|"
    r"namespace|end|section|open|import|#check|#eval|#print)\b"
)

# Matches a theorem declaration line; captures the theorem identifier.
THEOREM_NAME_RE = re.compile(
    r"^\s*" + _MODIFIERS + r"theorem\s+([\w']+)",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Misc patterns
# ---------------------------------------------------------------------------

# Matches the word ``sorry`` at a word boundary.
SORRY_RE = re.compile(r"\bsorry\b")

# Matches ``:= by`` in theorem statements.
BY_CLAUSE_RE = re.compile(r":=\s*by\b", re.MULTILINE)

# Matches term-mode `:= sorry` (no `by`) — may span a newline between `:=` and `sorry`.
TERM_SORRY_RE = re.compile(r":=\s*sorry\b", re.MULTILINE)

# ---------------------------------------------------------------------------
# Fenced code block patterns
# ---------------------------------------------------------------------------

# Tried in order; most specific (lean4, optional trailing whitespace) first.
LEAN_BLOCK_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"```lean4\s*\n(.*?)\n```", re.DOTALL),
    re.compile(r"```lean4\s*\n(.*?)```",   re.DOTALL),
    re.compile(r"```lean\s*\n(.*?)\n```",  re.DOTALL),
    re.compile(r"```lean\s*\n(.*?)```",    re.DOTALL),
]

# ---------------------------------------------------------------------------
# Code block extraction
# ---------------------------------------------------------------------------

def extract_lean_code_block(model_text: str) -> Optional[str]:
    """Return the last fenced Lean code block from *model_text*, stripped.

    Tries ``lean4`` blocks before falling back to generic ``lean`` blocks.
    Returns ``None`` if no block is found.
    """
    if not model_text:
        return None
    for pat in LEAN_BLOCK_PATTERNS:
        matches = pat.findall(model_text)
        if matches:
            return matches[-1].strip()
    return None


def extract_lean_code_block_or_text(text: str) -> str:
    """Return the last Lean code block, or the full text (stripped) if none."""
    code = extract_lean_code_block(text)
    return code if code is not None else (text or "").strip()

# ---------------------------------------------------------------------------
# Comment stripping
# ---------------------------------------------------------------------------

def remove_lean_comments(text: str) -> str:
    """Strip ``/- … -/`` block comments and ``--`` line comments."""
    text = re.sub(r"/-.*?-/", "", text, flags=re.DOTALL)
    lines = text.split("\n")
    return "\n".join(line.split("--", 1)[0] for line in lines).strip()

# ---------------------------------------------------------------------------
# Abbrev utilities
# ---------------------------------------------------------------------------

def extract_abbrev_name_from_statement(lean_text: str) -> Optional[str]:
    """Return the identifier of the first (or only) abbrev in *lean_text*."""
    m = ABBREV_NAME_RE.search(lean_text or "")
    return m.group(1) if m else None


def extract_rhs_from_abbrev_declaration(abbrev_declaration: str) -> Optional[str]:
    """Return everything after ``:=`` in a single abbrev declaration string."""
    if ":=" not in abbrev_declaration:
        return None
    return abbrev_declaration.split(":=", 1)[1].strip()


def extract_preamble(lean_statement: str) -> str:
    """Return the import/open lines that precede the abbrev placeholder."""
    preamble: List[str] = []
    for line in lean_statement.splitlines():
        if ABBREV_LINE_RE.match(line):
            break
        preamble.append(line)
    return "\n".join(preamble).rstrip()


def extract_ground_truth_comment_and_strip_line(lean4_full_contents: str) -> tuple[str, str]:
    """Extract and remove the ground-truth answer comment following the abbrev.

    Expected shape::

        abbrev foo_solution : ... := sorry
        -- <ground truth answer>

        theorem ...

    Returns ``(ground_truth_answer, scaffold_without_comment_line)``.
    Raises ``ValueError`` if exactly one match is not found.
    """
    matches = list(ABBREV_WITH_COMMENT_RE.finditer(lean4_full_contents or ""))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one abbrev+comment match, found {len(matches)}"
        )
    m = matches[0]
    comment_line = m.group("comment_line")
    ground_truth_answer = comment_line.split("--", 1)[1].strip()

    stripped = (
        lean4_full_contents[: m.start("comment_line")]
        + lean4_full_contents[m.end("comment_line"):]
    )
    if comment_line in stripped:
        raise ValueError("Failed to remove the matched abbrev answer comment line")

    return ground_truth_answer, stripped


def replace_abbrev_in_statement(
    lean_statement: str,
    new_abbrev_declaration: str,
    *,
    required_abbrev_name: Optional[str] = None,
) -> str:
    """Splice *new_abbrev_declaration* into the scaffold, replacing the placeholder.

    Raises ``ValueError`` if the abbrev names do not match or if the scaffold
    does not contain exactly one single-line abbrev placeholder.
    """
    current_name = extract_abbrev_name_from_statement(lean_statement)
    if current_name is None:
        raise ValueError("Could not find abbrev name in lean_statement")
    if required_abbrev_name is not None and current_name != required_abbrev_name:
        raise ValueError(
            f"Lean statement abbrev name {current_name!r} does not match "
            f"required_abbrev_name={required_abbrev_name!r}"
        )

    new_name = extract_abbrev_name_from_statement(new_abbrev_declaration)
    if new_name is None:
        raise ValueError("Could not find abbrev name in new_abbrev_declaration")
    if new_name != current_name:
        raise ValueError(
            f"Generated abbrev name {new_name!r} does not match scaffold "
            f"abbrev name {current_name!r}"
        )

    matches = list(ABBREV_SINGLE_LINE_RE.finditer(lean_statement))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one single-line abbrev placeholder in scaffold, "
            f"found {len(matches)}"
        )
    m = matches[0]
    return lean_statement[: m.start()] + new_abbrev_declaration.strip() + lean_statement[m.end():]


def negate_theorem_statement(lean_text: str) -> str:
    """Transform ``theorem T (args) : P := by sorry`` into
    ``theorem T_neg (args) : ¬ (P) := by sorry``.

    Works on a full Lean file (preamble + abbrev + theorem).  Only the
    theorem declaration is modified; the preamble and abbrev are left intact.

    The input must contain exactly one ``theorem`` keyword.  The theorem
    must end with ``:= by sorry`` (the GoedelProver prompt format — use
    ``normalize_for_prompt`` from ``lean_utils.py`` to enforce this).

    Raises ``ValueError`` if the theorem cannot be found or parsed.
    """
    m = THEOREM_NAME_RE.search(lean_text)
    if m is None:
        raise ValueError("negate_theorem_statement: no theorem found in lean_text")

    theorem_name = m.group(1)
    name_start = m.start(1)
    name_end = m.end(1)

    # Scan for the first ':' at paren-depth 0 after the theorem name
    # that is NOT ':=' — this marks the start of the type annotation.
    depth = 0
    prop_colon_pos: Optional[int] = None
    i = name_end
    while i < len(lean_text):
        c = lean_text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ":" and depth == 0:
            if i + 1 < len(lean_text) and lean_text[i + 1] == "=":
                i += 1  # skip ':='
            else:
                prop_colon_pos = i
                break
        i += 1

    if prop_colon_pos is None:
        raise ValueError(
            "negate_theorem_statement: could not find proposition ':' in theorem"
        )

    # Scan for ':=' at depth 0 after the proposition colon.
    depth = 0
    assign_pos: Optional[int] = None
    i = prop_colon_pos + 1
    while i < len(lean_text):
        c = lean_text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ":" and depth == 0 and i + 1 < len(lean_text) and lean_text[i + 1] == "=":
            assign_pos = i
            break
        i += 1

    if assign_pos is None:
        raise ValueError(
            "negate_theorem_statement: could not find ':=' in theorem"
        )

    proposition = lean_text[prop_colon_pos + 1:assign_pos].strip()
    args_text = lean_text[name_end:prop_colon_pos]

    return (
        lean_text[:name_start]
        + theorem_name + "_neg"
        + args_text.rstrip()
        + " : ¬ (" + proposition + ") := by sorry"
    )


def extract_last_abbrev_declaration(
    text: str,
    *,
    required_abbrev_name: Optional[str] = None,
) -> Optional[str]:
    """Extract the last plausible abbrev declaration from model output.

    Searches inside a ``lean4`` code block first, then raw text.
    Rejects any declaration that contains ``sorry``.
    """
    if not text:
        return None

    candidate_texts: List[str] = []
    code_block = extract_lean_code_block(text)
    if code_block:
        candidate_texts.append(code_block)
    stripped_text = text.strip()
    if code_block != stripped_text:
        candidate_texts.append(stripped_text)

    for candidate_text in candidate_texts:
        lines = candidate_text.splitlines()
        abbrev_indices = [
            i for i, line in enumerate(lines)
            if ABBREV_NAME_RE.match(line)
        ]
        for idx in reversed(abbrev_indices):
            head = lines[idx]
            m_name = ABBREV_NAME_RE.match(head)
            if not m_name:
                continue
            abbrev_name = m_name.group(1)
            if required_abbrev_name is not None and abbrev_name != required_abbrev_name:
                continue

            block_lines = [head]
            for j in range(idx + 1, len(lines)):
                line = lines[j]
                if TOP_LEVEL_DECL_RE.match(line):
                    break
                if line.strip().startswith("```"):
                    break
                block_lines.append(line)

            decl = "\n".join(block_lines).strip()
            if ":=" not in decl:
                continue
            if SORRY_RE.search(decl):
                continue
            return decl

    return None
