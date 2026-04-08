"""
Lean 4 code manipulation utilities for the Goedel prover.

Functions are kept verbatim from the original Goedel pipeline because the
fine-tuned model is sensitive to the exact prompt format it was trained on.
Original pipeline: https://github.com/Goedel-LM/Goedel-Prover-V2
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


_BY_CLAUSE_RE = re.compile(r":=\s*by\b", re.MULTILINE)


def _remove_comments(text: str) -> str:
    text = re.sub(r"/-.*?-/", "", text, flags=re.DOTALL)
    lines = text.split("\n")
    cleaned_lines = [line.split("--", 1)[0] for line in lines]
    return "\n".join(cleaned_lines).strip()


def _return_theorem_to_prove(text: str) -> Optional[Tuple[int, int]]:
    pattern = r"((?:theorem).*?:=\s*by\s*sorry)"
    match = re.search(pattern, text, re.DOTALL)
    return match.span() if match else None


def _return_theorem_to_replace(text: str) -> Optional[Tuple[int, int]]:
    pattern = r"((?:^|\s)theorem\s+.*?:=\s*by)"
    match = re.search(pattern, text, re.DOTALL)
    return match.span() if match else None


def normalize_for_prompt(statement: str) -> str:
    """Ensure the theorem statement ends with ``:= by sorry``."""
    m = _BY_CLAUSE_RE.search(statement)
    if not m:
        raise ValueError(
            "normalize_for_prompt: cannot find ':= by' in the input statement."
        )
    return statement[: m.start()] + ":= by sorry"


def replace_statement_in_proof(statement: str, proof: str) -> str:
    """
    Splice a generated proof block into the original theorem statement.
    Returns a string starting with ``**Error**`` on failure.
    """
    if ("apply?" in proof) or ("exact?" in proof):
        return "**Error**, 'apply?' or 'exact?' is used, which is not allowed."

    stats_re = _remove_comments(statement)
    stats_span_ = _return_theorem_to_prove(stats_re)
    if stats_span_ is None:
        error_app = "\n".join(["\n"] + ["-- " + x for x in statement.split("\n")])
        return f"**Error**, can not find 'theorem' and ':= sorry' in {error_app}"

    proof_str = _remove_comments(proof)
    span = _return_theorem_to_replace(proof_str)
    if span is None:
        error_app = "\n".join(["\n"] + ["-- " + x for x in proof.split("\n")])
        return f"**Error**, can not find 'theorem' and ':=' in {error_app}"

    return stats_re[: stats_span_[1]].replace("sorry", "") + proof_str[span[1] :]


def extract_lean4_code_block(model_text: str) -> Optional[str]:
    """Return the last ```lean4 / ```lean code block from model output."""
    patterns = [
        r"```lean4\n(.*?)\n```",
        r"```lean4\n(.*?)```",
        r"```lean\n(.*?)```",
    ]
    for pat in patterns:
        matches = re.findall(pat, model_text, re.DOTALL)
        if matches:
            return matches[-1]
    return None


def format_lean_errors(
    code: str,
    errors: List[Dict[str, Any]],
    *,
    truncate: bool = True,
) -> str:
    """
    Format Lean JSON errors with ``<error>...</error>`` markers.
    """
    err_str = ""
    code_lines = code.split("\n")
    max_errors = 8 if truncate else len(errors)

    for i, error in enumerate(errors[:max_errors]):
        start_line = error["pos"]["line"] - 1
        start_col = error["pos"]["column"]

        if error.get("endPos") is None:
            end_line = start_line
            end_col = (
                len(code_lines[start_line])
                if 0 <= start_line < len(code_lines)
                else start_col
            )
        else:
            end_line = error["endPos"]["line"] - 1
            end_col = error["endPos"]["column"]

        err_str += f"\nError {i + 1}:\n"
        err_str += "\nCorresponding Code:\n```lean4\n"

        error_code = ""
        for ii in range(-4, 0):
            if 0 <= start_line + ii < len(code_lines):
                error_code += f"{code_lines[start_line + ii]}\n"

        start_line = max(0, min(start_line, len(code_lines) - 1))
        end_line = max(0, min(end_line, len(code_lines) - 1))
        start_col = max(0, min(start_col, len(code_lines[start_line])))
        end_col = max(0, min(end_col, len(code_lines[end_line])))

        if start_line != end_line:
            error_code += (
                code_lines[start_line][:start_col]
                + "<error>"
                + code_lines[start_line][start_col:]
                + "\n"
            )
            show_line = 6
            for j in range(start_line + 1, min(end_line, start_line + show_line)):
                error_code += f"{code_lines[j]}\n"
            if end_line > start_line + show_line:
                last_j = min(end_line - 1, start_line + show_line - 1)
                leading = len(code_lines[last_j]) - len(code_lines[last_j].lstrip(" "))
                error_code += " " * leading + "... --[Truncated]-- ...\n"
            error_code += (
                code_lines[end_line][:end_col]
                + "</error>"
                + code_lines[end_line][end_col:]
                + "\n"
            )
        else:
            error_code += (
                code_lines[start_line][:start_col]
                + "<error>"
                + code_lines[start_line][start_col:end_col]
                + "</error>"
                + code_lines[start_line][end_col:]
                + "\n"
            )

        if end_line + 1 < len(code_lines):
            error_code += f"{code_lines[end_line + 1]}\n"

        err_str += error_code
        err_str += "\n```\n"
        err_str += f"\nError Message: {error.get('data', '')}\n"

    if len(errors) > max_errors:
        err_str += f"\n... [Omitted {len(errors) - max_errors} more errors] ...\n"

    return err_str
