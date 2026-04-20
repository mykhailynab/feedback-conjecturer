from typing import Any, Callable, Dict, List, Optional, Sequence

from .data_model import (
    LeanCompileResult,
    LeanCompilerConfig
)

def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(n, hi))


def format_lean_messages(
    code: str,
    messages: Sequence[Dict[str, Any]],
    *,
    max_messages: int = 8,
    truncate_middle_lines: bool = True,
) -> str:
    """
    Prompt-friendly formatting for Lean diagnostics.

    It highlights the approximate error span using <error>...</error> markers,
    following the same spirit as your previous theorem-proving utilities.
    """
    if not messages:
        return ""

    code_lines = code.split("\n")
    out_parts: List[str] = []
    shown = list(messages[:max_messages])

    for i, msg in enumerate(shown, start=1):
        severity = str(msg.get("severity", "unknown")).upper()
        message_text = str(msg.get("data", "")).strip()

        out_parts.append(f"\n{severity} {i}:\n")

        pos = msg.get("pos")
        if not isinstance(pos, dict) or "line" not in pos or "column" not in pos:
            out_parts.append("Message:\n")
            out_parts.append(f"{message_text}\n")
            continue

        start_line = int(pos["line"]) - 1
        start_col = int(pos["column"])

        end_pos = msg.get("endPos")
        if isinstance(end_pos, dict) and "line" in end_pos and "column" in end_pos:
            end_line = int(end_pos["line"]) - 1
            end_col = int(end_pos["column"])
        else:
            end_line = start_line
            if 0 <= start_line < len(code_lines):
                end_col = len(code_lines[start_line])
            else:
                end_col = start_col

        if not code_lines:
            out_parts.append("Corresponding Code:\n```lean4\n")
            out_parts.append("\n```\n")
            out_parts.append(f"Message:\n{message_text}\n")
            continue

        start_line = _clamp(start_line, 0, len(code_lines) - 1)
        end_line = _clamp(end_line, 0, len(code_lines) - 1)
        start_col = _clamp(start_col, 0, len(code_lines[start_line]))
        end_col = _clamp(end_col, 0, len(code_lines[end_line]))

        out_parts.append("Corresponding Code:\n```lean4\n")

        # Context before
        for j in range(max(0, start_line - 4), start_line):
            out_parts.append(code_lines[j] + "\n")

        if start_line != end_line:
            out_parts.append(
                code_lines[start_line][:start_col]
                + "<error>"
                + code_lines[start_line][start_col:]
                + "\n"
            )

            if truncate_middle_lines:
                show_line_budget = 6
                upper = min(end_line, start_line + show_line_budget)
                for j in range(start_line + 1, upper):
                    out_parts.append(code_lines[j] + "\n")

                if end_line > start_line + show_line_budget:
                    last_visible = upper - 1 if upper > start_line + 1 else start_line
                    leading_spaces = 0
                    if 0 <= last_visible < len(code_lines):
                        line = code_lines[last_visible]
                        leading_spaces = len(line) - len(line.lstrip(" "))
                    out_parts.append(" " * leading_spaces + "... --[Truncated]-- ...\n")
            else:
                for j in range(start_line + 1, end_line):
                    out_parts.append(code_lines[j] + "\n")

            out_parts.append(
                code_lines[end_line][:end_col]
                + "</error>"
                + code_lines[end_line][end_col:]
                + "\n"
            )
        else:
            out_parts.append(
                code_lines[start_line][:start_col]
                + "<error>"
                + code_lines[start_line][start_col:end_col]
                + "</error>"
                + code_lines[start_line][end_col:]
                + "\n"
            )

        if end_line + 1 < len(code_lines):
            out_parts.append(code_lines[end_line + 1] + "\n")

        out_parts.append("```\n")
        out_parts.append(f"Message:\n{message_text}\n")

    if len(messages) > max_messages:
        out_parts.append(f"\n... [Omitted {len(messages) - max_messages} more messages] ...\n")

    return "".join(out_parts).strip()


def build_tool_facing_feedback(
    result: LeanCompileResult,
    *,
    cfg: LeanCompilerConfig,
) -> str:
    """
    Build the textual response fed back to the agent/tool-calling model.
    """
    parts: List[str] = []

    status = "[OK]" if result.ok else "[ERROR]"
    parts.append(f"{status} Lean compilation {'succeeded' if result.ok else 'failed'}.")

    if result.relative_path:
        parts.append(f"File: {result.relative_path}")
    if result.returncode is not None:
        parts.append(f"Return code: {result.returncode}")
    parts.append(f"Elapsed ms: {result.elapsed_ms}{' [TIMED OUT]' if result.timed_out else ''}")

    if result.json_errors:
        parts.append(f"Errors: {len(result.json_errors)}")
    if result.json_warnings:
        parts.append(f"Warnings: {len(result.json_warnings)}")
    if result.sorry_warnings:
        parts.append(f"Sorry warnings: {len(result.sorry_warnings)}")

    if result.formatted_diagnostics:
        parts.append("")
        parts.append(result.formatted_diagnostics)

    if cfg.include_non_json_stdout_lines and result.non_json_stdout_lines:
        parts.append("")
        parts.append("Non-JSON stdout:")
        parts.append("\n".join(result.non_json_stdout_lines))

    if cfg.include_stdout_in_tool_response and result.stdout.strip():
        parts.append("")
        parts.append("Raw stdout:")
        parts.append(result.stdout)

    if cfg.include_stderr_in_tool_response and result.stderr.strip():
        parts.append("")
        parts.append("Raw stderr:")
        parts.append(result.stderr)

    return "\n".join(parts).strip()


__all__ = [
    "format_lean_messages",
    "build_tool_facing_feedback",
]