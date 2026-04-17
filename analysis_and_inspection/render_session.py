"""Renders a single ProverSession as a multi-turn conversation transcript."""
from __future__ import annotations

from typing import List, Optional

from analysis_and_inspection.display_utils import (
    WIDTH,
    bold, dim, green, red, yellow, cyan, blue,
    box_top, box_bottom, box_line, section_header,
    fmt_ms, truncate,
)
from analysis_and_inspection.goedel_data import ProverRound, ProverSession, parse_chat_messages


def _role_label(role: str) -> str:
    return {
        "user": "USER",
        "assistant": "ASSISTANT",
        "system": "SYSTEM",
    }.get(role.lower(), role.upper())


def _lean_status_str(rnd: ProverRound) -> str:
    if rnd.lean_ok is True:
        return green("LEAN OK")
    if rnd.lean_timed_out:
        return yellow("LEAN TIMEOUT")
    if rnd.lean_oom:
        return yellow("LEAN OOM")
    if rnd.lean_ok is False:
        errs = rnd.lean_error_count or 0
        return red(f"LEAN FAILED ({errs} error{'s' if errs != 1 else ''})")
    reason = rnd.termination_reason
    if reason == "no_code_block":
        return yellow("NO CODE BLOCK")
    if reason == "splice_error":
        return yellow("SPLICE ERROR")
    if reason == "context_exceeded":
        return yellow("CONTEXT EXCEEDED")
    return dim(reason or "?")


def render_session(
    sess: ProverSession,
    result_summary: Optional[str] = None,
    *,
    max_output_chars: int = 3000,
    no_truncate: bool = False,
) -> str:
    lines: List[str] = []
    mc = 0 if no_truncate else max_output_chars

    # ---- Session header ----
    outcome_fn = sess.outcome_color
    outcome_str = outcome_fn(bold(sess.outcome.upper()))
    checking_str = cyan("DISPROOF") if sess.checking == "disproof" else blue("PROOF")

    lines.append("")
    lines.append("═" * WIDTH)
    lines.append(
        f"  {bold(sess.problem_id)}  attempt {sess.attempt}  "
        f"[{checking_str}]  →  {outcome_str}"
    )
    if sess.termination_reason:
        lines.append(f"  termination: {dim(sess.termination_reason)}  "
                     f"rounds: {sess.rounds_used}  "
                     f"time: {fmt_ms(sess.elapsed_ms)}")
    else:
        lines.append(f"  {dim('(incomplete — no session_done event recorded)')}")

    if result_summary:
        lines.append(f"  {result_summary}")

    lines.append("═" * WIDTH)

    # ---- Theorem statement ----
    lines.append(section_header("THEOREM STATEMENT"))
    stmt, trunc = truncate(sess.theorem_statement.strip(), mc)
    lines.append(dim(stmt))
    if trunc:
        lines.append(dim(f"  ... [{len(sess.theorem_statement) - mc} chars truncated]"))
    lines.append("")

    # ---- Rounds ----
    if not sess.rounds:
        lines.append(dim("  (no rounds recorded)"))

    for rnd in sess.rounds:
        lean_status = _lean_status_str(rnd)
        elapsed = f"  {dim(fmt_ms(rnd.elapsed_ms))}" if rnd.elapsed_ms else ""
        lines.append(section_header(f"ROUND {rnd.round_idx}  {lean_status}{elapsed}"))

        messages = parse_chat_messages(rnd.prompt_text)
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if not content:
                continue
            role_label = _role_label(role)
            content_trunc, was_trunc = truncate(content, mc)
            lines.append(box_top(role_label, WIDTH))
            lines.append(box_line(content_trunc))
            if was_trunc:
                lines.append(box_line(dim(f"... [{len(content) - mc} chars truncated]")))
            lines.append(box_bottom(WIDTH))

        if rnd.raw_output:
            out_trunc, was_trunc = truncate(rnd.raw_output, mc)
            lines.append(box_top(_role_label("assistant") + " (output)", WIDTH))
            lines.append(box_line(out_trunc))
            if was_trunc:
                lines.append(box_line(dim(f"... [{len(rnd.raw_output) - mc} chars truncated]")))
            lines.append(box_bottom(WIDTH))
        else:
            lines.append(dim("  (no model output recorded)"))

        lines.append("")

    return "\n".join(lines)
