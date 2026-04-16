#!/usr/bin/env python3
"""
Inspect prove_formalizations runs: session traces and summary statistics.

Reads prove_goedel_events.jsonl and prove_results.jsonl to display per-session
conversation traces and/or aggregate statistics.

Usage:
    PYTHONPATH=. python -m analysis_and_inspection.inspect_prove_formalizations \\
        --goedel-events logs/.../prove_goedel_events.jsonl \\
        --results       logs/.../prove_results.jsonl \\
        [--trace]                          # show conversation traces (off by default)
        [--filter  proof|disproof|proved|failed|disproved|incomplete] \\
        [--problem-id <id>] \\
        [--max-sessions N] \\
        [--no-color] \\
        [--no-truncate] \\
        [--max-output-chars N]
"""
from __future__ import annotations

import sys
import argparse

from analysis_and_inspection.display_utils import bold, dim, green, red, yellow, set_color
from analysis_and_inspection.goedel_data import (
    load_prove_sessions,
    load_prove_results,
    apply_filter,
    FILTER_CHOICES,
)
from analysis_and_inspection.inspect_goedel_conversations.render_session import render_session
from analysis_and_inspection.inspect_prove_formalizations.render_stats import render_stats


def _prove_result_summary(r: dict) -> str:
    parts = []
    if r.get("proved"):
        parts.append(green("proved"))
    elif r.get("disproved"):
        parts.append(red("disproved"))
    elif r.get("incomplete"):
        pr = r.get("proof_result") or {}
        ctx = pr.get("total_context_tokens")
        ctx_str = f"  ctx={ctx}" if ctx else ""
        parts.append(yellow(f"incomplete{ctx_str}"))
    else:
        parts.append(dim("inconclusive"))
    if r.get("skipped"):
        parts.append(dim(f"skipped ({r.get('skip_reason', '?')})"))
    status = r.get("status", "")
    if status and status not in ("success",):
        parts.append(dim(f"status={status}"))
    return "prove_result: " + "  ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Inspect prove_formalizations runs: traces and statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--goedel-events", required=True,
                   help="Path to prove_goedel_events.jsonl.")
    p.add_argument("--results", required=True,
                   help="Path to prove_results.jsonl.")
    p.add_argument("--trace", action="store_true", default=False,
                   help="Print per-session conversation traces (off by default).")
    p.add_argument(
        "--filter",
        choices=FILTER_CHOICES,
        default=None,
        help=(
            "Show only a subset of sessions: "
            "proof / disproof / proved / disproved / failed / incomplete."
            "\n- proof/disproof filter by direction"
            "\n- proved/disproved include only successful sessions"
            "\n- failed are sessions where all rounds were exhausted without a proof"
            "\n- incomplete are sessions with no session_done event (killed mid-run)"
        ),
    )
    p.add_argument("--problem-id", default="",
                   help="Show only sessions for this problem ID.")
    p.add_argument("--max-sessions", type=int, default=20,
                   help="Maximum number of sessions to display with --trace (0 = all).")
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI colour output.")
    p.add_argument("--no-truncate", action="store_true",
                   help="Print full prompt and model output without truncation.")
    p.add_argument("--max-output-chars", type=int, default=3000,
                   help="Truncate model outputs / prompts to this many characters (default: 3000).")
    args = p.parse_args()

    if args.no_color or not sys.stdout.isatty():
        set_color(False)

    sessions = load_prove_sessions(args.goedel_events)
    results  = load_prove_results(args.results)

    filtered_sessions = sessions
    if args.problem_id:
        filtered_sessions = [s for s in filtered_sessions if s.problem_id == args.problem_id]
    filtered_sessions = apply_filter(filtered_sessions, args.filter)

    to_display = filtered_sessions
    if args.max_sessions > 0:
        to_display = to_display[:args.max_sessions]

    if args.trace:
        print(f"\n{bold('Prove Formalizations Inspector')}")
        print(dim(f"Events file : {args.goedel_events}"))
        print(dim(f"Results file: {args.results}"))
        filter_desc = []
        if args.problem_id:
            filter_desc.append(f"problem_id={args.problem_id}")
        if args.filter:
            filter_desc.append(f"filter={args.filter}")
        if filter_desc:
            print(dim(f"Filter      : {', '.join(filter_desc)}"))
        print(dim(f"Showing {len(to_display)} of {len(filtered_sessions)} filtered sessions "
                  f"({len(sessions)} total)"))

        for sess in to_display:
            r = results.get((sess.problem_id, sess.attempt))
            summary = _prove_result_summary(r) if r else None
            print(render_session(
                sess,
                summary,
                max_output_chars=args.max_output_chars,
                no_truncate=args.no_truncate,
            ))

    print(render_stats(sessions, results))


if __name__ == "__main__":
    main()
