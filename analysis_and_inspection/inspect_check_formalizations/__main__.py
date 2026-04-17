#!/usr/bin/env python3
"""
Inspect Goedel prover conversations from a check_formalizations run.

Reads check_goedel_events.jsonl and check_results.jsonl to display the full
multi-turn conversation each session had, along with summary stats.

Usage:
    PYTHONPATH=analysis_and_inspection python -m inspect_goedel_conversations \\
        --goedel-events logs/.../check_goedel_events.jsonl \\
        --results       logs/.../check_results.jsonl \\
        [--filter  proof|disproof|proved|failed|disproved|incomplete] \\
        [--problem-id <id>] \\
        [--max-sessions N] \\
        [--stats-only] \\
        [--no-color] \\
        [--no-truncate] \\
        [--max-output-chars N]
"""
from __future__ import annotations

import sys
import argparse

from analysis_and_inspection.display_utils import bold, dim, green, red, yellow, set_color
from analysis_and_inspection.goedel_data import load_goedel_sessions, load_check_results, apply_filter, FILTER_CHOICES
from analysis_and_inspection.inspect_goedel_conversations.render_session import render_session
from analysis_and_inspection.inspect_goedel_conversations.render_stats import render_stats


def _check_result_summary(cr: dict) -> str:
    eq = cr.get("equivalent")
    method = cr.get("method", "?")
    eq_str = green("True") if eq is True else (red("False") if eq is False else yellow("None"))
    return f"check_result: equivalent={eq_str}  method={dim(method)}"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Inspect Goedel prover conversations from a check_formalizations run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--goedel-events", required=True, help="Path to check_goedel_events.jsonl.")
    p.add_argument("--results", required=True, help="Path to check_results.jsonl.")
    p.add_argument(
        "--filter",
        choices=FILTER_CHOICES,
        default=None,
        help=(
            "Show only a subset of sessions: "
            "proof / disproof / proved / disproved / failed / incomplete."
            "\n- proof/disproof include unsuccessful sessions"
            "\n- proved/disproved include only successful sessions"
            "\n- incomplete are the ones w/o an end event (timeout exception, keyboardinterrupt, etc.)"
            "\n- failed are the ones that had issues with context/rounds/splice error"
        ),
    )
    p.add_argument("--problem-id", default="", help="Show only sessions for this problem ID.")
    p.add_argument("--max-sessions", type=int, default=20,
                   help="Maximum number of sessions to display (0 = all).")
    p.add_argument("--stats-only", action="store_true",
                   help="Skip conversation display; show only summary statistics.")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colour output.")
    p.add_argument("--no-truncate", action="store_true",
                   help="Print full prompt and model output without truncation.")
    p.add_argument("--max-output-chars", type=int, default=3000,
                   help="Truncate model outputs / prompts to this many characters (default: 3000).")
    args = p.parse_args()

    if args.no_color or not sys.stdout.isatty():
        set_color(False)

    sessions = load_goedel_sessions(args.goedel_events)
    results  = load_check_results(args.results)

    filtered_sessions = sessions
    if args.problem_id:
        filtered_sessions = [s for s in filtered_sessions if s.problem_id == args.problem_id]
    filtered_sessions = apply_filter(filtered_sessions, args.filter)

    to_display = filtered_sessions
    if args.max_sessions > 0:
        to_display = to_display[:args.max_sessions]

    if not args.stats_only:
        print(f"\n{bold('Goedel Conversation Inspector')}")
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
            cr = results.get((sess.problem_id, sess.attempt))
            summary = _check_result_summary(cr) if cr else None
            print(render_session(
                sess,
                summary,
                max_output_chars=args.max_output_chars,
                no_truncate=args.no_truncate,
            ))

    print(render_stats(sessions, results))


if __name__ == "__main__":
    main()
