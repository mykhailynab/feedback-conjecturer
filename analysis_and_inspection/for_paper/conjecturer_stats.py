#!/usr/bin/env python3
"""
Generate tables and plots for the conjecturer results.
  - Consolidated accuracy / tool-usage / context table
  - Termination reasons grouped bar chart
  - Turns-per-session histogram

Usage:
    PYTHONPATH=. python analysis_and_inspection/for_paper/conjecturer_stats.py
    PYTHONPATH=. python analysis_and_inspection/for_paper/conjecturer_stats.py \
        --run logs/my_run "My Run"
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[2]
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)


# ── helpers ────────────────────────────────────────────────────

def load_attempts(log_dir: Path, attempts_file: str | None = None) -> list[dict]:
    if attempts_file:
        path = log_dir / attempts_file
    elif (log_dir / "attempts_fixed.jsonl").exists():
        path = log_dir / "attempts_fixed.jsonl"
    else:
        path = log_dir / "attempts.jsonl"
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_correctness(
    log_dir: Path, solutions_file: str | None = None,
) -> dict[tuple[str, int], bool]:
    """
    Load per-attempt correctness from the solutions CSV.

    Returns {(problem_id, attempt): is_correct}.
    """
    if solutions_file:
        path = log_dir / solutions_file
    elif (log_dir / "solutions_fixed.csv").exists():
        path = log_dir / "solutions_fixed.csv"
    else:
        path = log_dir / "solutions.csv"
    result = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            pid = row["id"]
            cs = row.get("checker_summary", "")
            if not cs:
                continue
            parsed = json.loads(cs)
            per_attempt = parsed.get("per_attempt_truth", {})
            for att_str, info in per_attempt.items():
                result[(pid, int(att_str))] = bool(info.get("is_correct"))
    return result


def compute_stats(attempts: list[dict]) -> dict:
    n = len(attempts)
    if n == 0:
        return {
            "n": 0,
            "total_py_calls": 0, "total_py_errors": 0, "py_error_rate": 0,
            "calls_per_session_min": 0, "calls_per_session_mean": 0,
            "calls_per_session_median": 0, "calls_per_session_max": 0,
            "answer_count": 0, "answer_rate": 0,
            "resp_min": 0, "resp_max": 0, "resp_mean": 0, "resp_median": 0,
            "context_min": 0, "context_max": 0, "context_mean": 0, "context_median": 0,
            "turns": [], "turns_min": 0, "turns_mean": 0, "turns_median": 0, "turns_max": 0,
            "term_counts": Counter(),
        }
    py_calls = [a.get("python_calls", 0) or 0 for a in attempts]
    py_errors = [a.get("python_errors", 0) or 0 for a in attempts]
    resp_lens = [a.get("response_length", 0) or 0 for a in attempts]
    context_lens = [
        len(a.get("trace", {}).get("full_conversation_token_ids", []))
        if a["termination_reason"] != "context_exhausted" else 65536  # NOTE: this is specific to our gpt-pss:120B solver. Fix this
        for a in attempts
    ]
    # turns = number of tool calls + 1 (final assistant message)
    turns = []
    for a in attempts:
        tc = a.get("tool_calls")
        if isinstance(tc, list):
            turns.append(len(tc) + 1)
        else:
            turns.append((a.get("python_calls", 0) or 0) + 1)

    total_calls = sum(py_calls)
    total_errors = sum(py_errors)

    has_answer = sum(
        1 for a in attempts
        if a.get("attempt_answer") is not None
        and len(a.get("attempt_answer", "")) > 0
    )

    term_counts = Counter(
        str(a.get("termination_reason", "")) for a in attempts
    )

    return {
        "n": n,
        "total_py_calls": total_calls,
        "total_py_errors": total_errors,
        "py_error_rate": total_errors / total_calls if total_calls else 0,
        "calls_per_session_min": min(py_calls),
        "calls_per_session_mean": statistics.mean(py_calls),
        "calls_per_session_median": statistics.median(py_calls),
        "calls_per_session_max": max(py_calls),
        "answer_count": has_answer,
        "answer_rate": has_answer / n if n else 0,
        "resp_min": min(resp_lens),
        "resp_max": max(resp_lens),
        "resp_mean": statistics.mean(resp_lens),
        "resp_median": statistics.median(resp_lens),
        "context_min": min(context_lens),
        "context_max": max(context_lens),
        "context_mean": statistics.mean(context_lens),
        "context_median": statistics.median(context_lens),
        "turns": turns,
        "turns_min": min(turns),
        "turns_mean": statistics.mean(turns),
        "turns_median": statistics.median(turns),
        "turns_max": max(turns),
        "term_counts": term_counts,
    }


def split_by_correctness(
    attempts: list[dict],
    correctness: dict[tuple[str, int], bool],
) -> tuple[list[dict], list[dict]]:
    """Split attempts into (correct, incorrect) based on the solutions CSV."""
    correct, incorrect = [], []
    for a in attempts:
        pid = a.get("id") or a.get("problem_id")
        att = a.get("attempt")
        if pid is None or att is None:
            incorrect.append(a)
            continue
        is_correct = correctness.get((str(pid), int(att)))
        if is_correct:
            correct.append(a)
        else:
            incorrect.append(a)
    return correct, incorrect


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate tables and plots for conjecturer results.",
    )
    p.add_argument(
        "--run", nargs=2, action="append", metavar=("LOG_DIR", "LABEL"),
        help="Run: log directory and display label. Repeatable. "
             "Order matters for table column ordering.",
    )
    p.add_argument(
        "--attempts-file", type=str, default=None,
        help="Override attempts filename (default: attempts_fixed.jsonl if present, else attempts.jsonl).",
    )
    p.add_argument(
        "--solutions-file", type=str, default=None,
        help="Override solutions filename (default: solutions_fixed.csv if present, else solutions.csv).",
    )
    p.add_argument(
        "--output-dir", type=str, default=str(PLOTS),
        help="Directory for output plots (default: plots/).",
    )
    args = p.parse_args()

    if args.run is None:
        args.run = [
            [str(ROOT / "logs" / "putnam_120b_tir_pass4"), "5 min"],
            [str(ROOT / "logs" / "putnam_120b_tir_pass4_20min"), "20 min"],
        ]

    return args


# ── main ───────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all runs: for each run, compute stats for all / correct / incorrect
    # runs stores (label, all_stats) for charts; columns stores all 3×N stat dicts for the table
    runs: list[tuple[str, dict]] = []
    columns: list[tuple[str, dict]] = []  # (column_header, stats)
    for log_dir_str, label in args.run:
        log_dir = Path(log_dir_str)
        attempts = load_attempts(log_dir, args.attempts_file)
        correctness = load_correctness(log_dir, args.solutions_file)
        correct_att, incorrect_att = split_by_correctness(attempts, correctness)

        s_all = compute_stats(attempts)
        s_correct = compute_stats(correct_att)
        s_incorrect = compute_stats(incorrect_att)

        runs.append((label, s_all))
        columns.append((label, s_all))
        columns.append((f"{label} (+)", s_correct))
        columns.append((f"{label} (-)", s_incorrect))

    labels = [label for label, _ in runs]

    # ── 1) Print consolidated table (markdown) ─────────────────
    col_headers = [h for h, _ in columns]
    col_stats = [s for _, s in columns]
    hdr = " | ".join(col_headers)
    sep = " | ".join("------:" for _ in col_headers)
    print(f"\n### Consolidated solver agent stats ({' vs '.join(labels)})\n")
    print(f"| Metric | {hdr} |")
    print(f"|--------|{sep}|")

    def row(name: str, key: str, fmt: str = ","):
        vals = " | ".join(
            f"{s[key]:{fmt}}" if s["n"] else "—" for s in col_stats
        )
        print(f"| {name} | {vals} |")

    def row_frac(name: str, num_key: str, den_key: str):
        vals = " | ".join(
            f"{s[num_key]}/{s[den_key]} ({s[num_key]/s[den_key]:.2%})" if s[den_key] else "—"
            for s in col_stats
        )
        print(f"| {name} | {vals} |")

    row("N (attempts)", "n")
    row("Python tool calls (total)", "total_py_calls")
    row("Python tool errors (total)", "total_py_errors")
    row("Error rate", "py_error_rate", ".2%")
    row("Calls / session (min)", "calls_per_session_min", ".1f")
    row("Calls / session (mean)", "calls_per_session_mean", ".1f")
    row("Calls / session (median)", "calls_per_session_median", ".1f")
    row("Calls / session (max)", "calls_per_session_max", ".1f")
    row_frac("Sessions with answer", "answer_count", "n")
    row("Turns / session (min)", "turns_min", ".1f")
    row("Turns / session (mean)", "turns_mean", ".1f")
    row("Turns / session (median)", "turns_median", ".0f")
    row("Turns / session (max)", "turns_max", ".1f")
    row("Context tokens used — min", "context_min")
    row("Context tokens used — median", "context_median", ",.0f")
    row("Context tokens used — mean", "context_mean", ",.0f")
    row("Context tokens used — max", "context_max")
    row("Response tokens used — min", "resp_min")
    row("Response tokens used — median", "resp_median", ",.0f")
    row("Response tokens used — mean", "resp_mean", ",.0f")
    row("Response tokens used — max", "resp_max")

    # ── 2) Termination reasons bar chart ───────────────────────
    def simplify_reason(r: str) -> str:
        if r.startswith("exception:HarmonyError"):
            return "HarmonyError"
        return r

    term_counters = []
    for _, s in runs:
        tc = Counter()
        for r, c in s["term_counts"].items():
            tc[simplify_reason(r)] += c
        term_counters.append(tc)

    all_reasons = sorted(
        set().union(*(tc.keys() for tc in term_counters)),
        key=lambda r: -sum(tc.get(r, 0) for tc in term_counters),
    )

    n_runs = len(runs)
    colors = ["#5B9BD5", "#ED7D31", "#70AD47", "#FFC000", "#9B59B6", "#E74C3C"]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(all_reasons))
    w = 0.8 / n_runs
    for i, ((label, _), tc) in enumerate(zip(runs, term_counters)):
        offset = (i - (n_runs - 1) / 2) * w
        bars = [tc.get(r, 0) for r in all_reasons]
        ax.bar([j + offset for j in x], bars, w, label=label, color=colors[i % len(colors)])
    ax.set_xticks(list(x))
    wrapped = [r.replace("_", "\n") for r in all_reasons]
    ax.set_xticklabels(wrapped, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"Termination reasons ({' vs '.join(labels)})")
    ax.legend()
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    path = output_dir / "termination_reasons.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"\nSaved: {path}")

    # ── 3) Turns-per-session bar chart ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3.5))
    w = 0.8 / n_runs
    for i, ((label, _), (_, s)) in enumerate(zip(runs, runs)):
        offset = (i - (n_runs - 1) / 2) * w
        counts = Counter(s["turns"])
        xs = sorted(counts.keys())
        ys = [counts[k] for k in xs]
        ax.bar([xv + offset for xv in xs], ys, w, color=colors[i % len(colors)], label=label)
    ax.set_xlabel("Turns per session")
    ax.set_ylabel("Number of sessions")
    ax.set_title("Turns per session")
    ax.set_yscale('log')
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    path = output_dir / "turns_per_session.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
