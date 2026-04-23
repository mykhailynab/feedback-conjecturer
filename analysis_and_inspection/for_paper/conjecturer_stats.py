#!/usr/bin/env python3
"""
Generate tables and plots for the conjecturer results.
  - Consolidated accuracy / tool-usage / context table (5min vs 20min)
  - Termination reasons grouped bar chart
  - Turns-per-session histogram

Usage:
    PYTHONPATH=. python analysis_and_inspection/for_paper/conjecturer_stats.py
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── paths ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
LOGS_20 = ROOT / "logs" / "putnam_120b_tir_pass4_20min"
LOGS_5 = ROOT / "logs" / "putnam_120b_tir_pass4"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)


# ── helpers ────────────────────────────────────────────────────

def load_attempts(log_dir: Path) -> list[dict]:
    # Try both id field names
    fname = "attempts.jsonl"
    if (log_dir / "attempts_fixed.jsonl").exists():
        fname = "attempts_fixed.jsonl"
    rows = []
    with (log_dir / fname).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compute_stats(attempts: list[dict]) -> dict:
    n = len(attempts)
    py_calls = [a.get("python_calls", 0) or 0 for a in attempts]
    py_errors = [a.get("python_errors", 0) or 0 for a in attempts]
    resp_lens = [a.get("response_length", 0) or 0 for a in attempts]
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
        and str(a.get("attempt_answer", "")).strip() not in ("", "None")
    )

    term_counts = Counter(
        str(a.get("termination_reason", "")) for a in attempts
    )

    return {
        "n": n,
        "total_py_calls": total_calls,
        "total_py_errors": total_errors,
        "py_error_rate": total_errors / total_calls if total_calls else 0,
        "calls_per_session_mean": statistics.mean(py_calls),
        "calls_per_session_median": statistics.median(py_calls),
        "answer_count": has_answer,
        "answer_rate": has_answer / n if n else 0,
        "context_min": min(resp_lens),
        "context_max": max(resp_lens),
        "context_mean": statistics.mean(resp_lens),
        "context_median": statistics.median(resp_lens),
        "turns": turns,
        "turns_mean": statistics.mean(turns),
        "turns_median": statistics.median(turns),
        "term_counts": term_counts,
    }


# ── main ───────────────────────────────────────────────────────

def main():
    att_20 = load_attempts(LOGS_20)
    att_5 = load_attempts(LOGS_5)
    s20 = compute_stats(att_20)
    s5 = compute_stats(att_5)

    # ── 1) Print consolidated table (markdown) ─────────────────
    print("\n### Consolidated solver agent stats (5 min vs 20 min)\n")
    print("| Metric | 5 min | 20 min |")
    print("|--------|------:|-------:|")
    print(f"| Python tool calls (total) | {s5['total_py_calls']:,} | {s20['total_py_calls']:,} |")
    print(f"| Python tool errors (total) | {s5['total_py_errors']:,} | {s20['total_py_errors']:,} |")
    print(f"| Error rate | {s5['py_error_rate']:.2%} | {s20['py_error_rate']:.2%} |")
    print(f"| Calls / session (mean) | {s5['calls_per_session_mean']:.1f} | {s20['calls_per_session_mean']:.1f} |")
    print(f"| Calls / session (median) | {s5['calls_per_session_median']:.1f} | {s20['calls_per_session_median']:.1f} |")
    print(f"| Sessions with answer | {s5['answer_count']}/{s5['n']} ({s5['answer_rate']:.2%}) | {s20['answer_count']}/{s20['n']} ({s20['answer_rate']:.2%}) |")
    print(f"| Turns / session (mean) | {s5['turns_mean']:.1f} | {s20['turns_mean']:.1f} |")
    print(f"| Turns / session (median) | {s5['turns_median']:.0f} | {s20['turns_median']:.0f} |")
    print(f"| Context used — min | {s5['context_min']:,} | {s20['context_min']:,} |")
    print(f"| Context used — median | {s5['context_median']:,.0f} | {s20['context_median']:,.0f} |")
    print(f"| Context used — mean | {s5['context_mean']:,.0f} | {s20['context_mean']:,.0f} |")
    print(f"| Context used — max | {s5['context_max']:,} | {s20['context_max']:,} |")

    # ── 2) Termination reasons bar chart ───────────────────────
    # Normalize reason labels
    def simplify_reason(r: str) -> str:
        if r.startswith("exception:HarmonyError"):
            return "HarmonyError"
        return r

    tc5 = Counter()
    for r, c in s5["term_counts"].items():
        tc5[simplify_reason(r)] += c
    tc20 = Counter()
    for r, c in s20["term_counts"].items():
        tc20[simplify_reason(r)] += c

    all_reasons = sorted(
        set(tc5.keys()) | set(tc20.keys()),
        key=lambda r: -(tc5.get(r, 0) + tc20.get(r, 0)),
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(all_reasons))
    w = 0.38
    bars5 = [tc5.get(r, 0) for r in all_reasons]
    bars20 = [tc20.get(r, 0) for r in all_reasons]
    ax.bar([i - w / 2 for i in x], bars5, w, label="5 min", color="#5B9BD5")
    ax.bar([i + w / 2 for i in x], bars20, w, label="20 min", color="#ED7D31")
    ax.set_xticks(list(x))
    # wrap long labels
    wrapped = [r.replace("_", "\n") for r in all_reasons]
    ax.set_xticklabels(wrapped, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Termination reasons (5 min vs 20 min)")
    ax.legend()
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    path = PLOTS / "termination_reasons.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"\nSaved: {path}")

    # ── 3) Turns-per-session bar chart ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=False)
    for ax, turns, label in [
        (axes[0], s5["turns"], "5 min"),
        (axes[1], s20["turns"], "20 min"),
    ]:
        counts = Counter(turns)
        xs = sorted(counts.keys())
        ys = [counts[k] for k in xs]
        ax.bar(xs, ys, color="#5B9BD5" if label == "5 min" else "#ED7D31", width=0.8)
        ax.set_xlabel("Turns per session")
        ax.set_ylabel("Number of sessions")
        ax.set_title(f"Turns per session ({label})")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    path = PLOTS / "turns_per_session.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
