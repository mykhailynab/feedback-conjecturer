#!/usr/bin/env python3
"""
Scatter-plot comparing proof token count (from 'theorem' keyword) of two runs.

For Goedel: tokenize the last round's proof_text from 'theorem' onward.
For TIR: tokenize the last lean_final (or lean) tool call's code from 'theorem' onward.

For each problem_id, aggregates across retries:
  - Solved = any retry proved.  Proof tokens = shortest successful proof.
  - Unsolved = all retries failed.  Proof tokens = shortest last-attempt proof.

Dots are coloured by outcome:
  - Green:  both runs solved
  - Purple: run 1 solved only
  - Yellow: run 2 solved only
  - Grey:   neither solved

Usage:
    PYTHONPATH=. python analysis_and_inspection/for_paper/compare_proof_lengths.py
    PYTHONPATH=. python analysis_and_inspection/for_paper/compare_proof_lengths.py \
        --run1 goedel logs/run1/prove_results.jsonl "Goedel" \
        --run2 tir    logs/run2/prove_results.jsonl "TIR"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from analysis_and_inspection.for_paper.tokenize_utils import (
    aggregate_per_problem,
    goedel_proof_tokens,
    load_proved_status,
    tir_proof_tokens,
)

ROOT = Path(__file__).resolve().parents[2]
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)


def compute_proof_tokens(
    run_type: str,
    results_path: Path,
    tokenizer,
    keys: set[tuple[str, int]] | None = None,
) -> dict[tuple[str, int], int]:
    if run_type == "goedel":
        return goedel_proof_tokens(results_path, tokenizer, keys=keys)
    else:
        return tir_proof_tokens(results_path, tokenizer, keys=keys)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scatter-plot comparing proof token count of two prover runs.",
    )
    p.add_argument(
        "--run1", nargs=3, metavar=("TYPE", "PATH", "TITLE"),
        help='Run 1: type (goedel|tir), path to prove_results.jsonl, plot label.',
    )
    p.add_argument(
        "--run2", nargs=3, metavar=("TYPE", "PATH", "TITLE"),
        help='Run 2: type (goedel|tir), path to prove_results.jsonl, plot label.',
    )
    p.add_argument(
        "--goedel-tokenizer", type=str,
        default=str(ROOT / "tokenizers" / "goedel_prover_hf_tokenizer"),
    )
    p.add_argument(
        "--tir-tokenizer", type=str,
        default=str(ROOT / "tokenizers" / "Qwen3.6-35B-A3B"),
    )
    p.add_argument(
        "--no-neither", action="store_true",
        help="Exclude problems solved by neither run from the scatter plot.",
    )
    p.add_argument(
        "--aggregate-by-attempts", action="store_true",
        help="One dot per (problem_id, attempt) instead of aggregating retries per problem.",
    )
    p.add_argument(
        "--only-verified", action="store_true",
        help="Only treat attempts listed in each run's verified directory "
             "(included_attempts.jsonl) as proved.",
    )
    p.add_argument(
        "--output", type=str, default=str(PLOTS / "compare_proof_lengths.pdf"),
    )
    args = p.parse_args()

    if args.run1 is None:
        args.run1 = [
            "goedel",
            str(ROOT / "logs" / "prove_formalizations_40K_20mins" / "prove_results.jsonl"),
            "Goedel-Prover 8bit",
        ]
        args._run1_verified = str(ROOT / "data" / "goedel_proofs_equiv")
    else:
        args._run1_verified = None
    if args.run2 is None:
        args.run2 = [
            "tir",
            str(ROOT / "logs" / "short_4x5000" / "prove_results.jsonl"),
            "Add-Informal",
            # str(ROOT / "logs" / "full_20mins_tir_pass1_no_strip" / "prove_results.jsonl"),
            # "TIR-Prover Qwen3.6 5bit (no strip)",
        ]
        args._run2_verified = str(ROOT / "data" / "tir_prover_equiv")
    else:
        args._run2_verified = None

    for run_arg, name in [(args.run1, "--run1"), (args.run2, "--run2")]:
        if run_arg[0] not in ("goedel", "tir"):
            p.error(f"{name} TYPE must be 'goedel' or 'tir', got '{run_arg[0]}'")

    return args


def _load_verified_keys(directory: str) -> set[tuple[str, int]]:
    """Load (problem_id, attempt) pairs from included_attempts.jsonl."""
    path = Path(directory) / "included_attempts.jsonl"
    keys = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            keys.add((r["problem_id"], r["attempt"]))
    return keys


def _apply_verified_filter(
    proved: dict[tuple[str, int], bool],
    verified: set[tuple[str, int]],
) -> dict[tuple[str, int], bool]:
    """Override proved status: only keys in verified are True."""
    return {k: (v and k in verified) for k, v in proved.items()}


def main():
    args = parse_args()

    r1_type, r1_path_str, r1_title = args.run1
    r2_type, r2_path_str, r2_title = args.run2
    r1_path, r2_path = Path(r1_path_str), Path(r2_path_str)

    # Load proved status first (cheap) to find shared problem_ids
    print("Loading proved status...")
    r1_proved = load_proved_status(r1_path)
    r2_proved = load_proved_status(r2_path)

    if args.only_verified:
        if args._run1_verified:
            vk = _load_verified_keys(args._run1_verified)
            r1_proved = _apply_verified_filter(r1_proved, vk)
            print(f"  Run 1: filtered to {len(vk)} verified attempts from {args._run1_verified}")
        if args._run2_verified:
            vk = _load_verified_keys(args._run2_verified)
            r2_proved = _apply_verified_filter(r2_proved, vk)
            print(f"  Run 2: filtered to {len(vk)} verified attempts from {args._run2_verified}")

    r1_pids = {pid for pid, _ in r1_proved}
    r2_pids = {pid for pid, _ in r2_proved}
    shared_pids = r1_pids & r2_pids
    print(f"  Run 1: {len(r1_pids)} problems, Run 2: {len(r2_pids)} problems, shared: {len(shared_pids)}")

    if args.aggregate_by_attempts:
        shared_keys = set(r1_proved) & set(r2_proved)
        if args.no_neither:
            all_shared = shared_keys
            shared_keys = {
                k for k in shared_keys if r1_proved[k] or r2_proved[k]
            }
            print(f"  --no-neither: skipped {len(all_shared) - len(shared_keys)} neither-solved attempts")
        r1_keys = shared_keys
        r2_keys = shared_keys
    else:
        if args.no_neither:
            r1_solved_pids = {pid for (pid, _), v in r1_proved.items() if v}
            r2_solved_pids = {pid for (pid, _), v in r2_proved.items() if v}
            relevant_pids = shared_pids & (r1_solved_pids | r2_solved_pids)
            print(f"  --no-neither: {len(shared_pids) - len(relevant_pids)} neither-solved problems skipped")
        else:
            relevant_pids = shared_pids
        r1_keys = {k for k in r1_proved if k[0] in relevant_pids}
        r2_keys = {k for k in r2_proved if k[0] in relevant_pids}

    # Load tokenizers (only those needed)
    need_goedel = r1_type == "goedel" or r2_type == "goedel"
    need_tir = r1_type == "tir" or r2_type == "tir"
    goedel_tok = tir_tok = None
    if need_goedel:
        print("Loading Goedel tokenizer...")
        goedel_tok = AutoTokenizer.from_pretrained(args.goedel_tokenizer)
    if need_tir:
        print("Loading TIR tokenizer...")
        tir_tok = AutoTokenizer.from_pretrained(args.tir_tokenizer)

    def get_tokenizer(run_type: str):
        return goedel_tok if run_type == "goedel" else tir_tok

    print(f"Tokenizing proofs for run 1 ({r1_title}, {len(r1_keys)} attempts)...")
    r1_lengths = compute_proof_tokens(r1_type, r1_path, get_tokenizer(r1_type), keys=r1_keys)

    print(f"Tokenizing proofs for run 2 ({r2_title}, {len(r2_keys)} attempts)...")
    r2_lengths = compute_proof_tokens(r2_type, r2_path, get_tokenizer(r2_type), keys=r2_keys)

    # Classify and collect scatter data
    both_x, both_y = [], []
    r1_only_x, r1_only_y = [], []
    r2_only_x, r2_only_y = [], []
    neither_x, neither_y = [], []

    if args.aggregate_by_attempts:
        for k in sorted(shared_keys):
            if k not in r1_lengths or k not in r2_lengths:
                continue
            s1, s2 = r1_proved[k], r2_proved[k]
            l1, l2 = r1_lengths[k], r2_lengths[k]
            if s1 and s2:
                both_x.append(l1); both_y.append(l2)
            elif s1:
                r1_only_x.append(l1); r1_only_y.append(l2)
            elif s2:
                r2_only_x.append(l1); r2_only_y.append(l2)
            else:
                neither_x.append(l1); neither_y.append(l2)
    else:
        r1_agg = aggregate_per_problem(r1_lengths, r1_proved)
        r2_agg = aggregate_per_problem(r2_lengths, r2_proved)
        for pid in sorted(shared_pids):
            if pid not in r1_agg or pid not in r2_agg:
                continue
            s1, l1 = r1_agg[pid]
            s2, l2 = r2_agg[pid]
            if s1 and s2:
                both_x.append(l1); both_y.append(l2)
            elif s1:
                r1_only_x.append(l1); r1_only_y.append(l2)
            elif s2:
                r2_only_x.append(l1); r2_only_y.append(l2)
            else:
                neither_x.append(l1); neither_y.append(l2)

    n_both, n_r1, n_r2, n_neither = len(both_x), len(r1_only_x), len(r2_only_x), len(neither_x)
    unit = "attempts" if args.aggregate_by_attempts else "problems"
    print(f"\nResults on {n_both + n_r1 + n_r2 + n_neither} shared {unit}:")
    print(f"  Both solved:       {n_both}")
    print(f"  {r1_title} only:   {n_r1}")
    print(f"  {r2_title} only:   {n_r2}")
    print(f"  Neither:           {n_neither}")

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))

    if neither_x and not args.no_neither:
        ax.scatter(neither_x, neither_y, c="grey", alpha=0.4, s=20,
                   label=f"Neither (n={n_neither})", zorder=1)
    if r1_only_x:
        ax.scatter(r1_only_x, r1_only_y, c="#9B59B6", alpha=0.7, s=40,
                   label=f"{r1_title}\nonly (n={n_r1})", zorder=2)
    if r2_only_x:
        ax.scatter(r2_only_x, r2_only_y, c="#F1C40F", alpha=0.7, s=40,
                   edgecolors="#B7950B", linewidths=0.5,
                   label=f"{r2_title}\nonly (n={n_r2})", zorder=2)
    if both_x:
        ax.scatter(both_x, both_y, c="#27AE60", alpha=0.7, s=40,
                   label=f"Both solved (n={n_both})", zorder=3)

    # Diagonal reference line
    all_vals = both_x + both_y + r1_only_x + r1_only_y + r2_only_x + r2_only_y
    if not args.no_neither:
        all_vals += neither_x + neither_y
    if all_vals:
        lo, hi = 0, max(all_vals) * 1.05
        ax.plot([lo, hi], [lo, hi], ls="--", c="black", alpha=0.2, linewidth=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_xlabel(f"Proof tokens — {r1_title}", fontsize=13)
    ax.set_ylabel(f"Proof tokens — {r2_title}", fontsize=13)
    subtitle = "by attempt" if args.aggregate_by_attempts else "per problem, best"
    ax.set_title(f"Proof token count comparison ({subtitle})", fontsize=15)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=11, loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 0.95, 1])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
