#!/usr/bin/env python3
"""
Plot tokens used vs cumulative problems solved for Goedel and TIR provers.

For Goedel: tokenize the last round's (prompt + raw_output) per session using
the Goedel HF tokenizer — this gives the peak context window usage.

For TIR: sum all text content (system prompt, user messages, assistant thinking +
content, tool calls + results) across turns and tokenize with the Qwen tokenizer.

Usage:
    PYTHONPATH=. python analysis_and_inspection/for_paper/tokens_vs_solved.py
    PYTHONPATH=. python analysis_and_inspection/for_paper/tokens_vs_solved.py \
        --goedel logs/run1/prove_results.jsonl "Run 1" \
        --tir logs/tir1/prove_results.jsonl "TIR Run 1"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from analysis_and_inspection.for_paper.tokenize_utils import (
    aggregate_per_problem,
    goedel_session_tokens,
    load_proved_status,
    tir_session_tokens,
)

ROOT = Path(__file__).resolve().parents[2]
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

GOEDEL_COLORS = ["#ED7D31", "#C55A11", "#A04000", "#7F3300"]
TIR_COLORS = ["#5B9BD5", "#2E75B6", "#1F4E79", "#0D3B66"]


def build_curve(
    tokens_map: dict[tuple[str, int], int],
    proved_map: dict[tuple[str, int], bool],
) -> tuple[list[int], list[int]]:
    """x = token budget, y = cumulative attempts proved within that budget."""
    proved_tokens = sorted(
        tokens_map[k] for k, v in proved_map.items() if v and k in tokens_map
    )
    if not proved_tokens:
        return [0], [0]
    xs = [0] + proved_tokens
    ys = list(range(len(xs)))
    return xs, ys


def build_curve_from_aggregated(
    agg: dict[str, tuple[bool, int]],
) -> tuple[list[int], list[int]]:
    """x = token budget, y = cumulative problems proved within that budget."""
    proved_tokens = sorted(toks for solved, toks in agg.values() if solved)
    if not proved_tokens:
        return [0], [0]
    xs = [0] + proved_tokens
    ys = list(range(len(xs)))
    return xs, ys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot tokens used vs cumulative problems solved.",
    )
    p.add_argument(
        "--goedel", nargs=2, action="append", metavar=("PATH", "TITLE"),
        help="Goedel run: path to prove_results.jsonl and plot label. Repeatable.",
    )
    p.add_argument(
        "--tir", nargs=2, action="append", metavar=("PATH", "TITLE"),
        help="TIR run: path to prove_results.jsonl and plot label. Repeatable.",
    )
    p.add_argument(
        "--goedel-tokenizer", type=str,
        default=str(ROOT / "tokenizers" / "goedel_prover_hf_tokenizer"),
        help="Path to the Goedel HF tokenizer (default: tokenizers/goedel_prover_hf_tokenizer).",
    )
    p.add_argument(
        "--tir-tokenizer", type=str,
        default=str(ROOT / "tokenizers" / "Qwen3.6-35B-A3B"),
        help="Path to the TIR/Qwen HF tokenizer (default: tokenizers/Qwen3.6-35B-A3B).",
    )
    p.add_argument(
        "--aggregate-problems", action="store_true",
        help="Aggregate retries per problem (best successful token count). "
             "Curve shows problems solved instead of attempts solved.",
    )
    p.add_argument(
        "--output", type=str, default=str(PLOTS / "tokens_vs_solved.pdf"),
        help="Output plot path (default: plots/tokens_vs_solved.pdf).",
    )
    args = p.parse_args()

    # Defaults when no runs specified
    if args.goedel is None:
        args.goedel = [[
            str(ROOT / "logs" / "prove_formalizations_40K_20mins" / "prove_results.jsonl"),
            "Goedel-Prover 8bit",
        ]]
    if args.tir is None:
        args.tir = [
            # [
            #     str(ROOT / "logs" / "stripped_formalizations_goedel_pass_pass4x1_tir_qwen36_strip" / "prove_results.jsonl"),
            #     "TIR-Prover Qwen3.6 5bit (strip, on Goedel-proved)",
            # ],
            [
                str(ROOT / "logs" / "full_20mins_tir_pass1_strip" / "prove_results.jsonl"),
                "Base",
            ],
            [
                str(ROOT / "logs" / "full_20mins_tir_pass1_no_strip" / "prove_results.jsonl"),
                "Keep-CoT",
            ],
            [
                str(ROOT / "logs" / "short_4x5000" / "prove_results.jsonl"),
                "Add-Informal",
            ],
        ]

    return args


def main():
    args = parse_args()

    goedel_tok = None
    qwen_tok = None

    if args.goedel:
        print("Loading Goedel tokenizer...")
        goedel_tok = AutoTokenizer.from_pretrained(args.goedel_tokenizer)
    if args.tir:
        print("Loading TIR tokenizer...")
        qwen_tok = AutoTokenizer.from_pretrained(args.tir_tokenizer)

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, (path_str, title) in enumerate(args.goedel):
        results_path = Path(path_str)
        proved = load_proved_status(results_path)
        if args.aggregate_problems:
            # Need all attempts for per-problem aggregation
            proved_keys = {k for k, v in proved.items() if v}
            print(f"Computing Goedel session tokens for '{title}' ({len(proved)} attempts)...")
            tokens = goedel_session_tokens(results_path, goedel_tok, keys=proved_keys)
            agg = aggregate_per_problem(tokens, proved)
            xs, ys = build_curve_from_aggregated(agg)
        else:
            proved_keys = {k for k, v in proved.items() if v}
            print(f"Computing Goedel session tokens for '{title}' ({len(proved_keys)} proved attempts)...")
            tokens = goedel_session_tokens(results_path, goedel_tok, keys=proved_keys)
            xs, ys = build_curve(tokens, proved)
        print(f"  {ys[-1]} proved, max tokens: {max(xs)}")
        color = GOEDEL_COLORS[i % len(GOEDEL_COLORS)]
        ax.plot(xs, ys, label=f"{title} (n={ys[-1]})", color=color, linewidth=1.5)
        ax.hlines(y=ys[-1], xmin=xs[-1], xmax=40960, linestyle='--', color=color, linewidth=1.5)
        ax.scatter([40960], [ys[-1]], color=color)
        ax.scatter(xs, ys, color=color)

    for i, (path_str, title) in enumerate(args.tir):
        results_path = Path(path_str)
        proved = load_proved_status(results_path)
        if args.aggregate_problems:
            proved_keys = {k for k, v in proved.items() if v}
            print(f"Computing TIR session tokens for '{title}' ({len(proved)} attempts)...")
            tokens = tir_session_tokens(results_path, qwen_tok, keys=proved_keys)
            agg = aggregate_per_problem(tokens, proved)
            xs, ys = build_curve_from_aggregated(agg)
        else:
            proved_keys = {k for k, v in proved.items() if v}
            print(f"Computing TIR session tokens for '{title}' ({len(proved_keys)} proved attempts)...")
            tokens = tir_session_tokens(results_path, qwen_tok, keys=proved_keys)
            xs, ys = build_curve(tokens, proved)
        print(f"  {ys[-1]} proved, max tokens: {max(xs)}")
        color = TIR_COLORS[i % len(TIR_COLORS)]
        ax.plot(xs, ys, label=f"{title} (n={ys[-1]})", color=color, linewidth=1.5)
        ax.hlines(y=ys[-1], xmin=xs[-1], xmax=262144, linestyle='--', color=color, linewidth=1.5)
        ax.scatter([262144], [ys[-1]], color=color)
        ax.scatter(xs, ys, color=color)

    ax.set_xlabel("Token budget (tokens used)")
    unit = "Problems" if args.aggregate_problems else "Attempts"
    ax.set_ylabel(f"{unit} proved")
    ax.set_title(f"Proved {unit.lower()} vs token budget (pass@4)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5000, 60000)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
