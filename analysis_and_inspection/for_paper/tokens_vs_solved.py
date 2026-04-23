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
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

GOEDEL_COLORS = ["#ED7D31", "#C55A11", "#A04000", "#7F3300"]
TIR_COLORS = ["#5B9BD5", "#2E75B6", "#1F4E79", "#0D3B66"]


def goedel_session_tokens(results_path: Path, tokenizer) -> dict[tuple[str, int], int]:
    """
    For each (problem_id, attempt), tokenize the last round's prompt + output.

    The prompt field already contains the fully-rendered chat template, so we
    tokenize it as raw text (add_special_tokens=False).  The peak context usage
    of a session is the last round, which includes all prior rounds' content.
    """
    result = {}
    with results_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            pid, att = d["problem_id"], d["attempt"]
            pr = d.get("proof_result")
            if not pr:
                continue
            rounds = pr.get("rounds") or []
            if not rounds:
                continue
            last = rounds[-1]
            prompt = last.get("prompt", "") or ""
            output = last.get("raw_output", "") or ""
            text = prompt + output
            tokens = len(tokenizer.encode(text, add_special_tokens=False))
            result[(pid, att)] = tokens
    return result


def tir_session_tokens(results_path: Path, tokenizer) -> dict[tuple[str, int], int]:
    """
    For each (problem_id, attempt), sum all conversation text and tokenize.

    TIR sessions record turns with thinking, content, and tool_calls (each with
    arguments and result).  We concatenate all text and tokenize.  The proved_lean
    field contains the initial theorem (system/user prompt content).
    """
    result = {}
    with results_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            pid, att = d["problem_id"], d["attempt"]
            pr = d.get("proof_result")
            if not pr:
                continue
            turns = pr.get("turns") or []
            if not turns:
                continue
            # Start with the input theorem (user prompt)
            parts = [d.get("proved_lean", "") or ""]
            for t in turns:
                parts.append(t.get("thinking", "") or "")
                parts.append(t.get("content", "") or "")
                for tc in t.get("tool_calls", []):
                    args = tc.get("arguments", {})
                    if isinstance(args, dict):
                        parts.append(json.dumps(args))
                    else:
                        parts.append(str(args))
                    parts.append(tc.get("result", "") or "")
            text = "\n".join(parts)
            tokens = len(tokenizer.encode(text, add_special_tokens=False))
            result[(pid, att)] = tokens
    return result


def load_proved_status(path: Path) -> dict[tuple[str, int], bool]:
    results = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            results[(d["problem_id"], d["attempt"])] = bool(d.get("proved"))
    return results


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
                "TIR-Prover Qwen3.6 5bit (strip)",
            ],
            [
                str(ROOT / "logs" / "full_20mins_tir_pass1_no_strip" / "prove_results.jsonl"),
                "TIR-Prover Qwen3.6 5bit (no strip)",
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
        print(f"Computing Goedel session tokens for '{title}'...")
        tokens = goedel_session_tokens(results_path, goedel_tok)
        proved = load_proved_status(results_path)
        n_proved = sum(1 for k, v in proved.items() if v and k in tokens)
        print(f"  {len(tokens)} sessions tokenized, {n_proved} proved with token data")

        xs, ys = build_curve(tokens, proved)
        print(f"  max tokens: {max(xs)}")
        color = GOEDEL_COLORS[i % len(GOEDEL_COLORS)]
        ax.plot(xs, ys, label=f"{title} (n={ys[-1]})", color=color, linewidth=1.5)
        ax.scatter(xs, ys, color=color)

    for i, (path_str, title) in enumerate(args.tir):
        results_path = Path(path_str)
        print(f"Computing TIR session tokens for '{title}'...")
        tokens = tir_session_tokens(results_path, qwen_tok)
        proved = load_proved_status(results_path)
        n_proved = sum(1 for k, v in proved.items() if v and k in tokens)
        print(f"  {len(tokens)} sessions tokenized, {n_proved} proved with token data")

        xs, ys = build_curve(tokens, proved)
        print(f"  max tokens: {max(xs)}")
        color = TIR_COLORS[i % len(TIR_COLORS)]
        ax.plot(xs, ys, label=f"{title} (n={ys[-1]})", color=color, linewidth=1.5)
        ax.scatter(xs, ys, color=color)

    ax.set_xlabel("Token budget (tokens used)")
    ax.set_ylabel("Attempts proved")
    ax.set_title("Proved attempts vs token budget")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
