import json
import argparse
from pathlib import Path
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

KNOWN_REASONS = {
    "no_tokens",
    "max_turns_exhausted",
    "deadline_exceeded",
    "proved",
    "exception:APIError",
    "token_limit",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prove-results-path", type=str, required=True)
    p.add_argument(
        "--tir-tokenizer", type=str,
        default=str(ROOT / "tokenizers" / "Qwen3.6-35B-A3B"),
    )
    args = p.parse_args()
    args.prove_results_path = Path(args.prove_results_path)

    tir_tok = AutoTokenizer.from_pretrained(args.tir_tokenizer)

    # Load termination reasons per (problem_id, attempt)
    term_reason_counts = {}
    status_counts = {}
    term_reason_map: dict[tuple[str, int], str] = {}
    result_lines = open(args.prove_results_path, "r").readlines()
    total_attempts = len(result_lines)
    incomplete_count = 0
    no_proof_result_count = 0
    skipped_count = 0
    skipped_is_null_count = 0
    skip_reason_is_null = 0
    skip_reason_counts = {}
    for line in result_lines:
        rec = json.loads(line)
        term_reason = (rec.get("proof_result") or {}).get("termination_reason", "None")
        term_reason_counts.setdefault(term_reason, 0)
        term_reason_counts[term_reason] += 1
        status_counts.setdefault(rec["status"], 0)
        status_counts[rec["status"]] += 1
        if rec['incomplete']:
            incomplete_count += 1
        if rec['proof_result'] is None:
            no_proof_result_count += 1
        if 'skipped' not in rec:
            skipped_is_null_count += 1
        else:
            if rec['skipped']:
                skipped_count += 1
        if 'skip_reason' not in rec:
            skip_reason_is_null += 1
        else:
            skip_reason_counts.setdefault(rec['skip_reason'], 0)
            skip_reason_counts[rec['skip_reason']] += 1
        term_reason_map[(rec["problem_id"], rec["attempt"])] = term_reason

    print(f"{incomplete_count} / {total_attempts} are incomplete")
    print(f"{no_proof_result_count} / {total_attempts} have no proof result")
    print(f"{skipped_is_null_count} / {total_attempts} where skipped is null")
    print(f"{skipped_count} / {total_attempts} where skipped is true")
    print(f"{skip_reason_is_null} / {total_attempts} without a skip reason")
    print(f"{term_reason_counts = }")
    print(f"{status_counts = }")
    print(f"{skip_reason_counts = }")

    proof_tokens_dict = tir_proof_tokens(args.prove_results_path, tir_tok)

    # --- Plot 1: histogram of proof token counts ---
    token_values = sorted(proof_tokens_dict.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(token_values, bins=50, edgecolor="black")
    ax.set_xlabel("Proof tokens")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of proof tokens per attempt")
    fig.tight_layout()
    fig.savefig(PLOTS / "proof_tokens_histogram.pdf")
    plt.close(fig)
    print(f"Saved {PLOTS / 'proof_tokens_histogram.pdf'}")

    # --- Plot 2: boxplot of proof tokens grouped by termination reason ---
    # Bucket each attempt into a known category or "others"
    other_reasons: set[str] = set()
    tokens_by_category: dict[str, list[int]] = {r: [] for r in KNOWN_REASONS}
    tokens_by_category["others"] = []
    for key, toks in proof_tokens_dict.items():
        reason = term_reason_map.get(key, "None")
        if reason in KNOWN_REASONS:
            tokens_by_category[reason].append(toks)
        else:
            tokens_by_category["others"].append(toks)
            other_reasons.add(reason)

    # Only keep categories with data, order by median descending
    labels_data = [
        (cat, vals) for cat, vals in tokens_by_category.items() if vals
    ]
    labels_data.sort(key=lambda x: -sorted(x[1])[len(x[1]) // 2])
    labels = [cat for cat, _ in labels_data]
    data = [vals for _, vals in labels_data]

    others_str = ", ".join(sorted(other_reasons)) if other_reasons else "none"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, tick_labels=labels, vert=True)
    ax.set_ylabel("Proof tokens")
    ax.set_title(f"Proof tokens by termination reason (others = {others_str})")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(PLOTS / "proof_tokens_by_reason.pdf")
    plt.close(fig)
    print(f"Saved {PLOTS / 'proof_tokens_by_reason.pdf'}")



if __name__ == "__main__":
    main()
