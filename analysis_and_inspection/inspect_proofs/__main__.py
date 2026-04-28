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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prove-results-path", type=str, required=True)
    p.add_argument(
        "--tir-tokenizer", type=str,
        default=str(ROOT / "tokenizers" / "Qwen3.6-35B-A3B"),
    )
    args = p.parse_args()

    tir_tok = AutoTokenizer.from_pretrained(args.tir_tokenizer)

    term_reason_counts = {}
    for line in open(args.prove_results_path, "r").readlines():
        line = json.loads(line)
        term_reason = (line.get('proof_result') or {}).get('termination_reason', "None")
        term_reason_counts.setdefault(term_reason, 0)
        term_reason_counts[term_reason] += 1

    print(term_reason_counts)
    
    proof_tokens_dict = tir_proof_tokens(args.prove_results_path, tir_tok)

    proof_tokens_dict.values()

    # TODO: add a barplot here


if __name__ == "__main__":
    main()
