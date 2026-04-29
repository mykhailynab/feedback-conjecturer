import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from collections import defaultdict

from analysis_and_inspection.for_paper.tokenize_utils import (
    aggregate_per_problem,
    goedel_proof_tokens,
    load_proved_status,
    tir_proof_tokens,
    tir_session_tokens,
)
from conjecturing_agents.inference_backends.llamacpp_tir import parse_tool_calls_from_thinking

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
    "final_answer",
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
    term_reason_counts = defaultdict(int)
    status_counts = defaultdict(int)
    term_reason_map: dict[tuple[str, int], str] = {}
    result_lines = open(args.prove_results_path, "r").readlines()
    total_attempts = len(result_lines)
    incomplete_count = 0
    no_proof_result_count = 0
    skipped_count = 0
    skipped_is_null_count = 0
    skip_reason_is_null = 0
    skip_reason_counts = defaultdict(int)
    n_proved = 0
    proved_term_reason_counts = defaultdict(int)
    called_lean_final_count = 0
    called_lean_final_term_reason_counts = defaultdict(int)
    called_lean_count = 0
    called_lean_term_reason_counts = defaultdict(int)
    called_python_count = 0
    called_python_term_reason_counts = defaultdict(int)
    called_nothing_count = 0
    called_nothing_term_reason_counts = defaultdict(int)
    all_records = {}
    n_zero_turns = 0
    zero_turns_term_reason_counts = defaultdict(int)
    non_skipped_keys = []
    no_tokens_limited_count = 0
    unparsed_tool_call_count = 0
    unparsed_tool_call_no_start = 0
    thinking_tc_parsed_count = 0
    thinking_tc_parsed_total = 0
    no_tokens_other_reasons = 0
    for line in result_lines:
        rec = json.loads(line)
        rec_key = (rec["problem_id"], rec["attempt"])
        all_records[rec_key] = rec
        term_reason = (rec.get("proof_result") or {}).get("termination_reason", "None")
        term_reason_counts[term_reason] += 1
        status_counts[rec["status"]] += 1
        if rec['incomplete']:
            incomplete_count += 1
        if rec['proof_result'] is None:
            no_proof_result_count += 1
        else:
            proof_result = rec['proof_result']
            n_proved += proof_result['proved']
            if proof_result['proved']:
                proved_term_reason_counts[term_reason] += 1
            called_lean_final = any(any(tc['name'] == "lean_final" for tc in t['tool_calls']) for t in proof_result['turns'])
            if called_lean_final:
                called_lean_final_term_reason_counts[term_reason] += 1
            called_lean_final_count += called_lean_final
            called_lean = any(any(tc['name'] == "lean" for tc in t['tool_calls']) for t in proof_result['turns'])
            if called_lean:
                called_lean_term_reason_counts[term_reason] += 1
            called_lean_count += called_lean
            called_python = any(any(tc['name'] == "python" for tc in t['tool_calls']) for t in proof_result['turns'])
            if called_python:
                called_python_term_reason_counts[term_reason] += 1
            called_python_count += called_python
            called_nothing = not called_lean and not called_lean and not called_lean_final
            called_nothing_count += called_nothing
            if called_nothing:
                called_nothing_term_reason_counts[term_reason] += 1
            if len(proof_result['turns']) == 0:
                n_zero_turns += 1
                zero_turns_term_reason_counts[term_reason] += 1

        if term_reason == "no_tokens":
            assert proof_result['turns'][-1]['content'] == ""
            assert proof_result['turns'][-1]['tool_calls'] == []
            last_thinking = proof_result['turns'][-1]['thinking']
            if last_thinking.endswith("</tool_call>"):
                unparsed_tool_call_count += 1
                tool_call_start = last_thinking.find("<tool_call>")
                if tool_call_start == -1:
                    unparsed_tool_call_no_start += 1
                else:
                    parsed_tcs = parse_tool_calls_from_thinking(last_thinking)
                    if parsed_tcs:
                        thinking_tc_parsed_count += 1
                        thinking_tc_parsed_total += len(parsed_tcs)
            elif len(tir_tok.encode(last_thinking)) > (16384 - 16):  # include buffer
                no_tokens_limited_count += 1
            else:
                no_tokens_other_reasons += 1
        if 'skipped' not in rec:
            skipped_is_null_count += 1
            non_skipped_keys.append(rec_key)
        else:
            if rec['skipped']:
                skipped_count += 1
        if 'skip_reason' not in rec:
            skip_reason_is_null += 1
        else:
            skip_reason_counts[rec['skip_reason']] += 1
        term_reason_map[(rec["problem_id"], rec["attempt"])] = term_reason
    print(f"{incomplete_count} / {total_attempts} are incomplete")
    print(f"{no_proof_result_count} / {total_attempts} have no proof result")
    print()
    print(f"{skipped_is_null_count} / {total_attempts} where skipped is null")
    print(f"{skipped_count} / {total_attempts} where skipped is true")
    print(f"{skip_reason_is_null} / {total_attempts} without a skip reason")
    print(f"{skip_reason_is_null} / {total_attempts} without a skip reason")
    print()
    print(f"{called_lean_final_count} / {total_attempts} called lean_final")
    print(f"called_lean_final_term_reason_counts = {dict(called_lean_final_term_reason_counts)}")
    print(f"{called_lean_count} / {total_attempts} called lean")
    print(f"called_lean_term_reason_counts = {dict(called_lean_term_reason_counts)}")
    print(f"{called_python_count} / {total_attempts} called python")
    print(f"called_python_term_reason_counts = {dict(called_python_term_reason_counts)}")
    print(f"{called_nothing_count} / {total_attempts} called nothing")
    print(f"called_nothing_term_reason_counts = {dict(called_nothing_term_reason_counts)}")
    print()
    print(f"{n_zero_turns} / {total_attempts} had 0 turns")
    print(f"zero_turns_term_reason_counts = {dict(zero_turns_term_reason_counts)}")
    print()
    print(f"{no_tokens_limited_count} / {term_reason_counts['no_tokens']} were token-limited")
    print(f"{unparsed_tool_call_count} / {term_reason_counts['no_tokens']} had unparsed tool calls")
    print(f"  {unparsed_tool_call_no_start} / {unparsed_tool_call_count} had no <tool_call> start tag")
    print(f"  {thinking_tc_parsed_count} / {unparsed_tool_call_count - unparsed_tool_call_no_start} were parseable ({thinking_tc_parsed_total} tool calls total)")
    print(f"{no_tokens_other_reasons} / {term_reason_counts['no_tokens']} had no tokens for other reasons")
    print()
    print(f"proved_term_reason_counts = {dict(proved_term_reason_counts)}")
    print(f"term_reason_counts = {dict(term_reason_counts)}")
    print(f"status_counts = {dict(status_counts)}")
    print(f"skip_reason_counts = {dict(skip_reason_counts)}")
    print()

    # raise SystemExit(0) 
    # NOTE: tir_session_tokens does not work properly for V1
    # NOTE: the conversion of turns to messages is not possible since we don't have system/user messages
    # NOTE: even if we add them, apply_chat_template still fails for some reason.
    # NOTE: I reverted the tir_session_tokens to have the old V1 schema hack w/o turns_to_chat
    # TODO: Indicate that we use a hack in case of a V1 schema. Use the conversation history otherwise.

    proof_tokens_dict = tir_session_tokens(args.prove_results_path, tir_tok, exclude_last_tool_call_result=True, keys=non_skipped_keys)
    assert len(proof_tokens_dict) == (total_attempts - skipped_count)

    min_tokens = min([v for k, v in proof_tokens_dict.items() if v != 0])
    for k, v in proof_tokens_dict.items():
        rec = all_records[k]
        proof_result = rec.get("proof_result") or {}
        term_reason = proof_result.get("termination_reason", "None")
        if v == min_tokens:
            print("Shortest non-zero attempt:", k)
            print(f"Term reason: {term_reason}")
            # print(f"Turns: {proof_result['turns']}")
        # if term_reason == "no_tokens":
        #     assert proof_result['turns'][-1]['content'] == ""
        #     assert proof_result['turns'][-1]['tool_calls'] == []
        #     print(proof_result['turns'][-1]['thinking'])
        #     print("==========================================================================================")

    proof_tokens_no_turns = defaultdict(int)
    for k, rec in all_records.items():
        proof_result = rec.get("proof_result") or {}
        if not proof_result:
            continue
        if len(proof_result['turns']) == 0:
            proof_tokens_no_turns[k] = proof_tokens_dict[k]
    if proof_tokens_no_turns:
        print(f"{min(proof_tokens_no_turns.values()) = }")

    print()

    # --- Plot 1: histogram of proof token counts ---
    token_values = sorted(proof_tokens_dict.values())
    print(f"")
    print(f"Min proof tokens: {min(token_values)}")
    print(f"Proof tokens=0: {token_values.count(0)} / {len(token_values)}")
    print(f"Max proof tokens: {max(token_values)}")
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
    labels = [f"{cat} ({term_reason_counts[cat]})" for cat, _ in labels_data]
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
