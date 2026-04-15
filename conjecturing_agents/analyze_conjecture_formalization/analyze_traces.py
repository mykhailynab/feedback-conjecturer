#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from conjecturing_agents.tools import load_jsonl
from conjecturing_agents.lean_regex import ABBREV_RHS_RE as _ABBREV_RHS_RE
import math
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Helpers
# ============================================================


def short(x: Any, max_len: int = 160) -> str:
    if x is None:
        return "None"
    s = str(x).replace("\n", "\\n")
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"...[+{len(s) - max_len} chars]"


def normalize_math_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    s = s.replace("\u221a", "√")
    s = re.sub(r"\s+", "", s)
    return s


def extract_rhs_from_abbrev_declaration(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = _ABBREV_RHS_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def compute_exact_match(record: Dict[str, Any]) -> bool:
    final_rhs = normalize_math_text(
        extract_rhs_from_abbrev_declaration(record.get("final_abbrev_declaration"))
    )
    truth_rhs = normalize_math_text(record.get("ground_truth_extracted_answer"))
    return bool(final_rhs and truth_rhs and final_rhs == truth_rhs)


# ============================================================
# Failure mode classification
# ============================================================

def classify_compile_feedback(feedback: str) -> str:
    if not feedback:
        return "no_feedback"

    s = feedback.lower()

    if "no valid abbrev declaration was parsed" in s:
        return "no_valid_abbrev_output"
    if "failed to replace the scaffold abbrev" in s:
        return "scaffold_replacement_failed"
    if "[timeout]" in s or "timed out" in s:
        return "lean_timeout"
    if "unknown constant" in s or "unknown identifier" in s:
        return "unknown_identifier_or_constant"
    if "unexpected token" in s or "parser" in s or "expected token" in s:
        return "syntax_or_parser_error"
    if "application type mismatch" in s or "type mismatch" in s:
        return "type_mismatch"
    if "failed to synthesize" in s:
        return "typeclass_synthesis_failure"
    if "don't know how to synthesize placeholder" in s:
        return "placeholder_synthesis_failure"
    if "invalid field" in s:
        return "invalid_field"
    if "unsolved goals" in s:
        return "unsolved_goals"
    if "declaration uses 'sorry'" in s:
        return "unexpected_sorry_warning"
    if "invalid constructor" in s:
        return "invalid_constructor"
    if "function expected at" in s:
        return "function_expected"
    if "invalid argument name" in s:
        return "invalid_argument_name"
    if "ambiguous" in s:
        return "ambiguity"
    if "not a proposition" in s:
        return "not_a_proposition"
    if "invalid {...} notation" in s or "set notation" in s:
        return "set_notation_issue"
    return "other_compile_failure"


def classify_round_failure(round_record: Dict[str, Any]) -> str:
    abbrev_declaration = round_record.get("abbrev_declaration")
    if not abbrev_declaration:
        term = str(round_record.get("termination_reason") or "")
        if term:
            return f"no_abbrev::{term}"
        return "no_abbrev::unknown"

    compile_ok = round_record.get("compile_ok")
    if compile_ok is True:
        return "success"

    feedback = str(round_record.get("compile_formatted_diagnostics") or "")
    return classify_compile_feedback(feedback)


def classify_record(record: Dict[str, Any]) -> str:
    status = str(record.get("status", "unknown"))

    if status == "success":
        return "success_exact" if compute_exact_match(record) else "success_compile_only"

    if status == "skipped":
        return f"skipped::{record.get('skip_reason', 'unknown')}"

    rounds = record.get("rounds", []) or []
    if rounds:
        return classify_round_failure(rounds[-1])

    return "failed::no_rounds"


# ============================================================
# Example selection
# ============================================================

def summarize_example(record: Dict[str, Any]) -> str:
    rounds = record.get("rounds", []) or []
    final_round = rounds[-1] if rounds else {}
    return (
        f"id={record.get('problem_id')} "
        f"attempt={record.get('attempt')} "
        f"status={record.get('status')} "
        f"rounds={record.get('rounds_used')} "
        f"compile_ok={record.get('final_compile_ok')} "
        f"term={final_round.get('termination_reason')} "
        f"truth={short(record.get('ground_truth_extracted_answer'), 80)} "
        f"abbrev={short(record.get('final_abbrev_declaration'), 120)}"
    )


def build_example_catalog(
    records: List[Dict[str, Any]],
    *,
    limit_modes: int,
    success_examples: int,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    failure_groups: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
    success_pool: List[Dict[str, Any]] = []

    failure_counts = Counter()

    for rec in records:
        mode = classify_record(rec)
        if str(rec.get("status")) == "success":
            success_pool.append(rec)
        else:
            failure_counts[mode] += 1

    top_failure_modes = [mode for mode, _ in failure_counts.most_common(limit_modes)]

    for mode in top_failure_modes:
        failure_groups[mode] = [
            rec for rec in records
            if str(rec.get("status")) != "success" and classify_record(rec) == mode
        ]

    exact_successes = [r for r in success_pool if classify_record(r) == "success_exact"]
    compile_only_successes = [r for r in success_pool if classify_record(r) == "success_compile_only"]

    selected_successes = exact_successes[:success_examples]
    remaining = success_examples - len(selected_successes)
    if remaining > 0:
        selected_successes.extend(compile_only_successes[:remaining])

    catalog: List[Dict[str, Any]] = []
    by_id: Dict[int, Dict[str, Any]] = {}
    next_id = 1

    for mode, examples in failure_groups.items():
        for rec in examples:
            entry = {
                "analysis_id": next_id,
                "bucket": "failure",
                "mode": mode,
                "record": rec,
            }
            catalog.append(entry)
            by_id[next_id] = entry
            next_id += 1

    for rec in selected_successes:
        entry = {
            "analysis_id": next_id,
            "bucket": "success",
            "mode": classify_record(rec),
            "record": rec,
        }
        catalog.append(entry)
        by_id[next_id] = entry
        next_id += 1

    return catalog, by_id


# ============================================================
# Printing summary
# ============================================================

def print_failure_examples(
    catalog: List[Dict[str, Any]],
    *,
    limit_modes: int,
    examples_per_mode: int,
) -> None:
    print("=" * 80)
    print("FAILURE EXAMPLES")
    print("=" * 80)

    mode_to_entries: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
    for entry in catalog:
        if entry["bucket"] != "failure":
            continue
        mode_to_entries.setdefault(entry["mode"], []).append(entry)

    if not mode_to_entries:
        print("(none)")
        print()
        return

    shown = 0
    total_entries = sum([len(e) for e in mode_to_entries.values()])
    for mode, entries in mode_to_entries.items():
        print(f"{len(entries) / total_entries:.2%} === {mode} === [examples shown: {examples_per_mode} / {len(entries)}]")
        for entry in entries[:examples_per_mode]:
            rec = entry["record"]
            print(f"  [{entry['analysis_id']}] {summarize_example(rec)}")
        print()
        shown += 1
        if shown >= limit_modes:
            break


def print_success_examples(catalog: List[Dict[str, Any]]) -> None:
    print("=" * 80)
    print("SUCCESS EXAMPLES")
    print("=" * 80)

    entries = [e for e in catalog if e["bucket"] == "success"]
    if not entries:
        print("(none)")
        print()
        return

    for entry in entries:
        rec = entry["record"]
        print(f"  [{entry['analysis_id']}] mode={entry['mode']}  {summarize_example(rec)}")
    print()


# ============================================================
# Detailed trace printing
# ============================================================

def print_tool_call(tool_call: Dict[str, Any], idx: int) -> None:
    recipient = tool_call.get("recipient")
    print(f"    Tool call {idx}: recipient={recipient}")

    # Print commonly useful fields first if present.
    preferred_keys = [
        "request_text",
        "executed_code",
        "compiled_code",
        "output",
        "formatted_diagnostics",
        "ok",
        "timed_out",
        "elapsed_ms",
        "returncode",
        "relative_path",
        "source_path",
        "json_error_count",
        "json_warning_count",
        "sorry_warning_count",
    ]

    printed = set()
    for key in preferred_keys:
        if key in tool_call:
            printed.add(key)
            value = tool_call[key]
            if isinstance(value, str) and "\n" in value:
                print(f"      {key}:")
                print(value)
            else:
                print(f"      {key}: {value}")

    # Then print the remaining keys.
    for key, value in tool_call.items():
        if key in printed or key == "recipient":
            continue
        if isinstance(value, str) and "\n" in value:
            print(f"      {key}:")
            print(value)
        else:
            print(f"      {key}: {json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}")

def _render_tool_response_text(tool_call: Dict[str, Any]) -> str:
    """
    Best-effort reconstruction of the exact text the tool returned to the model.

    For python tool calls, this is usually stored under `output`.
    For lean tool calls, newer records may store `tool_response_text`.
    Otherwise, we reconstruct a readable approximation from the recorded fields.
    """
    if tool_call.get("tool_response_text"):
        return str(tool_call["tool_response_text"])

    if tool_call.get("output"):
        return str(tool_call["output"])

    parts: List[str] = []

    if "ok" in tool_call:
        ok = bool(tool_call.get("ok"))
        parts.append(f"[{'OK' if ok else 'ERROR'}] Lean compilation {'succeeded' if ok else 'failed'}.")

    if tool_call.get("relative_path"):
        parts.append(f"File: {tool_call['relative_path']}")
    if tool_call.get("returncode") is not None:
        parts.append(f"Return code: {tool_call['returncode']}")
    if tool_call.get("elapsed_ms") is not None:
        parts.append(f"Elapsed ms: {tool_call['elapsed_ms']}")
    if tool_call.get("json_error_count") is not None:
        parts.append(f"Errors: {tool_call['json_error_count']}")
    if tool_call.get("json_warning_count") is not None:
        parts.append(f"Warnings: {tool_call['json_warning_count']}")
    if tool_call.get("sorry_warning_count") is not None:
        parts.append(f"Sorry warnings: {tool_call['sorry_warning_count']}")

    if tool_call.get("formatted_diagnostics"):
        parts.append("")
        parts.append(str(tool_call["formatted_diagnostics"]))

    if tool_call.get("stderr"):
        parts.append("")
        parts.append("Raw stderr:")
        parts.append(str(tool_call["stderr"]))

    if tool_call.get("stdout"):
        parts.append("")
        parts.append("Raw stdout:")
        parts.append(str(tool_call["stdout"]))

    return "\n".join(parts).strip()


def _print_chat_block(role: str, text: Optional[str]) -> None:
    text = (text or "").strip()
    if not text:
        return
    print(f"[{role}]")
    print(text)
    print()


def _print_tool_exchange(tool_call: Dict[str, Any], idx: int) -> None:
    recipient = str(tool_call.get("recipient", "tool"))
    request_text = (
        tool_call.get("request_text")
        or tool_call.get("executed_code")
        or tool_call.get("compiled_code")
        or ""
    )
    response_text = _render_tool_response_text(tool_call)

    print(f"[assistant -> {recipient} #{idx}]")
    if request_text:
        print(str(request_text).strip())
    else:
        print("(no recorded request text)")
    print()

    print(f"[{recipient} -> assistant #{idx}]")
    if response_text:
        print(response_text)
    else:
        print("(no recorded tool response text)")
    print()


def _print_reconstructed_round_chat(round_record: Dict[str, Any]) -> None:
    raw_output = str(round_record.get("raw_output") or "")
    tool_calls = round_record.get("tool_calls", []) or []

    if not raw_output and not tool_calls:
        print("(no recorded assistant/tool trace)")
        return

    # Optional future fields: if later you persist them, this function will show them.
    if round_record.get("user_prompt"):
        _print_chat_block("user", round_record.get("user_prompt"))

    if round_record.get("error_feedback"):
        _print_chat_block("user (correction feedback)", round_record.get("error_feedback"))

    if "<|call|>" not in raw_output:
        # Fallback: no call markers, so just show assistant output, then any extra tools.
        _print_chat_block("assistant", raw_output)
        for i, tool_call in enumerate(tool_calls, start=1):
            _print_tool_exchange(tool_call, i)
        return

    parts = raw_output.split("<|call|>")

    # The convention is:
    #   assistant_text_0 <|call|> assistant_text_1 <|call|> assistant_text_2 ...
    # and each <|call|> corresponds to one tool invocation inserted between parts.
    for i, assistant_chunk in enumerate(parts):
        _print_chat_block("assistant", assistant_chunk)

        if i < len(tool_calls):
            _print_tool_exchange(tool_calls[i], i + 1)

    # If, for any reason, there are more tool calls than markers, show the extras.
    if len(tool_calls) > max(0, len(parts) - 1):
        for i in range(max(0, len(parts) - 1), len(tool_calls)):
            _print_tool_exchange(tool_calls[i], i + 1)


def print_round_trace(round_record: Dict[str, Any], idx: int) -> None:
    print("-" * 80)
    print(f"Round {idx}")
    print(f"  termination_reason      : {round_record.get('termination_reason')}")
    print(f"  abbrev_name             : {round_record.get('abbrev_name')}")
    print(f"  rhs                     : {round_record.get('rhs')}")
    print(f"  python_calls            : {round_record.get('python_calls')}")
    print(f"  python_errors           : {round_record.get('python_errors')}")
    print(f"  lean_calls              : {round_record.get('lean_calls')}")
    print(f"  lean_errors             : {round_record.get('lean_errors')}")
    print(f"  compile_ok              : {round_record.get('compile_ok')}")
    print(f"  compile_relative_path   : {round_record.get('compile_relative_path')}")
    print()

    if round_record.get("abbrev_declaration"):
        print("[parsed final abbrev]")
        print(round_record.get("abbrev_declaration"))
        print()

    print("[reconstructed transcript]")
    _print_reconstructed_round_chat(round_record)

    compile_diag = round_record.get("compile_formatted_diagnostics")
    if compile_diag:
        print("[external validator feedback]")
        print(compile_diag)
        print()

    assembled_lean = round_record.get("assembled_lean")
    if assembled_lean:
        print("[assembled lean]")
        print(assembled_lean)
        print()

def print_round_trace_old(round_record: Dict[str, Any], idx: int) -> None:
    print("-" * 80)
    print(f"Round {idx}")
    print(f"  termination_reason: {round_record.get('termination_reason')}")
    print(f"  abbrev_name       : {round_record.get('abbrev_name')}")
    print(f"  abbrev_declaration:")
    print(round_record.get("abbrev_declaration"))
    print(f"  rhs               : {round_record.get('rhs')}")
    print(f"  python_calls      : {round_record.get('python_calls')}")
    print(f"  python_errors     : {round_record.get('python_errors')}")
    print(f"  lean_calls        : {round_record.get('lean_calls')}")
    print(f"  lean_errors       : {round_record.get('lean_errors')}")
    print(f"  compile_ok        : {round_record.get('compile_ok')}")
    print(f"  compile_relative_path: {round_record.get('compile_relative_path')}")

    raw_output = round_record.get("raw_output")
    if raw_output:
        print("\n  raw_output:")
        print(raw_output)

    tool_calls = round_record.get("tool_calls", []) or []
    if tool_calls:
        print("\n  tool_calls:")
        for i, tool_call in enumerate(tool_calls, start=1):
            print_tool_call(tool_call, i)

    compile_diag = round_record.get("compile_formatted_diagnostics")
    if compile_diag:
        print("\n  compile_formatted_diagnostics:")
        print(compile_diag)

    assembled_lean = round_record.get("assembled_lean")
    if assembled_lean:
        print("\n  assembled_lean:")
        print(assembled_lean)


def print_detailed_record(entry: Dict[str, Any], print_raw: bool) -> None:
    rec = entry["record"]

    print("=" * 80)
    print(f"DETAILED ANALYSIS FOR ID {entry['analysis_id']}")
    print("=" * 80)
    print(f"bucket                    : {entry['bucket']}")
    print(f"mode                      : {entry['mode']}")
    print(f"problem_id                : {rec.get('problem_id')}")
    print(f"attempt                   : {rec.get('attempt')}")
    print(f"status                    : {rec.get('status')}")
    print(f"skip_reason               : {rec.get('skip_reason')}")
    print(f"rounds_used               : {rec.get('rounds_used')}")
    print(f"final_compile_ok          : {rec.get('final_compile_ok')}")
    print(f"final_compile_relative_path: {rec.get('final_compile_relative_path')}")
    print(f"started_ts                : {rec.get('started_ts')}")
    print(f"finished_ts               : {rec.get('finished_ts')}")
    print(f"elapsed_ms                : {rec.get('elapsed_ms')}")
    print(f"required_abbrev_name      : {rec.get('required_abbrev_name')}")
    print(f"extracted_row_name        : {rec.get('extracted_row_name')}")
    print(f"extracted_row_tags        : {rec.get('extracted_row_tags')}")
    print()

    print("attempt_answer:")
    print(rec.get("attempt_answer"))
    print()

    print("ground_truth_extracted_answer:")
    print(rec.get("ground_truth_extracted_answer"))
    print()

    print("final_abbrev_declaration:")
    print(rec.get("final_abbrev_declaration"))
    print()

    if rec.get("attempt_raw_output_tail"):
        print("attempt_raw_output_tail:")
        print(rec.get("attempt_raw_output_tail"))
        print()

    if rec.get("lean_statement_without_comment"):
        print("lean_statement_without_comment:")
        print(rec.get("lean_statement_without_comment"))
        print()

    if rec.get("final_compile_formatted_diagnostics"):
        print("final_compile_formatted_diagnostics:")
        print(rec.get("final_compile_formatted_diagnostics"))
        print()

    rounds = rec.get("rounds", []) or []
    if rounds:
        print("=" * 80)
        print("ROUND TRACE")
        print("=" * 80)
        for idx, rr in enumerate(rounds, start=1):
            print_round_trace(rr, idx)
            print()
    else:
        print("No round trace available.")

    if print_raw:
        print("=" * 80)
        print("RAW JSON")
        print("=" * 80)
        print(json.dumps(rec, ensure_ascii=False, indent=2))


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Browse failure and success examples from formalizations.jsonl."
    )
    p.add_argument(
        "formalizations_jsonl",
        help="Path to formalizations.jsonl",
    )
    p.add_argument(
        "--limit-modes",
        type=int,
        default=100,
        help="How many failure modes to print",
    )
    p.add_argument(
        "--examples-per-mode",
        type=int,
        default=3,
        help="How many examples to print per failure mode",
    )
    p.add_argument(
        "--success-examples",
        type=int,
        default=10,
        help="How many successful examples to print",
    )
    p.add_argument(
        "--analyze-id",
        type=int,
        default=None,
        help="Print the full trace for the selected example id",
    )
    p.add_argument(
        "--print-raw",
        action="store_true",
        help="Print the raw trace for the selected example id",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    path = Path(args.formalizations_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    records = load_jsonl(path)
    if not records:
        print("No records found.")
        return

    catalog, by_id = build_example_catalog(
        records,
        limit_modes=args.limit_modes,
        success_examples=args.success_examples,
    )

    print_failure_examples(catalog, limit_modes=args.limit_modes, examples_per_mode=args.examples_per_mode)
    print_success_examples(catalog)

    if args.analyze_id is not None:
        print()
        entry = by_id.get(args.analyze_id)
        if entry is None:
            print(f"[warn] No example with analysis id {args.analyze_id}.")
            available = sorted(by_id.keys())
            print(f"Available ids: {available}")
            return
        print_detailed_record(entry, args.print_raw)


if __name__ == "__main__":
    main()