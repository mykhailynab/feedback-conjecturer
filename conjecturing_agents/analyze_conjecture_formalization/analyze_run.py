#!/usr/bin/env python3
from __future__ import annotations

import re
import json
import math

from conjecturing_agents.tools import load_jsonl
from conjecturing_agents.lean_regex import ABBREV_RHS_RE as _ABBREV_RHS_RE
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional


# ============================================================
# Helpers
# ============================================================


def extract_rhs_from_abbrev_declaration(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = _ABBREV_RHS_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()


def normalize_math_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    s = s.replace("\u221a", "√")
    s = re.sub(r"\s+", "", s)
    return s


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return default
        return val
    except Exception:
        return default


def pct(n: int, d: int) -> str:
    if d == 0:
        return "0.00%"
    return f"{100.0 * n / d:.2f}%"


def short(s: Any, max_len: int = 140) -> str:
    if s is None:
        return "None"
    t = str(s).replace("\n", "\\n")
    if len(t) <= max_len:
        return t
    return t[:max_len] + f"...[+{len(t) - max_len} chars]"


def print_counter(
    title: str,
    counter: Counter,
    *,
    total: Optional[int] = None,
    limit: int = 20,
    indent: str = "  ",
) -> None:
    print(title)
    if not counter:
        print(f"{indent}(none)")
        return
    for key, count in counter.most_common(limit):
        if total is None:
            print(f"{indent}{count:>6}  {key}")
        else:
            print(f"{indent}{count:>6}  {pct(count, total):>8}  {key}")


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


# ============================================================
# Formalization analysis
# ============================================================

def analyze_formalizations(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)

    status_counter = Counter()
    skip_reason_counter = Counter()
    final_failure_mode_counter = Counter()
    round_failure_mode_counter = Counter()
    final_termination_counter = Counter()
    round_termination_counter = Counter()
    rounds_used_counter = Counter()
    tag_counter = Counter()

    success_compile_and_exact = 0
    success_compile_but_not_exact = 0
    success_compile_total = 0

    elapsed_ms_all: List[int] = []
    elapsed_ms_success: List[int] = []
    elapsed_ms_failed: List[int] = []

    python_call_total = 0
    python_error_total = 0
    lean_call_total = 0
    lean_error_total = 0

    rounds_total = 0
    rounds_with_compile = 0
    rounds_compile_success = 0

    no_rounds_failed_records = 0

    example_final_failures: Dict[str, List[str]] = defaultdict(list)

    for rec in records:
        status = str(rec.get("status", "unknown"))
        status_counter[status] += 1

        elapsed_ms = safe_int(rec.get("elapsed_ms"), default=0)
        if elapsed_ms > 0:
            elapsed_ms_all.append(elapsed_ms)
            if status == "success":
                elapsed_ms_success.append(elapsed_ms)
            elif status == "failed":
                elapsed_ms_failed.append(elapsed_ms)

        for tag in rec.get("extracted_row_tags", []) or []:
            tag_counter[str(tag)] += 1

        if status == "skipped":
            skip_reason_counter[str(rec.get("skip_reason", "unknown"))] += 1

        rounds = rec.get("rounds", []) or []
        rounds_used_counter[safe_int(rec.get("rounds_used"), default=len(rounds))] += 1

        if not rounds and status == "failed":
            no_rounds_failed_records += 1

        if rounds:
            final_round = rounds[-1]
            term = str(final_round.get("termination_reason") or "")
            if term:
                final_termination_counter[term] += 1

            final_mode = classify_round_failure(final_round)
            if status != "success":
                final_failure_mode_counter[final_mode] += 1
                if len(example_final_failures[final_mode]) < 3:
                    example_final_failures[final_mode].append(
                        f"id={rec.get('problem_id')} attempt={rec.get('attempt')} "
                        f"abbrev={short(final_round.get('abbrev_declaration'))} "
                        f"diag={short(final_round.get('compile_formatted_diagnostics'))}"
                    )

        for rr in rounds:
            rounds_total += 1

            rr_term = str(rr.get("termination_reason") or "")
            if rr_term:
                round_termination_counter[rr_term] += 1

            python_call_total += safe_int(rr.get("python_calls"), 0)
            python_error_total += safe_int(rr.get("python_errors"), 0)
            lean_call_total += safe_int(rr.get("lean_calls"), 0)
            lean_error_total += safe_int(rr.get("lean_errors"), 0)

            if rr.get("compile_ok") is not None:
                rounds_with_compile += 1
                if rr.get("compile_ok") is True:
                    rounds_compile_success += 1

            r_failure = classify_round_failure(rr)
            if r_failure != "success":
                round_failure_mode_counter[r_failure] += 1

        if rec.get("final_compile_ok") is True:
            success_compile_total += 1
            final_rhs = normalize_math_text(extract_rhs_from_abbrev_declaration(rec.get("final_abbrev_declaration")))
            truth_rhs = normalize_math_text(rec.get("ground_truth_extracted_answer"))
            if final_rhs and truth_rhs and final_rhs == truth_rhs:
                success_compile_and_exact += 1
            else:
                success_compile_but_not_exact += 1

    avg_elapsed_all = (sum(elapsed_ms_all) / len(elapsed_ms_all)) if elapsed_ms_all else 0.0
    avg_elapsed_success = (sum(elapsed_ms_success) / len(elapsed_ms_success)) if elapsed_ms_success else 0.0
    avg_elapsed_failed = (sum(elapsed_ms_failed) / len(elapsed_ms_failed)) if elapsed_ms_failed else 0.0

    return {
        "total": total,
        "status_counter": status_counter,
        "skip_reason_counter": skip_reason_counter,
        "final_failure_mode_counter": final_failure_mode_counter,
        "round_failure_mode_counter": round_failure_mode_counter,
        "final_termination_counter": final_termination_counter,
        "round_termination_counter": round_termination_counter,
        "rounds_used_counter": rounds_used_counter,
        "tag_counter": tag_counter,
        "success_compile_total": success_compile_total,
        "success_compile_and_exact": success_compile_and_exact,
        "success_compile_but_not_exact": success_compile_but_not_exact,
        "elapsed_ms_all": elapsed_ms_all,
        "elapsed_ms_success": elapsed_ms_success,
        "elapsed_ms_failed": elapsed_ms_failed,
        "avg_elapsed_all": avg_elapsed_all,
        "avg_elapsed_success": avg_elapsed_success,
        "avg_elapsed_failed": avg_elapsed_failed,
        "python_call_total": python_call_total,
        "python_error_total": python_error_total,
        "lean_call_total": lean_call_total,
        "lean_error_total": lean_error_total,
        "rounds_total": rounds_total,
        "rounds_with_compile": rounds_with_compile,
        "rounds_compile_success": rounds_compile_success,
        "no_rounds_failed_records": no_rounds_failed_records,
        "example_final_failures": example_final_failures,
    }


# ============================================================
# Event analysis
# ============================================================

def analyze_events(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    event_counter = Counter()
    warn_messages = Counter()
    formalization_done_status = Counter()
    run_end_payloads: List[Dict[str, Any]] = []

    for rec in records:
        event = str(rec.get("event", "unknown"))
        event_counter[event] += 1

        if event == "warn":
            warn_messages[str(rec.get("msg", "unknown"))] += 1

        if event == "formalization_done":
            formalization_done_status[str(rec.get("status", "unknown"))] += 1

        if event == "run_end":
            run_end_payloads.append(rec)

    return {
        "event_counter": event_counter,
        "warn_messages": warn_messages,
        "formalization_done_status": formalization_done_status,
        "run_end_payloads": run_end_payloads,
    }


# ============================================================
# Reporting
# ============================================================

def print_elapsed_stats(label: str, values: List[int]) -> None:
    print(label)
    if not values:
        print("  (none)")
        return
    values_sorted = sorted(values)
    print(f"  count:   {len(values_sorted)}")
    print(f"  mean:    {sum(values_sorted) / len(values_sorted):.1f} ms")
    print(f"  median:  {median(values_sorted):.1f} ms")
    print(f"  min:     {values_sorted[0]} ms")
    print(f"  max:     {values_sorted[-1]} ms")


def print_examples_by_mode(example_final_failures: Dict[str, List[str]], limit_modes: int = 10) -> None:
    print("Example failed attempts by final failure mode")
    if not example_final_failures:
        print("  (none)")
        return
    shown = 0
    for mode, examples in example_final_failures.items():
        print(f"  {mode}")
        for ex in examples[:3]:
            print(f"    - {ex}")
        shown += 1
        if shown >= limit_modes:
            break


def print_formalization_report(stats: Dict[str, Any]) -> None:
    total = stats["total"]
    status_counter: Counter = stats["status_counter"]

    print("=" * 80)
    print("CONJECTURE FORMALIZATION ANALYSIS")
    print("=" * 80)
    print(f"Total records: {total}")
    print()

    print("Top-level outcomes")
    for status in ["success", "failed", "skipped"]:
        count = status_counter.get(status, 0)
        print(f"  {status:>18}: {count:>6}  {pct(count, total)}")
    for status, count in status_counter.items():
        if status not in {"success", "failed", "skipped"}:
            print(f"  {status:>18}: {count:>6}  {pct(count, total)}")
    print()

    success_compile_total = stats["success_compile_total"]
    success_compile_and_exact = stats["success_compile_and_exact"]
    success_compile_but_not_exact = stats["success_compile_but_not_exact"]

    print("Compile-success quality")
    print(f"  compile-success total:             {success_compile_total:>6}  {pct(success_compile_total, total)}")
    print(f"  exact RHS match to ground truth:   {success_compile_and_exact:>6}  {pct(success_compile_and_exact, total)}")
    print(f"  compile-success but RHS differs:   {success_compile_but_not_exact:>6}  {pct(success_compile_but_not_exact, total)}")
    print()
    print(f"  compile-success among non-skipped: {status_counter['success']:>6} / {status_counter['success'] + status_counter['failed']:>6}  {pct(status_counter['success'], status_counter['success'] + status_counter['failed'])}")
    if success_compile_total:
        print(f"  exact among compile-success:       {success_compile_and_exact:>6} / {success_compile_total:>6}  {pct(success_compile_and_exact, success_compile_total)}")
    print()

    print_counter("Skip reasons", stats["skip_reason_counter"], total=status_counter.get("skipped", 0), limit=20)
    print()
    print_counter("Final failure modes", stats["final_failure_mode_counter"], total=status_counter.get("failed", 0), limit=20)
    print()
    print_counter("Round-level failure modes", stats["round_failure_mode_counter"], total=stats["rounds_total"] - stats["rounds_compile_success"], limit=25)
    print()
    print_counter("Final round termination reasons", stats["final_termination_counter"], total=status_counter.get("failed", 0) + status_counter.get("success", 0), limit=20)
    print()
    print_counter("All round termination reasons", stats["round_termination_counter"], total=stats["rounds_total"], limit=25)
    print()
    print_counter("Rounds used", stats["rounds_used_counter"], total=total, limit=20)
    print()

    rounds_total = stats["rounds_total"]
    rounds_with_compile = stats["rounds_with_compile"]
    rounds_compile_success = stats["rounds_compile_success"]
    print("Round-level tool usage")
    print(f"  total rounds:                  {rounds_total}")
    print(f"  rounds with compile result:    {rounds_with_compile}")
    print(f"  compile-success rounds:        {rounds_compile_success}  {pct(rounds_compile_success, rounds_with_compile)}")
    print(f"  python calls total:            {stats['python_call_total']}")
    print(f"  python errors total:           {stats['python_error_total']}")
    print(f"  lean calls total:              {stats['lean_call_total']}")
    print(f"  lean errors total:             {stats['lean_error_total']}")
    print()

    print_elapsed_stats("Elapsed time: all records", stats["elapsed_ms_all"])
    print()
    print_elapsed_stats("Elapsed time: successes", stats["elapsed_ms_success"])
    print()
    print_elapsed_stats("Elapsed time: failures", stats["elapsed_ms_failed"])
    print()

    print_counter("Most common tags", stats["tag_counter"], total=total, limit=20)
    print()

    print(f"Failed records with zero rounds: {stats['no_rounds_failed_records']}")
    print()

    print_examples_by_mode(stats["example_final_failures"])
    print()


def print_event_report(stats: Dict[str, Any], total_formalizations: int) -> None:
    print("=" * 80)
    print("EVENT LOG ANALYSIS")
    print("=" * 80)
    print_counter("Event counts", stats["event_counter"], limit=50)
    print()
    print_counter("Warning messages", stats["warn_messages"], limit=20)
    print()
    print_counter("formalization_done statuses", stats["formalization_done_status"], total=total_formalizations, limit=20)
    print()

    run_end_payloads = stats["run_end_payloads"]
    if run_end_payloads:
        last = run_end_payloads[-1]
        print("Last run_end payload")
        for k in sorted(last.keys()):
            if k == "event":
                continue
            print(f"  {k}: {last[k]}")
        print()


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse conjecture formalization results from a run folder."
    )
    p.add_argument(
        "run_dir",
        help="Folder containing formalizations.jsonl and events.jsonl",
    )
    p.add_argument(
        "--formalizations-file",
        default="formalizations.jsonl",
        help="Formalizations JSONL filename inside run_dir",
    )
    p.add_argument(
        "--events-file",
        default="events.jsonl",
        help="Events JSONL filename inside run_dir",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    formalizations_path = run_dir / args.formalizations_file
    events_path = run_dir / args.events_file

    if not formalizations_path.exists():
        raise FileNotFoundError(f"Missing file: {formalizations_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Missing file: {events_path}")

    formalizations = load_jsonl(formalizations_path)
    events = load_jsonl(events_path)

    formalization_stats = analyze_formalizations(formalizations)
    event_stats = analyze_events(events)

    print(f"Run directory: {run_dir}")
    print(f"Formalizations file: {formalizations_path}")
    print(f"Events file: {events_path}")
    print()

    print_formalization_report(formalization_stats)
    print_event_report(event_stats, total_formalizations=len(formalizations))


if __name__ == "__main__":
    main()