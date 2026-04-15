#!/usr/bin/env python3
"""
Check formalized answers against ground-truth answers.

Reads a formalizations.jsonl produced by run_conjecture_formalization.py
and applies two heuristics per record:

  1. String match (whitespace-normalized)
  2. Lean equivalence proof via canned tactics

Results are written to the output file continuously as they complete.

Pass --continue to resume from an existing output file: already-decided
records (equivalent is not None) are kept and written back immediately;
only the undecided records (equivalent=None) are re-checked.

Usage:
  python -m conjecturing_agents.check_formalizations \
    --formalizations-path logs/conjecture_formalization_logs/formalizations.jsonl \
    --lean-project-dir /path/to/mathlib4 \
    --parallelism 8
"""
from __future__ import annotations

import json
import traceback
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from conjecturing_agents.answer_checking.checker import AnswerChecker
from conjecturing_agents.tools import load_jsonl
from conjecturing_agents.check_formalizations.config import (
    make_checker_config,
    parse_args_and_validate,
)
from conjecturing_agents.check_formalizations.logging import CheckFormalizationsLogger


def check_record(checker: AnswerChecker, record: Dict[str, Any]) -> Dict[str, Any]:
    check = checker.check(record)
    return {
        "problem_id": record.get("problem_id"),
        "attempt": record.get("attempt"),
        "status": record.get("status"),
        "required_abbrev_name": record.get("required_abbrev_name"),
        **check,
    }


def _record_key(r: Dict[str, Any]) -> Tuple[Any, Any]:
    return (r.get("problem_id"), r.get("attempt"))


def main() -> None:
    cfg = parse_args_and_validate()

    records = load_jsonl(cfg.formalizations_path)
    if cfg.max_records > 0:
        records = records[: cfg.max_records]

    output_path = cfg.output_path or str(
        Path(cfg.formalizations_path).parent / "check_results.jsonl"
    )

    checker_cfg = make_checker_config(cfg)

    events_path = str(Path(output_path).parent / "check_goedel_events.jsonl")
    event_logger = (
        CheckFormalizationsLogger(events_path)
        if (checker_cfg.use_goedel_prover or checker_cfg.use_goedel_disprover)
        else None
    )

    # tok/s tracking — accumulated from raw_generation_done events
    tok_lock = threading.Lock()
    total_output_tokens = 0
    total_generation_ms = 0

    def _log_event_intercepted(event_type: str, payload: Any) -> None:
        nonlocal total_output_tokens, total_generation_ms
        if event_type == "raw_generation_done":
            gen_toks = payload.get("generated_tokens")
            elapsed = payload.get("elapsed_ms")
            if gen_toks is not None and elapsed is not None and elapsed > 0:
                with tok_lock:
                    total_output_tokens += gen_toks
                    total_generation_ms += elapsed
        if event_logger is not None:
            event_logger.log_event(event_type, payload)

    decided_results: List[Dict[str, Any]] = []
    records_to_check = records

    if cfg.resume:
        existing_path = Path(output_path)
        if existing_path.exists():
            existing = load_jsonl(existing_path)
            decided_results = [r for r in existing if r.get("equivalent") is not None]
            undecided_keys = {_record_key(r) for r in existing if r.get("equivalent") is None}
            existing_keys = {_record_key(r) for r in existing}
            records_to_check = [
                r for r in records
                if _record_key(r) in undecided_keys or _record_key(r) not in existing_keys
            ]
            if cfg.verbose:
                print(
                    f"Loaded {len(existing)} existing results from {output_path}: "
                    f"{len(decided_results)} decided, {len(records_to_check)} to re-check"
                )

    total = len(records_to_check)

    if cfg.verbose:
        print(f"Checking {total} records from {cfg.formalizations_path}")
        print(f"Output: {output_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()

    def make_checker() -> AnswerChecker:
        return AnswerChecker(
            checker_cfg,
            event_logger=_log_event_intercepted,
        )

    # Summary counters (decided records already counted in)
    equiv_counts: Dict[Optional[bool], int] = defaultdict(int)
    by_method: Dict[str, int] = defaultdict(int)
    by_error_details: Dict[str, int] = defaultdict(int)
    success_total = 0
    lean_timed_out = 0
    lean_oom = 0

    for r in decided_results:
        equiv_counts[r.get("equivalent")] += 1
        if r.get("status") == "success":
            success_total += 1
            if r.get("equivalent") is True:
                by_method[r.get("method", "unknown")] += 1

    all_new_results: List[Dict[str, Any]] = []

    with open(output_path, "w", encoding="utf-8") as out_f:
        # Write carried-over decided records first
        for r in decided_results:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out_f.flush()

        def write_result(result: Dict[str, Any]) -> None:
            with write_lock:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

        progress = tqdm(total=total, desc="Checking", unit="rec")

        def _record_done(result: Dict[str, Any]) -> None:
            nonlocal lean_timed_out, lean_oom
            equiv_counts[result.get("equivalent")] += 1
            if result.get("status") == "success":
                if result.get("equivalent") is True:
                    by_method[result.get("method", "unknown")] += 1
                elif result.get("equivalent") is None:
                    by_error_details[(
                        result.get("check_result", {})
                              .get("details", {})
                              .get("error", "unknown")
                    )] += 1
            for heuristic in result.get("all_results", []):
                if heuristic.get("method") != "lean_equiv":
                    continue
                for attempt in heuristic.get("details", {}).get("attempts", []):
                    if attempt.get("timed_out"):
                        lean_timed_out += 1
                    if attempt.get("oom"):
                        lean_oom += 1
            with tok_lock:
                _toks = total_output_tokens
                _ms = total_generation_ms
            tok_s = _toks / (_ms / 1000) if _ms > 0 else 0
            progress.set_postfix(
                equiv=equiv_counts[True],
                not_equiv=equiv_counts[False],
                unknown=equiv_counts[None],
                lean_tout=lean_timed_out,
                lean_oom=lean_oom,
                tok_s=f"{tok_s:.1f}",
            )
            progress.update(1)

        with ThreadPoolExecutor(max_workers=cfg.parallelism) as pool:
            future_to_idx = {
                pool.submit(check_record, make_checker(), rec): i
                for i, rec in enumerate(records_to_check)
            }

            done = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                rec = records_to_check[idx]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "problem_id": rec.get("problem_id"),
                        "attempt": rec.get("attempt"),
                        "status": rec.get("status"),
                        "equivalent": None,
                        "method": "error",
                        "check_result": {
                            "equivalent": None,
                            "method": "error",
                            "details": {
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                                "type": "terminating",  # indicate that heuristics were interrupted
                            },
                        },
                        "all_results": [],
                    }

                write_result(result)
                _record_done(result)
                all_new_results.append(result)
                done += 1
                if cfg.verbose and done % 50 == 0:
                    print(f"  {done}/{total} done")

        progress.close()

    all_results = decided_results + all_new_results
    success_records = [r for r in all_results if r.get("status") == "success"]
    equiv_true = sum(1 for r in success_records if r.get("equivalent") is True)
    equiv_false = sum(1 for r in success_records if r.get("equivalent") is False)
    inconclusive = sum(1 for r in success_records if r.get("equivalent") is None)
    total_success = len(success_records)

    print(f"\nResults ({total_success} success records):")
    print(f"  equivalent=True : {equiv_true} ({100*equiv_true/max(total_success,1):.1f}%)")
    print(f"  equivalent=False: {equiv_false} ({100*equiv_false/max(total_success,1):.1f}%)")
    print(f"  inconclusive    : {inconclusive} ({100*inconclusive/max(total_success,1):.1f}%)")
    print(f"  by method: {dict(by_method)}")
    if by_error_details:
        print(f"  errors: {dict(by_error_details)}")
    if lean_timed_out or lean_oom:
        print(f"  lean_equiv timed out: {lean_timed_out}, OOM: {lean_oom}")
    print(f"\nWrote {len(all_results)} records to {output_path}")
    if event_logger is not None:
        event_logger.close()
        print(f"Event log: {events_path}")


if __name__ == "__main__":
    main()
