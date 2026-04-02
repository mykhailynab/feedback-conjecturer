#!/usr/bin/env python3
"""
Check formalized answers against ground-truth answers.

Reads a formalizations.jsonl produced by run_conjecture_formalization.py
and applies two heuristics per record:

  1. String match (whitespace-normalized)
  2. Lean equivalence proof via canned tactics

Writes results to <log_dir>/check_results.jsonl and prints a summary.

Usage:
  python -m conjecturing_agents.check_formalizations \
    --formalizations-path logs/conjecture_formalization_logs/formalizations.jsonl \
    --lean-project-dir /path/to/mathlib4 \
    --parallelism 8
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from conjecturing_agents.answer_checking.checker import AnswerChecker
from conjecturing_agents.tools import load_jsonl, write_jsonl
from conjecturing_agents.check_formalizations.config import (
    make_checker_config,
    parse_args_and_validate,
)



def check_record(checker: AnswerChecker, record: Dict[str, Any]) -> Dict[str, Any]:
    check = checker.check(record)
    return {
        "problem_id": record.get("problem_id"),
        "attempt": record.get("attempt"),
        "status": record.get("status"),
        "required_abbrev_name": record.get("required_abbrev_name"),
        **check,
    }


def main() -> None:
    cfg = parse_args_and_validate()

    records = load_jsonl(cfg.formalizations_path)
    if cfg.max_records > 0:
        records = records[: cfg.max_records]

    output_path = cfg.output_path or str(
        Path(cfg.formalizations_path).parent / "check_results.jsonl"
    )

    checker_cfg = make_checker_config(cfg)

    # One checker per thread to avoid sharing the compiler's file counter.
    def make_checker() -> AnswerChecker:
        return AnswerChecker(checker_cfg)

    results: List[Dict[str, Any]] = [{}] * len(records)
    total = len(records)

    if cfg.verbose:
        print(f"Checking {total} records from {cfg.formalizations_path}")
        print(f"Output: {output_path}")

    progress = tqdm(total=total, desc="Checking", unit="rec")
    equiv_counts: Dict[Any, int] = {True: 0, False: 0, None: 0}

    def _update_counts(result: Dict[str, Any]) -> None:
        equiv_counts[result.get("equivalent")] = equiv_counts.get(result.get("equivalent"), 0) + 1
        progress.set_postfix(
            equiv=equiv_counts[True],
            not_equiv=equiv_counts[False],
            unknown=equiv_counts[None],
        )
        progress.update(1)

    with ThreadPoolExecutor(max_workers=cfg.parallelism) as pool:
        future_to_idx = {
            pool.submit(check_record, make_checker(), rec): i
            for i, rec in enumerate(records)
        }

        done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                rec = records[idx]
                results[idx] = {
                    "problem_id": rec.get("problem_id"),
                    "attempt": rec.get("attempt"),
                    "status": rec.get("status"),
                    "equivalent": None,
                    "method": "error",
                    "check_result": {
                        "equivalent": None,
                        "method": "error",
                        "details": {"error": str(exc)},
                    },
                    "all_results": [],
                }
            _update_counts(results[idx])
            done += 1
            if cfg.verbose and done % 50 == 0:
                print(f"  {done}/{total} done")

    progress.close()

    write_jsonl(output_path, results)

    # Summary
    success_records = [r for r in results if r.get("status") == "success"]
    equiv_true = sum(1 for r in success_records if r.get("equivalent") is True)
    inconclusive = sum(1 for r in success_records if r.get("equivalent") is None)
    total_success = len(success_records)

    by_method: Dict[str, int] = {}
    for r in success_records:
        if r.get("equivalent") is True:
            m = r.get("method", "unknown")
            by_method[m] = by_method.get(m, 0) + 1

    print(f"\nResults ({total_success} success records):")
    print(f"  equivalent=True : {equiv_true} ({100*equiv_true/max(total_success,1):.1f}%)")
    print(f"  inconclusive    : {inconclusive} ({100*inconclusive/max(total_success,1):.1f}%)")
    print(f"  by method: {by_method}")
    print(f"\nWrote {len(results)} records to {output_path}")


if __name__ == "__main__":
    main()
