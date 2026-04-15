#!/usr/bin/env python3
"""
Attempt to prove formalized conjectures using the Goedel prover.

Reads a formalizations.jsonl produced by run_conjecture_formalization.py,
substitutes each formalized abbrev into the Lean scaffold, and runs
GoedelProverAgent to try to prove (and optionally disprove) the theorem.

Results are written to prove_results.jsonl alongside the input file.

Usage:
  python -m conjecturing_agents.prove_formalizations \\
    --formalizations-path logs/.../formalizations.jsonl \\
    --lean-project-dir /path/to/mathlib4 \\
    --parallelism 8 \\
    --proof-retries 4 \\
    --parallel-proof-disproof

Pass --continue to resume from an existing prove_results.jsonl: already-decided
records (proved=True or disproved=True) are kept; undecided records are re-run.
"""
from __future__ import annotations

import json
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from conjecturing_agents.tools import load_jsonl
from conjecturing_agents.prove_formalizations.config import parse_args_and_validate
from conjecturing_agents.prove_formalizations.logger import ProveFormalizationsLogger
from conjecturing_agents.prove_formalizations.scheduler import ProveFormalizationsScheduler


def _record_key(r: Dict[str, Any]) -> Tuple[Any, Any]:
    return (r.get("problem_id"), r.get("attempt"))


def _is_decided(r: Dict[str, Any]) -> bool:
    return bool(r.get("proved")) or bool(r.get("disproved"))


def main() -> None:
    cfg = parse_args_and_validate()

    records = load_jsonl(cfg.formalizations_path)
    if cfg.max_records > 0:
        records = records[: cfg.max_records]

    output_path = cfg.output_path or str(
        Path(cfg.formalizations_path).parent / "prove_results.jsonl"
    )

    events_path = str(Path(output_path).parent / "prove_goedel_events.jsonl")
    event_logger = ProveFormalizationsLogger(events_path)

    tok_lock = threading.Lock()
    total_output_tokens = 0
    total_generation_ms = 0

    def _intercepted_event_logger(event_type: str, payload: Any) -> None:
        nonlocal total_output_tokens, total_generation_ms
        if event_type == "raw_generation_done":
            gen_toks = payload.get("generated_tokens")
            elapsed = payload.get("elapsed_ms")
            if gen_toks is not None and elapsed is not None and elapsed > 0:
                with tok_lock:
                    total_output_tokens += gen_toks
                    total_generation_ms += elapsed
        event_logger.log_event(event_type, payload)

    decided_results: List[Dict[str, Any]] = []
    records_to_run = records

    if cfg.resume:
        existing_path = Path(output_path)
        if existing_path.exists():
            existing = load_jsonl(existing_path)
            decided_results = [r for r in existing if _is_decided(r)]
            undecided_keys = {_record_key(r) for r in existing if not _is_decided(r)}
            existing_keys = {_record_key(r) for r in existing}
            records_to_run = [
                r for r in records
                if _record_key(r) in undecided_keys or _record_key(r) not in existing_keys
            ]
            if cfg.verbose:
                print(
                    f"Loaded {len(existing)} existing results from {output_path}: "
                    f"{len(decided_results)} decided, {len(records_to_run)} to re-run"
                )

    total = len(records_to_run)

    if cfg.verbose:
        print(f"Processing {total} records from {cfg.formalizations_path}")
        print(f"Output: {output_path}")
        if cfg.parallel_proof_disproof:
            outer = max(1, cfg.parallelism // 2)
            print(f"Mode: parallel proof+disproof ({outer} outer workers × 2 sub-threads)")
        else:
            print(f"Mode: proof only ({cfg.parallelism} workers)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()

    proved_count = 0
    disproved_count = 0
    inconclusive_count = 0
    error_count = 0

    for r in decided_results:
        if r.get("proved"):
            proved_count += 1
        elif r.get("disproved"):
            disproved_count += 1
        else:
            inconclusive_count += 1

    progress = tqdm(total=total, desc="Proving", unit="rec")

    all_new_results: List[Dict[str, Any]] = []

    with open(output_path, "w", encoding="utf-8") as out_f:
        for r in decided_results:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out_f.flush()

        def write_result(result: Dict[str, Any]) -> None:
            with write_lock:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

        def on_result(result: Dict[str, Any]) -> None:
            nonlocal proved_count, disproved_count, inconclusive_count, error_count
            write_result(result)
            all_new_results.append(result)
            if result.get("error"):
                error_count += 1
            elif result.get("proved"):
                proved_count += 1
            elif result.get("disproved"):
                disproved_count += 1
            else:
                inconclusive_count += 1

            with tok_lock:
                _toks = total_output_tokens
                _ms = total_generation_ms
            tok_s = _toks / (_ms / 1000) if _ms > 0 else 0
            progress.set_postfix(
                proved=proved_count,
                disproved=disproved_count,
                incon=inconclusive_count,
                err=error_count,
                tok_s=f"{tok_s:.1f}",
            )
            progress.update(1)

        with ProveFormalizationsScheduler(cfg, event_logger=_intercepted_event_logger) as scheduler:
            scheduler.run(records_to_run, on_result=on_result)

    progress.close()

    all_results = decided_results + all_new_results
    success_records = [r for r in all_results if r.get("status") == "success" and not r.get("skipped")]
    proved_total = sum(1 for r in success_records if r.get("proved"))
    disproved_total = sum(1 for r in success_records if r.get("disproved"))
    incon_total = sum(1 for r in success_records if not r.get("proved") and not r.get("disproved") and not r.get("error"))
    err_total = sum(1 for r in success_records if r.get("error"))
    n = len(success_records)

    print(f"\nResults ({n} success records):")
    print(f"  proved     : {proved_total} ({100*proved_total/max(n,1):.1f}%)")
    print(f"  disproved  : {disproved_total} ({100*disproved_total/max(n,1):.1f}%)")
    print(f"  inconclusive: {incon_total} ({100*incon_total/max(n,1):.1f}%)")
    if err_total:
        print(f"  errors     : {err_total}")
    print(f"\nWrote {len(all_results)} records to {output_path}")
    print(f"Event log: {events_path}")

    event_logger.close()


if __name__ == "__main__":
    main()
