#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import polars as pl

from conjecturing_agents.inference_backends.vllm_harmony import (
    VLLMHarmonyBackend,
)
from conjecturing_agents.run_conjecture_formalization.config import (
    parse_args_and_validate,
    make_backend_config,
)
from conjecturing_agents.run_conjecture_formalization.formalization_scheduler import (
    FormalizationScheduler,
)
from conjecturing_agents.run_conjecture_formalization.logger import (
    RunLogger,
)


# ============================================================
# Data loading helpers
# ============================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                print(f"[warn] Failed to parse JSONL line {line_num} in {path}: {exc}")
                continue
            if not isinstance(obj, dict):
                print(f"[warn] Non-dict JSONL line {line_num} in {path}; skipping.")
                continue
            records.append(obj)
    return records


def load_references_putnam(path: str) -> List[Dict[str, Any]]:
    df = pl.read_csv(path)

    required_cols = {"id", "problem", "answer"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"references_putnam.csv must contain columns {sorted(required_cols)}; "
            f"missing {sorted(missing)}"
        )

    return [dict(row) for row in df.iter_rows(named=True)]


def validate_extracted_putnam_rows(rows: List[Dict[str, Any]], path: str) -> None:
    required_keys = {"lean4_full_contents", "informal_statement"}
    for i, row in enumerate(rows, start=1):
        missing = [k for k in required_keys if k not in row]
        if missing:
            raise ValueError(
                f"extracted_putnam.jsonl row {i} in {path} is missing required keys: {missing}"
            )


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg = parse_args_and_validate()

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    formalizations_path = str(log_dir / cfg.formalizations_filename)

    logger = RunLogger(
        formalizations_path=formalizations_path,
        log_dir=str(log_dir),
        verbose=cfg.verbose,
    )

    attempts = load_jsonl(cfg.attempts_path)
    references_putnam_rows = load_references_putnam(cfg.references_putnam_path)
    extracted_putnam_rows = load_jsonl(cfg.extracted_putnam_path)
    validate_extracted_putnam_rows(extracted_putnam_rows, cfg.extracted_putnam_path)

    if cfg.verbose:
        print(f"Loaded attempts: {len(attempts)}")
        print(f"Loaded references_putnam rows: {len(references_putnam_rows)}")
        print(f"Loaded extracted_putnam rows: {len(extracted_putnam_rows)}")

    backend_cfg = make_backend_config(cfg)
    backend = VLLMHarmonyBackend(
        backend_cfg,
        event_logger=logger.log_event
    )

    scheduler = FormalizationScheduler(
        cfg=cfg,
        backend=backend,
        logger=logger,
        attempts=attempts,
        reference_rows=references_putnam_rows,
        extracted_rows=extracted_putnam_rows,
    )

    logger.log_event(
        "run_start",
        {
            "attempts_path": cfg.attempts_path,
            "references_putnam_path": cfg.references_putnam_path,
            "extracted_putnam_path": cfg.extracted_putnam_path,
            "formalizations_path": formalizations_path,
            "msg": "Conjecture formalization run started",
        },
    )

    try:
        backend.start()

        if cfg.verbose:
            print("Base URL:", backend.cfg.base_url)
            print("Models list:", backend.client.models.list())

        results = scheduler.run_all()
    finally:
        backend.close()

    success_count = sum(1 for r in results if r.get("status") == "success")
    skipped_count = sum(1 for r in results if r.get("status") == "skipped")
    failed_count = sum(1 for r in results if r.get("status") == "failed")

    logger.log_event(
        "run_end",
        {
            "results_total": len(results),
            "success_count": success_count,
            "skipped_count": skipped_count,
            "failed_count": failed_count,
            "formalizations_path": formalizations_path,
            "msg": (
                f"Conjecture formalization run finished: total={len(results)} "
                f"success={success_count} skipped={skipped_count} failed={failed_count}"
            ),
        },
    )

    print(f"\nWrote formalization results to: {formalizations_path}")
    print(
        {
            "total": len(results),
            "success": success_count,
            "skipped": skipped_count,
            "failed": failed_count,
        }
    )


if __name__ == "__main__":
    main()