#!/usr/bin/env python3
from __future__ import annotations

import os
import csv
import json
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import polars as pl

from conjecturing_agents.inference_backends.vllm_harmony import (
    VLLMHarmonyBackend,
)

from conjecturing_agents.run_conjecturing.problem_scheduler import (
    ProblemState,
    ProblemScheduler
)

from conjecturing_agents.run_conjecturing.config import (
    parse_args_and_validate,
    make_backend_config,
)

from conjecturing_agents.run_conjecturing.logger import (
    RunLogger,
)

# ============================================================
# Data helpers
# ============================================================

def iter_reference(reference_df: pl.DataFrame) -> Iterable[Tuple[str, str, str]]:
    for row in reference_df.iter_rows(named=True):
        pid = str(row["id"])
        ptxt = str(row["problem"])
        true_answer_text = str(row["answer"])
        yield pid, ptxt, true_answer_text

# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg = parse_args_and_validate()

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    attempts_path = str(log_dir / cfg.attempts_filename)
    solutions_path = str(log_dir / cfg.solutions_filename)
    submission_path = str(log_dir / cfg.submission_filename)

    reference_df = pl.read_csv(cfg.reference_path)
    if "id" not in reference_df.columns or "problem" not in reference_df.columns or "answer" not in reference_df.columns:
        raise ValueError("reference.csv must contain columns: id, problem, answer")

    if cfg.max_problems > 0:
        reference_df = reference_df.head(cfg.max_problems)

    logger = RunLogger(
        attempts_path=attempts_path,
        solutions_path=solutions_path,
        log_dir=str(log_dir),
        verbose=cfg.verbose,
    )

    backend_cfg = make_backend_config(cfg)
    backend = VLLMHarmonyBackend(backend_cfg)

    problems: List[ProblemState] = [
        ProblemState(
            id_value=pid,
            problem_text=ptxt,
            true_answer_text=true_ans_text,
            total_attempts=cfg.attempts_per_problem,
        )
        for pid, ptxt, true_ans_text in iter_reference(reference_df)
    ]

    scheduler = ProblemScheduler(
        cfg=cfg,
        backend=backend,
        logger=logger,
        problems=problems,
    )

    try:
        backend.start()

        if cfg.verbose:
            print("Base URL:", backend.cfg.base_url)
            print("Models list:", backend.client.models.list())

        submission_rows = scheduler.run_all()
    finally:
        backend.close()

    order = [str(x) for x in reference_df["id"].to_list()]
    pred_by_id = {str(r["id"]): str(r["answer"]) for r in submission_rows}
    submission_out = [{"id": pid, "answer": pred_by_id.get(pid, "")} for pid in order]

    submission_df = pl.DataFrame(submission_out)
    submission_df.write_csv(submission_path)

    print(f"\nWrote submission to: {submission_path}")
    print(submission_df.head(5))


if __name__ == "__main__":
    main()
