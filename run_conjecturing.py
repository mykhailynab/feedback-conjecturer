#!/usr/bin/env python3
from __future__ import annotations

import csv
import gc
import json
import math
import os
import threading
from argparse import ArgumentParser
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import polars as pl

from conjecturering_agents.agents.informal_correctness_checker import (
    InformalCorrectnessCheckerAgent,
    InformalCorrectnessCheckerConfig,
)
from conjecturering_agents.agents.solver import SolverAgent, SolverAgentConfig
from conjecturering_agents.inference_backends.vllm_harmony import (
    VLLMHarmonyBackend,
    VLLMHarmonyBackendConfig,
)
from conjecturering_agents.tool_calling_backends.jupyter import JupyterKernelConfig


# ============================================================
# Script config
# ============================================================

@dataclass
class RunConfig:
    # Paths
    reference_path: str = "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv"
    log_dir: str = "/kaggle/working/aimo3_logs"
    attempts_filename: str = "attempts.jsonl"
    solutions_filename: str = "solutions.csv"
    submission_filename: str = "submission.csv"

    # vLLM / model serving
    served_model_name: str = "gpt-oss"
    model_path: str = "/kaggle/input/models/danielhanchen/gpt-oss-20b/transformers/default/1"
    port: int = 8000
    api_key: str = "sk-local"
    host: str = "0.0.0.0"
    client_host: str = "0.0.0.0"
    dtype: str = "auto"
    kv_cache_dtype: str = "fp8_e4m3"
    gpu_memory_utilization: float = 0.96
    batch_size: int = 256
    context_tokens: int = 65536
    stream_interval: int = 200
    server_timeout: int = 180
    session_timeout: int = 960
    preload_workers: int = 32
    preload_model_weights: bool = True
    manage_server: bool = True
    server_log_path: str = "vllm_server.log"

    # Global orchestration
    agent_parallelism: int = 8
    attempts_per_problem: int = 16
    max_problems: int = 0
    seed: int = 42

    # Shared tool runtime defaults
    jupyter_timeout: float = 6.0

    # Solver agent
    solver_temperature: float = 0.5
    solver_min_p: float = 0.02
    solver_top_logprobs: int = 5
    solver_max_turns: int = 128
    solver_timeout_seconds: int = 600
    solver_buffer_tokens: int = 512
    solver_stream_text_window: int = 32

    # Checker agent
    checker_temperature: float = 0.0
    checker_min_p: float = 0.0
    checker_max_turns: int = 64
    checker_timeout_seconds: int = 120
    checker_buffer_tokens: int = 512
    checker_stream_text_window: int = 32

    # Logging
    verbose: bool = True


# ============================================================
# Logging
# ============================================================

class RunLogger:
    """
    Writes:
      - attempts.jsonl
      - solutions.csv
      - events.jsonl
    """

    def __init__(
        self,
        *,
        attempts_path: str,
        solutions_path: str,
        log_dir: str,
        verbose: bool = True,
    ):
        self.attempts_path = attempts_path
        self.solutions_path = solutions_path
        self.events_path = str(Path(log_dir) / "events.jsonl")
        self.verbose = verbose
        self._lock = threading.Lock()
        self._init_solutions_csv()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _append_jsonl(self, path: str, records: List[Dict[str, Any]]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @staticmethod
    def _safe_one_line(x: Any, max_len: int = 200) -> str:
        if x is None:
            return "None"
        s = str(x).replace("\n", "\\n")
        if len(s) > max_len:
            return s[:max_len] + f"...[truncated:{len(s) - max_len}]"
        return s

    def _init_solutions_csv(self) -> None:
        if os.path.exists(self.solutions_path):
            return

        header = [
            "id",
            "true_answer_text",
            "solve_started_ts",
            "solve_finished_ts",
            "solve_elapsed_ms",
            "attempts_total",
            "attempts_with_answer",
            "selected_attempt",
            "selected_answer_text",
            "selected_entropy",
            "selected_is_correct",
            "checker_summary",
        ]
        with open(self.solutions_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        rec = {"ts": self._now_iso(), "event": event_type, **payload}
        with self._lock:
            self._append_jsonl(self.events_path, [rec])

        if self.verbose:
            msg = payload.get("msg") or ""
            print(f"[LOG:{event_type}] {msg}".rstrip())

    def log_attempt(self, attempt_record: Dict[str, Any]) -> None:
        ts = self._now_iso()
        summary = (
            f"id={attempt_record.get('id')} attempt={attempt_record.get('attempt')} "
            f"ans={self._safe_one_line(attempt_record.get('attempt_answer'))} "
            f"ent={attempt_record.get('entropy')} "
            f"py={attempt_record.get('python_calls')}/{attempt_record.get('python_errors')} "
            f"termination={attempt_record.get('termination_reason')} "
            f"elapsed_ms={attempt_record.get('attempt_elapsed_ms')}"
        )

        record = dict(attempt_record)
        record["ts"] = ts
        record["summary"] = summary

        with self._lock:
            self._append_jsonl(self.attempts_path, [record])

        if self.verbose:
            print(f"[LOG:attempts] Attempt summary: {summary}")

    def log_agent_call(self, *, kind: str, payload: Dict[str, Any]) -> None:
        rec = {"ts": self._now_iso(), "event": f"agent_{kind}", **payload}
        with self._lock:
            self._append_jsonl(self.events_path, [rec])

        if self.verbose:
            rid = payload.get("problem_id")
            att = payload.get("attempt")
            eq = payload.get("agent_call", {}).get("result", {}).get("equivalent", None)
            print(f"[LOG:agent_{kind}] id={rid} attempt={att} equivalent={eq}")

    def log_solution_row(
        self,
        *,
        id_value: str,
        true_answer_text: str,
        solve_started_ts: Optional[str],
        solve_finished_ts: Optional[str],
        solve_elapsed_ms: Optional[int],
        attempts_total: int,
        attempts_with_answer: int,
        selected_attempt: Optional[int],
        selected_answer_text: str,
        selected_entropy: Optional[float],
        selected_is_correct: Optional[bool],
        checker_summary: Dict[str, Any],
    ) -> None:
        row = [
            id_value,
            true_answer_text,
            solve_started_ts,
            solve_finished_ts,
            solve_elapsed_ms,
            attempts_total,
            attempts_with_answer,
            selected_attempt,
            selected_answer_text,
            selected_entropy,
            selected_is_correct,
            json.dumps(checker_summary or {}, ensure_ascii=False),
        ]

        with self._lock:
            with open(self.solutions_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

        if self.verbose:
            print(f"[LOG:attempts_finished] id={id_value}")


# ============================================================
# Data helpers
# ============================================================

def iter_reference(reference_df: pl.DataFrame) -> Iterable[Tuple[str, str, str]]:
    for row in reference_df.iter_rows(named=True):
        pid = str(row["id"])
        ptxt = str(row["problem"])
        true_answer_text = str(row["answer"])
        yield pid, ptxt, true_answer_text


def to_float_or_inf(x: Any) -> float:
    try:
        value = float(x)
    except Exception:
        return float("inf")
    if math.isnan(value):
        return float("inf")
    return value


def validate_cfg(cfg: RunConfig) -> None:
    errs: List[str] = []

    if cfg.attempts_per_problem <= 0:
        errs.append("attempts_per_problem must be >= 1")
    if cfg.agent_parallelism <= 0:
        errs.append("agent_parallelism must be >= 1")
    if cfg.batch_size <= 0:
        errs.append("batch_size must be >= 1")
    if cfg.context_tokens <= 0:
        errs.append("context_tokens must be >= 1")
    if cfg.preload_workers <= 0:
        errs.append("preload_workers must be >= 1")
    if cfg.jupyter_timeout <= 0:
        errs.append("jupyter_timeout must be > 0")
    if cfg.solver_max_turns <= 0:
        errs.append("solver_max_turns must be >= 1")
    if cfg.checker_max_turns <= 0:
        errs.append("checker_max_turns must be >= 1")
    if cfg.solver_timeout_seconds <= 0:
        errs.append("solver_timeout_seconds must be >= 1")
    if cfg.checker_timeout_seconds <= 0:
        errs.append("checker_timeout_seconds must be >= 1")
    if cfg.solver_buffer_tokens <= 0:
        errs.append("solver_buffer_tokens must be >= 1")
    if cfg.checker_buffer_tokens <= 0:
        errs.append("checker_buffer_tokens must be >= 1")
    if cfg.solver_temperature < 0:
        errs.append("solver_temperature must be >= 0")
    if cfg.checker_temperature < 0:
        errs.append("checker_temperature must be >= 0")
    if not (0.0 <= cfg.solver_min_p <= 1.0):
        errs.append("solver_min_p must be in [0, 1]")
    if not (0.0 <= cfg.checker_min_p <= 1.0):
        errs.append("checker_min_p must be in [0, 1]")
    if cfg.max_problems < 0:
        errs.append("max_problems must be >= 0")

    if errs:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errs))


# ============================================================
# Agent factory helpers
# ============================================================

def make_solver_agent(cfg: RunConfig) -> SolverAgent:
    jupyter_cfg = JupyterKernelConfig(
        timeout_seconds=cfg.jupyter_timeout,
        init_on_create=False,
    )
    solver_cfg = SolverAgentConfig(
        temperature=cfg.solver_temperature,
        min_p=cfg.solver_min_p,
        top_logprobs=cfg.solver_top_logprobs,
        max_turns=cfg.solver_max_turns,
        timeout_seconds=cfg.solver_timeout_seconds,
        buffer_tokens=cfg.solver_buffer_tokens,
        stream_text_window=cfg.solver_stream_text_window,
        jupyter=jupyter_cfg,
    )
    return SolverAgent(cfg=solver_cfg)


def make_checker_agent(cfg: RunConfig) -> InformalCorrectnessCheckerAgent:
    jupyter_cfg = JupyterKernelConfig(
        timeout_seconds=cfg.jupyter_timeout,
        init_on_create=False,
    )
    checker_cfg = InformalCorrectnessCheckerConfig(
        temperature=cfg.checker_temperature,
        min_p=cfg.checker_min_p,
        max_turns=cfg.checker_max_turns,
        timeout_seconds=cfg.checker_timeout_seconds,
        buffer_tokens=cfg.checker_buffer_tokens,
        stream_text_window=cfg.checker_stream_text_window,
        jupyter=jupyter_cfg,
    )
    return InformalCorrectnessCheckerAgent(cfg=checker_cfg)


def make_backend_config(cfg: RunConfig) -> VLLMHarmonyBackendConfig:
    return VLLMHarmonyBackendConfig(
        served_model_name=cfg.served_model_name,
        model_path=cfg.model_path,
        port=cfg.port,
        api_key=cfg.api_key,
        host=cfg.host,
        client_host=cfg.client_host,
        session_timeout=cfg.session_timeout,
        server_timeout=cfg.server_timeout,
        stream_interval=cfg.stream_interval,
        context_tokens=cfg.context_tokens,
        batch_size=cfg.batch_size,
        dtype=cfg.dtype,
        kv_cache_dtype=cfg.kv_cache_dtype,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        preload_workers=cfg.preload_workers,
        preload_model_weights=cfg.preload_model_weights,
        manage_server=cfg.manage_server,
        server_log_path=cfg.server_log_path,
    )


# ============================================================
# Scheduler state
# ============================================================

@dataclass
class ProblemState:
    id_value: str
    problem_text: str
    true_answer_text: str
    total_attempts: int

    next_attempt_idx: int = 0
    finished_attempts: List[Dict[str, Any]] = field(default_factory=list)

    truth_done: Set[int] = field(default_factory=set)
    truth_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    finalized: bool = False

    solve_started_ts: str = ""
    solve_finished_ts: str = ""
    solve_elapsed_ms: int = 0
    started_logged: bool = False


@dataclass(frozen=True)
class SchedulerTaskInfo:
    kind: str  # "attempt" | "truth_check"
    problem_idx: int
    attempt_idx: Optional[int] = None


# ============================================================
# Scheduler
# ============================================================

class ProblemScheduler:
    """
    Global task loop across problems.

    Policy:
      - Fill capacity with attempt futures in problem order, exhausting each
        problem's attempts before moving on.
      - When an attempt finishes, immediately schedule a truth-check future.
      - Finalize a problem when all attempts are done and all per-attempt
        truth-checks are done.
      - Submission answer is the lowest-entropy attempt with a non-empty answer.
    """

    def __init__(
        self,
        *,
        cfg: RunConfig,
        backend: VLLMHarmonyBackend,
        logger: RunLogger,
        problems: List[ProblemState],
    ):
        self.cfg = cfg
        self.backend = backend
        self.logger = logger
        self.problems = problems

    # --------------------------------------------------------
    # Task implementations
    # --------------------------------------------------------

    def _attempt_task(self, problem_idx: int, attempt_idx: int) -> Dict[str, Any]:
        ps = self.problems[problem_idx]
        agent = make_solver_agent(self.cfg)
        try:
            return agent.run_attempt(
                backend=self.backend,
                problem_id=ps.id_value,
                problem_text=ps.problem_text,
                attempt_index=attempt_idx,
                base_seed=self.cfg.seed,
                metadata={
                    "problem_id": ps.id_value,
                    "attempt_index": attempt_idx,
                    "task_kind": "attempt",
                },
            )
        finally:
            agent.close()

    def _truth_check_task(self, problem_idx: int, attempt_idx: int) -> Dict[str, Any]:
        ps = self.problems[problem_idx]
        attempt_record = self._find_finished_attempt_record_in_problem(ps, attempt_idx) or {}
        pred = str(attempt_record.get("Answer") or "").strip()

        if not pred:
            return {
                "problem_idx": problem_idx,
                "attempt_idx": attempt_idx,
                "skipped": True,
                "skipped_reason": "no_prediction",
                "agent_call": {
                    "raw_output": "",
                    "result": {"equivalent": False, "confidence": 0.0, "reason": "no_prediction"},
                    "termination_reason": "skipped",
                    "tool_calls": [],
                    "turns": [],
                },
            }

        agent = make_checker_agent(self.cfg)
        try:
            agent_call = agent.check_vs_truth(
                backend=self.backend,
                problem_text=ps.problem_text,
                truth_answer_text=ps.true_answer_text,
                candidate_answer_text=pred,
                base_seed=self.cfg.seed,
                metadata={
                    "problem_id": ps.id_value,
                    "attempt_index": attempt_idx,
                    "task_kind": "truth_check",
                },
            )
        finally:
            agent.close()

        return {
            "problem_idx": problem_idx,
            "attempt_idx": attempt_idx,
            "skipped": False,
            "agent_call": agent_call,
        }

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def run_all(self) -> List[Dict[str, Any]]:
        submission_rows: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.cfg.agent_parallelism) as pool:
            inflight: Dict[Future, SchedulerTaskInfo] = {}

            self._fill_with_attempts(pool=pool, inflight=inflight)

            while inflight:
                if self.cfg.verbose and self.problems:
                    completed = sum(1 for p in self.problems if p.finalized)
                    pct = 100.0 * completed / len(self.problems)
                    print(f"Completed: {pct:.2f}%")

                done_fut = next(as_completed(list(inflight.keys())))
                info = inflight.pop(done_fut)
                ps = self.problems[info.problem_idx]

                if info.kind == "attempt":
                    self._handle_attempt_done(
                        done_fut=done_fut,
                        pool=pool,
                        inflight=inflight,
                        ps=ps,
                        problem_idx=info.problem_idx,
                        attempt_idx=int(info.attempt_idx),
                    )
                elif info.kind == "truth_check":
                    self._handle_truth_check_done(
                        done_fut=done_fut,
                        ps=ps,
                        attempt_idx=int(info.attempt_idx),
                    )
                else:
                    raise ValueError(f"Unknown task kind: {info.kind}")

                for pstate in self.problems:
                    if self._ready_to_finalize_problem(pstate):
                        submission_rows.append(self._finalize_problem_and_log_solution(pstate))

                self._fill_with_attempts(pool=pool, inflight=inflight)

        return submission_rows

    # --------------------------------------------------------
    # Submission / scheduling helpers
    # --------------------------------------------------------

    def _log_problem_start_if_needed(self, ps: ProblemState) -> None:
        if ps.started_logged:
            return
        ps.solve_started_ts = datetime.now(timezone.utc).isoformat()
        self.logger.log_event("problem_start", {"id": ps.id_value, "msg": f"Problem id: {ps.id_value}"})
        ps.started_logged = True

    def _has_more_attempts(self, ps: ProblemState) -> bool:
        return ps.next_attempt_idx < ps.total_attempts

    def _can_submit_more_tasks_globally(self, inflight: Dict[Future, SchedulerTaskInfo]) -> bool:
        return len(inflight) < self.cfg.agent_parallelism

    def _submit_attempt(
        self,
        *,
        pool: ThreadPoolExecutor,
        inflight: Dict[Future, SchedulerTaskInfo],
        problem_idx: int,
        ps: ProblemState,
    ) -> bool:
        if not self._has_more_attempts(ps):
            return False

        self._log_problem_start_if_needed(ps)

        attempt_idx = ps.next_attempt_idx
        ps.next_attempt_idx += 1

        fut = pool.submit(self._attempt_task, problem_idx, attempt_idx)
        inflight[fut] = SchedulerTaskInfo(kind="attempt", problem_idx=problem_idx, attempt_idx=attempt_idx)
        return True

    def _fill_with_attempts(
        self,
        *,
        pool: ThreadPoolExecutor,
        inflight: Dict[Future, SchedulerTaskInfo],
    ) -> None:
        if not self._can_submit_more_tasks_globally(inflight):
            return

        for pidx, ps in enumerate(self.problems):
            if not self._can_submit_more_tasks_globally(inflight):
                break
            while self._can_submit_more_tasks_globally(inflight) and self._has_more_attempts(ps):
                self._submit_attempt(pool=pool, inflight=inflight, problem_idx=pidx, ps=ps)

    # --------------------------------------------------------
    # Attempt completion
    # --------------------------------------------------------

    def _process_attempt_future_result(self, attempt_idx: int, done_fut: Future) -> Dict[str, Any]:
        try:
            return done_fut.result()
        except Exception as exc:
            exc_text = f"future_exception:{type(exc).__name__} msg={exc}"
            print(f"[warn] {exc_text}")
            return {
                "Problem ID": "",
                "Attempt": attempt_idx,
                "Response Length": 0,
                "Python Calls": 0,
                "Python Errors": 1,
                "Entropy": float("inf"),
                "Answer": None,
                "Trace": {
                    "prompt_token_ids_initial": [],
                    "prompt_text_initial": "",
                    "turns": [],
                    "full_completion_token_ids": [],
                    "full_conversation_token_ids": [],
                    "raw_output": "",
                    "last_assistant_channel": None,
                    "last_assistant_recipient": None,
                },
                "Termination Reason": exc_text,
                "Tool Calls": [],
                "Attempt Started TS": "",
                "Attempt Finished TS": datetime.now(timezone.utc).isoformat(),
                "Attempt Elapsed MS": 0,
            }

    @staticmethod
    def _find_finished_attempt_record_in_problem(ps: ProblemState, attempt_idx: int) -> Optional[Dict[str, Any]]:
        for attempt_record in ps.finished_attempts:
            if int(attempt_record.get("Attempt", -1)) == int(attempt_idx):
                return attempt_record
        return None

    def _handle_attempt_done(
        self,
        *,
        done_fut: Future,
        pool: ThreadPoolExecutor,
        inflight: Dict[Future, SchedulerTaskInfo],
        ps: ProblemState,
        problem_idx: int,
        attempt_idx: int,
    ) -> None:
        attempt_record = self._process_attempt_future_result(attempt_idx, done_fut)
        ps.finished_attempts.append(attempt_record)

        self.logger.log_attempt(
            {
                "id": ps.id_value,
                "attempt": attempt_record.get("Attempt", attempt_idx),
                "attempt_answer": attempt_record.get("Answer", None),
                "entropy": attempt_record.get("Entropy", None),
                "response_length": attempt_record.get("Response Length", None),
                "python_calls": attempt_record.get("Python Calls", None),
                "python_errors": attempt_record.get("Python Errors", None),
                "termination_reason": attempt_record.get("Termination Reason", "unknown"),
                "attempt_started_ts": attempt_record.get("Attempt Started TS"),
                "attempt_finished_ts": attempt_record.get("Attempt Finished TS"),
                "attempt_elapsed_ms": attempt_record.get("Attempt Elapsed MS"),
                "trace": attempt_record.get("Trace", {}),
                "tool_calls": attempt_record.get("Tool Calls", []),
            }
        )

        tfut = pool.submit(self._truth_check_task, problem_idx, attempt_idx)
        inflight[tfut] = SchedulerTaskInfo(kind="truth_check", problem_idx=problem_idx, attempt_idx=attempt_idx)

    # --------------------------------------------------------
    # Truth-check completion
    # --------------------------------------------------------

    def _handle_truth_check_done(self, *, done_fut: Future, ps: ProblemState, attempt_idx: int) -> None:
        try:
            payload = done_fut.result()
        except Exception as exc:
            payload = {
                "skipped": False,
                "agent_call": {
                    "raw_output": "",
                    "result": {
                        "equivalent": False,
                        "confidence": 0.0,
                        "reason": f"truth_future_exception:{type(exc).__name__}",
                    },
                    "termination_reason": f"truth_future_exception:{type(exc).__name__} msg={exc}",
                    "tool_calls": [],
                    "turns": [],
                },
            }

        agent_call = payload.get("agent_call", {}) or {}

        self.logger.log_agent_call(
            kind="truth_check",
            payload={
                "problem_id": ps.id_value,
                "attempt": attempt_idx,
                "agent_call": agent_call,
            },
        )

        skipped = bool(payload.get("skipped", False))
        if skipped:
            ps.truth_results[attempt_idx] = {
                "attempt_idx": attempt_idx,
                "skipped": True,
                "skipped_reason": payload.get("skipped_reason", "unknown_reason"),
                "is_correct": None,
                "parsed_result": agent_call.get("result", {}),
                "termination_reason": agent_call.get("termination_reason", ""),
            }
        else:
            is_correct = bool(agent_call.get("result", {}).get("equivalent", False))
            ps.truth_results[attempt_idx] = {
                "attempt_idx": attempt_idx,
                "skipped": False,
                "is_correct": is_correct,
                "parsed_result": agent_call.get("result", {}),
                "termination_reason": agent_call.get("termination_reason", ""),
            }

        ps.truth_done.add(attempt_idx)

    # --------------------------------------------------------
    # Finalization
    # --------------------------------------------------------

    def _ready_to_finalize_problem(self, ps: ProblemState) -> bool:
        if ps.finalized:
            return False
        if len(ps.finished_attempts) < ps.total_attempts:
            return False
        return all(i in ps.truth_done for i in range(ps.total_attempts))

    @staticmethod
    def _select_best_attempt(ps: ProblemState) -> Optional[Dict[str, Any]]:
        candidates = [
            r for r in ps.finished_attempts
            if str(r.get("Answer") or "").strip() != ""
        ]
        if not candidates:
            return None

        return min(
            candidates,
            key=lambda r: (
                to_float_or_inf(r.get("Entropy")),
                int(r.get("Attempt", 10**9)),
            ),
        )

    def _finalize_problem_and_log_solution(self, ps: ProblemState) -> Dict[str, Any]:
        ps.solve_finished_ts = datetime.now(timezone.utc).isoformat()

        if ps.solve_started_ts:
            started_dt = datetime.fromisoformat(ps.solve_started_ts)
            finished_dt = datetime.fromisoformat(ps.solve_finished_ts)
            ps.solve_elapsed_ms = int((finished_dt - started_dt).total_seconds() * 1000)
        else:
            ps.solve_elapsed_ms = 0

        attempts_total = len(ps.finished_attempts)
        attempts_with_answer = sum(
            1 for r in ps.finished_attempts if str(r.get("Answer") or "").strip() != ""
        )

        selected_attempt_record = self._select_best_attempt(ps)
        if selected_attempt_record is None:
            selected_attempt_idx: Optional[int] = None
            selected_answer_text = ""
            selected_entropy: Optional[float] = None
            selected_is_correct: Optional[bool] = None
        else:
            selected_attempt_idx = int(selected_attempt_record.get("Attempt"))
            selected_answer_text = str(selected_attempt_record.get("Answer") or "")
            selected_entropy = to_float_or_inf(selected_attempt_record.get("Entropy"))
            selected_is_correct = ps.truth_results.get(selected_attempt_idx, {}).get("is_correct", None)

        checker_summary = {
            "per_attempt_truth": {
                str(attempt_idx): {
                    "is_correct": call_result.get("is_correct", None),
                    "skipped": call_result.get("skipped", False),
                    "skipped_reason": call_result.get("skipped_reason", ""),
                    "parsed_result": call_result.get("parsed_result", {}),
                    "termination_reason": call_result.get("termination_reason", ""),
                }
                for attempt_idx, call_result in sorted(ps.truth_results.items(), key=lambda kv: kv[0])
            }
        }

        self.logger.log_solution_row(
            id_value=ps.id_value,
            true_answer_text=ps.true_answer_text,
            solve_started_ts=ps.solve_started_ts,
            solve_finished_ts=ps.solve_finished_ts,
            solve_elapsed_ms=ps.solve_elapsed_ms,
            attempts_total=attempts_total,
            attempts_with_answer=attempts_with_answer,
            selected_attempt=selected_attempt_idx,
            selected_answer_text=selected_answer_text,
            selected_entropy=selected_entropy,
            selected_is_correct=selected_is_correct,
            checker_summary=checker_summary,
        )
        self.logger.log_event("problem_end", {"id": ps.id_value, "msg": f"Problem id: {ps.id_value}"})
        ps.finalized = True

        return {
            "id": ps.id_value,
            "answer": selected_answer_text,
            "solve_elapsed_ms": ps.solve_elapsed_ms,
            "selected_attempt": selected_attempt_idx,
            "selected_is_correct": selected_is_correct,
        }


# ============================================================
# CLI
# ============================================================

def parse_args() -> RunConfig:
    p = ArgumentParser(
        description="Run conjecturing pipeline: solver attempts plus parallel informal correctness checks."
    )

    # Paths
    p.add_argument("--reference-path", default=RunConfig.reference_path)
    p.add_argument("--log-dir", default=RunConfig.log_dir)
    p.add_argument("--attempts-log", dest="attempts_filename", default=RunConfig.attempts_filename)
    p.add_argument("--solutions-log", dest="solutions_filename", default=RunConfig.solutions_filename)
    p.add_argument("--submission-out", dest="submission_filename", default=RunConfig.submission_filename)

    # Backend / server
    p.add_argument("--served-model-name", default=RunConfig.served_model_name)
    p.add_argument("--model-path", default=RunConfig.model_path)
    p.add_argument("--port", type=int, default=RunConfig.port)
    p.add_argument("--api-key", default=RunConfig.api_key)
    p.add_argument("--host", default=RunConfig.host)
    p.add_argument("--client-host", default=RunConfig.client_host)
    p.add_argument("--dtype", default=RunConfig.dtype)
    p.add_argument("--kv-cache-dtype", default=RunConfig.kv_cache_dtype)
    p.add_argument("--gpu-memory-utilization", type=float, default=RunConfig.gpu_memory_utilization)
    p.add_argument("--max-num-seqs", dest="batch_size", type=int, default=RunConfig.batch_size)
    p.add_argument("--context-tokens", type=int, default=RunConfig.context_tokens)
    p.add_argument("--stream-interval", type=int, default=RunConfig.stream_interval)
    p.add_argument("--server-startup-timeout-seconds", dest="server_timeout", type=int, default=RunConfig.server_timeout)
    p.add_argument("--openai-client-timeout-seconds", dest="session_timeout", type=int, default=RunConfig.session_timeout)
    p.add_argument("--preload-workers", type=int, default=RunConfig.preload_workers)
    p.add_argument("--server-log-path", default=RunConfig.server_log_path)

    p.add_argument(
        "--no-preload-model-weights",
        dest="preload_model_weights",
        action="store_false",
        default=RunConfig.preload_model_weights,
        help="Disable page-caching model weights before server startup.",
    )
    p.add_argument(
        "--use-existing-server",
        dest="manage_server",
        action="store_false",
        default=RunConfig.manage_server,
        help="Do not start vLLM; connect to an already running OpenAI-compatible endpoint.",
    )

    # Global orchestration
    p.add_argument("--agent-parallelism", type=int, default=RunConfig.agent_parallelism)
    p.add_argument("--attempts-per-problem", type=int, default=RunConfig.attempts_per_problem)
    p.add_argument("--max-problems", type=int, default=RunConfig.max_problems)
    p.add_argument("--seed", type=int, default=RunConfig.seed)

    # Shared tool runtime
    p.add_argument("--jupyter-exec-timeout-seconds", dest="jupyter_timeout", type=float, default=RunConfig.jupyter_timeout)

    # Solver
    p.add_argument("--solver-temperature", type=float, default=RunConfig.solver_temperature)
    p.add_argument("--solver-min-p", type=float, default=RunConfig.solver_min_p)
    p.add_argument("--solver-top-logprobs", type=int, default=RunConfig.solver_top_logprobs)
    p.add_argument("--solver-max-turns", type=int, default=RunConfig.solver_max_turns)
    p.add_argument("--solver-timeout-seconds", type=int, default=RunConfig.solver_timeout_seconds)
    p.add_argument("--solver-buffer-tokens", type=int, default=RunConfig.solver_buffer_tokens)
    p.add_argument("--solver-stream-text-window", type=int, default=RunConfig.solver_stream_text_window)

    # Checker
    p.add_argument("--checker-temperature", type=float, default=RunConfig.checker_temperature)
    p.add_argument("--checker-min-p", type=float, default=RunConfig.checker_min_p)
    p.add_argument("--checker-max-turns", type=int, default=RunConfig.checker_max_turns)
    p.add_argument("--checker-timeout-seconds", type=int, default=RunConfig.checker_timeout_seconds)
    p.add_argument("--checker-buffer-tokens", type=int, default=RunConfig.checker_buffer_tokens)
    p.add_argument("--checker-stream-text-window", type=int, default=RunConfig.checker_stream_text_window)

    # Logging
    p.add_argument("--verbose", action="store_true", default=RunConfig.verbose)

    args = p.parse_args()

    return RunConfig(
        reference_path=args.reference_path,
        log_dir=args.log_dir,
        attempts_filename=args.attempts_filename,
        solutions_filename=args.solutions_filename,
        submission_filename=args.submission_filename,
        served_model_name=args.served_model_name,
        model_path=args.model_path,
        port=args.port,
        api_key=args.api_key,
        host=args.host,
        client_host=args.client_host,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        batch_size=args.batch_size,
        context_tokens=args.context_tokens,
        stream_interval=args.stream_interval,
        server_timeout=args.server_timeout,
        session_timeout=args.session_timeout,
        preload_workers=args.preload_workers,
        preload_model_weights=args.preload_model_weights,
        manage_server=args.manage_server,
        server_log_path=args.server_log_path,
        agent_parallelism=args.agent_parallelism,
        attempts_per_problem=args.attempts_per_problem,
        max_problems=args.max_problems,
        seed=args.seed,
        jupyter_timeout=args.jupyter_timeout,
        solver_temperature=args.solver_temperature,
        solver_min_p=args.solver_min_p,
        solver_top_logprobs=args.solver_top_logprobs,
        solver_max_turns=args.solver_max_turns,
        solver_timeout_seconds=args.solver_timeout_seconds,
        solver_buffer_tokens=args.solver_buffer_tokens,
        solver_stream_text_window=args.solver_stream_text_window,
        checker_temperature=args.checker_temperature,
        checker_min_p=args.checker_min_p,
        checker_max_turns=args.checker_max_turns,
        checker_timeout_seconds=args.checker_timeout_seconds,
        checker_buffer_tokens=args.checker_buffer_tokens,
        checker_stream_text_window=args.checker_stream_text_window,
        verbose=args.verbose,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg = parse_args()
    validate_cfg(cfg)

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

    gc.disable()
    try:
        backend.start()

        if cfg.verbose:
            print("Base URL:", backend.cfg.base_url)
            print("Models list:", backend.client.models.list())

        submission_rows = scheduler.run_all()
    finally:
        gc.enable()
        gc.collect()
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