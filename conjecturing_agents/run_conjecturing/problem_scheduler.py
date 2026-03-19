from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

from conjecturing_agents.inference_backends.vllm_harmony import (
    VLLMHarmonyBackend,
)
from conjecturing_agents.tools import to_float_or_inf

from conjecturing_agents.run_conjecturing.config import (
    RunConfig,
    make_solver_agent,
    make_checker_agent,
)

from conjecturing_agents.run_conjecturing.logger import (
    RunLogger,
)

from conjecturing_agents.agents.solver import (
    SolverAgent,
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
        pred = str(attempt_record.get("attempt_answer") or "").strip()

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
            return SolverAgent.make_empty_attempt_record(
                attempt_idx=attempt_idx,
                termination_reason=exc_text
            )

    @staticmethod
    def _find_finished_attempt_record_in_problem(ps: ProblemState, attempt_idx: int) -> Optional[Dict[str, Any]]:
        for attempt_record in ps.finished_attempts:
            if int(attempt_record.get("attempt", -1)) == int(attempt_idx):
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

        self.logger.log_attempt(attempt_record)

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
            if str(r.get("attempt_answer") or "").strip() != ""
        ]
        if not candidates:
            return None

        return min(
            candidates,
            key=lambda r: (
                to_float_or_inf(r.get("entropy")),
                int(r.get("attempt", 10**9)),
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
            1 for r in ps.finished_attempts if str(r.get("attempt_answer") or "").strip() != ""
        )

        selected_attempt_record = self._select_best_attempt(ps)
        if selected_attempt_record is None:
            selected_attempt_idx: Optional[int] = None
            selected_answer_text = ""
            selected_entropy: Optional[float] = None
            selected_is_correct: Optional[bool] = None
        else:
            selected_attempt_idx = int(selected_attempt_record.get("attempt"))
            selected_answer_text = str(selected_attempt_record.get("attempt_answer") or "")
            selected_entropy = to_float_or_inf(selected_attempt_record.get("entropy"))
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

