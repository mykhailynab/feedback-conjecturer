from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from conjecturing_agents.agents.conjecture_formalizer import (
    extract_abbrev_name_from_statement,
    extract_ground_truth_comment_and_strip_line,
    replace_abbrev_in_statement,
)
from conjecturing_agents.inference_backends.vllm_harmony import (
    VLLMHarmonyBackend,
)
from conjecturing_agents.run_conjecture_formalization.config import (
    RunConjectureFormalizationConfig,
    make_formalizer_agent,
    make_lean_compiler_config,
)
from conjecturing_agents.run_conjecture_formalization.logger import (
    RunLogger,
)
from conjecturing_agents.tool_calling_backends.lean4_compiler import (
    Lean4CompilerBackend,
    build_tool_facing_feedback,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attempt_field(record: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return default


def _extract_problem_id(record: Dict[str, Any]) -> str:
    return str(
        _attempt_field(record, "problem_id", "Problem ID", "id", default="")
    )


def _extract_attempt_index(record: Dict[str, Any]) -> int:
    value = _attempt_field(record, "attempt", "Attempt", default=0)
    try:
        return int(value)
    except Exception:
        return 0


def _extract_attempt_answer(record: Dict[str, Any]) -> str:
    value = _attempt_field(record, "attempt_answer", "Answer", default="")
    return str(value or "").strip()


def _extract_trace_raw_output(record: Dict[str, Any]) -> str:
    trace = _attempt_field(record, "trace", "Trace", default={}) or {}
    if not isinstance(trace, dict):
        return ""
    return str(trace.get("raw_output", "") or "")


def _build_correction_user_prompt(
    *,
    initial_user_prompt: str,
    previous_abbrev: Optional[str],
    error_feedback: str,
    correction_round_num: int,
) -> str:
    prev_block = previous_abbrev or "<no valid abbrev was parsed in the previous round>"
    return (
        f"{initial_user_prompt}\n\n"
        f"Previous failed formalization attempt (Round {correction_round_num - 1}):\n"
        "```lean4\n"
        f"{prev_block}\n"
        "```\n\n"
        "Compiler feedback for that attempt:\n"
        f"{error_feedback}\n\n"
        "Repair the abbrev declaration.\n"
        "- Keep the abbrev name exactly the same.\n"
        "- Output only a single ```lean4``` block containing the corrected abbrev declaration.\n"
        "- Do not output the theorem.\n"
        "- Do not output `sorry`.\n"
    )


@dataclass(frozen=True)
class FormalizationTaskState:
    task_idx: int
    attempt_record: Dict[str, Any]
    problem_id: str
    attempt_idx: int


class FormalizationScheduler:
    """
    Parallel manager for conjecture formalization tasks.

    Each task:
      1. resolves problem_id -> reference row -> extracted_putnam row
      2. strips the ground-truth answer comment from the Lean scaffold
      3. queries the conjecture formalizer agent
      4. inserts the produced abbrev into the scaffold
      5. runs external Lean validation
      6. optionally performs correction rounds

    The scheduler returns one result record per input attempt.
    """

    def __init__(
        self,
        *,
        cfg: RunConjectureFormalizationConfig,
        backend: VLLMHarmonyBackend,
        logger: RunLogger,
        attempts: Sequence[Dict[str, Any]],
        reference_rows: Sequence[Dict[str, Any]],
        extracted_rows: Sequence[Dict[str, Any]],
    ):
        self.cfg = cfg
        self.backend = backend
        self.logger = logger

        attempts_list = list(attempts)
        if self.cfg.max_attempts > 0:
            attempts_list = attempts_list[: self.cfg.max_attempts]

        self.states: List[FormalizationTaskState] = [
            FormalizationTaskState(
                task_idx=i,
                attempt_record=attempt_record,
                problem_id=_extract_problem_id(attempt_record),
                attempt_idx=_extract_attempt_index(attempt_record),
            )
            for i, attempt_record in enumerate(attempts_list)
        ]

        self.reference_rows_by_id: Dict[str, List[Dict[str, Any]]] = {}
        for row in reference_rows:
            pid = str(row["id"])
            self.reference_rows_by_id.setdefault(pid, []).append(dict(row))

        self.extracted_rows_by_informal_statement: Dict[str, List[Dict[str, Any]]] = {}
        for row in extracted_rows:
            informal_statement = str(row["informal_statement"])
            self.extracted_rows_by_informal_statement.setdefault(informal_statement, []).append(dict(row))

        self.validation_lean_cfg = make_lean_compiler_config(cfg)
        self.validation_lean_backend = Lean4CompilerBackend(self.validation_lean_cfg)

    # --------------------------------------------------------
    # Resolution helpers
    # --------------------------------------------------------

    def _resolve_reference_row(self, problem_id: str) -> Optional[Dict[str, Any]]:
        rows = self.reference_rows_by_id.get(problem_id, [])
        if len(rows) != 1:
            self.logger.log_warn(
                f"Expected exactly one row in references_putnam.csv for id={problem_id!r}, found {len(rows)}. Skipping."
            )
            return None
        return rows[0]

    def _resolve_extracted_row(self, problem_text: str, *, problem_id: str) -> Optional[Dict[str, Any]]:
        rows = self.extracted_rows_by_informal_statement.get(problem_text, [])
        if len(rows) != 1:
            self.logger.log_warn(
                f"Expected exactly one row in extracted_putnam.jsonl for informal_statement of id={problem_id!r}, found {len(rows)}. Skipping."
            )
            return None
        return rows[0]

    def _resolve_attempt_context(self, state: FormalizationTaskState) -> Optional[Dict[str, Any]]:
        attempt_record = state.attempt_record
        problem_id = state.problem_id

        self.logger.log_event("formalization_resolve_start", {
            "problem_id": state.problem_id,
            "attempt": state.attempt_idx,
            "msg": f"Resolving formalization context: id={state.problem_id} attempt={state.attempt_idx}",
        })

        attempt_answer = _extract_attempt_answer(attempt_record)
        if self.cfg.skip_empty_answers and not attempt_answer:
            return {
                "status": "skipped",
                "skip_reason": "empty_attempt_answer",
            }

        ref_row = self._resolve_reference_row(problem_id)
        if ref_row is None:
            return None

        problem_text = str(ref_row["problem"])
        extracted_row = self._resolve_extracted_row(problem_text, problem_id=problem_id)
        if extracted_row is None:
            return None

        lean4_full_contents = str(extracted_row["lean4_full_contents"])
        try:
            extracted_answer_comment, lean_statement_without_comment = (
                extract_ground_truth_comment_and_strip_line(lean4_full_contents)
            )
        except Exception as exc:
            self.logger.log_warn(
                f"Failed to remove abbrev answer comment for id={problem_id!r}: {exc}. Skipping."
            )
            return None

        required_abbrev_name = extract_abbrev_name_from_statement(lean_statement_without_comment)
        if required_abbrev_name is None:
            self.logger.log_warn(
                f"Could not extract target abbrev name from lean scaffold for id={problem_id!r}. Skipping."
            )
            return None

        raw_output = _extract_trace_raw_output(attempt_record)

        self.logger.log_event("formalization_resolve_done", {
            "problem_id": problem_id,
            "attempt": state.attempt_idx
        })

        return {
            "status": "ready",
            "problem_text": problem_text,
            "attempt_answer": attempt_answer,
            "attempt_raw_output": raw_output,
            "lean4_full_contents": lean4_full_contents,
            "lean_statement_without_comment": lean_statement_without_comment,
            "ground_truth_extracted_answer": extracted_answer_comment,
            "required_abbrev_name": required_abbrev_name,
            "extracted_row_name": extracted_row.get("name"),
            "extracted_row_tags": extracted_row.get("tags", []),
        }

    # --------------------------------------------------------
    # Single task
    # --------------------------------------------------------

    def _run_formalization_task(self, state: FormalizationTaskState) -> Dict[str, Any]:
        started_ts = _now_iso()
        t0 = datetime.now(timezone.utc)

        self.logger.log_event("formalization_start", {
            "problem_id": state.problem_id,
            "attempt": state.attempt_idx,
            "msg": f"Formalization start: id={state.problem_id} attempt={state.attempt_idx}",
        })

        context = self._resolve_attempt_context(state)
        if context is None:
            finished_ts = _now_iso()
            elapsed_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
            return {
                "problem_id": state.problem_id,
                "attempt": state.attempt_idx,
                "status": "skipped",
                "skip_reason": "resolution_failed",
                "rounds_used": 0,
                "final_compile_ok": False,
                "final_abbrev_declaration": None,
                "ground_truth_extracted_answer": None,
                "started_ts": started_ts,
                "finished_ts": finished_ts,
                "elapsed_ms": elapsed_ms,
            }

        if context["status"] == "skipped":
            finished_ts = _now_iso()
            elapsed_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
            return {
                "problem_id": state.problem_id,
                "attempt": state.attempt_idx,
                "status": "skipped",
                "skip_reason": context["skip_reason"],
                "rounds_used": 0,
                "final_compile_ok": False,
                "final_abbrev_declaration": None,
                "ground_truth_extracted_answer": None,
                "started_ts": started_ts,
                "finished_ts": finished_ts,
                "elapsed_ms": elapsed_ms,
            }

        formalizer = make_formalizer_agent(self.cfg)
        rounds: List[Dict[str, Any]] = []

        initial_user_prompt = formalizer.build_user_prompt(
            informal_problem_text=context["problem_text"],
            proposed_solution_text=context["attempt_raw_output"],
            boxed_answer_text=context["attempt_answer"],
            lean_statement=context["lean_statement_without_comment"],
        )

        final_status = "failed"
        final_skip_reason: Optional[str] = None
        final_abbrev_declaration: Optional[str] = None
        final_compile_ok = False
        final_compile_relative_path: Optional[str] = None
        final_compile_formatted_diagnostics: str = ""

        def _append_round_and_log(round_num: int, round_record: Dict[str, Any]):
            rounds.append(round_record)
            self.logger.log_event("formalization_round_done", {
                "problem_id": state.problem_id,
                "attempt": state.attempt_idx,
                "round": round_num,
                "compile_ok": round_record.get("compile_ok"),
                "termination_reason": round_record.get("termination_reason"),
            })

        try:
            error_feedback = ""
            previous_abbrev: Optional[str] = None

            for round_num in range(1, self.cfg.max_correction_rounds + 2):
                self.logger.log_event("formalization_round_start", {
                    "problem_id": state.problem_id,
                    "attempt": state.attempt_idx,
                    "round": round_num,
                })
                if round_num == 1:
                    formalizer_result = formalizer.formalize_conjecture(
                        backend=self.backend,
                        informal_problem_text=context["problem_text"],
                        proposed_solution_text=context["attempt_raw_output"],
                        boxed_answer_text=context["attempt_answer"],
                        lean_statement=context["lean_statement_without_comment"],
                        base_seed=self.cfg.seed + 100000 * state.task_idx + round_num,
                        metadata={
                            "problem_id": state.problem_id,
                            "attempt": state.attempt_idx,
                            "round": round_num,
                            "task_kind": "conjecture_formalization",
                        },
                    )
                else:
                    correction_user_prompt = _build_correction_user_prompt(
                        initial_user_prompt=initial_user_prompt,
                        previous_abbrev=previous_abbrev,
                        error_feedback=error_feedback,
                        correction_round_num=round_num,
                    )
                    run_result = self.backend.run_user_prompt(
                        agent=formalizer.build_agent_spec(),
                        user_prompt=correction_user_prompt,
                        seed=self.cfg.seed + 100000 * state.task_idx + round_num,
                        metadata={
                            "problem_id": state.problem_id,
                            "attempt": state.attempt_idx,
                            "round": round_num,
                            "task_kind": "conjecture_formalization_correction",
                            "required_abbrev_name": context["required_abbrev_name"],
                        },
                    )
                    formalizer_result = formalizer._to_formalization_result(run_result)
                
                self.logger.log_event("formalization_agent_call_done", {
                    "problem_id": state.problem_id,
                    "attempt": state.attempt_idx,
                    "round": round_num,
                    "termination_reason": formalizer_result.get("termination_reason"),
                    "python_calls": formalizer_result.get("python_calls", 0),
                    "lean_calls": formalizer_result.get("lean_calls", 0),
                })

                parsed = formalizer_result.get("result", {}) or {}
                abbrev_declaration = parsed.get("abbrev_declaration")
                previous_abbrev = abbrev_declaration

                round_record: Dict[str, Any] = {
                    "round": round_num,
                    "termination_reason": formalizer_result.get("termination_reason"),
                    "abbrev_declaration": abbrev_declaration,
                    "abbrev_name": parsed.get("abbrev_name"),
                    "rhs": parsed.get("rhs"),
                    "python_calls": formalizer_result.get("python_calls", 0),
                    "python_errors": formalizer_result.get("python_errors", 0),
                    "lean_calls": formalizer_result.get("lean_calls", 0),
                    "lean_errors": formalizer_result.get("lean_errors", 0),
                    "tool_calls": formalizer_result.get("tool_calls", []),
                    "raw_output": formalizer_result.get("raw_output", ""),
                    "compile_ok": None,
                    "compile_relative_path": None,
                    "compile_formatted_diagnostics": "",
                }

                if not abbrev_declaration:
                    error_feedback = (
                        "No valid abbrev declaration was parsed from your final answer.\n"
                        "You must output a single ```lean4``` block containing only the abbrev declaration.\n"
                        f"The target abbrev name is `{context['required_abbrev_name']}`."
                    )
                    round_record["compile_ok"] = False
                    round_record["compile_formatted_diagnostics"] = error_feedback
                    _append_round_and_log(round_num, round_record)
                    continue

                try:
                    assembled_lean = replace_abbrev_in_statement(
                        context["lean_statement_without_comment"],
                        abbrev_declaration,
                        required_abbrev_name=context["required_abbrev_name"],
                    )
                except Exception as exc:
                    error_feedback = f"Failed to replace the scaffold abbrev with your generated declaration: {exc}"
                    round_record["compile_ok"] = False
                    round_record["compile_formatted_diagnostics"] = error_feedback
                    _append_round_and_log(round_num, round_record)
                    continue

                self.logger.log_event("formalization_external_compile_start", {
                    "problem_id": state.problem_id,
                    "attempt": state.attempt_idx,
                    "round": round_num,
                })
                compile_result = self.validation_lean_backend.compile_code(assembled_lean)
                self.logger.log_event("formalization_external_compile_done", {
                    "problem_id": state.problem_id,
                    "attempt": state.attempt_idx,
                    "round": round_num,
                    "ok": compile_result.ok,
                    "timed_out": compile_result.timed_out,
                    "returncode": compile_result.returncode,
                    "json_error_count": len(compile_result.json_errors),
                    "json_warning_count": len(compile_result.json_warnings),
                    "sorry_warning_count": len(compile_result.sorry_warnings),
                })
                round_record["compile_ok"] = compile_result.ok
                round_record["compile_relative_path"] = compile_result.relative_path
                round_record["compile_formatted_diagnostics"] = compile_result.formatted_diagnostics
                round_record["assembled_lean"] = assembled_lean
                _append_round_and_log(round_num, round_record)

                if compile_result.ok:
                    final_status = "success"
                    final_abbrev_declaration = abbrev_declaration
                    final_compile_ok = True
                    final_compile_relative_path = compile_result.relative_path
                    final_compile_formatted_diagnostics = compile_result.formatted_diagnostics
                    break

                error_feedback = build_tool_facing_feedback(
                    compile_result,
                    cfg=self.validation_lean_cfg,
                )
                final_compile_relative_path = compile_result.relative_path
                final_compile_formatted_diagnostics = compile_result.formatted_diagnostics

            if final_status != "success" and rounds:
                final_abbrev_declaration = rounds[-1].get("abbrev_declaration")

        finally:
            self.logger.log_event("formalization_close_start", {
                "problem_id": state.problem_id,
                "attempt": state.attempt_idx,
            })
            formalizer.close()
            self.logger.log_event("formalization_close_done", {
                "problem_id": state.problem_id,
                "attempt": state.attempt_idx,
            })

        finished_ts = _now_iso()
        elapsed_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)

        result_record = {
            "problem_id": state.problem_id,
            "attempt": state.attempt_idx,
            "status": final_status,
            "skip_reason": final_skip_reason,
            "attempt_answer": context["attempt_answer"],
            "attempt_raw_output_tail": context["attempt_raw_output"][-self.cfg.solution_tail_chars :],
            "ground_truth_extracted_answer": context["ground_truth_extracted_answer"],
            "required_abbrev_name": context["required_abbrev_name"],
            "lean_statement_without_comment": context["lean_statement_without_comment"],
            "extracted_row_name": context["extracted_row_name"],
            "extracted_row_tags": context["extracted_row_tags"],
            "rounds_used": len(rounds),
            "rounds": rounds,
            "final_abbrev_declaration": final_abbrev_declaration,
            "final_compile_ok": final_compile_ok,
            "final_compile_relative_path": final_compile_relative_path,
            "final_compile_formatted_diagnostics": final_compile_formatted_diagnostics,
            "started_ts": started_ts,
            "finished_ts": finished_ts,
            "elapsed_ms": elapsed_ms,
        }

        self.logger.log_event("formalization_done", {
            "problem_id": state.problem_id,
            "attempt": state.attempt_idx,
            "status": final_status,
            "msg": f"Formalization done: id={state.problem_id} attempt={state.attempt_idx} status={final_status}",
        })
        return result_record

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def run_all(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.cfg.agent_parallelism) as pool:
            futures: Dict[Future, FormalizationTaskState] = {
                pool.submit(self._run_formalization_task, state): state
                for state in self.states
            }

            for fut in as_completed(futures):
                state = futures[fut]
                try:
                    record = fut.result()
                except Exception as exc:
                    self.logger.log_warn(
                        f"Formalization task crashed for id={state.problem_id!r} attempt={state.attempt_idx}: {exc}"
                    )
                    record = {
                        "problem_id": state.problem_id,
                        "attempt": state.attempt_idx,
                        "status": "failed",
                        "skip_reason": None,
                        "attempt_answer": _extract_attempt_answer(state.attempt_record),
                        "attempt_raw_output_tail": _extract_trace_raw_output(state.attempt_record)[-self.cfg.solution_tail_chars :],
                        "ground_truth_extracted_answer": None,
                        "required_abbrev_name": None,
                        "lean_statement_without_comment": None,
                        "extracted_row_name": None,
                        "extracted_row_tags": [],
                        "rounds_used": 0,
                        "rounds": [],
                        "final_abbrev_declaration": None,
                        "final_compile_ok": False,
                        "final_compile_relative_path": None,
                        "final_compile_formatted_diagnostics": str(exc),
                        "started_ts": "",
                        "finished_ts": _now_iso(),
                        "elapsed_ms": 0,
                    }

                results.append(record)
                self.logger.log_formalization_result(record)

                if self.cfg.verbose and self.states:
                    pct = 100.0 * len(results) / len(self.states)
                    print(f"Completed: {pct:.2f}%")

        return results