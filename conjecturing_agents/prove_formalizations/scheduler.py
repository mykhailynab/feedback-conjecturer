"""
Scheduler for the prove_formalizations pipeline.

Each record from formalizations.jsonl is processed as follows:

1. Build the proved Lean file:
   ``replace_abbrev_in_statement(lean_statement_without_comment, final_abbrev_declaration)``

2. Run ``GoedelProverAgent.prove_theorem()`` on the proved Lean file.
   If ``--proof-retries > 1``, run up to that many independent attempts
   (different seeds), stopping as soon as one succeeds.

3. With ``--enable-parallel-disproof``:
   - Negate the theorem with ``negate_theorem_statement()``
   - Run proof and disproof in separate sub-threads sharing a
     ``threading.Event``; the first to succeed cancels the other.
   - Use ``parallelism // 2`` outer workers (each worker runs 2 sub-threads).

Without ``--enable-parallel-disproof``, use ``parallelism`` workers, one per
record, running only the proof direction.
"""
from __future__ import annotations

import dataclasses
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from conjecturing_agents.lean_regex import (
    negate_theorem_statement,
    replace_abbrev_in_statement,
)
from conjecturing_agents.agents.goedel_prover import (
    GoedelProverAgent,
    GoedelProverResult,
)
from conjecturing_agents.inference_backends.raw_backend import RawBackend, EventLoggerFn

from .config import ProveFormalizationsConfig, make_goedel_prover_config, make_goedel_backend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record_key(r: Dict[str, Any]) -> Tuple[Any, Any]:
    return (r.get("problem_id"), r.get("attempt"))


def _result_to_dict(r: GoedelProverResult) -> Dict[str, Any]:
    return dataclasses.asdict(r)


def _run_prove_attempts(
    agent: GoedelProverAgent,
    backend: RawBackend,
    theorem_text: str,
    *,
    base_seed: int,
    retries: int,
    event_logger: Optional[EventLoggerFn],
    metadata: Dict[str, Any],
    stop_event: Optional[threading.Event] = None,
    token_limit: int = 0,
    initial_messages: Optional[List[Dict[str, Any]]] = None,
    initial_partial_response: str = "",
) -> GoedelProverResult:
    """Run up to ``retries`` proof attempts, stopping as soon as one succeeds.

    ``initial_messages`` and ``initial_partial_response`` are used only for
    retry 0 (resuming a saved incomplete session).  Subsequent retries always
    start fresh with a new seed.
    """
    last_result: Optional[GoedelProverResult] = None
    for i in range(retries):
        if stop_event is not None and stop_event.is_set():
            break
        result = agent.prove_theorem(
            theorem_text,
            backend,
            seed=(base_seed + i) & 0x7FFFFFFF,
            event_logger=event_logger,
            metadata={**metadata, "retry": i},
            stop_event=stop_event,
            token_limit=token_limit,
            initial_messages=initial_messages if i == 0 else None,
            partial_response=initial_partial_response if i == 0 else "",
        )
        last_result = result
        if result.proved:
            break
    assert last_result is not None
    return last_result


# ---------------------------------------------------------------------------
# Resume-state extraction
# ---------------------------------------------------------------------------

def _extract_resume_state(
    incomplete_result: Optional[Dict[str, Any]],
    direction: str,  # "proof_result" or "disproof_result"
) -> Tuple[Optional[List[Dict[str, str]]], str]:
    """Return ``(initial_messages, partial_response)`` for a resumed session.

    Pulls the saved conversation history and partial response from the
    ``proof_result`` or ``disproof_result`` sub-dict of an incomplete record.
    Returns ``(None, "")`` when there is nothing to resume from.
    """
    if incomplete_result is None:
        return None, ""
    sub: Dict[str, Any] = incomplete_result.get(direction, {})
    msgs = sub.get("conversation_history", None)
    partial = sub.get("partial_response", "")
    return msgs, partial


# ---------------------------------------------------------------------------
# Single-record processing
# ---------------------------------------------------------------------------

def process_record_sequential(
    record: Dict[str, Any],
    *,
    proof_agent: GoedelProverAgent,
    proof_backend: RawBackend,
    proof_retries: int,
    base_seed: int,
    event_logger: Optional[EventLoggerFn],
    token_limit: int = 0,
    incomplete_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Prove only (no disproof).  Returns the result dict to be written to JSONL.
    """
    problem_id = record.get("problem_id")
    attempt = record.get("attempt")

    lean_statement = str(record.get("lean_statement_without_comment") or "")
    abbrev_decl = str(record.get("final_abbrev_declaration") or "")

    try:
        proved_lean = replace_abbrev_in_statement(lean_statement, abbrev_decl)
    except Exception as exc:
        return _error_record(record, "replace_abbrev_error", str(exc))

    init_msgs, init_partial = _extract_resume_state(incomplete_result, "proof_result")
    meta = {"problem_id": problem_id, "attempt": attempt, "direction": "proof"}
    proof_result = _run_prove_attempts(
        proof_agent, proof_backend, proved_lean,
        base_seed=base_seed,
        retries=proof_retries,
        event_logger=event_logger,
        metadata=meta,
        token_limit=token_limit,
        initial_messages=init_msgs,
        initial_partial_response=init_partial,
    )

    return _build_output_record(
        record,
        proved_lean=proved_lean,
        negated_lean=None,
        proof_result=proof_result,
        disproof_result=None,
    )


def process_record_sequential_with_disproof(
    record: Dict[str, Any],
    *,
    proof_agent: GoedelProverAgent,
    proof_backend: RawBackend,
    disproof_agent: GoedelProverAgent,
    disproof_backend: RawBackend,
    proof_retries: int,
    disproof_retries: int,
    base_seed: int,
    event_logger: Optional[EventLoggerFn],
    token_limit: int = 0,
    incomplete_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run proof first; if every retry fails, run disproof in the same thread.
    Uses the full ``parallelism`` worker count (no division by 2).
    """
    problem_id = record.get("problem_id")
    attempt = record.get("attempt")

    lean_statement = str(record.get("lean_statement_without_comment") or "")
    abbrev_decl = str(record.get("final_abbrev_declaration") or "")

    try:
        proved_lean = replace_abbrev_in_statement(lean_statement, abbrev_decl)
    except Exception as exc:
        return _error_record(record, "replace_abbrev_error", str(exc))

    init_proof_msgs, init_proof_partial = _extract_resume_state(incomplete_result, "proof_result")
    meta = {"problem_id": problem_id, "attempt": attempt, "direction": "proof"}
    proof_result = _run_prove_attempts(
        proof_agent, proof_backend, proved_lean,
        base_seed=base_seed,
        retries=proof_retries,
        event_logger=event_logger,
        metadata=meta,
        token_limit=token_limit,
        initial_messages=init_proof_msgs,
        initial_partial_response=init_proof_partial,
    )

    # If proof is still incomplete (token-limited), skip disproof for now.
    disproof_result: Optional[GoedelProverResult] = None
    if not proof_result.proved and not proof_result.incomplete:
        try:
            negated_lean = negate_theorem_statement(proved_lean)
        except Exception as exc:
            return _error_record(record, "negate_theorem_error", str(exc))

        # Resume disproof only if the prior run had one; otherwise start fresh.
        init_dis_msgs, init_dis_partial = _extract_resume_state(incomplete_result, "disproof_result")
        dis_meta = {"problem_id": problem_id, "attempt": attempt, "direction": "disproof"}
        disproof_result = _run_prove_attempts(
            disproof_agent, disproof_backend, negated_lean,
            base_seed=(base_seed + 0x10000) & 0x7FFFFFFF,
            retries=disproof_retries,
            event_logger=event_logger,
            metadata=dis_meta,
            token_limit=token_limit,
            initial_messages=init_dis_msgs,
            initial_partial_response=init_dis_partial,
        )
        return _build_output_record(
            record,
            proved_lean=proved_lean,
            negated_lean=negated_lean,
            proof_result=proof_result,
            disproof_result=disproof_result,
        )

    return _build_output_record(
        record,
        proved_lean=proved_lean,
        negated_lean=None,
        proof_result=proof_result,
        disproof_result=None,
    )


def process_record_parallel(
    record: Dict[str, Any],
    *,
    proof_agent: GoedelProverAgent,
    proof_backend: RawBackend,
    disproof_agent: GoedelProverAgent,
    disproof_backend: RawBackend,
    proof_retries: int,
    disproof_retries: int,
    base_seed: int,
    event_logger: Optional[EventLoggerFn],
    token_limit: int = 0,
    incomplete_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run proof and disproof concurrently.  The first to succeed signals the
    other to cancel via a shared ``threading.Event``.
    """
    problem_id = record.get("problem_id")
    attempt = record.get("attempt")

    lean_statement = str(record.get("lean_statement_without_comment") or "")
    abbrev_decl = str(record.get("final_abbrev_declaration") or "")

    try:
        proved_lean = replace_abbrev_in_statement(lean_statement, abbrev_decl)
    except Exception as exc:
        return _error_record(record, "replace_abbrev_error", str(exc))

    try:
        negated_lean = negate_theorem_statement(proved_lean)
    except Exception as exc:
        return _error_record(record, "negate_theorem_error", str(exc))

    stop_proof = threading.Event()
    stop_disproof = threading.Event()

    init_proof_msgs, init_proof_partial = _extract_resume_state(incomplete_result, "proof_result")
    init_dis_msgs, init_dis_partial = _extract_resume_state(incomplete_result, "disproof_result")

    proof_result: Optional[GoedelProverResult] = None
    disproof_result: Optional[GoedelProverResult] = None

    def run_proof() -> None:
        nonlocal proof_result
        meta = {"problem_id": problem_id, "attempt": attempt, "direction": "proof"}
        proof_result = _run_prove_attempts(
            proof_agent, proof_backend, proved_lean,
            base_seed=base_seed,
            retries=proof_retries,
            event_logger=event_logger,
            metadata=meta,
            stop_event=stop_proof,
            token_limit=token_limit,
            initial_messages=init_proof_msgs,
            initial_partial_response=init_proof_partial,
        )
        if proof_result.proved:
            stop_disproof.set()

    def run_disproof() -> None:
        nonlocal disproof_result
        meta = {"problem_id": problem_id, "attempt": attempt, "direction": "disproof"}
        disproof_result = _run_prove_attempts(
            disproof_agent, disproof_backend, negated_lean,
            base_seed=(base_seed + 0x10000) & 0x7FFFFFFF,
            retries=disproof_retries,
            event_logger=event_logger,
            metadata=meta,
            stop_event=stop_disproof,
            token_limit=token_limit,
            initial_messages=init_dis_msgs,
            initial_partial_response=init_dis_partial,
        )
        if disproof_result.proved:
            stop_proof.set()

    proof_thread = threading.Thread(target=run_proof, daemon=True)
    disproof_thread = threading.Thread(target=run_disproof, daemon=True)
    proof_thread.start()
    disproof_thread.start()
    proof_thread.join()
    disproof_thread.join()

    return _build_output_record(
        record,
        proved_lean=proved_lean,
        negated_lean=negated_lean,
        proof_result=proof_result,
        disproof_result=disproof_result,
    )


# ---------------------------------------------------------------------------
# Output record helpers
# ---------------------------------------------------------------------------

def _build_output_record(
    record: Dict[str, Any],
    *,
    proved_lean: str,
    negated_lean: Optional[str],
    proof_result: Optional[GoedelProverResult],
    disproof_result: Optional[GoedelProverResult],
) -> Dict[str, Any]:
    proved = proof_result.proved if proof_result is not None else False
    disproved = disproof_result.proved if disproof_result is not None else False

    incomplete = (
        (proof_result is not None and proof_result.incomplete) or
        (disproof_result is not None and disproof_result.incomplete)
    )
    out: Dict[str, Any] = {
        "problem_id": record.get("problem_id"),
        "attempt": record.get("attempt"),
        "status": record.get("status"),
        "required_abbrev_name": record.get("required_abbrev_name"),
        "final_abbrev_declaration": record.get("final_abbrev_declaration"),
        "proved_lean": proved_lean,
        "negated_lean": negated_lean,
        "proved": proved,
        "disproved": disproved,
        "incomplete": incomplete,
        "proof_result": _result_to_dict(proof_result) if proof_result is not None else None,
        "disproof_result": _result_to_dict(disproof_result) if disproof_result is not None else None,
    }
    return out


def _error_record(record: Dict[str, Any], error_type: str, message: str) -> Dict[str, Any]:
    return {
        "problem_id": record.get("problem_id"),
        "attempt": record.get("attempt"),
        "status": record.get("status"),
        "required_abbrev_name": record.get("required_abbrev_name"),
        "final_abbrev_declaration": record.get("final_abbrev_declaration"),
        "proved_lean": None,
        "negated_lean": None,
        "proved": False,
        "disproved": False,
        "proof_result": None,
        "disproof_result": None,
        "error": {"type": error_type, "message": message, "traceback": traceback.format_exc()},
    }


# ---------------------------------------------------------------------------
# Top-level scheduler
# ---------------------------------------------------------------------------

class ProveFormalizationsScheduler:
    """
    Runs the prove_formalizations pipeline over a list of records.

    ``on_result`` is called (thread-safely) for each completed record; the
    caller is responsible for writing it to JSONL.
    """

    def __init__(
        self,
        cfg: ProveFormalizationsConfig,
        *,
        event_logger: Optional[EventLoggerFn] = None,
    ) -> None:
        self.cfg = cfg
        self.event_logger = event_logger

        # Build proof agent (always needed)
        goedel_proof_cfg = make_goedel_prover_config(cfg, workspace_suffix="proof")
        self._proof_agent = GoedelProverAgent(goedel_proof_cfg)
        self._proof_backend = make_goedel_backend(cfg)

        if cfg.print_agent_conv:
            self._proof_backend.set_verbose(True)
        if event_logger is not None:
            self._proof_backend.set_event_logger(event_logger)

        # Build disproof agent (when either disproof mode is active)
        self._disproof_agent: Optional[GoedelProverAgent] = None
        self._disproof_backend: Optional[RawBackend] = None
        if cfg.enable_parallel_disproof or cfg.enable_sequential_disproof:
            goedel_disproof_cfg = make_goedel_prover_config(cfg, workspace_suffix="disproof")
            self._disproof_agent = GoedelProverAgent(goedel_disproof_cfg)
            self._disproof_backend = make_goedel_backend(cfg)
            if cfg.print_agent_conv:
                self._disproof_backend.set_verbose(True)
            if event_logger is not None:
                self._disproof_backend.set_event_logger(event_logger)

    def run(
        self,
        records: List[Dict[str, Any]],
        on_result: Callable[[Dict[str, Any]], None],
        incomplete_map: Optional[Dict[Tuple, Dict[str, Any]]] = None,
    ) -> None:
        """
        Process all ``records`` and call ``on_result`` for each completed
        output record (may be called from multiple threads).

        ``incomplete_map`` maps ``(problem_id, attempt)`` to a previously
        saved incomplete result.  When present, the saved conversation history
        and partial response are threaded into the prover so it resumes from
        where the prior run left off.
        """
        cfg = self.cfg
        incomplete_map = incomplete_map or {}

        # Records with status != "success" or missing fields are passed through
        # immediately without running the prover.
        skippable, provable = [], []
        for rec in records:
            if (
                rec.get("status") == "success"
                and rec.get("final_abbrev_declaration")
                and rec.get("lean_statement_without_comment")
            ):
                provable.append(rec)
            else:
                skippable.append(rec)

        for rec in skippable:
            on_result({
                "problem_id": rec.get("problem_id"),
                "attempt": rec.get("attempt"),
                "status": rec.get("status"),
                "required_abbrev_name": rec.get("required_abbrev_name"),
                "final_abbrev_declaration": rec.get("final_abbrev_declaration"),
                "proved_lean": None,
                "negated_lean": None,
                "proved": False,
                "disproved": False,
                "incomplete": False,
                "proof_result": None,
                "disproof_result": None,
                "skipped": True,
                "skip_reason": f"status={rec.get('status')}",
            })

        outer_workers = cfg.parallelism // 2 if cfg.enable_parallel_disproof else cfg.parallelism
        outer_workers = max(1, outer_workers)

        def process(rec: Dict[str, Any]) -> Dict[str, Any]:
            problem_id = rec.get("problem_id", 0)
            attempt = rec.get("attempt", 0)
            base_seed = hash((problem_id, attempt)) & 0x7FFFFFFF
            saved = incomplete_map.get((problem_id, attempt))

            if cfg.enable_parallel_disproof:
                assert self._disproof_agent is not None
                assert self._disproof_backend is not None
                return process_record_parallel(
                    rec,
                    proof_agent=self._proof_agent,
                    proof_backend=self._proof_backend,
                    disproof_agent=self._disproof_agent,
                    disproof_backend=self._disproof_backend,
                    proof_retries=cfg.proof_retries,
                    disproof_retries=cfg.disproof_retries,
                    base_seed=base_seed,
                    event_logger=self.event_logger,
                    token_limit=cfg.limit_prover_tokens,
                    incomplete_result=saved,
                )
            elif cfg.enable_sequential_disproof:
                assert self._disproof_agent is not None
                assert self._disproof_backend is not None
                return process_record_sequential_with_disproof(
                    rec,
                    proof_agent=self._proof_agent,
                    proof_backend=self._proof_backend,
                    disproof_agent=self._disproof_agent,
                    disproof_backend=self._disproof_backend,
                    proof_retries=cfg.proof_retries,
                    disproof_retries=cfg.disproof_retries,
                    base_seed=base_seed,
                    event_logger=self.event_logger,
                    token_limit=cfg.limit_prover_tokens,
                    incomplete_result=saved,
                )
            else:
                return process_record_sequential(
                    rec,
                    proof_agent=self._proof_agent,
                    proof_backend=self._proof_backend,
                    proof_retries=cfg.proof_retries,
                    base_seed=base_seed,
                    event_logger=self.event_logger,
                    token_limit=cfg.limit_prover_tokens,
                    incomplete_result=saved,
                )

        with ThreadPoolExecutor(max_workers=outer_workers) as pool:
            future_to_rec: Dict[Future[Dict[str, Any]], Dict[str, Any]] = {
                pool.submit(process, rec): rec
                for rec in provable
            }
            for future in as_completed(future_to_rec):
                rec = future_to_rec[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = _error_record(
                        rec,
                        "unhandled_exception",
                        f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                    )
                on_result(result)

    def close(self) -> None:
        self._proof_agent.close()
        self._proof_backend.close()
        if self._disproof_agent is not None:
            self._disproof_agent.close()
        if self._disproof_backend is not None:
            self._disproof_backend.close()

    def __enter__(self) -> "ProveFormalizationsScheduler":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
