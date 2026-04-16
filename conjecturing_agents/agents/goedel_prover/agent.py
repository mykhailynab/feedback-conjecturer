"""
GoedelProverAgent — drives multi-round proof generation + Lean compilation.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from conjecturing_agents.inference_backends.raw_backend import (
    EventLoggerFn,
    RawBackend,
    RawGenerationConfig,
)
from conjecturing_agents.tool_calling_backends.lean4_compiler import Lean4CompilerBackend

from .config import GoedelProverConfig, GoedelProverResult
from .lean_utils import (
    extract_lean_code_block,
    format_lean_errors,
    normalize_for_prompt,
    replace_statement_in_proof,
)
from .prompts import (
    CORRECTION_USER_PROMPT,
    INITIAL_USER_PROMPT,
    render_with_template,
)


class GoedelProverAgent:
    """
    Agent that attempts to prove a Lean theorem using the Goedel prover model.

    Usage::

        agent = GoedelProverAgent(cfg)
        result = agent.prove_theorem(theorem_statement, backend, seed=42)

    The theorem statement must contain a ``theorem ... := by sorry`` placeholder.
    The caller is responsible for providing a ``RawBackend`` instance (either
    ``VLLMRawBackend`` or ``OllamaBackend``).
    """

    def __init__(self, cfg: Optional[GoedelProverConfig] = None) -> None:
        self.cfg = cfg or GoedelProverConfig()
        if self.cfg.lean is None:
            raise ValueError("GoedelProverConfig.lean must be set")
        self.lean_backend = Lean4CompilerBackend(self.cfg.lean)
        self.chat_template = Path(self.cfg.chat_template_path).read_text(
            encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_initial_messages(
        self, formal_statement: str
    ) -> List[Dict[str, str]]:
        content = INITIAL_USER_PROMPT.format(formal_statement=formal_statement)
        return [{"role": "user", "content": content}]

    def _build_correction_messages(
        self,
        prev_messages: List[Dict[str, str]],
        prev_assistant_output: str,
        error_feedback: str,
        failed_round_num: int,
    ) -> List[Dict[str, str]]:
        msgs = list(prev_messages)
        msgs.append({"role": "assistant", "content": prev_assistant_output})
        msgs.append({
            "role": "user",
            "content": CORRECTION_USER_PROMPT.format(
                round_num=failed_round_num,
                error_feedback=error_feedback,
            ),
        })
        return msgs

    def _render_prompt(self, messages: List[Dict[str, str]]) -> str:
        return render_with_template(
            self.chat_template,
            messages,
            add_generation_prompt=True,
            enable_thinking=self.cfg.enable_thinking,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(
        self,
        prompt: str,
        backend: RawBackend,
        gen_cfg: RawGenerationConfig,
        stream_callback: Optional[Callable[[str], None]],
        stop_event: Optional[threading.Event] = None,
        token_limit: int = 0,
        prompt_tokens: int = 0,
    ) -> Tuple[str, bool]:
        """Generate text, returning ``(text, hit_token_limit)``.

        Uses streaming whenever ``stop_event`` or ``token_limit`` is set.
        A single ``_streaming_stop`` event is passed to the backend; it is
        set internally when the token budget is exhausted (or when the
        external ``stop_event`` fires), so the backend loop exits promptly.
        """
        use_streaming = stop_event is not None or token_limit > 0 or stream_callback is not None
        if not use_streaming:
            return backend.generate(prompt, gen_cfg).text, False

        _streaming_stop = threading.Event()
        chunks: List[str] = []
        chars_since_recount = 0
        hit_token_limit = False

        for chunk in backend.generate_streaming(prompt, gen_cfg, stop_event=_streaming_stop):
            # Check external cancellation first
            if stop_event is not None and stop_event.is_set():
                _streaming_stop.set()
                break
            chunks.append(chunk)
            if stream_callback is not None:
                stream_callback(chunk)
            # Periodically count tokens to check the budget
            if token_limit > 0:
                chars_since_recount += len(chunk)
                if chars_since_recount >= 200:
                    chars_since_recount = 0
                    gen_tokens = backend.count_tokens("".join(chunks))
                    if prompt_tokens + gen_tokens >= token_limit:
                        hit_token_limit = True
                        _streaming_stop.set()
                        break

        return "".join(chunks), hit_token_limit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prove_theorem(
        self,
        theorem_statement: str,
        backend: RawBackend,
        *,
        seed: int = 0,
        stream_callback: Optional[Callable[[str], None]] = None,
        event_logger: Optional[EventLoggerFn] = None,
        metadata: Optional[Dict[str, Any]] = None,
        stop_event: Optional[threading.Event] = None,
        token_limit: int = 0,
        initial_messages: Optional[List[Dict[str, str]]] = None,
        partial_response: str = "",
    ) -> GoedelProverResult:
        """
        Attempt to prove ``theorem_statement`` using up to
        ``cfg.max_rounds + 1`` generation + Lean compilation cycles.

        Args:
            theorem_statement: Full Lean file ending in ``theorem ... := by sorry``.
            backend: A ``RawBackend`` instance (vLLM or Ollama).
            seed: Base seed; each round adds the round index.
            stream_callback: If provided, called with each streamed text chunk.
            event_logger: Optional callable for structured event logging.
            metadata: Extra key-value pairs merged into every logged event
                (e.g. ``{"problem_id": 42, "attempt": 1}``).
            stop_event: When set, the current generation round is cancelled.
            token_limit: Stop (incompletely) when the rendered prompt reaches
                this many tokens.  0 = unlimited.
            initial_messages: Resume from a saved conversation history instead
                of starting fresh.  The first round uses this history as-is.
            partial_response: If the prior run was cut off mid-generation,
                this is the assistant text produced before the cut.  On the
                first round the model continues from this point.
        """
        _meta = metadata or {}

        def _log(event_type: str, payload: Dict[str, Any]) -> None:
            if event_logger is not None:
                event_logger(event_type, {**payload, **_meta})

        t0 = time.time()
        rounds: List[Dict[str, Any]] = []

        try:
            formal_statement = normalize_for_prompt(theorem_statement)
        except ValueError as exc:
            return GoedelProverResult(
                proved=False,
                termination_reason="splice_error",
                raw_output="",
                proof_text="",
                full_code="",
                rounds_used=0,
                elapsed_ms=int((time.time() - t0) * 1000),
                rounds=[{"error": str(exc)}],
            )

        _log("prover_session_start", {
            "theorem_statement": theorem_statement,
            "formal_statement": formal_statement,
            "seed": seed,
            "max_rounds": self.cfg.max_rounds,
            "token_limit": token_limit,
            "resuming": initial_messages is not None,
        })

        if initial_messages is not None:
            messages: List[Dict[str, str]] = list(initial_messages)
            _resume_partial = partial_response
        else:
            messages = self._build_initial_messages(formal_statement)
            _resume_partial = ""

        # Mutable snapshots updated at the top of each round; used to build
        # the resume state when termination_reason == "token_limit".
        save_messages: List[Dict[str, str]] = list(messages)
        save_partial: str = ""

        termination_reason = "max_rounds_exhausted"
        raw_output = ""
        proof_text = ""
        full_code = ""
        prompt = ""

        for round_idx in range(self.cfg.max_rounds + 1):
            if stop_event is not None and stop_event.is_set():
                termination_reason = "cancelled"
                break

            # Snapshot messages before this round so we can save them if the
            # round is interrupted mid-generation by the token limit.
            save_messages = list(messages)

            round_record: Dict[str, Any] = {"round": round_idx}
            round_t0 = time.time()

            # --- Render prompt.  On the first round of a resumed session,
            #     append the saved partial response so the model continues
            #     from exactly where it left off. ---
            base_prompt = self._render_prompt(messages)
            if round_idx == 0 and _resume_partial:
                prompt = base_prompt + _resume_partial
            else:
                prompt = base_prompt

            prompt_tokens = backend.count_tokens(prompt)
            round_record["prompt"] = prompt
            round_record["prompt_tokens"] = prompt_tokens

            # --- Token limit check (between-round) ---
            if token_limit > 0 and prompt_tokens >= token_limit:
                save_partial = ""
                round_record["termination_reason"] = "token_limit"
                round_record["elapsed_ms"] = int((time.time() - round_t0) * 1000)
                rounds.append(round_record)
                _log("prover_round_done", {
                    "round": round_idx,
                    "prompt_tokens": prompt_tokens,
                    "prompt_text": prompt,
                    "raw_output": "",
                    "lean_ok": None,
                    "lean_timed_out": None,
                    "lean_oom": None,
                    "lean_error_count": None,
                    "elapsed_ms": round_record["elapsed_ms"],
                    "termination_reason": "token_limit",
                })
                termination_reason = "token_limit"
                break

            # --- Existing context budget check ---
            if prompt_tokens + self.cfg.max_tokens > self.cfg.context_tokens:
                round_record["termination_reason"] = "context_exceeded"
                round_record["elapsed_ms"] = int((time.time() - round_t0) * 1000)
                rounds.append(round_record)
                _log("prover_round_done", {
                    "round": round_idx,
                    "prompt_tokens": prompt_tokens,
                    "prompt_text": prompt,
                    "raw_output": "",
                    "lean_ok": None,
                    "lean_timed_out": None,
                    "lean_oom": None,
                    "lean_error_count": None,
                    "elapsed_ms": round_record["elapsed_ms"],
                    "termination_reason": "context_exceeded",
                })
                termination_reason = "context_exceeded"
                break

            # --- Generate ---
            gen_cfg_round = RawGenerationConfig(
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                seed=seed + round_idx,
                repeat_penalty=self.cfg.repeat_penalty,
            )
            raw_output_new, hit_limit = self._generate(
                prompt, backend, gen_cfg_round, stream_callback,
                stop_event=stop_event,
                token_limit=token_limit,
                prompt_tokens=prompt_tokens,
            )

            # On the first round of a resumed session, prepend the saved
            # partial so downstream sees the full assistant turn.
            if round_idx == 0 and _resume_partial:
                raw_output = _resume_partial + raw_output_new
                _resume_partial = ""
            else:
                raw_output = raw_output_new
            round_record["raw_output"] = raw_output

            # --- Mid-stream token limit ---
            if hit_limit:
                save_partial = raw_output
                round_record["termination_reason"] = "token_limit"
                round_record["elapsed_ms"] = int((time.time() - round_t0) * 1000)
                rounds.append(round_record)
                _log("prover_round_done", {
                    "round": round_idx,
                    "prompt_tokens": prompt_tokens,
                    "prompt_text": prompt,
                    "raw_output": raw_output,
                    "lean_ok": None,
                    "lean_timed_out": None,
                    "lean_oom": None,
                    "lean_error_count": None,
                    "elapsed_ms": round_record["elapsed_ms"],
                    "termination_reason": "token_limit",
                })
                termination_reason = "token_limit"
                break

            # --- Post-generation cancellation check ---
            if stop_event is not None and stop_event.is_set():
                round_record["termination_reason"] = "cancelled"
                round_record["elapsed_ms"] = int((time.time() - round_t0) * 1000)
                rounds.append(round_record)
                termination_reason = "cancelled"
                break

            # --- Extract code block ---
            code_block = extract_lean_code_block(raw_output)
            if not code_block:
                round_record["termination_reason"] = "no_code_block"
                round_record["elapsed_ms"] = int((time.time() - round_t0) * 1000)
                rounds.append(round_record)
                _log("prover_round_done", {
                    "round": round_idx,
                    "prompt_tokens": prompt_tokens,
                    "prompt_text": prompt,
                    "raw_output": raw_output,
                    "lean_ok": None,
                    "lean_timed_out": None,
                    "lean_oom": None,
                    "lean_error_count": None,
                    "elapsed_ms": round_record["elapsed_ms"],
                    "termination_reason": "no_code_block",
                })
                if round_idx < self.cfg.max_rounds:
                    # No code to correct with; rebuild from scratch next round
                    messages = self._build_initial_messages(formal_statement)
                else:
                    termination_reason = "no_code_block"
                continue

            proof_text = code_block
            round_record["proof_text"] = proof_text

            # --- Splice proof into statement ---
            full_code = replace_statement_in_proof(formal_statement, code_block)
            round_record["full_code"] = full_code

            if full_code.startswith("**Error**"):
                round_record["termination_reason"] = "splice_error"
                round_record["elapsed_ms"] = int((time.time() - round_t0) * 1000)
                rounds.append(round_record)
                _log("prover_round_done", {
                    "round": round_idx,
                    "prompt_tokens": prompt_tokens,
                    "prompt_text": prompt,
                    "raw_output": raw_output,
                    "lean_ok": None,
                    "lean_timed_out": None,
                    "lean_oom": None,
                    "lean_error_count": None,
                    "elapsed_ms": round_record["elapsed_ms"],
                    "termination_reason": "splice_error",
                })
                termination_reason = "splice_error"
                break

            # --- Lean compilation ---
            compile_result = self.lean_backend.compile_code(full_code)
            round_record["lean_ok"] = compile_result.ok
            round_record["lean_timed_out"] = compile_result.timed_out
            round_record["lean_oom"] = compile_result.oom
            round_record["lean_error_count"] = len(compile_result.json_errors)
            round_record["lean_relative_path"] = compile_result.relative_path
            round_record["elapsed_ms"] = int((time.time() - round_t0) * 1000)
            rounds.append(round_record)

            _lean_reason = "proved" if compile_result.ok else "lean_failed"
            _log("prover_round_done", {
                "round": round_idx,
                "prompt_tokens": prompt_tokens,
                "prompt_text": prompt,
                "raw_output": raw_output,
                "lean_ok": compile_result.ok,
                "lean_timed_out": compile_result.timed_out,
                "lean_oom": compile_result.oom,
                "lean_error_count": len(compile_result.json_errors),
                "elapsed_ms": round_record["elapsed_ms"],
                "termination_reason": _lean_reason,
            })

            if compile_result.ok:
                termination_reason = "proved"
                break

            # --- Build correction for next round ---
            if round_idx < self.cfg.max_rounds:
                all_errors = compile_result.json_errors + compile_result.sorry_warnings
                error_feedback = format_lean_errors(
                    full_code,
                    all_errors,
                    truncate=self.cfg.truncate_errors,
                    max_message_chars=self.cfg.max_error_message_chars,
                )
                messages = self._build_correction_messages(
                    messages,
                    raw_output,
                    error_feedback,
                    failed_round_num=round_idx,
                )

        elapsed_ms = int((time.time() - t0) * 1000)

        # Build resume state when stopped by the token limit.
        incomplete = termination_reason == "token_limit"
        if incomplete:
            _resume_prompt = self._render_prompt(save_messages)
            if save_partial:
                _resume_prompt = _resume_prompt + save_partial
            total_context_tokens = backend.count_tokens(_resume_prompt)
        else:
            total_context_tokens = 0

        _log("prover_session_done", {
            "proved": termination_reason == "proved",
            "termination_reason": termination_reason,
            "rounds_used": len(rounds),
            "elapsed_ms": elapsed_ms,
            "incomplete": incomplete,
            "total_context_tokens": total_context_tokens,
        })
        return GoedelProverResult(
            proved=(termination_reason == "proved"),
            termination_reason=termination_reason,
            raw_output=raw_output,
            proof_text=proof_text,
            full_code=full_code,
            rounds_used=len(rounds),
            elapsed_ms=elapsed_ms,
            rounds=rounds,
            incomplete=incomplete,
            total_context_tokens=total_context_tokens,
            conversation_history=list(save_messages) if incomplete else [],
            partial_response=save_partial if incomplete else "",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.lean_backend.close()

    def __enter__(self) -> "GoedelProverAgent":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()