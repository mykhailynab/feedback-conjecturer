"""
GoedelProverAgent — drives multi-round proof generation + Lean compilation.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from conjecturing_agents.inference_backends.raw_backend import (
    RawBackend,
    RawGenerationConfig,
)
from conjecturing_agents.tool_calling_backends.lean4_compiler import Lean4CompilerBackend

from .config import GoedelProverConfig, GoedelProverResult
from .lean_utils import (
    extract_lean4_code_block,
    format_lean_errors,
    normalize_for_prompt,
    replace_statement_in_proof,
)
from .prompts import (
    _CORRECTION_USER_PROMPT,
    _INITIAL_USER_PROMPT,
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
        content = _INITIAL_USER_PROMPT.format(formal_statement=formal_statement)
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
            "content": _CORRECTION_USER_PROMPT.format(
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
    ) -> str:
        if stream_callback is not None:
            chunks: List[str] = []
            for chunk in backend.generate_streaming(prompt, gen_cfg):
                chunks.append(chunk)
                stream_callback(chunk)
            return "".join(chunks)
        return backend.generate(prompt, gen_cfg).text

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
    ) -> GoedelProverResult:
        """
        Attempt to prove ``theorem_statement`` using up to
        ``cfg.max_rounds + 1`` generation + Lean compilation cycles.

        Args:
            theorem_statement: Full Lean file ending in ``theorem ... := by sorry``.
            backend: A ``RawBackend`` instance (vLLM or Ollama).
            seed: Base seed; each round adds the round index.
            stream_callback: If provided, called with each streamed text chunk.
        """
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

        messages = self._build_initial_messages(formal_statement)

        termination_reason = "max_rounds_exhausted"
        raw_output = ""
        proof_text = ""
        full_code = ""

        for round_idx in range(self.cfg.max_rounds + 1):
            round_record: Dict[str, Any] = {"round": round_idx}
            round_t0 = time.time()

            # --- Render + context budget check ---
            prompt = self._render_prompt(messages)
            prompt_tokens = backend.count_tokens(prompt)
            round_record["prompt_tokens"] = prompt_tokens

            if prompt_tokens + self.cfg.max_tokens > self.cfg.context_tokens:
                round_record["termination_reason"] = "context_exceeded"
                rounds.append(round_record)
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
            raw_output = self._generate(prompt, backend, gen_cfg_round, stream_callback)
            round_record["raw_output"] = raw_output

            # --- Extract code block ---
            code_block = extract_lean4_code_block(raw_output)
            if not code_block:
                round_record["termination_reason"] = "no_code_block"
                round_record["elapsed_ms"] = int((time.time() - round_t0) * 1000)
                rounds.append(round_record)
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
                termination_reason = "splice_error"
                break

            # --- Lean compilation ---
            compile_result = self.lean_backend.compile_code(full_code)
            round_record["lean_ok"] = compile_result.ok
            round_record["lean_timed_out"] = compile_result.timed_out
            round_record["lean_error_count"] = len(compile_result.json_errors)
            round_record["lean_relative_path"] = compile_result.relative_path
            round_record["elapsed_ms"] = int((time.time() - round_t0) * 1000)
            rounds.append(round_record)

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
                )
                messages = self._build_correction_messages(
                    messages,
                    raw_output,
                    error_feedback,
                    failed_round_num=round_idx,
                )

        return GoedelProverResult(
            proved=(termination_reason == "proved"),
            termination_reason=termination_reason,
            raw_output=raw_output,
            proof_text=proof_text,
            full_code=full_code,
            rounds_used=len(rounds),
            elapsed_ms=int((time.time() - t0) * 1000),
            rounds=rounds,
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