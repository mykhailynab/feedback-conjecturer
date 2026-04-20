"""
TIRProverAgent — proves Lean 4 theorems via Tool-Integrated Reasoning.

Unlike GoedelProverAgent (which pre-renders a raw prompt and loops over
generate / compile / correct rounds), TIRProverAgent lets the model drive
the iteration itself through native tool calls:

  - The model reads the theorem, calls **lean** with proof attempts, reads
    error diagnostics, and revises until it submits a successful proof via
    **lean_final**.
  - Optionally, the model can call **python** for mathematical exploration
    before or during the Lean proof search.

The multi-turn conversation loop and tool dispatch are handled by
TIRBackend.run_session(); the agent is responsible only for:
  1. Building the initial message list.
  2. Wiring up the Lean, lean_final, and Python tool handlers.
  3. Validating lean_final submissions (theorem signature check).
  4. Detecting proof success and stopping the session immediately.
  5. Returning a TIRProverResult.

Thread safety
-------------
The Lean tool backend (LeanCompilerToolAdapter) is stateless and owned by the
agent instance — safe to share across concurrent prove_theorem() calls.
The Jupyter kernel backend is created fresh for each prove_theorem() call
and closed in a finally block, so concurrent calls each have their own
isolated Python environment.
"""
from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from conjecturing_agents.inference_backends.raw_backend import EventLoggerFn
from conjecturing_agents.inference_backends.tir_base import (
    TIRBackend,
    TIRGenerationConfig,
)
from conjecturing_agents.lean_regex import contains_theorem_signature
from conjecturing_agents.tool_calling_backends.lean4_compiler import LeanCompilerToolAdapter
from conjecturing_agents.tool_calling_backends.lean4_compiler.prompt_formatting import build_tool_facing_feedback
from conjecturing_agents.lean_regex import extract_lean_code_block_or_text
from conjecturing_agents.tool_calling_backends.jupyter import (
    JupyterTIRToolBackend,
)

from .config import TIRProverConfig, TIRProverResult
from .prompts import INITIAL_USER_MESSAGE, LEAN_FINAL_MISSING_THEOREM_CORRECTION


class TIRProverAgent:
    """
    Agent that attempts to prove a Lean theorem using Tool-Integrated Reasoning.

    Usage::

        cfg = TIRProverConfig(lean=LeanCompilerConfig(project_dir="/path/to/mathlib4"))
        agent = TIRProverAgent(cfg)
        backend = OllamaTIRBackend(OllamaTIRConfig(model="qwen3.5"))

        result = agent.prove_theorem(theorem_statement, backend, seed=42)
        if result.proved:
            print(result.proved_lean)

    The theorem_statement should be a complete Lean 4 file ending in
    ``theorem ... := by sorry`` or ``theorem ... := sorry``.
    """

    def __init__(self, cfg: Optional[TIRProverConfig] = None) -> None:
        self.cfg = cfg or TIRProverConfig()
        if self.cfg.lean is None:
            raise ValueError("TIRProverConfig.lean must be set")

        self._lean_tool = LeanCompilerToolAdapter(
            description=self.cfg.lean_tool_description,
            cfg=self.cfg.lean,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prove_theorem(
        self,
        theorem_statement: str,
        backend: TIRBackend,
        *,
        seed: int = 0,
        event_logger: Optional[EventLoggerFn] = None,
        metadata: Optional[Dict[str, Any]] = None,
        stop_event: Optional[threading.Event] = None,
        token_limit: int = 0,
        initial_messages: Optional[List[Dict[str, Any]]] = None,
        partial_response: str = "",
    ) -> TIRProverResult:
        """
        Attempt to prove ``theorem_statement`` using native tool-call iterations.

        Args:
            theorem_statement: Full Lean 4 file ending with ``theorem ... := by sorry``.
            backend: A ``TIRBackend`` instance (e.g. ``OllamaTIRBackend``).
            seed: Base random seed passed to TIRGenerationConfig.
            event_logger: Optional callable for structured event logging.
            metadata: Extra key-value pairs merged into every logged event.
            stop_event: When set, the current tool-call turn is cancelled and
                the session terminates early.
            token_limit: Stop the session (marking it incomplete) when the
                rendered prompt reaches this many tokens.  0 = unlimited.
                TIR checks the token count both *between* turns (before each
                LLM call) and *during* streaming (every ~200 chars of generated
                text), so the cutoff may fire mid-stream. Any partial assistant
                turn in progress when the limit fires is stored in
                TIRSessionResult.partial_assistant_turn for logging; it is not
                included in messages_at_cutoff, so the resumed session restarts
                cleanly from the beginning of the interrupted turn.
                Requires the backend to have a token counter registered
                (OllamaTIRConfig.tokenizer_path).
            initial_messages: If provided, resume the session from this saved
                conversation history rather than starting fresh.  Used by the
                scheduler to continue an incomplete session from a prior run.
            partial_response: Not used by TIR.  Accepted for API compatibility
                with GoedelProverAgent.  Unlike GoedelProverAgent, the TIR
                resume always restarts from the beginning of the interrupted turn.

        Returns:
            TIRProverResult with proved=True iff a lean or lean_final tool call returned [OK].
        """
        _meta = metadata or {}

        def _log(event_type: str, payload: Dict[str, Any]) -> None:
            if event_logger is not None:
                event_logger(event_type, {**payload, **_meta})

        t0 = time.time()

        # ------------------------------------------------------------------ #
        # Proof-success tracking (mutated by the lean handler closure).
        # Using a dict rather than nonlocal booleans so the closure captures
        # the container, which is compatible with all Python 3.x scoping rules.
        # ------------------------------------------------------------------ #
        _state: Dict[str, Any] = {"proved": False, "proved_lean": ""}
        # Event set by lean_final_handle when a proof is accepted; used to
        # stop the session immediately without starting another LLM turn.
        _proof_stop = threading.Event()

        # ------------------------------------------------------------------ #
        # Lean tool handler — wraps LeanCompilerToolAdapter to intercept [OK].
        # ------------------------------------------------------------------ #
        lean_cfg = self._lean_tool.cfg

        def lean_handle(tool_name: str, arguments: Dict[str, Any]) -> str:
            code_raw = arguments.get("code", "")
            code = (
                extract_lean_code_block_or_text(code_raw)
                if lean_cfg.auto_extract_code_block
                else code_raw.strip()
            )
            _log("tir_lean_compile_start", {"code_chars": len(code)})
            result = self._lean_tool.backend.compile_code(code)
            _log("tir_lean_compile_done", {
                "ok": result.ok,
                "error_count": len(result.json_errors),
                "warning_count": len(result.json_warnings),
                "timed_out": result.timed_out,
                "oom": result.oom,
                "elapsed_ms": result.elapsed_ms,
            })
            # The lean tool is for intermediate checks only — the model may
            # compile helper lemmas or partial files that succeed without
            # proving the full theorem.  Only lean_final sets proved=True.
            return build_tool_facing_feedback(result, cfg=lean_cfg)

        # ------------------------------------------------------------------ #
        # lean_final handler — validates the theorem signature, compiles,
        # marks proved, and stops the session on success.
        # ------------------------------------------------------------------ #
        lean_final_tool_def: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": "lean_final",
                "description": self.cfg.lean_final_tool_description,
                "parameters": {
                    "type": "object",
                    "required": ["code"],
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Complete Lean 4 file with sorry replaced by the proof",
                        }
                    },
                },
            },
        }

        def lean_final_handle(tool_name: str, arguments: Dict[str, Any]) -> str:
            code_raw = arguments.get("code", "")
            code = (
                extract_lean_code_block_or_text(code_raw)
                if lean_cfg.auto_extract_code_block
                else code_raw.strip()
            )
            # Verify the submitted file contains the original theorem signature
            # (whitespace-normalized).  The model must send the complete Lean
            # file with sorry replaced — not a different theorem or a bare
            # tactic block.
            if not contains_theorem_signature(theorem_statement, code):
                return LEAN_FINAL_MISSING_THEOREM_CORRECTION.format(
                    theorem_statement=theorem_statement,
                )
            _log("tir_lean_final_start", {"code_chars": len(code)})
            result = self._lean_tool.backend.compile_code(code)
            _log("tir_lean_final_done", {
                "ok": result.ok,
                "error_count": len(result.json_errors),
                "warning_count": len(result.json_warnings),
                "timed_out": result.timed_out,
                "oom": result.oom,
                "elapsed_ms": result.elapsed_ms,
            })
            if result.ok and not _state["proved"]:
                _state["proved"] = True
                _state["proved_lean"] = code
                # Stop the session immediately — no further LLM turns needed.
                _proof_stop.set()
            return build_tool_facing_feedback(result, cfg=lean_cfg)

        # ------------------------------------------------------------------ #
        # Assemble tools list and handlers
        # ------------------------------------------------------------------ #
        tools = [lean_final_tool_def]
        tool_handlers: Dict[str, Any] = {"lean_final": lean_final_handle}

        if self.cfg.use_lean_tool:
            tools.append(self._lean_tool.tool_def)
            tool_handlers[lean_cfg.tool_name] = lean_handle

        # ------------------------------------------------------------------ #
        # Initial messages
        # ------------------------------------------------------------------ #
        if initial_messages is not None:
            messages: List[Dict[str, Any]] = list(initial_messages)
        else:
            messages = [
                {"role": "system", "content": self.cfg.system_prompt},
                {
                    "role": "user",
                    "content": INITIAL_USER_MESSAGE.format(
                        theorem_statement=theorem_statement
                    ),
                },
            ]

        # ------------------------------------------------------------------ #
        # TIR generation config
        # ------------------------------------------------------------------ #
        gen_cfg = TIRGenerationConfig(
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            seed=seed,
            max_turns=self.cfg.max_turns,
            timeout_seconds=self.cfg.timeout_seconds,
            token_limit=token_limit,
        )

        _log("tir_prover_session_start", {
            "theorem_chars": len(theorem_statement),
            "seed": seed,
            "max_turns": self.cfg.max_turns,
            "timeout_seconds": self.cfg.timeout_seconds,
            "use_python_tool": self.cfg.use_python_tool,
            "resuming": initial_messages is not None,
        })

        # ------------------------------------------------------------------ #
        # Combined stop event — fires when either a proof is found (lean_final
        # returned [OK]) or the external stop_event fires.  Passed to
        # run_session so the session terminates immediately after a successful
        # proof without starting another LLM turn.
        # ------------------------------------------------------------------ #
        class _CombinedStopEvent:
            def is_set(self) -> bool:
                return _proof_stop.is_set() or (
                    stop_event is not None and stop_event.is_set()
                )

        _combined_stop = _CombinedStopEvent()

        # ------------------------------------------------------------------ #
        # Jupyter tool — created fresh per call for thread safety.
        # Each prove_theorem() call gets its own isolated Python kernel.
        # ------------------------------------------------------------------ #
        jupyter_tool: Optional[JupyterTIRToolBackend] = None
        try:
            if self.cfg.use_python_tool:
                jupyter_tool = JupyterTIRToolBackend(
                    description=self.cfg.python_tool_description,
                    cfg=self.cfg.jupyter,
                )
                tools.append(jupyter_tool.tool_def)
                tool_handlers[self.cfg.jupyter.tool_name] = jupyter_tool.handle_call

            # ------------------------------------------------------------------ #
            # Run the TIR session
            # ------------------------------------------------------------------ #
            session_result = backend.run_session(
                messages=messages,
                tools=tools,
                tool_handlers=tool_handlers,
                cfg=gen_cfg,
                stop_event=_combined_stop,
            )

        finally:
            if jupyter_tool is not None:
                try:
                    jupyter_tool.close()
                except Exception:
                    pass

        # ------------------------------------------------------------------ #
        # Build result
        # ------------------------------------------------------------------ #
        elapsed_ms = int((time.time() - t0) * 1000)
        proved = _state["proved"]

        # When a Lean call succeeded, override the backend's termination reason
        # with "proved" regardless of whether the model made further tool calls
        # afterward.
        if proved:
            termination_reason = "proved"
        elif stop_event is not None and stop_event.is_set():
            termination_reason = "cancelled"
        else:
            termination_reason = session_result.termination_reason

        incomplete = session_result.incomplete

        _log("tir_prover_session_done", {
            "proved": proved,
            "termination_reason": termination_reason,
            "turns_used": len(session_result.turns),
            "elapsed_ms": elapsed_ms,
            "incomplete": incomplete,
            "exception": session_result.exception,
        })

        turns_as_dicts = [dataclasses.asdict(t) for t in session_result.turns]

        return TIRProverResult(
            proved=proved,
            termination_reason=termination_reason,
            proved_lean=_state["proved_lean"],
            turns_used=len(session_result.turns),
            elapsed_ms=elapsed_ms,
            turns=turns_as_dicts,
            exception=session_result.exception,
            incomplete=incomplete,
            # Save the full message list so the scheduler can resume the
            # session via initial_messages in a subsequent --continue run.
            conversation_history=session_result.messages_at_cutoff if incomplete else [],
            # NOTE: not used in --continue
            partial_assistant_turn=session_result.partial_assistant_turn,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._lean_tool.close()

    def __enter__(self) -> "TIRProverAgent":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
