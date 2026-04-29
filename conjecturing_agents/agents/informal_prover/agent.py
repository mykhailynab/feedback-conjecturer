"""
InformalProverAgent — generates structured informal certification proofs.

Takes a solver's output (problem statement, solution trace, extracted answer,
and a formal Lean 4 theorem statement) and produces a clean, rigorous informal
proof that the answer is correct.  The output is later fed into the formal
prover agent's prompt to simplify its reasoning.

The agent uses the TIR backend but is primarily a single-turn text generator.
An optional Python tool can be enabled for computational verification.

Thread safety
-------------
When ``use_python_tool`` is enabled, a fresh JupyterTIRToolBackend is created
per ``generate_proof()`` call and closed in a finally block, so concurrent
calls each have their own isolated Python environment.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Dict, List, Optional

from conjecturing_agents.inference_backends.raw_base import EventLoggerFn
from conjecturing_agents.inference_backends.tir_base import (
    TIRBackend,
    TIRGenerationConfig,
)
from conjecturing_agents.tool_calling_backends.jupyter import JupyterTIRToolBackend

from .config import InformalProverConfig, InformalProverResult
from .prompts import INITIAL_USER_MESSAGE


class InformalProverAgent:
    """
    Agent that generates an informal certification proof for a given answer.

    Usage::

        cfg = InformalProverConfig()
        agent = InformalProverAgent(cfg)
        backend = OllamaTIRBackend(OllamaTIRConfig(
            model="qwen3.5",
            top_k=20, min_p=0.0, presence_penalty=1.5, repeat_penalty=1.0,
        ))

        result = agent.generate_proof(
            problem_statement="Find the value of ...",
            solution_trace="The solver reasoned that ...",
            answer="-1",
            lean_statement="import Mathlib\\n...",
            backend=backend,
            seed=42,
        )
        print(result.proof_text)
    """

    def __init__(self, cfg: Optional[InformalProverConfig] = None) -> None:
        self.cfg = cfg or InformalProverConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_proof(
        self,
        problem_statement: str,
        solution_trace: str,
        answer: str,
        lean_statement: str,
        backend: TIRBackend,
        *,
        seed: int = 0,
        event_logger: Optional[EventLoggerFn] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InformalProverResult:
        """
        Generate an informal certification proof.

        Args:
            problem_statement: The informal math problem text.
            solution_trace: The solver's reasoning (raw output or tail).
            answer: The extracted answer to certify.
            lean_statement: The formal Lean 4 theorem statement (for context).
                            Should have the formalized answer inserted.
            backend: A ``TIRBackend`` instance (e.g. ``OllamaTIRBackend``).
            seed: Random seed passed to TIRGenerationConfig.
            event_logger: Optional callable for structured event logging.
            metadata: Extra key-value pairs merged into every logged event.

        Returns:
            InformalProverResult with the generated proof text.
        """
        _meta = metadata or {}

        def _log(event_type: str, payload: Dict[str, Any]) -> None:
            if event_logger is not None:
                event_logger(event_type, {**payload, **_meta})

        t0 = time.time()

        # ------------------------------------------------------------------ #
        # Messages
        # ------------------------------------------------------------------ #
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.cfg.system_prompt},
            {
                "role": "user",
                "content": INITIAL_USER_MESSAGE.format(
                    problem_statement=problem_statement,
                    solution_trace=solution_trace,
                    answer=answer,
                    lean_statement=lean_statement,
                ),
            },
        ]

        # ------------------------------------------------------------------ #
        # Tools (optional Python tool)
        # ------------------------------------------------------------------ #
        tools: List[Dict[str, Any]] = []
        tool_handlers: Dict[str, Any] = {}
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
            # Generation config
            # ------------------------------------------------------------------ #
            gen_cfg = TIRGenerationConfig(
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                seed=seed,
                max_turns=self.cfg.max_turns,
                timeout_seconds=self.cfg.timeout_seconds,
            )

            _log("informal_prover_session_start", {
                "problem_chars": len(problem_statement),
                "trace_chars": len(solution_trace),
                "answer": answer,
                "seed": seed,
                "use_python_tool": self.cfg.use_python_tool,
            })

            # ------------------------------------------------------------------ #
            # Run session
            # ------------------------------------------------------------------ #
            session_result = backend.run_session(
                messages=messages,
                tools=tools,
                tool_handlers=tool_handlers,
                cfg=gen_cfg,
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
        turns_as_dicts = [dataclasses.asdict(t) for t in session_result.turns]

        _log("informal_prover_session_done", {
            "termination_reason": session_result.termination_reason,
            "turns_used": len(session_result.turns),
            "elapsed_ms": elapsed_ms,
            "proof_chars": len(session_result.final_text),
            "exception": session_result.exception,
        })

        return InformalProverResult(
            proof_text=session_result.final_text,
            termination_reason=session_result.termination_reason,
            elapsed_ms=elapsed_ms,
            turns=turns_as_dicts,
            exception=session_result.exception,
        )
    
        # TODO: new schema
        InformalProverResult(
            elapsed_ms=elapsed_ms,
            session_result=..., # we add this instead of the others
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        pass

    def __enter__(self) -> "InformalProverAgent":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
