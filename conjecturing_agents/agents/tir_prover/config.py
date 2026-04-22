"""
Configuration and result types for the TIR prover agent.
"""
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from conjecturing_agents.tool_calling_backends.lean4_compiler import LeanCompilerConfig
from conjecturing_agents.tool_calling_backends.jupyter import JupyterKernelConfig

from .prompts import (
    DEFAULT_TIR_PROVER_SYSTEM_PROMPT,
    DEFAULT_TIR_PROVER_LEAN_TOOL_DESCRIPTION,
    DEFAULT_TIR_PROVER_LEAN_FINAL_TOOL_DESCRIPTION,
    DEFAULT_TIR_PROVER_PYTHON_TOOL_DESCRIPTION,
)


@dataclass
class TIRProverConfig:
    name: str = "tir_prover"

    # Prompts
    system_prompt: str = DEFAULT_TIR_PROVER_SYSTEM_PROMPT
    lean_tool_description: str = DEFAULT_TIR_PROVER_LEAN_TOOL_DESCRIPTION
    lean_final_tool_description: str = DEFAULT_TIR_PROVER_LEAN_FINAL_TOOL_DESCRIPTION
    python_tool_description: str = DEFAULT_TIR_PROVER_PYTHON_TOOL_DESCRIPTION

    # Generation
    max_tokens: int = 16384
    temperature: float = 0.6
    top_p: float = 0.95

    # Session limits
    # max_turns is the upper bound on tool-call + response turns within one
    # prove_theorem() call.  Each lean or python invocation counts as one turn.
    max_turns: int = 32
    timeout_seconds: float = 600.0

    # Context management
    strip_thinking: bool = False  # strip CoT from prior turns to save context

    # Tools
    use_lean_tool: bool = True    # intermediate lean calls (lean_final is always present)
    use_python_tool: bool = True

    # Lean compiler config — REQUIRED (must be set by the caller).
    lean: Optional[LeanCompilerConfig] = None

    # Jupyter kernel config — used only when use_python_tool=True.
    jupyter: JupyterKernelConfig = field(default_factory=JupyterKernelConfig)

    # ------------------------------------------------------------------
    # CLI integration
    # ------------------------------------------------------------------

    @classmethod
    def add_cli_args(
        cls,
        parser: ArgumentParser,
        prefix: str = "tir",
        defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register CLI args for user-facing TIRProverConfig fields.

        Internal fields (name, prompts, lean, jupyter) stay at defaults and
        are overridden in factory helpers.
        """
        d = defaults or {}
        pre = prefix
        dst = prefix.replace("-", "_")
        defs = cls()

        parser.add_argument(
            f"--{pre}-max-tokens",
            dest=f"{dst}_max_tokens",
            type=int,
            default=d.get("max_tokens", defs.max_tokens),
            help="Max tokens to generate per TIR turn. Default: %(default)s.",
        )
        parser.add_argument(
            f"--{pre}-temperature",
            dest=f"{dst}_temperature",
            type=float,
            default=d.get("temperature", defs.temperature),
            help="Sampling temperature for the TIR prover. Default: %(default)s.",
        )
        parser.add_argument(
            f"--{pre}-top-p",
            dest=f"{dst}_top_p",
            type=float,
            default=d.get("top_p", defs.top_p),
            help="Top-p (nucleus) sampling for the TIR prover. Default: %(default)s.",
        )
        parser.add_argument(
            f"--{pre}-max-turns",
            dest=f"{dst}_max_turns",
            type=int,
            default=d.get("max_turns", defs.max_turns),
            help="Maximum tool-call turns per TIR session. Default: %(default)s.",
        )
        parser.add_argument(
            f"--{pre}-timeout-seconds",
            dest=f"{dst}_timeout_seconds",
            type=float,
            default=d.get("timeout_seconds", defs.timeout_seconds),
            help="Wall-clock timeout (seconds) per TIR session. Default: %(default)s.",
        )
        parser.add_argument(
            f"--{pre}-no-lean-tool",
            dest=f"{dst}_use_lean_tool",
            action="store_false",
            default=d.get("use_lean_tool", defs.use_lean_tool),
            help="Disable the intermediate lean tool (lean_final is always available).",
        )
        parser.add_argument(
            f"--{pre}-no-python-tool",
            dest=f"{dst}_use_python_tool",
            action="store_false",
            default=d.get("use_python_tool", defs.use_python_tool),
            help="Disable the Python (Jupyter) tool for the TIR prover.",
        )
        parser.add_argument(
            f"--{pre}-strip-thinking",
            dest=f"{dst}_strip_thinking",
            action="store_true",
            default=d.get("strip_thinking", defs.strip_thinking),
            help="Strip reasoning/thinking content from prior turns to save context.",
        )
        parser.add_argument(
            f"--{pre}-no-strip-thinking",
            dest=f"{dst}_strip_thinking",
            action="store_false",
            help="Keep reasoning/thinking content in prior turns (default).",
        )

    @classmethod
    def from_parsed_args(
        cls,
        args: Any,
        prefix: str = "tir",
        **overrides: Any,
    ) -> "TIRProverConfig":
        """Construct from an argparse namespace.

        Internal fields (name, prompts, lean, jupyter) stay at defaults
        unless provided via ``overrides``.
        """
        dst = prefix.replace("-", "_")
        kwargs: Dict[str, Any] = {
            "max_tokens": getattr(args, f"{dst}_max_tokens"),
            "temperature": getattr(args, f"{dst}_temperature"),
            "top_p": getattr(args, f"{dst}_top_p"),
            "max_turns": getattr(args, f"{dst}_max_turns"),
            "timeout_seconds": getattr(args, f"{dst}_timeout_seconds"),
            "use_lean_tool": getattr(args, f"{dst}_use_lean_tool"),
            "use_python_tool": getattr(args, f"{dst}_use_python_tool"),
            "strip_thinking": getattr(args, f"{dst}_strip_thinking"),
        }
        kwargs.update(overrides)
        return cls(**kwargs)


@dataclass
class TIRProverResult:
    proved: bool

    # "proved" | "final_answer" | "max_turns_exhausted" | "deadline_exceeded" |
    # "stop_event" | "no_tokens" | "exception:..." | "cancelled"
    termination_reason: str

    # Full Lean file that compiled successfully ([OK] from the lean tool), or "".
    proved_lean: str

    turns_used: int    # total TIR turns (each tool call + assistant response = 1 turn)
    elapsed_ms: int

    turns: List[Dict[str, Any]] = field(default_factory=list)
    exception: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Scheduler-compatibility fields (mirror GoedelProverResult layout so
    # the existing scheduler and _extract_resume_state work unchanged).
    # ------------------------------------------------------------------ #
    incomplete: bool = False
    total_context_tokens: int = 0
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    partial_response: str = ""

    # Partial assistant turn that was in progress when the stream was
    # interrupted (e.g. by the token limit).  Logged for debugging but not
    # used for resume — TIR always restarts from the beginning of the
    # interrupted turn.
    partial_assistant_turn: Optional[Dict[str, Any]] = None
