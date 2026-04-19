"""
Configuration and result types for the TIR prover agent.
"""
from __future__ import annotations

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

    # Tools
    use_lean_tool: bool = True    # intermediate lean calls (lean_final is always present)
    use_python_tool: bool = True

    # Lean compiler config — REQUIRED (must be set by the caller).
    lean: Optional[LeanCompilerConfig] = None

    # Jupyter kernel config — used only when use_python_tool=True.
    jupyter: JupyterKernelConfig = field(default_factory=JupyterKernelConfig)


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
    # TIR sessions are not interrupted mid-stream, so incomplete is always
    # False and the resume fields are always empty.
    incomplete: bool = False
    total_context_tokens: int = 0
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    partial_response: str = ""
