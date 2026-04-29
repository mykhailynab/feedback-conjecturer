"""
Configuration and result types for the informal proof agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from conjecturing_agents.inference_backends.tir_base import TIRSessionResult

from conjecturing_agents.tool_calling_backends.jupyter import JupyterKernelConfig

from .prompts import (
    DEFAULT_INFORMAL_PROVER_SYSTEM_PROMPT,
    DEFAULT_INFORMAL_PROVER_PYTHON_TOOL_DESCRIPTION,
)


@dataclass
class InformalProverConfig:
    """Configuration for the informal proof agent.

    Recommended OllamaTIRConfig backend defaults for this agent:
        top_k=20, min_p=0.0, presence_penalty=1.5, repeat_penalty=1.0
    These are set on the backend, not per-session.
    """

    name: str = "informal_prover"

    # Prompts
    system_prompt: str = DEFAULT_INFORMAL_PROVER_SYSTEM_PROMPT
    python_tool_description: str = DEFAULT_INFORMAL_PROVER_PYTHON_TOOL_DESCRIPTION

    # Generation
    max_tokens: int = 262144
    temperature: float = 1.0
    top_p: float = 0.95

    # Session limits
    max_turns: int = 1
    timeout_seconds: float = 600.0

    # Tools
    use_python_tool: bool = False
    jupyter: JupyterKernelConfig = field(default_factory=JupyterKernelConfig)


@dataclass
class InformalProverResult:
    """Result of an informal proof generation."""

    # The generated informal certification proof text.
    proof_text: str

    elapsed_ms: int

    # Full TIR session result (conversation_history, partial_assistant_turn, etc.).
    session_result: Optional[TIRSessionResult] = None
