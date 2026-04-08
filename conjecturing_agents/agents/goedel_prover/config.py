"""
Configuration and result types for the Goedel prover agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from conjecturing_agents.tool_calling_backends.lean4_compiler import LeanCompilerConfig


@dataclass
class GoedelProverConfig:
    name: str = "goedel_prover"

    # Jinja2 chat template — path to file (e.g. goedel_template.jinja)
    chat_template_path: str = "goedel_template.jinja"
    enable_thinking: bool = True

    # Generation
    max_tokens: int = 16384
    temperature: float = 0.6
    top_p: float = 0.95
    repeat_penalty: float = 1.0

    # Context budget — must match the model's max context length.
    # Prompts + max_tokens that exceed this are skipped rather than sent.
    context_tokens: int = 40960

    # Number of self-correction rounds after the initial attempt.
    # Total attempts = max_rounds + 1.
    max_rounds: int = 2

    # Lean compiler used inside the generation loop (required).
    lean: Optional[LeanCompilerConfig] = None

    # Whether to truncate error lists to 8 (matches the default pipeline).
    truncate_errors: bool = True


@dataclass
class GoedelProverResult:
    proved: bool

    # "proved" | "max_rounds_exhausted" | "context_exceeded" |
    # "no_code_block" | "splice_error" | "exception"
    termination_reason: str

    raw_output: str          # last model generation
    proof_text: str          # last extracted lean4 code block (or "")
    full_code: str           # last assembled theorem+proof file (or "")

    rounds_used: int         # total rounds attempted (1 = initial only)
    elapsed_ms: int

    rounds: List[Dict[str, Any]] = field(default_factory=list)
