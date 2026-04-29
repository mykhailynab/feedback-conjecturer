"""
Configuration and result types for the Goedel prover agent.
"""
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from conjecturing_agents.tool_calling_backends.lean4_compiler import LeanCompilerConfig


@dataclass
class GoedelProverConfig:
    name: str = "goedel_prover"

    # HF tokenizer directory — used to render prompts via apply_chat_template.
    tokenizer_path: str = "tokenizers/goedel_prover_hf_tokenizer"
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

    # Maximum characters per individual error message (error['data']).
    # Prevents tactics like interval_cases from producing thousands of
    # unsolved-goal entries that blow up the correction prompt.
    # 0 = no truncation.
    max_error_message_chars: int = 0

    # ------------------------------------------------------------------
    # CLI integration
    # ------------------------------------------------------------------

    @classmethod
    def add_cli_args(
        cls,
        parser: ArgumentParser,
        prefix: str = "goedel",
        defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register CLI args for user-facing GoedelProverConfig fields."""
        d = defaults or {}
        pre = prefix
        dst = prefix.replace("-", "_")
        defs = cls()

        parser.add_argument(
            f"--{pre}-tokenizer-path",
            dest=f"{dst}_tokenizer_path",
            default=d.get("tokenizer_path", defs.tokenizer_path),
            help="HF tokenizer path for exact token counting.",
        )
        parser.add_argument(
            f"--{pre}-max-rounds",
            dest=f"{dst}_max_rounds",
            type=int,
            default=d.get("max_rounds", defs.max_rounds),
            help="Self-correction rounds for Goedel prover (0 = initial attempt only).",
        )
        parser.add_argument(
            f"--{pre}-max-tokens",
            dest=f"{dst}_max_tokens",
            type=int,
            default=d.get("max_tokens", defs.max_tokens),
            help="Max tokens to generate per Goedel round.",
        )
        parser.add_argument(
            f"--{pre}-temperature",
            dest=f"{dst}_temperature",
            type=float,
            default=d.get("temperature", defs.temperature),
            help="Sampling temperature for the Goedel model.",
        )
        parser.add_argument(
            f"--{pre}-top-p",
            dest=f"{dst}_top_p",
            type=float,
            default=d.get("top_p", defs.top_p),
            help="Top-p (nucleus) sampling parameter for the Goedel model.",
        )
        parser.add_argument(
            f"--{pre}-repeat-penalty",
            dest=f"{dst}_repeat_penalty",
            type=float,
            default=d.get("repeat_penalty", defs.repeat_penalty),
            help="Repetition penalty for the Goedel model (1.0 = no penalty).",
        )
        parser.add_argument(
            f"--{pre}-context-tokens",
            dest=f"{dst}_context_tokens",
            type=int,
            default=d.get("context_tokens", defs.context_tokens),
            help="Model max context length for pre-flight token budget checks.",
        )
        parser.add_argument(
            f"--{pre}-max-error-message-chars",
            dest=f"{dst}_max_error_message_chars",
            type=int,
            default=d.get("max_error_message_chars", defs.max_error_message_chars),
            help=(
                "Truncate each Lean error message (error['data']) to this many characters "
                "in the correction prompt. 0 = no truncation (default). Suggested value: 2000."
            ),
        )

    @classmethod
    def from_parsed_args(
        cls,
        args: Any,
        prefix: str = "goedel",
        **overrides: Any,
    ) -> "GoedelProverConfig":
        """Construct from an argparse namespace.

        Internal fields (name, enable_thinking, lean, truncate_errors) stay at
        defaults unless provided via ``overrides``.
        """
        dst = prefix.replace("-", "_")
        kwargs: Dict[str, Any] = {
            "tokenizer_path": getattr(args, f"{dst}_tokenizer_path"),
            "max_rounds": getattr(args, f"{dst}_max_rounds"),
            "max_tokens": getattr(args, f"{dst}_max_tokens"),
            "temperature": getattr(args, f"{dst}_temperature"),
            "top_p": getattr(args, f"{dst}_top_p"),
            "repeat_penalty": getattr(args, f"{dst}_repeat_penalty"),
            "context_tokens": getattr(args, f"{dst}_context_tokens"),
            "max_error_message_chars": getattr(args, f"{dst}_max_error_message_chars"),
        }
        kwargs.update(overrides)
        return cls(**kwargs)


@dataclass
class GoedelProverResult:
    proved: bool

    # "proved" | "max_rounds_exhausted" | "context_exceeded" |
    # "no_code_block" | "splice_error" | "exception" |
    # "token_limit" | "cancelled"
    termination_reason: str

    raw_output: str          # last model generation
    proof_text: str          # last extracted lean4 code block (or "")
    full_code: str           # last assembled theorem+proof file (or "")

    rounds_used: int         # total rounds attempted (1 = initial only)
    elapsed_ms: int

    rounds: List[Dict[str, Any]] = field(default_factory=list)

    # Progressive proving / resume support.
    # When termination_reason == "token_limit":
    #   token_limit_triggered=True, conversation_history holds the messages snapshot
    #   before the interrupted round, partial_response holds the assistant
    #   text generated before the cut-off, and total_context_tokens is the
    #   token count of render(conversation_history)+partial_response.
    # On resume, the caller rebuilds that prompt and continues generation.
    token_limit_triggered: bool = False
    total_context_tokens: int = 0
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    partial_response: str = ""
