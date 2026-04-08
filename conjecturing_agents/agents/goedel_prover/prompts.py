"""
Prompt constants and chat-template rendering for the Goedel prover.

Prompts are kept verbatim from the original Goedel pipeline because the
fine-tuned model is sensitive to the exact prompt format it was trained on.
Original pipeline: https://github.com/Goedel-LM/Goedel-Prover-V2
"""
from __future__ import annotations

from typing import Any, Dict, List

from jinja2 import Environment


# ============================================================
# Prompt constants
# (kept verbatim from the original pipeline)
# ============================================================

# build_initial_messages content after .strip()
INITIAL_USER_PROMPT = (
    "Complete the following Lean 4 code:\n\n"
    "```lean4\n"
    "{formal_statement}```\n\n"
    "Before producing the Lean 4 code to formally prove the given theorem, "
    "provide a detailed proof plan outlining the main proof steps and strategies.\n"
    "The plan should highlight key ideas, intermediate lemmas, and proof structures "
    "that will guide the construction of the final formal proof."
)

# build_correction_messages user content
CORRECTION_USER_PROMPT = (
    "The proof (Round {round_num}) is not correct. "
    "Following is the compilation error message, where we use "
    "<error></error> to signal the position of the error.\n\n"
    "{error_feedback}\n\n"
    "Before producing the Lean 4 code to formally prove the given theorem, "
    "provide a detailed analysis of the error message."
)


# ============================================================
# Template rendering
# ============================================================

def render_with_template(
    chat_template: str,
    messages: List[Dict[str, Any]],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
) -> str:
    """
    Render an HF-style Jinja2 chat template into a raw prompt string.
    """
    class _Obj:
        def __init__(self, d: Dict[str, Any]) -> None:
            for k, v in d.items():
                setattr(self, k, v)

    msg_objs = [_Obj(m) for m in messages]
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    tmpl = env.from_string(chat_template)
    return tmpl.render(
        tools=[],
        messages=msg_objs,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
