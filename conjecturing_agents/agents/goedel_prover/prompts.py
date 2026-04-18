"""
Prompt constants for the Goedel prover.

Prompts are kept verbatim from the original Goedel pipeline because the
fine-tuned model is sensitive to the exact prompt format it was trained on.
Original pipeline: https://github.com/Goedel-LM/Goedel-Prover-V2
"""
from __future__ import annotations


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
