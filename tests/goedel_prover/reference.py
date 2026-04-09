"""
Thin wrappers around the original Goedel pipeline code in
``goedel_original_scripts/src/utils.py``.

We import from the original source rather than copying it, so that any drift
between our implementation and the reference is caught by tests even if the
reference file changes.

The helpers below expose a stable interface used by the test modules so that
the path manipulation lives in one place.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make goedel_original_scripts/src importable without installing it.
_REF_SRC = Path(__file__).parents[2] / "goedel_original_scripts" / "src"
if str(_REF_SRC) not in sys.path:
    sys.path.insert(0, str(_REF_SRC))

from utils import DeepSeekCoTHandler, get_error_str  # noqa: E402  (path set above)

_handler = DeepSeekCoTHandler()


def reference_get_error_str(code: str, errors: list, error_thres: bool = True) -> str:
    """
    Verbatim reference error formatter.

    ``error_thres=True`` matches the default used in ``inference.py``
    (``args.error_thres = True``).
    """
    return get_error_str(code, errors, error_thres)


def reference_initial_prompt(lean4_code: str, tokenizer) -> tuple[str, list]:
    """
    Reference initial prompt string and message list.

    Equivalent to ``DeepSeekCoTHandler.prover_inference``.
    Returns ``(rendered_prompt_str, messages)``.
    """
    return _handler.prover_inference(lean4_code, tokenizer)


def reference_correction_prompt(
    lean4_code: str,
    history_messages: list,
    prev_output: str,
    error_str: str,
    tokenizer,
    correction_round: int,
) -> tuple[str, list]:
    """
    Reference correction prompt string and message list.

    ``correction_round`` is 1-based (1 = fixing the initial attempt).
    Equivalent to ``DeepSeekCoTHandler.generate_correction_prompt``.
    Returns ``(rendered_prompt_str, messages)``.
    """
    return _handler.generate_correction_prompt(
        lean4_code_original_stmt=lean4_code,
        history_messages_from_prev_round=history_messages,
        prev_round_llm_raw_output=prev_output,
        error_message_for_prev_round=error_str,
        tokenizer=tokenizer,
        current_correction_round_num=correction_round,
    )
