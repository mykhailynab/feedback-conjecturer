"""
Token-for-token equivalence tests for GoedelProverAgent prompt building.

Two layers are tested for both the initial and correction prompts:

  1. Message-list content — pure Python, no model required.
     Verifies that the ``role`` and ``content`` fields we build match those
     produced by the reference ``DeepSeekCoTHandler``.

  2. Rendered prompt string — requires the HF tokenizer.
     Compares the output of ``render_with_template(goedel_template.jinja, msgs)``
     against ``tokenizer.apply_chat_template(msgs)``.  Skipped when the
     tokenizer directory is absent.

Note on round numbering
-----------------------
The reference ``generate_correction_prompt`` is called with
``current_correction_round_num=N`` (1-based) and labels the failed attempt as
"Round N-1".  Our agent calls ``_build_correction_messages`` with
``failed_round_num=round_idx`` (0-based) and labels it "Round round_idx".
After round_idx=0 fails → label "Round 0"; reference correction_round=1 →
label "Round 0".  They match.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from conjecturing_agents.agents.goedel_prover.agent import GoedelProverAgent
from conjecturing_agents.agents.goedel_prover.config import GoedelProverConfig
from conjecturing_agents.agents.goedel_prover.lean_utils import format_lean_errors
from tests.goedel_prover.fixtures import (
    CODE_15_LINES,
    ERROR_SINGLE_LINE,
    LEAN_STMT_SIMPLE,
    MODEL_OUTPUT_WITH_PROOF,
)
from tests.goedel_prover.reference import (
    reference_correction_prompt,
    reference_get_error_str,
    reference_initial_prompt,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[2]
_TOKENIZER_PATH = _REPO_ROOT / "tokenizers" / "goedel_prover_hf_tokenizer"

_needs_tokenizer = pytest.mark.skipif(
    not _TOKENIZER_PATH.exists(),
    reason=f"HF tokenizer not found at {_TOKENIZER_PATH}",
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(_TOKENIZER_PATH), use_fast=True)


@pytest.fixture(scope="module")
def agent(tmp_path_factory):
    """GoedelProverAgent with a dummy lean config (no actual compilation)."""
    from conjecturing_agents.tool_calling_backends.lean4_compiler import LeanCompilerConfig
    tmp = tmp_path_factory.mktemp("lean")
    lean_cfg = LeanCompilerConfig(project_dir=str(tmp))
    cfg = GoedelProverConfig(
        tokenizer_path=str(_TOKENIZER_PATH),
        lean=lean_cfg,
    )
    return GoedelProverAgent(cfg)


# ---------------------------------------------------------------------------
# Initial prompt — message content
# ---------------------------------------------------------------------------

def test_initial_message_role(agent):
    from conjecturing_agents.agents.goedel_prover.lean_utils import normalize_for_prompt
    formal = normalize_for_prompt(LEAN_STMT_SIMPLE)
    msgs = agent._build_initial_messages(formal)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


def test_initial_message_content_matches_reference(agent, tokenizer):
    """
    The user message content must be identical to the reference.
    The reference builds ``formal_statement = lean4_code.split(':= by')[0] + ':= by sorry'``
    then inserts it into the prompt template.
    """
    from conjecturing_agents.agents.goedel_prover.lean_utils import normalize_for_prompt
    formal = normalize_for_prompt(LEAN_STMT_SIMPLE)
    our_msgs = agent._build_initial_messages(formal)

    _, ref_msgs = reference_initial_prompt(LEAN_STMT_SIMPLE, tokenizer)

    assert our_msgs[0]["content"] == ref_msgs[0]["content"]


# ---------------------------------------------------------------------------
# Initial prompt — rendered string
# ---------------------------------------------------------------------------

@_needs_tokenizer
def test_initial_rendered_prompt_matches_reference(agent, tokenizer):
    from conjecturing_agents.agents.goedel_prover.lean_utils import normalize_for_prompt
    formal = normalize_for_prompt(LEAN_STMT_SIMPLE)
    our_msgs = agent._build_initial_messages(formal)

    our_rendered = tokenizer.apply_chat_template(our_msgs, tokenize=False, add_generation_prompt=True)
    ref_rendered, _ = reference_initial_prompt(LEAN_STMT_SIMPLE, tokenizer)

    assert our_rendered == ref_rendered


# ---------------------------------------------------------------------------
# Correction prompt — message content
# ---------------------------------------------------------------------------

def _make_correction_inputs(agent, tokenizer):
    """
    Build the inputs for a round-0 correction (i.e., the first self-correction
    after the initial attempt fails).

    Returns (our_msgs, ref_msgs, error_str) so both test layers can reuse them.
    """
    from conjecturing_agents.agents.goedel_prover.lean_utils import normalize_for_prompt
    formal = normalize_for_prompt(LEAN_STMT_SIMPLE)
    initial_msgs = agent._build_initial_messages(formal)

    error_str = format_lean_errors(CODE_15_LINES, [ERROR_SINGLE_LINE])
    ref_error_str = reference_get_error_str(CODE_15_LINES, [ERROR_SINGLE_LINE])

    # Round 0 failed: build correction messages
    our_msgs = agent._build_correction_messages(
        prev_messages=initial_msgs,
        prev_assistant_output=MODEL_OUTPUT_WITH_PROOF,
        error_feedback=error_str,
        failed_round_num=0,
    )

    # Reference: correction_round=1 (1-based) → labels failure as "Round 0"
    _, ref_msgs = reference_correction_prompt(
        lean4_code=LEAN_STMT_SIMPLE,
        history_messages=list(initial_msgs),
        prev_output=MODEL_OUTPUT_WITH_PROOF,
        error_str=ref_error_str,
        tokenizer=tokenizer,
        correction_round=1,
    )

    return our_msgs, ref_msgs, error_str, ref_error_str


def test_correction_messages_length(agent, tokenizer):
    our_msgs, ref_msgs, _, _ = _make_correction_inputs(agent, tokenizer)
    assert len(our_msgs) == len(ref_msgs)


def test_correction_messages_roles(agent, tokenizer):
    our_msgs, ref_msgs, _, _ = _make_correction_inputs(agent, tokenizer)
    for our, ref in zip(our_msgs, ref_msgs):
        assert our["role"] == ref["role"]


def test_correction_assistant_content(agent, tokenizer):
    """The assistant turn must carry the previous model output verbatim."""
    our_msgs, ref_msgs, _, _ = _make_correction_inputs(agent, tokenizer)
    our_assistant = next(m for m in our_msgs if m["role"] == "assistant")
    ref_assistant = next(m for m in ref_msgs if m["role"] == "assistant")
    assert our_assistant["content"] == ref_assistant["content"]


def test_correction_user_feedback_content(agent, tokenizer):
    """
    The final user message (error feedback + request) must match the reference.
    This is the most sensitive part: it contains the formatted error string and
    the exact wording the model was trained on.
    """
    our_msgs, ref_msgs, _, _ = _make_correction_inputs(agent, tokenizer)
    our_feedback = our_msgs[-1]["content"]
    ref_feedback = ref_msgs[-1]["content"]
    assert our_feedback == ref_feedback


# ---------------------------------------------------------------------------
# Correction prompt — rendered string
# ---------------------------------------------------------------------------

@_needs_tokenizer
def test_correction_rendered_prompt_matches_reference(agent, tokenizer):
    our_msgs, _, _, ref_error_str = _make_correction_inputs(agent, tokenizer)

    our_rendered = tokenizer.apply_chat_template(our_msgs, tokenize=False, add_generation_prompt=True)

    ref_rendered, _ = reference_correction_prompt(
        lean4_code=LEAN_STMT_SIMPLE,
        history_messages=agent._build_initial_messages(
            __import__(
                "conjecturing_agents.agents.goedel_prover.lean_utils",
                fromlist=["normalize_for_prompt"],
            ).normalize_for_prompt(LEAN_STMT_SIMPLE)
        ),
        prev_output=MODEL_OUTPUT_WITH_PROOF,
        error_str=ref_error_str,
        tokenizer=tokenizer,
        correction_round=1,
    )

    assert our_rendered == ref_rendered
