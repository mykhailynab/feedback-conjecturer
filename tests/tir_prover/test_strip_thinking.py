"""
Unit tests for strip_thinking logic in the TIR prover pipeline.

Verifies that when strip_thinking=True, the reasoning_content field is removed
from prior assistant messages before rendering via apply_chat_template, and that
when strip_thinking=False, the reasoning content is preserved in the rendered
prompt.

Requirements:
- The Qwen3.6-35B-A3B tokenizer at tokenizers/Qwen3.6-35B-A3B

Run with:
    PYTHONPATH=. python -m pytest tests/tir_prover/test_strip_thinking.py -v
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest


_TOKENIZER_PATH = "tokenizers/Qwen3.6-35B-A3B"

# A simple Lean-like tool definition that the model will recognise.
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lean",
            "description": "Compile a Lean 4 file and return diagnostics.",
            "parameters": {
                "type": "object",
                "required": ["code"],
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Lean 4 file content.",
                    }
                },
            },
        },
    },
]

# Sentinel strings used to verify presence/absence in rendered prompts.
_REASONING_TEXT = "SENTINEL_REASONING: I should try norm_num to close this goal."
_ASSISTANT_CONTENT = "Let me try to prove this theorem."
_TOOL_CALL_CODE = "import Mathlib\n\ntheorem one_plus_one : 1 + 1 = 2 := by norm_num"
_TOOL_RESULT = "[OK] Compilation successful."


def _build_messages() -> List[Dict[str, Any]]:
    """Build a realistic multi-turn message list as it would look after turn 0.

    Sequence: system → user → assistant (with reasoning_content + tool_calls) → tool
    This is what ``messages`` looks like at the start of turn 1 in run_session().
    """
    return [
        {"role": "system", "content": "You are a Lean 4 theorem prover."},
        {
            "role": "user",
            "content": (
                "Prove the following theorem by replacing `sorry` with a "
                "valid proof.\n\n"
                "```lean\nimport Mathlib\n\n"
                "theorem one_plus_one : 1 + 1 = 2 := by sorry\n```"
            ),
        },
        {
            "role": "assistant",
            "content": _ASSISTANT_CONTENT,
            "reasoning_content": _REASONING_TEXT,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "lean",
                        "arguments": {"code": _TOOL_CALL_CODE},
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "name": "lean",
            "content": _TOOL_RESULT,
        },
    ]


def _strip_thinking(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply the same strip logic as tir_base.py:290-298."""
    return [
        {k: v for k, v in m.items() if k != "reasoning_content"}
        if m.get("role") == "assistant" and "reasoning_content" in m
        else m
        for m in messages
    ]


@pytest.fixture(scope="module")
def tokenizer():
    if not os.path.isdir(_TOKENIZER_PATH):
        pytest.skip(f"Tokenizer not found at {_TOKENIZER_PATH}")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(_TOKENIZER_PATH)


def _render(tokenizer, messages: List[Dict[str, Any]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tools=_TOOLS,
        tokenize=False,
        add_generation_prompt=True,
    )


class TestStripThinking:
    def test_no_strip_preserves_reasoning(self, tokenizer):
        """With strip_thinking=False, the rendered prompt contains the reasoning text."""
        messages = _build_messages()
        rendered = _render(tokenizer, messages)

        assert _REASONING_TEXT in rendered, (
            "Expected reasoning text in rendered prompt when strip_thinking=False"
        )
        # Should have a non-empty <think> block
        assert "<think>\n" + _REASONING_TEXT in rendered

    def test_strip_removes_reasoning(self, tokenizer):
        """With strip_thinking=True, the rendered prompt does NOT contain the reasoning text."""
        messages = _build_messages()
        stripped = _strip_thinking(messages)
        rendered = _render(tokenizer, stripped)

        assert _REASONING_TEXT not in rendered, (
            "Reasoning text should NOT appear in rendered prompt when strip_thinking=True"
        )
        # The template still emits an empty <think></think> block for assistant
        # messages after last_query_index.
        assert "<think>\n\n</think>" in rendered, (
            "Expected empty <think> block for the prior assistant turn"
        )

    def test_strip_preserves_content_and_tool_calls(self, tokenizer):
        """Stripping reasoning_content must not affect the rest of the message."""
        messages = _build_messages()
        stripped = _strip_thinking(messages)
        rendered = _render(tokenizer, stripped)

        assert _ASSISTANT_CONTENT in rendered, (
            "Assistant content text should survive stripping"
        )
        assert _TOOL_CALL_CODE in rendered, (
            "Tool call arguments should survive stripping"
        )
        assert _TOOL_RESULT in rendered, (
            "Tool result should survive stripping"
        )

    def test_stripped_vs_unstripped_differ(self, tokenizer):
        """The two renderings must actually differ (the test is not vacuous)."""
        messages = _build_messages()
        rendered_full = _render(tokenizer, messages)
        rendered_stripped = _render(tokenizer, _strip_thinking(messages))

        assert rendered_full != rendered_stripped, (
            "Rendered prompts with and without strip_thinking should differ"
        )
        # The only difference should be the reasoning content.
        # Verify that removing the reasoning text from the full rendering
        # makes them equal.
        normalised_full = rendered_full.replace(_REASONING_TEXT, "")
        normalised_stripped = rendered_stripped.replace(_REASONING_TEXT, "")
        assert normalised_full == normalised_stripped, (
            "The only difference between stripped and unstripped should be "
            "the reasoning text itself"
        )
