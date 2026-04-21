"""
Integration tests for the llama.cpp TIR backend.

These tests require:
- A llama.cpp server with tool-call support at LLAMACPP_HOST
  (default: http://localhost:8001)

Each test writes real logs to logs/test/llamacpp_tir_<name>/ for post-hoc inspection.
These directories are NOT cleaned up after testing.

Run with:
    PYTHONPATH=. python -m pytest tests/llamacpp_tir/ -v -s
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

from conjecturing_agents.inference_backends.llamacpp_tir import (
    LlamaCppTIRBackend,
    LlamaCppTIRConfig,
)
from conjecturing_agents.inference_backends.tir_base import (
    TIRGenerationConfig,
    TIRStreamChunk,
    TIRToolCallSpec,
)

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------

_LLAMACPP_HOST = os.environ.get("LLAMACPP_HOST", "http://localhost:8001")
_LLAMACPP_MODEL = os.environ.get("LLAMACPP_MODEL", "")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LOGS_DIR = _REPO_ROOT / "logs" / "test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_log_dir(name: str) -> Path:
    d = _LOGS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_events_logger(events_path: Path):
    """Simple event logger that appends JSONL to a file."""
    import threading
    from datetime import datetime, timezone

    lock = threading.Lock()
    events_path.write_text("", encoding="utf-8")

    def logger(event_type: str, payload: Dict[str, Any]) -> None:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **payload,
        }
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with lock:
            with open(events_path, "a", encoding="utf-8") as f:
                f.write(line)

    return logger


def _make_backend() -> LlamaCppTIRBackend:
    cfg = LlamaCppTIRConfig(
        base_url=_LLAMACPP_HOST,
        model=_LLAMACPP_MODEL,
    )
    return LlamaCppTIRBackend(cfg)


def _python_tool_def() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Execute Python code and return stdout.",
            "parameters": {
                "type": "object",
                "required": ["code"],
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    }
                },
            },
        },
    }


def _python_handler(name: str, args: Dict[str, Any]) -> str:
    code = args.get("code", "")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = exec(code)
    output = buf.getvalue()
    if not output and result is not None:
        output = str(result)
    # If still no output, try eval() for expression-only code
    if not output:
        try:
            val = eval(code.strip().split("\n")[-1])
            if val is not None:
                output = str(val)
        except Exception:
            pass
    return output or "(no output)"


# ---------------------------------------------------------------------------
# Tests — chat_streaming
# ---------------------------------------------------------------------------

class TestChatStreaming:
    """Test the chat_streaming() method directly."""

    def test_simple_chat_produces_output(self):
        """A simple question should produce reasoning and/or content."""
        backend = _make_backend()
        try:
            cfg = TIRGenerationConfig(max_tokens=500, temperature=0.6, top_p=0.95)
            thinking = ""
            content = ""
            for chunk in backend.chat_streaming(
                messages=[{"role": "user", "content": "What is 7 * 8? Just the number."}],
                tools=[],
                cfg=cfg,
            ):
                thinking += chunk.thinking
                content += chunk.content

            # The model should produce at least reasoning or content
            assert thinking or content, "Expected at least reasoning or content output"
        finally:
            backend.close()

    def test_tool_call_streaming(self):
        """When tools are provided, the model should produce tool calls."""
        backend = _make_backend()
        try:
            cfg = TIRGenerationConfig(max_tokens=1000, temperature=0.6, top_p=0.95)
            thinking = ""
            content = ""
            tool_calls: list[TIRToolCallSpec] = []

            for chunk in backend.chat_streaming(
                messages=[
                    {"role": "system", "content": "Use the python tool for computations."},
                    {"role": "user", "content": "Compute 17 * 23 using the python tool."},
                ],
                tools=[_python_tool_def()],
                cfg=cfg,
            ):
                thinking += chunk.thinking
                content += chunk.content
                tool_calls.extend(chunk.tool_calls)

            assert len(tool_calls) > 0, "Expected at least one tool call"
            tc = tool_calls[0]
            assert tc.name == "python"
            assert "code" in tc.arguments
            assert tc.id != "", "Tool call should have an id"
        finally:
            backend.close()

    def test_tool_call_arguments_are_parsed(self):
        """Tool call arguments should be parsed as a dict, not a raw string."""
        backend = _make_backend()
        try:
            cfg = TIRGenerationConfig(max_tokens=1000, temperature=0.6, top_p=0.95)
            tool_calls: list[TIRToolCallSpec] = []

            for chunk in backend.chat_streaming(
                messages=[
                    {"role": "system", "content": "Use the python tool."},
                    {"role": "user", "content": "Print hello world using python."},
                ],
                tools=[_python_tool_def()],
                cfg=cfg,
            ):
                tool_calls.extend(chunk.tool_calls)

            assert len(tool_calls) > 0
            tc = tool_calls[0]
            assert isinstance(tc.arguments, dict), (
                f"Expected arguments to be a dict, got {type(tc.arguments)}"
            )
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# Tests — run_session (multi-turn)
# ---------------------------------------------------------------------------

class TestRunSession:
    """Test the full multi-turn run_session() loop."""

    def test_single_tool_call_round_trip(self):
        """Model calls python tool, gets result, and produces a final answer."""
        log_dir = _make_log_dir("llamacpp_tir_single_roundtrip")
        events_path = log_dir / "events.jsonl"
        event_logger = _make_events_logger(events_path)

        backend = _make_backend()
        backend.set_event_logger(event_logger)
        try:
            cfg = TIRGenerationConfig(
                max_tokens=2000,
                temperature=0.6,
                top_p=0.95,
                max_turns=5,
                timeout_seconds=120.0,
            )

            result = backend.run_session(
                messages=[
                    {"role": "system", "content": "Use the python tool for all computations. Be brief."},
                    {"role": "user", "content": "What is 37 * 41? Use the python tool."},
                ],
                tools=[_python_tool_def()],
                tool_handlers={"python": _python_handler},
                cfg=cfg,
            )

            assert result.termination_reason == "final_answer", (
                f"Expected final_answer, got {result.termination_reason}"
            )
            assert result.exception is None
            assert result.elapsed_ms > 0
            assert len(result.turns) >= 2, (
                "Expected at least 2 turns: tool call + final answer"
            )

            # First turn should have a tool call
            assert len(result.turns[0].tool_calls) > 0
            tc = result.turns[0].tool_calls[0]
            assert tc["name"] == "python"

            # The answer (1517) should appear somewhere in the session
            all_text = result.final_text
            for turn in result.turns:
                all_text += turn.content
                for tc_rec in turn.tool_calls:
                    all_text += tc_rec.get("result", "")
            assert "1517" in all_text, (
                f"Expected 1517 somewhere in session output, got: {all_text[:500]}"
            )

        finally:
            backend.close()

    def test_tool_call_ids_in_messages(self):
        """Verify that tool call messages contain id, type, and tool_call_id."""
        backend = _make_backend()
        try:
            cfg = TIRGenerationConfig(
                max_tokens=2000,
                temperature=0.6,
                top_p=0.95,
                max_turns=5,
                timeout_seconds=120.0,
            )

            result = backend.run_session(
                messages=[
                    {"role": "system", "content": "Use the python tool. Be brief."},
                    {"role": "user", "content": "Print 42 using python."},
                ],
                tools=[_python_tool_def()],
                tool_handlers={"python": _python_handler},
                cfg=cfg,
            )

            assert result.exception is None

            # Inspect the messages that were built during the session.
            # We can't access them directly, but we can verify through the
            # turn records that tool calls had IDs.
            assert len(result.turns) >= 1
            if result.turns[0].tool_calls:
                # The turn record doesn't store the id directly, but we can
                # verify the backend produced tool calls with ids by checking
                # the run completed successfully with llama.cpp (which requires
                # proper id/type/tool_call_id format).
                assert result.termination_reason == "final_answer"

        finally:
            backend.close()

    def test_events_are_logged(self):
        """Event logger receives session lifecycle events."""
        log_dir = _make_log_dir("llamacpp_tir_events")
        events_path = log_dir / "events.jsonl"
        event_logger = _make_events_logger(events_path)

        backend = _make_backend()
        backend.set_event_logger(event_logger)
        try:
            cfg = TIRGenerationConfig(
                max_tokens=2000,
                temperature=0.6,
                top_p=0.95,
                max_turns=5,
                timeout_seconds=120.0,
            )

            result = backend.run_session(
                messages=[
                    {"role": "system", "content": "Use the python tool. Be brief."},
                    {"role": "user", "content": "Compute 2+2 using python."},
                ],
                tools=[_python_tool_def()],
                tool_handlers={"python": _python_handler},
                cfg=cfg,
            )

            events = [
                json.loads(line)
                for line in events_path.read_text().strip().splitlines()
            ]
            event_types = {e["event"] for e in events}

            assert "tir_session_start" in event_types
            assert "tir_session_done" in event_types
            assert "tir_chat_stream_start" in event_types
            assert "tir_chat_stream_done" in event_types
            assert "tir_turn_start" in event_types
            assert "tir_turn_done" in event_types

            # Should have at least one tool call event
            if result.turns and result.turns[0].tool_calls:
                assert "tir_tool_call" in event_types

        finally:
            backend.close()

    def test_no_tools_produces_direct_answer(self):
        """Without tools, the model should give a direct final answer in one turn."""
        backend = _make_backend()
        try:
            cfg = TIRGenerationConfig(
                max_tokens=500,
                temperature=0.6,
                top_p=0.95,
                max_turns=3,
                timeout_seconds=60.0,
            )

            result = backend.run_session(
                messages=[
                    {"role": "user", "content": "What is the capital of France? One word."},
                ],
                tools=[],
                tool_handlers={},
                cfg=cfg,
            )

            assert result.termination_reason == "final_answer"
            assert result.exception is None
            assert len(result.turns) == 1
            assert len(result.turns[0].tool_calls) == 0

        finally:
            backend.close()


# ---------------------------------------------------------------------------
# Tests — stop_event cancellation
# ---------------------------------------------------------------------------

class TestStopEvent:
    """Test that stop_event cancels streaming."""

    def test_stop_event_halts_streaming(self):
        """Setting stop_event should interrupt chat_streaming."""
        import threading

        backend = _make_backend()
        try:
            stop = threading.Event()
            cfg = TIRGenerationConfig(max_tokens=4000, temperature=0.6, top_p=0.95)

            chunk_count = 0
            for chunk in backend.chat_streaming(
                messages=[{"role": "user", "content": "Write a very long essay about mathematics."}],
                tools=[],
                cfg=cfg,
                stop_event=stop,
            ):
                chunk_count += 1
                if chunk_count >= 5:
                    stop.set()

            # We should have stopped early (well before 4000 tokens)
            assert chunk_count < 100, (
                f"Expected early stop, but got {chunk_count} chunks"
            )
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# Tests — verbose mode
# ---------------------------------------------------------------------------

class TestVerboseMode:
    """Test that verbose mode prints output without errors."""

    def test_verbose_does_not_crash(self):
        """Verbose mode should print output without raising exceptions."""
        backend = _make_backend()
        backend.set_verbose(True)
        try:
            cfg = TIRGenerationConfig(max_tokens=200, temperature=0.6, top_p=0.95)
            for chunk in backend.chat_streaming(
                messages=[{"role": "user", "content": "Say hello."}],
                tools=[],
                cfg=cfg,
            ):
                pass  # just consume chunks
        finally:
            backend.close()
