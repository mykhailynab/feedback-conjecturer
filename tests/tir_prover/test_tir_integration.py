"""
Integration tests for the TIR prover agent.

These tests require:
- A local Ollama server with the qwen3.5 model
- A local Lean 4 / Mathlib installation at /Users/mila/lean/mathlib4
- The Qwen3.5 HuggingFace tokenizer at tokenizers/Qwen3.5-27B

Each test writes real logs to logs/test/tir_<name>/ for post-hoc inspection.
These directories are NOT cleaned up after testing.

Run with:
    PYTHONPATH=. python -m pytest tests/tir_prover/test_tir_integration.py -v -s
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Paths — adjust if your setup differs
# ---------------------------------------------------------------------------

_LEAN_PROJECT_DIR = "/Users/mila/lean/mathlib4"
_OLLAMA_MODEL = "qwen3.5"
_TOKENIZER_PATH = "tokenizers/Qwen3.5-27B"
_OLLAMA_HOST = "http://localhost:11434"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LOGS_DIR = _REPO_ROOT / "logs" / "test"

# ---------------------------------------------------------------------------
# Simple theorem statements
# ---------------------------------------------------------------------------

# Trivially provable: 1 + 1 = 2
TRIVIAL_THEOREM = """\
import Mathlib

theorem one_plus_one : 1 + 1 = 2 := by sorry"""

# Slightly harder but still easy: n + 0 = n
NAT_ADD_ZERO_THEOREM = """\
import Mathlib

theorem nat_add_zero (n : ℕ) : n + 0 = n := by sorry"""

# With an abbrev (mirrors the formalization pipeline format)
ABBREV_THEOREM = """\
import Mathlib

abbrev simple_solution : ℕ := 2

theorem simple_thm : 1 + 1 = simple_solution := by sorry"""


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
        from datetime import datetime, timezone
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


def _make_agent_and_backend(
    log_dir: Path,
    *,
    token_limit: int = 0,
    max_turns: int = 32,
    timeout_seconds: float = 300.0,
    use_python_tool: bool = False,
):
    """Create a TIRProverAgent and OllamaTIRBackend wired for testing."""
    from conjecturing_agents.agents.tir_prover import TIRProverAgent, TIRProverConfig
    from conjecturing_agents.tool_calling_backends.lean4_compiler import LeanCompilerConfig
    from conjecturing_agents.inference_backends.ollama_tir import (
        OllamaTIRBackend,
        OllamaTIRConfig,
    )

    lean_cfg = LeanCompilerConfig(
        project_dir=_LEAN_PROJECT_DIR,
        workspace_subdir=str(log_dir / ".lean_workspace"),
        timeout_seconds=120,
        lean_jobs=4,
        max_memory_megabytes=8 * 1024,
        treat_sorry_warning_as_failure=True,
        treat_any_warning_as_failure=False,
    )

    agent_cfg = TIRProverConfig(
        lean=lean_cfg,
        max_tokens=16384,
        temperature=0.6,
        top_p=0.95,
        max_turns=max_turns,
        timeout_seconds=timeout_seconds,
        use_lean_tool=True,
        use_python_tool=use_python_tool,
    )

    backend_cfg = OllamaTIRConfig(
        model=_OLLAMA_MODEL,
        host=_OLLAMA_HOST,
        client_timeout=960,
        think=True,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repeat_penalty=1.0,
        tokenizer_path=_TOKENIZER_PATH,
    )

    agent = TIRProverAgent(agent_cfg)
    backend = OllamaTIRBackend(backend_cfg)
    backend.set_verbose(True)

    return agent, backend


def _count_initial_prompt_tokens(agent, backend, theorem_statement: str) -> int:
    """Count tokens in the initial prompt that prove_theorem would build.

    Builds the same messages + tool definitions that the agent assembles
    internally, then uses the backend's token counter.
    """
    from conjecturing_agents.agents.tir_prover.prompts import INITIAL_USER_MESSAGE

    messages = [
        {"role": "system", "content": agent.cfg.system_prompt},
        {
            "role": "user",
            "content": INITIAL_USER_MESSAGE.format(
                theorem_statement=theorem_statement
            ),
        },
    ]
    # Mirror the tools list built in prove_theorem().
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lean_final",
                "description": agent.cfg.lean_final_tool_description,
                "parameters": {
                    "type": "object",
                    "required": ["code"],
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": (
                                "Complete Lean 4 file with sorry replaced by the proof"
                            ),
                        }
                    },
                },
            },
        },
    ]
    if agent.cfg.use_lean_tool:
        tools.append(agent._lean_tool.tool_def)
    if agent.cfg.use_python_tool:
        tools.append({
            "type": "function",
            "function": {
                "name": agent.cfg.jupyter.tool_name,
                "description": agent.cfg.python_tool_description,
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
        })
    return backend.count_tokens_messages(messages, tools)


def _save_result(log_dir: Path, result, filename: str = "result.json") -> None:
    """Save a TIRProverResult as JSON."""
    import dataclasses
    d = dataclasses.asdict(result)
    (log_dir / filename).write_text(
        json.dumps(d, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTIRProverSuccessfulProof:
    """Test that the TIR agent can prove a trivially simple theorem."""

    def test_trivial_theorem_proved(self):
        """Prove 1 + 1 = 2 — should succeed within a few turns."""
        log_dir = _make_log_dir("tir_trivial_proof")
        events_path = log_dir / "events.jsonl"
        event_logger = _make_events_logger(events_path)

        agent, backend = _make_agent_and_backend(log_dir)
        try:
            result = agent.prove_theorem(
                TRIVIAL_THEOREM,
                backend,
                seed=42,
                event_logger=event_logger,
            )
            _save_result(log_dir, result)

            # Core assertion: the theorem should be proved
            assert result.proved, (
                f"Expected trivial theorem to be proved, "
                f"got termination_reason={result.termination_reason}"
            )
            assert result.termination_reason == "proved"
            assert result.proved_lean != ""
            assert "sorry" not in result.proved_lean
            assert "one_plus_one" in result.proved_lean

            # Verify events were logged
            events = [
                json.loads(line)
                for line in events_path.read_text().strip().splitlines()
            ]
            assert len(events) > 0
            event_types = {e["event"] for e in events}
            assert "tir_prover_session_start" in event_types
            assert "tir_prover_session_done" in event_types

            # Session done event should record proved=True
            done_events = [e for e in events if e["event"] == "tir_prover_session_done"]
            assert done_events[-1]["proved"] is True

        finally:
            agent.close()
            backend.close()


class TestTIRProverTokenLimitAndResume:
    """Test token-limited interruption and --continue resume logic."""

    def test_token_limit_creates_incomplete_record(self):
        """Run with a token limit below the initial prompt size so the
        between-turns check fires before any generation starts, producing
        a clean incomplete record.
        """
        log_dir = _make_log_dir("tir_token_limit_incomplete")
        events_path = log_dir / "events.jsonl"
        event_logger = _make_events_logger(events_path)

        agent, backend = _make_agent_and_backend(log_dir)
        try:
            prompt_tokens = _count_initial_prompt_tokens(
                agent, backend, NAT_ADD_ZERO_THEOREM
            )
            result = agent.prove_theorem(
                NAT_ADD_ZERO_THEOREM,
                backend,
                seed=42,
                event_logger=event_logger,
                token_limit=prompt_tokens - 1,
            )
            _save_result(log_dir, result, "result_incomplete.json")

            # The session should be token_limit_triggered (not proved)
            assert result.token_limit_triggered, (
                f"Expected token_limit_triggered=True with token_limit={prompt_tokens - 1}, "
                f"got termination_reason={result.termination_reason}"
            )
            assert not result.proved
            assert result.termination_reason == "token_limit"

            # conversation_history should be non-empty for resume
            assert result.session_result is not None
            assert len(result.session_result.conversation_history) > 0, (
                "conversation_history should be non-empty for incomplete sessions"
            )
            # Should contain at least system + user messages
            roles = [m["role"] for m in result.session_result.conversation_history]
            assert "system" in roles
            assert "user" in roles

            # With a limit below the prompt size, generation never starts,
            # so partial_assistant_turn should be None.
            assert result.session_result.partial_assistant_turn is None

            # Events should include a token_limit termination
            events = [
                json.loads(line)
                for line in events_path.read_text().strip().splitlines()
            ]
            done_events = [e for e in events if e["event"] == "tir_prover_session_done"]
            assert len(done_events) == 1
            assert done_events[0]["token_limit_triggered"] is True

        finally:
            agent.close()
            backend.close()

    def test_resume_from_incomplete_session(self):
        """Interrupt with a low token limit, then resume with a higher limit."""
        log_dir = _make_log_dir("tir_token_limit_resume")

        # --- Phase 1: run with low token limit ---
        events_path_1 = log_dir / "events_phase1.jsonl"
        event_logger_1 = _make_events_logger(events_path_1)

        agent, backend = _make_agent_and_backend(log_dir)
        try:
            prompt_tokens = _count_initial_prompt_tokens(
                agent, backend, TRIVIAL_THEOREM
            )
            result_1 = agent.prove_theorem(
                TRIVIAL_THEOREM,
                backend,
                seed=42,
                event_logger=event_logger_1,
                token_limit=prompt_tokens - 1,
            )
            _save_result(log_dir, result_1, "result_phase1.json")

            assert result_1.token_limit_triggered, (
                f"Phase 1 should be token_limit_triggered, got {result_1.termination_reason}"
            )
            assert result_1.session_result is not None
            assert len(result_1.session_result.conversation_history) > 0

            # --- Phase 2: resume with a much higher token limit ---
            events_path_2 = log_dir / "events_phase2.jsonl"
            event_logger_2 = _make_events_logger(events_path_2)

            result_2 = agent.prove_theorem(
                TRIVIAL_THEOREM,
                backend,
                seed=42,
                event_logger=event_logger_2,
                token_limit=0,  # unlimited
                initial_messages=result_1.session_result.conversation_history,
            )
            _save_result(log_dir, result_2, "result_phase2.json")

            # The resumed session should complete (proved or at least not token_limit_triggered)
            assert not result_2.token_limit_triggered, (
                f"Phase 2 should not be token_limit_triggered, got {result_2.termination_reason}"
            )

            # Verify phase 2 events include a session_start with resuming=True
            events_2 = [
                json.loads(line)
                for line in events_path_2.read_text().strip().splitlines()
            ]
            start_events = [
                e for e in events_2 if e["event"] == "tir_prover_session_start"
            ]
            assert len(start_events) == 1
            assert start_events[0]["resuming"] is True

        finally:
            agent.close()
            backend.close()

    def test_resume_reaches_proof(self):
        """Interrupt then resume — verify the theorem is actually proved."""
        log_dir = _make_log_dir("tir_resume_proves")

        agent, backend = _make_agent_and_backend(log_dir)
        try:
            # Phase 1: token limit below prompt size → immediate cutoff
            prompt_tokens = _count_initial_prompt_tokens(
                agent, backend, TRIVIAL_THEOREM
            )
            result_1 = agent.prove_theorem(
                TRIVIAL_THEOREM,
                backend,
                seed=42,
                token_limit=prompt_tokens - 1,
            )
            _save_result(log_dir, result_1, "result_phase1.json")

            assert result_1.token_limit_triggered, (
                f"Phase 1 should be token_limit_triggered, got {result_1.termination_reason}"
            )

            # Phase 2: unlimited
            events_path = log_dir / "events_final.jsonl"
            event_logger = _make_events_logger(events_path)

            result_2 = agent.prove_theorem(
                TRIVIAL_THEOREM,
                backend,
                seed=42,
                event_logger=event_logger,
                token_limit=0,
                initial_messages=result_1.session_result.conversation_history,
            )
            _save_result(log_dir, result_2, "result_final.json")

            assert result_2.proved, (
                f"Expected trivial theorem to be proved after resume, "
                f"got termination_reason={result_2.termination_reason}"
            )
            assert result_2.termination_reason == "proved"
            assert "sorry" not in result_2.proved_lean

        finally:
            agent.close()
            backend.close()


class TestTIRProverResultLogging:
    """Verify result fields and log files are populated correctly."""

    def test_result_fields_complete(self):
        """All expected fields in TIRProverResult are populated."""
        log_dir = _make_log_dir("tir_result_fields")
        events_path = log_dir / "events.jsonl"
        event_logger = _make_events_logger(events_path)

        agent, backend = _make_agent_and_backend(log_dir, max_turns=4)
        try:
            result = agent.prove_theorem(
                TRIVIAL_THEOREM,
                backend,
                seed=42,
                event_logger=event_logger,
            )
            _save_result(log_dir, result)

            # Basic field checks
            assert isinstance(result.proved, bool)
            assert isinstance(result.termination_reason, str)
            assert result.termination_reason != ""
            assert isinstance(result.elapsed_ms, int)
            assert result.elapsed_ms > 0

            # session_result should be populated
            assert result.session_result is not None
            assert isinstance(result.session_result.conversation_history, list)
            assert len(result.session_result.conversation_history) > 0
            assert result.session_result.exception is None

            # conversation_history should contain assistant messages with tool_calls
            assistant_msgs = [
                m for m in result.session_result.conversation_history
                if m.get("role") == "assistant"
            ]
            assert len(assistant_msgs) > 0

        finally:
            agent.close()
            backend.close()

    def test_abbrev_theorem_proved(self):
        """A theorem with an abbrev (formalization pipeline format) can be proved."""
        log_dir = _make_log_dir("tir_abbrev_proof")
        events_path = log_dir / "events.jsonl"
        event_logger = _make_events_logger(events_path)

        agent, backend = _make_agent_and_backend(log_dir)
        try:
            result = agent.prove_theorem(
                ABBREV_THEOREM,
                backend,
                seed=42,
                event_logger=event_logger,
            )
            _save_result(log_dir, result)

            assert result.proved, (
                f"Expected abbrev theorem to be proved, "
                f"got termination_reason={result.termination_reason}"
            )
            assert "simple_thm" in result.proved_lean
            assert "simple_solution" in result.proved_lean

        finally:
            agent.close()
            backend.close()
