"""
Integration tests for the informal proof agent.

These tests require:
- A local Ollama server with the qwen3.5 model

Each test writes real logs to logs/test/informal_<name>/ for post-hoc inspection.
These directories are NOT cleaned up after testing.

Run with:
    PYTHONPATH=. python -m pytest tests/informal_prover/test_informal_prover_integration.py -v -s
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Paths — adjust if your setup differs
# ---------------------------------------------------------------------------

_OLLAMA_MODEL = "qwen3.5"
_OLLAMA_HOST = "http://localhost:11434"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LOGS_DIR = _REPO_ROOT / "logs" / "test"

# ---------------------------------------------------------------------------
# Test fixtures — a real formalization record (Putnam 2008 B2)
# ---------------------------------------------------------------------------

PROBLEM_STATEMENT = (
    "Let $F_0(x)=\\ln x$. For $n \\geq 0$ and $x>0$, let "
    "$F_{n+1}(x)=\\int_0^x F_n(t)\\,dt$. Evaluate "
    "$\\lim_{n \\to \\infty} \\frac{n!F_n(1)}{\\ln n}$."
)

SOLUTION_TRACE = """\
We prove by induction that F_n(x) = 1/(n-1)! ∫_0^x (x - s)^{n-1} ln s ds for n >= 1.

Base case n=1: F_1(x) = ∫_0^x ln t dt = x ln x - x. And 1/0! ∫_0^x (x-s)^0 ln s ds \
= ∫_0^x ln s ds = x ln x - x. ✓

Inductive step: Suppose F_n(x) = 1/(n-1)! ∫_0^x (x - s)^{n-1} ln s ds. Then \
F_{n+1}(x) = ∫_0^x F_n(t) dt = 1/(n-1)! ∫_0^x ∫_0^t (t-s)^{n-1} ln s ds dt. \
Swapping order: = 1/(n-1)! ∫_0^x ln s ∫_s^x (t-s)^{n-1} dt ds \
= 1/(n-1)! ∫_0^x ln s [(x-s)^n / n] ds = 1/n! ∫_0^x (x-s)^n ln s ds.

Thus F_n(1) = 1/(n-1)! ∫_0^1 (1-t)^{n-1} ln t dt.

Using Beta function: ∫_0^1 (1-t)^{n-1} ln t dt = ∂/∂a B(a,n)|_{a=1}.

B(a,n) = Γ(a)Γ(n)/Γ(a+n). Derivative at a=1 gives B(1,n)(ψ(1) - ψ(1+n)).

Since B(1,n) = 1/n, ψ(1) = -γ, ψ(1+n) = H_n - γ, the integral = -H_n/n.

So n! F_n(1) = n · (-H_n/n) = -H_n.

Since H_n ~ ln n + γ, we get n! F_n(1) / ln n → -1.

The answer is \\boxed{-1}."""

ANSWER = "-1"

LEAN_STATEMENT = """\
import Mathlib

open Filter Topology Set Nat

abbrev putnam_2008_b2_solution : ℝ := -1
/--
Let $F_0(x)=\\ln x$. For $n \\geq 0$ and $x>0$, let $F_{n+1}(x)=\\int_0^x F_n(t)\\,dt$. Evaluate $\\lim_{n \\to \\infty} \\frac{n!F_n(1)}{\\ln n}$.
-/
theorem putnam_2008_b2
(F : ℕ → ℝ → ℝ)
(hF0 : ∀ x : ℝ, F 0 x = Real.log x)
(hFn : ∀ n : ℕ, ∀ x > 0, F (n + 1) x = ∫ t in Set.Ioo 0 x, F n t)
: Tendsto (fun n : ℕ => ((n)! * F n 1) / Real.log n) atTop (𝓝 putnam_2008_b2_solution) :=
sorry"""

# A simple problem for quick tests
SIMPLE_PROBLEM = "Solve the equation $4x^2 + 4x + 1 = 0$."
SIMPLE_TRACE = """\
We recognize 4x^2 + 4x + 1 = (2x + 1)^2. Setting (2x + 1)^2 = 0 gives \
2x + 1 = 0, so x = -1/2. This is a repeated root.

The answer is \\boxed{-1/2}."""
SIMPLE_ANSWER = "-1/2"
SIMPLE_LEAN = """\
import Mathlib

abbrev simple_test_solution : ℚ := -1/2

theorem simple_test :
    4 * simple_test_solution ^ 2 + 4 * simple_test_solution + 1 = 0 :=
sorry"""


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


def _make_agent_and_backend():
    """Create an InformalProverAgent and OllamaTIRBackend wired for testing."""
    from conjecturing_agents.agents.informal_prover import (
        InformalProverAgent,
        InformalProverConfig,
    )
    from conjecturing_agents.inference_backends.ollama_tir import (
        OllamaTIRBackend,
        OllamaTIRConfig,
    )

    agent_cfg = InformalProverConfig()

    backend_cfg = OllamaTIRConfig(
        model=_OLLAMA_MODEL,
        host=_OLLAMA_HOST,
        client_timeout=960,
        think=True,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repeat_penalty=1.0,
    )

    agent = InformalProverAgent(agent_cfg)
    backend = OllamaTIRBackend(backend_cfg)
    backend.set_verbose(True)

    return agent, backend


def _save_result(log_dir: Path, result, filename: str = "result.json") -> None:
    """Save an InformalProverResult as JSON."""
    import dataclasses
    d = dataclasses.asdict(result)
    (log_dir / filename).write_text(
        json.dumps(d, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInformalProverBasic:
    """Test that the informal proof agent produces non-empty proof text."""

    def test_simple_problem(self):
        """Generate an informal proof for 1+1=2 — should succeed quickly."""
        log_dir = _make_log_dir("informal_simple")
        events_path = log_dir / "events.jsonl"
        event_logger = _make_events_logger(events_path)

        agent, backend = _make_agent_and_backend()
        try:
            result = agent.generate_proof(
                problem_statement=SIMPLE_PROBLEM,
                solution_trace=SIMPLE_TRACE,
                answer=SIMPLE_ANSWER,
                lean_statement=SIMPLE_LEAN,
                backend=backend,
                seed=42,
                event_logger=event_logger,
            )
            _save_result(log_dir, result)

            assert result.proof_text != "", "Expected non-empty proof text"
            assert result.exception is None
            assert result.elapsed_ms > 0
            assert len(result.turns) > 0
        finally:
            agent.close()
            backend.close()

    def test_putnam_problem(self):
        """Generate an informal proof for a real Putnam problem."""
        log_dir = _make_log_dir("informal_putnam")
        events_path = log_dir / "events.jsonl"
        event_logger = _make_events_logger(events_path)

        agent, backend = _make_agent_and_backend()
        try:
            result = agent.generate_proof(
                problem_statement=PROBLEM_STATEMENT,
                solution_trace=SOLUTION_TRACE,
                answer=ANSWER,
                lean_statement=LEAN_STATEMENT,
                backend=backend,
                seed=42,
                event_logger=event_logger,
            )
            _save_result(log_dir, result)

            assert result.proof_text != "", "Expected non-empty proof text"
            assert result.exception is None
            assert result.elapsed_ms > 0
            # The proof should mention the answer somewhere
            assert "-1" in result.proof_text, (
                "Expected proof to reference the answer -1"
            )
        finally:
            agent.close()
            backend.close()


class TestInformalProverResultLogging:
    """Verify result fields and event logs are populated correctly."""

    def test_result_fields_complete(self):
        """All expected fields in InformalProverResult are populated."""
        log_dir = _make_log_dir("informal_result_fields")
        events_path = log_dir / "events.jsonl"
        event_logger = _make_events_logger(events_path)

        agent, backend = _make_agent_and_backend()
        try:
            result = agent.generate_proof(
                problem_statement=SIMPLE_PROBLEM,
                solution_trace=SIMPLE_TRACE,
                answer=SIMPLE_ANSWER,
                lean_statement=SIMPLE_LEAN,
                backend=backend,
                seed=42,
                event_logger=event_logger,
            )
            _save_result(log_dir, result)

            # Basic field checks
            assert isinstance(result.proof_text, str)
            assert isinstance(result.termination_reason, str)
            assert result.termination_reason != ""
            assert isinstance(result.elapsed_ms, int)
            assert result.elapsed_ms > 0
            assert isinstance(result.turns, list)
            assert len(result.turns) > 0
            assert result.exception is None

            # Each turn should have expected keys
            for turn in result.turns:
                assert "turn" in turn
                assert "thinking" in turn
                assert "content" in turn
                assert "tool_calls" in turn

        finally:
            agent.close()
            backend.close()

    def test_events_logged(self):
        """Event logger receives session_start and session_done events."""
        log_dir = _make_log_dir("informal_events")
        events_path = log_dir / "events.jsonl"
        event_logger = _make_events_logger(events_path)

        agent, backend = _make_agent_and_backend()
        try:
            result = agent.generate_proof(
                problem_statement=SIMPLE_PROBLEM,
                solution_trace=SIMPLE_TRACE,
                answer=SIMPLE_ANSWER,
                lean_statement=SIMPLE_LEAN,
                backend=backend,
                seed=42,
                event_logger=event_logger,
            )
            _save_result(log_dir, result)

            events = [
                json.loads(line)
                for line in events_path.read_text().strip().splitlines()
            ]
            assert len(events) > 0
            event_types = {e["event"] for e in events}
            assert "informal_prover_session_start" in event_types
            assert "informal_prover_session_done" in event_types

            # session_start should have the expected fields
            start_events = [
                e for e in events
                if e["event"] == "informal_prover_session_start"
            ]
            assert len(start_events) == 1
            assert "seed" in start_events[0]
            assert "answer" in start_events[0]

            # session_done should record proof_chars
            done_events = [
                e for e in events
                if e["event"] == "informal_prover_session_done"
            ]
            assert len(done_events) == 1
            assert done_events[0]["proof_chars"] > 0

        finally:
            agent.close()
            backend.close()
