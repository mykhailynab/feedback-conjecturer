"""
Data model and parsing for Goedel prover event logs.

Reads check_goedel_events.jsonl and check_results.jsonl into typed Python objects.
"""
from __future__ import annotations

import re
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from analysis_and_inspection.display_utils import green, red, yellow, dim

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ProverRound:
    round_idx: int
    prompt_text: str
    raw_output: str
    lean_ok: Optional[bool]
    lean_timed_out: Optional[bool]
    lean_oom: Optional[bool]
    lean_error_count: Optional[int]
    elapsed_ms: Optional[int]
    termination_reason: str


@dataclass
class ProverSession:
    problem_id: str
    attempt: int
    checking: str           # "proof" or "disproof"
    theorem_statement: str
    formal_statement: str
    seed: int
    max_rounds: int
    start_ts: str
    # Fields from prover_session_done (None if incomplete)
    theorem_proved: Optional[bool]
    termination_reason: Optional[str]
    rounds_used: Optional[int]
    elapsed_ms: Optional[int]
    done_ts: Optional[str]
    # Rounds
    rounds: List[ProverRound] = field(default_factory=list)
    # prove_formalizations only (0 for check_formalizations sessions)
    retry: int = 0
    # token_limit set for this session (0 = unlimited); prove_formalizations only
    token_limit: int = 0

    @property
    def complete(self) -> bool:
        return self.termination_reason is not None

    @property
    def outcome(self) -> str:
        """Human-readable outcome string."""
        if not self.complete:
            return "incomplete"
        if self.theorem_proved:
            return "disproved" if self.checking == "disproof" else "proved"
        return "failed"

    @property
    def outcome_color(self):
        o = self.outcome
        if o == "proved":
            return green
        if o == "disproved":
            return red
        if o == "failed":
            return yellow
        return dim  # incomplete


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_chat_messages(prompt_text: str) -> List[Dict[str, str]]:
    """
    Split a chat-template rendered string into [{role, content}, ...] messages.
    The format used by the Goedel template is:
        <|im_start|>role\\ncontent<|im_end|>\\n...
    The final (assistant) turn has no content — it marks the generation start.
    """
    parts = re.split(r"<\|im_start\|>", prompt_text)
    messages = []
    for part in parts:
        if not part.strip():
            continue
        nl = part.find("\n")
        if nl == -1:
            continue
        role = part[:nl].strip()
        content = part[nl + 1:]
        content = re.sub(r"<\|im_end\|>\s*$", "", content).rstrip()
        messages.append({"role": role, "content": content})
    return messages


def load_goedel_sessions(goedel_events_path: str) -> List[ProverSession]:
    """
    Build ProverSession objects from check_goedel_events.jsonl.

    NOTE: Only works for pass@1 so far
    """
    events: List[Dict[str, Any]] = []
    with open(goedel_events_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    # Group events by (problem_id, attempt, checking).
    by_key: Dict[Tuple, List[Dict]] = defaultdict(list)
    for e in events:
        pid = e.get("problem_id")
        att = e.get("attempt")
        chk = e.get("checking")
        evt = e.get("event")
        if pid is None or att is None:
            continue
        if evt in ("prover_session_start", "prover_round_done", "prover_session_done"):
            by_key[(pid, att, chk)].append(e)

    sessions: List[ProverSession] = []
    for (pid, att, chk), evts in by_key.items():
        starts = [e for e in evts if e["event"] == "prover_session_start"]
        dones  = [e for e in evts if e["event"] == "prover_session_done"]
        rounds_evts = [e for e in evts if e["event"] == "prover_round_done"]

        # assert pass@1
        assert len(starts) <= 1
        assert len(dones) <= 1

        if not starts:
            continue

        start_evt = starts[-1]
        done_evt  = dones[-1] if dones else None

        rounds = [
            ProverRound(
                round_idx=r.get("round", i),
                prompt_text=r.get("prompt_text", ""),
                raw_output=r.get("raw_output", ""),
                lean_ok=r.get("lean_ok"),
                lean_timed_out=r.get("lean_timed_out"),
                lean_oom=r.get("lean_oom"),
                lean_error_count=r.get("lean_error_count"),
                elapsed_ms=r.get("elapsed_ms"),
                termination_reason=r.get("termination_reason", ""),
            )
            for i, r in enumerate(rounds_evts)
        ]
        rounds.sort(key=lambda r: r.round_idx)

        sess = ProverSession(
            problem_id=pid,
            attempt=att,
            checking=chk,
            theorem_statement=start_evt.get("theorem_statement", ""),
            formal_statement=start_evt.get("formal_statement", ""),
            seed=start_evt.get("seed", 0),
            max_rounds=start_evt.get("max_rounds", 0),
            start_ts=start_evt.get("ts", ""),
            theorem_proved=done_evt.get("proved") if done_evt else None,
            termination_reason=done_evt.get("termination_reason") if done_evt else None,
            rounds_used=done_evt.get("rounds_used") if done_evt else (len(rounds) if rounds else None),
            elapsed_ms=done_evt.get("elapsed_ms") if done_evt else None,
            done_ts=done_evt.get("ts") if done_evt else None,
            rounds=rounds,
        )
        sessions.append(sess)

    sessions.sort(key=lambda s: s.start_ts)
    return sessions


def load_prove_sessions(goedel_events_path: str) -> List[ProverSession]:
    """
    Build ProverSession objects from prover_events.jsonl.

    Groups events by (problem_id, attempt, direction, retry).
    The ``checking`` field is populated from ``direction`` ("proof"/"disproof").
    """
    events: List[Dict[str, Any]] = []
    with open(goedel_events_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    by_key: Dict[Tuple, List[Dict]] = defaultdict(list)
    for e in events:
        pid = e.get("problem_id")
        att = e.get("attempt")
        direction = e.get("direction", "proof")
        retry = e.get("retry", 0)
        evt = e.get("event")
        if pid is None or att is None:
            continue
        if evt in ("prover_session_start", "prover_round_done", "prover_session_done"):
            by_key[(pid, att, direction, retry)].append(e)

    sessions: List[ProverSession] = []
    for (pid, att, direction, retry), evts in by_key.items():
        starts      = [e for e in evts if e["event"] == "prover_session_start"]
        dones       = [e for e in evts if e["event"] == "prover_session_done"]
        rounds_evts = [e for e in evts if e["event"] == "prover_round_done"]

        if not starts:
            continue

        start_evt = starts[-1]
        done_evt  = dones[-1] if dones else None

        rounds = [
            ProverRound(
                round_idx=r.get("round", i),
                prompt_text=r.get("prompt_text", ""),
                raw_output=r.get("raw_output", ""),
                lean_ok=r.get("lean_ok"),
                lean_timed_out=r.get("lean_timed_out"),
                lean_oom=r.get("lean_oom"),
                lean_error_count=r.get("lean_error_count"),
                elapsed_ms=r.get("elapsed_ms"),
                termination_reason=r.get("termination_reason", ""),
            )
            for i, r in enumerate(rounds_evts)
        ]
        rounds.sort(key=lambda r: r.round_idx)

        sess = ProverSession(
            problem_id=pid,
            attempt=att,
            checking=direction,
            theorem_statement=start_evt.get("theorem_statement", ""),
            formal_statement=start_evt.get("formal_statement", ""),
            seed=start_evt.get("seed", 0),
            max_rounds=start_evt.get("max_rounds", 0),
            start_ts=start_evt.get("ts", ""),
            theorem_proved=done_evt.get("proved") if done_evt else None,
            termination_reason=done_evt.get("termination_reason") if done_evt else None,
            rounds_used=done_evt.get("rounds_used") if done_evt else (len(rounds) if rounds else None),
            elapsed_ms=done_evt.get("elapsed_ms") if done_evt else None,
            done_ts=done_evt.get("ts") if done_evt else None,
            rounds=rounds,
            retry=retry,
            token_limit=start_evt.get("token_limit", 0),
        )
        sessions.append(sess)

    sessions.sort(key=lambda s: s.start_ts)
    return sessions


def load_prove_results(results_path: str) -> Dict[Tuple, Dict]:
    """Return a dict keyed by (problem_id, attempt) → prove_results record."""
    out: Dict[Tuple, Dict] = {}
    with open(results_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[(r.get("problem_id"), r.get("attempt"))] = r
    return out


def load_check_results(results_path: str) -> Dict[Tuple, Dict]:
    """Return a dict keyed by (problem_id, attempt) → result record."""
    out: Dict[Tuple, Dict] = {}
    with open(results_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[(r.get("problem_id"), r.get("attempt"))] = r
    return out


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

FILTER_CHOICES = ("proof", "disproof", "proved", "disproved", "failed", "incomplete")


def apply_filter(sessions: List[ProverSession], filter_val: Optional[str]) -> List[ProverSession]:
    if not filter_val:
        return sessions
    f = filter_val.lower()
    if f == "proof":
        return [s for s in sessions if s.checking == "proof"]
    if f == "disproof":
        return [s for s in sessions if s.checking == "disproof"]
    if f == "proved":
        return [s for s in sessions if s.outcome == "proved"]
    if f == "disproved":
        return [s for s in sessions if s.outcome == "disproved"]
    if f == "failed":
        return [s for s in sessions if s.outcome == "failed"]
    if f == "incomplete":
        return [s for s in sessions if not s.complete]
    raise ValueError(f"--filter must be one of {FILTER_CHOICES}, got {filter_val}")
