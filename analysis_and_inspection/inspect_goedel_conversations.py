#!/usr/bin/env python3
"""
Inspect Goedel prover conversations from a check_formalizations run.

Reads goedel_events.jsonl (and optionally check_results.jsonl) to display
the full multi-turn conversation each session had, along with summary stats.

Usage:
    PYTHONPATH=. python inspect_goedel_conversations.py \\
        --events  logs/.../goedel_events.jsonl \\
        [--results logs/.../check_results.jsonl] \\
        [--filter  proof|disproof|proved|failed|disproved|incomplete] \\
        [--problem-id <id>] \\
        [--max-sessions N] \\
        [--stats-only] \\
        [--no-color] \\
        [--no-truncate] \\
        [--max-output-chars N]
"""
from __future__ import annotations

import re
import sys
import json
import textwrap
import argparse
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------

_USE_COLOR = True  # overridden by --no-color

def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def green(t: str)  -> str: return _c("32", t)
def red(t: str)    -> str: return _c("31", t)
def yellow(t: str) -> str: return _c("33", t)
def cyan(t: str)   -> str: return _c("36", t)
def bold(t: str)   -> str: return _c("1",  t)
def dim(t: str)    -> str: return _c("2",  t)
def blue(t: str)   -> str: return _c("34", t)
def magenta(t: str)-> str: return _c("35", t)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

WIDTH = 100

def hline(char: str = "─", width: int = WIDTH) -> str:
    return char * width

def box_top(title: str = "", width: int = WIDTH, char: str = "─") -> str:
    if title:
        pad = width - len(title) - 4
        left = pad // 2
        right = pad - left
        return f"┌─ {title} " + "─" * right + "┐" if pad >= 0 else f"┌ {title} ┐"
    return "┌" + char * (width - 2) + "┐"

def box_bottom(width: int = WIDTH, char: str = "─") -> str:
    return "└" + char * (width - 2) + "┘"

def box_line(text: str, width: int = WIDTH) -> str:
    inner = width - 4
    lines = []
    for raw in text.split("\n"):
        if len(raw) <= inner:
            lines.append("│ " + raw + " " * (inner - len(raw)) + " │")
        else:
            # wrap long lines
            for chunk in textwrap.wrap(raw, inner) or [""]:
                lines.append("│ " + chunk + " " * (inner - len(chunk)) + " │")
    return "\n".join(lines)

def section_header(title: str, width: int = WIDTH) -> str:
    pad = width - len(title) - 4
    return f"  {bold(title)}  " + dim("─" * max(pad, 0))

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

    @property
    def complete(self) -> bool:
        # the termination ireason is not none if we found an end event
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

def _parse_chat_messages(prompt_text: str) -> List[Dict[str, str]]:
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
    Build ProverSession objects from goedel_events.jsonl.

    NOTE: Only works for pass@1 so far
    """
    events: List[Dict[str, Any]] = []
    with open(goedel_events_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    # Group events by (problem_id, attempt, checking).
    # Events without these fields (raw_generation_*) are ignored here.
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
        # If multiple session_start/done events for the same key, take the last
        # complete run (has both start and done); otherwise take the last start.
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
# Rendering
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int) -> Tuple[str, bool]:
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars], True
    return text, False


def render_session(
    sess: ProverSession,
    check_result: Optional[Dict],
    *,
    max_output_chars: int = 3000,
    no_truncate: bool = False,
) -> str:
    lines: List[str] = []
    mc = 0 if no_truncate else max_output_chars

    # ---- Session header ----
    outcome_fn = sess.outcome_color
    outcome_str = outcome_fn(bold(sess.outcome.upper()))
    checking_str = cyan("DISPROOF") if sess.checking == "disproof" else blue("PROOF")

    lines.append("")
    lines.append("═" * WIDTH)
    lines.append(
        f"  {bold(sess.problem_id)}  attempt {sess.attempt}  "
        f"[{checking_str}]  →  {outcome_str}"
    )
    if sess.termination_reason:
        lines.append(f"  termination: {dim(sess.termination_reason)}  "
                     f"rounds: {sess.rounds_used}  "
                     f"time: {_fmt_ms(sess.elapsed_ms)}")
    else:
        lines.append(f"  {dim('(incomplete — no session_done event recorded)')}")

    if check_result:
        eq = check_result.get("equivalent")
        method = check_result.get("method", "?")
        eq_str = green("True") if eq is True else (red("False") if eq is False else yellow("None"))
        lines.append(f"  check_result: equivalent={eq_str}  method={dim(method)}")

    lines.append("═" * WIDTH)

    # ---- Theorem statement ----
    lines.append(section_header("THEOREM STATEMENT"))
    stmt, trunc = _truncate(sess.theorem_statement.strip(), mc)
    lines.append(dim(stmt))
    if trunc:
        lines.append(dim(f"  ... [{len(sess.theorem_statement) - mc} chars truncated]"))
    lines.append("")

    # ---- Rounds ----
    if not sess.rounds:
        lines.append(dim("  (no rounds recorded)"))

    for rnd in sess.rounds:
        # Round header
        lean_status = _lean_status_str(rnd)
        elapsed = f"  {dim(_fmt_ms(rnd.elapsed_ms))}" if rnd.elapsed_ms else ""
        lines.append(section_header(f"ROUND {rnd.round_idx}  {lean_status}{elapsed}"))

        # Parse prompt into messages and display each turn
        messages = _parse_chat_messages(rnd.prompt_text)
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            if not content:
                # Empty assistant turn = generation start marker
                continue
            role_label = _role_label(role)
            content_trunc, was_trunc = _truncate(content, mc)
            lines.append(box_top(role_label, WIDTH))
            lines.append(box_line(content_trunc))
            if was_trunc:
                lines.append(box_line(dim(f"... [{len(content) - mc} chars truncated]")))
            lines.append(box_bottom(WIDTH))

        # Model output
        if rnd.raw_output:
            out_trunc, was_trunc = _truncate(rnd.raw_output, mc)
            lines.append(box_top(_role_label("assistant") + " (output)", WIDTH))
            lines.append(box_line(out_trunc))
            if was_trunc:
                lines.append(box_line(dim(f"... [{len(rnd.raw_output) - mc} chars truncated]")))
            lines.append(box_bottom(WIDTH))
        else:
            lines.append(dim("  (no model output recorded)"))

        lines.append("")

    return "\n".join(lines)


def _role_label(role: str) -> str:
    return {
        "user": "USER",
        "assistant": "ASSISTANT",
        "system": "SYSTEM",
    }.get(role.lower(), role.upper())


def _lean_status_str(rnd: ProverRound) -> str:
    if rnd.lean_ok is True:
        return green("LEAN OK")
    if rnd.lean_timed_out:
        return yellow("LEAN TIMEOUT")
    if rnd.lean_oom:
        return yellow("LEAN OOM")
    if rnd.lean_ok is False:
        errs = rnd.lean_error_count or 0
        return red(f"LEAN FAILED ({errs} error{'s' if errs != 1 else ''})")
    reason = rnd.termination_reason
    if reason == "no_code_block":
        return yellow("NO CODE BLOCK")
    if reason == "splice_error":
        return yellow("SPLICE ERROR")
    if reason == "context_exceeded":
        return yellow("CONTEXT EXCEEDED")
    return dim(reason or "?")


def _fmt_ms(ms: Optional[int]) -> str:
    if ms is None:
        return "?"
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms/1000:.1f}s"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def render_stats(sessions: List[ProverSession], results: Dict[Tuple, Dict]) -> str:
    lines: List[str] = []

    complete   = [s for s in sessions if s.complete]
    incomplete = [s for s in sessions if not s.complete]

    # Split all sessions (complete + incomplete) by checking type so that
    # incomplete sessions are still reflected in section totals.
    all_proofs    = [s for s in sessions if s.checking == "proof"]
    all_disproofs = [s for s in sessions if s.checking == "disproof"]

    proofs    = [s for s in all_proofs    if s.complete]
    disproofs = [s for s in all_disproofs if s.complete]

    incomplete_proofs    = [s for s in all_proofs    if not s.complete]
    incomplete_disproofs = [s for s in all_disproofs if not s.complete]

    proved    = [s for s in proofs    if s.theorem_proved]
    failed_p  = [s for s in proofs    if not s.theorem_proved]
    disproved = [s for s in disproofs if s.theorem_proved]
    failed_d  = [s for s in disproofs if not s.theorem_proved]

    lines.append("")
    lines.append("═" * WIDTH)
    lines.append(bold("  SUMMARY STATISTICS"))
    lines.append("═" * WIDTH)

    lines.append(f"  {'Total sessions tracked:':<40} {len(sessions)}")
    lines.append(f"  {'  complete:':<40} {len(complete)}")
    lines.append(f"  {'  incomplete (process killed mid-run):':<40} {dim(str(len(incomplete)))}")
    lines.append("")

    # Proof sessions
    lines.append(bold("  Proof attempts  ") + dim(f"(theorem: proposed = gt)"))
    lines.append(hline("─"))
    _pct = lambda n, d: f"{100*n//max(d,1)}%" if d else "n/a"

    lines.append(f"  {'Total proof sessions:':<40} {len(all_proofs)}")
    lines.append(f"  {'  proved (equivalent=True):':<40} "
                 f"{green(str(len(proved)))}  {dim(_pct(len(proved), len(all_proofs)))}")
    lines.append(f"  {'  failed (inconclusive):':<40} "
                 f"{yellow(str(len(failed_p)))}  {dim(_pct(len(failed_p), len(all_proofs)))}")
    if incomplete_proofs:
        lines.append(f"  {'  incomplete (killed mid-run):':<40} "
                     f"{dim(str(len(incomplete_proofs)))}  "
                     f"{dim(_pct(len(incomplete_proofs), len(all_proofs)))}")

    # Disproof sessions
    lines.append("")
    lines.append(bold("  Disproof attempts  ") + dim(f"(theorem: proposed ≠ gt)"))
    lines.append(hline("─"))
    lines.append(f"  {'Total disproof sessions:':<40} {len(all_disproofs)}")
    lines.append(f"  {'  disproved (equivalent=False):':<40} "
                 f"{red(str(len(disproved)))}  {dim(_pct(len(disproved), len(all_disproofs)))}")
    lines.append(f"  {'  failed (inconclusive):':<40} "
                 f"{yellow(str(len(failed_d)))}  {dim(_pct(len(failed_d), len(all_disproofs)))}")
    if incomplete_disproofs:
        lines.append(f"  {'  incomplete (killed mid-run):':<40} "
                     f"{dim(str(len(incomplete_disproofs)))}  "
                     f"{dim(_pct(len(incomplete_disproofs), len(all_disproofs)))}")

    def _attempt_check_outcome(pid: str, att: int, checking: str) -> str:
        """
        Returns the outcome for a single (pid, attempt, checking) session:
          "proved"/"disproved" - session proved it
          "failed"             - session complete but didn't prove
          "terminated"         - session incomplete (process killed)
          "none"               - no session of this type recorded for this attempt
        """
        sess_for = [s for s in sessions if s.problem_id == pid and s.attempt == att and s.checking == checking]
        # assert pass@1 (for now)
        assert len(sess_for) <= 1
        if not sess_for:
            return "none"
        s = sess_for[0]
        if s.theorem_proved:
            return "proved" if checking == "proof" else "disproved"
        if s.complete:
            return "failed"
        return "terminated"
    
    all_attempts: set = set(results.keys())

    n_resolved_goedel = 0
    n_resolved_otherwise = 0
    cat_counts: Dict[str, int] = defaultdict(int)
    for pid, att in all_attempts:
        p_out = _attempt_check_outcome(pid, att, "proof")
        d_out = _attempt_check_outcome(pid, att, "disproof")
        if p_out == "proved" or d_out == "disproved":
            n_resolved_goedel += 1
            continue
        otherwise_eq = results[(pid, att)].get("equivalent") is True
        otherwise_neq = results[(pid, att)].get("equivalent") is False
        if otherwise_eq or otherwise_neq:
            n_resolved_otherwise += 1
            continue
        # Inconclusive — bucket by the combination of outcomes
        if p_out == "none" and d_out == "none":
            cat_counts["skipped"] += 1
        elif p_out == "failed" and d_out == "failed":
            cat_counts["proof failed, disproof failed"] += 1
        elif p_out == "failed" and d_out == "terminated":
            cat_counts["proof failed, disproof terminated"] += 1
        elif p_out == "terminated" and d_out == "failed":
            cat_counts["proof terminated, disproof failed"] += 1
        elif p_out == "terminated" and d_out == "terminated":
            cat_counts["proof terminated, disproof terminated"] += 1
        else:
            cat_counts[f"proof {p_out}, disproof {d_out}"] += 1

    n_total_attempts = len(all_attempts)
    n_inconclusive = n_total_attempts - n_resolved_goedel - n_resolved_otherwise
    lines.append("")
    lines.append(bold("  Inconclusive attempt breakdown"))
    lines.append(hline("─"))
    lines.append(f"  {'Total attempts tracked:':<44} {n_total_attempts}")
    lines.append(f"  {'  resolved (goedel, proved or disproved):':<44} "
                 f"{green(str(n_resolved_goedel))}  {dim(_pct(n_resolved_goedel, n_total_attempts))}")
    lines.append(f"  {'  resolved (other methods):':<44} "
                 f"{green(str(n_resolved_otherwise))}  {dim(_pct(n_resolved_otherwise, n_total_attempts))}")
    lines.append(f"  {'  inconclusive:':<44} "
                 f"{yellow(str(n_inconclusive))}  {dim(_pct(n_inconclusive, n_total_attempts))}")
    _ordered_cats = [
        "skipped",
        "proof failed, disproof failed",
        "proof failed, disproof terminated",
        "proof terminated, disproof failed",
        "proof terminated, disproof terminated",
    ]
    remaining = dict(cat_counts)
    for cat in _ordered_cats:
        count = remaining.pop(cat, 0)
        label = f"    {cat}:"
        lines.append(f"  {label:<44} {dim(str(count))}  {dim(_pct(count, n_inconclusive))}")
    for cat, count in sorted(remaining.items(), key=lambda x: -x[1]):
        label = f"    {cat}:"
        lines.append(f"  {label:<44} {dim(str(count))}  {dim(_pct(count, n_inconclusive))}")

    # Failure reason breakdown (for complete sessions)
    lines.append("")
    lines.append(bold("  Failure reasons (complete proof sessions)"))
    lines.append(hline("─"))
    reason_counts: Dict[str, int] = defaultdict(int)
    for s in proofs:
        reason_counts[s.termination_reason or "unknown"] += 1
    max_count = max(reason_counts.values())
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(count / max_count * 40)
        lines.append(f"  {reason:<35} {count:>4}  {dim(bar)}")

    all_goedel_results = [  # includes results before --continue
        r
        for rs in results.values()
        for r in rs["all_results"]
        if r.get("method") in ("goedel_prover", "goedel_disprover")
    ]

    session_keys = set((s.problem_id, s.attempt) for s in sessions)
    goedel_results_from_current_run = [
        r
        for rs in results.values()
        for r in rs["all_results"]
        if (rs["problem_id"], rs["attempt"]) in session_keys
        if r.get("method") in ("goedel_prover", "goedel_disprover")
    ]
    goedel_killed_reasons = [
        r['details']['traceback'].split('\n')[-2]
        for r in goedel_results_from_current_run
        if r.get('details', {}).get("error") is not None
    ]
    goedel_killed_reason_counts: Dict[str, int] = defaultdict(int)
    for reason in goedel_killed_reasons:
        goedel_killed_reason_counts[reason] += 1

    lines.append("")
    lines.append(bold("  Incomplete session reasons"))
    lines.append(hline("─"))
    lines.append(f"  {'Goedel-results from current run:':<40}  {len(goedel_results_from_current_run)}")
    lines.append(f"  {'Total killed mid-run:':<40}  {len(goedel_killed_reasons)}")
    lines.append(f"  Reason counts:")
    max_count = max(goedel_killed_reason_counts.values())
    for reason, count in sorted(goedel_killed_reason_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(count / max_count * 40)
        reason = f"  {reason}"
        lines.append(f"  {reason:<40} {count:>4}  {dim(bar)}")

    # Round distribution (complete proof sessions)
    if proofs:
        lines.append("")
        lines.append(bold("  Rounds used (complete proof sessions)"))
        lines.append(hline("─"))
        round_counts: Dict[int, int] = defaultdict(int)
        for s in proofs:
            round_counts[s.rounds_used or 0] += 1
        max_count = max(reason_counts.values())
        for n_rounds in sorted(round_counts):
            label = f"{n_rounds} round{'s' if n_rounds != 1 else ''}"
            count = round_counts[n_rounds]
            bar = "█" * int(count / max_count * 40)
            lines.append(f"  {label:<35} {count:>4}  {dim(bar)}")
    
    # Round distribution (complete disproof sessions)
    if disproofs:
        lines.append("")
        lines.append(bold("  Rounds used (complete disproof sessions)"))
        lines.append(hline("─"))
        round_counts: Dict[int, int] = defaultdict(int)
        for s in disproofs:
            round_counts[s.rounds_used or 0] += 1
        max_count = max(reason_counts.values())
        for n_rounds in sorted(round_counts):
            label = f"{n_rounds} round{'s' if n_rounds != 1 else ''}"
            count = round_counts[n_rounds]
            bar = "█" * int(count / max_count * 40)
            lines.append(f"  {label:<35} {count:>4}  {dim(bar)}")

    # Round distribution (incomplete proof sessions)
    if incomplete_proofs:
        lines.append("")
        lines.append(bold("  Rounds used (incomplete proof sessions)"))
        lines.append(hline("─"))
        round_counts: Dict[int, int] = defaultdict(int)
        for s in incomplete_proofs:
            round_counts[s.rounds_used or 0] += 1
        max_count = max(reason_counts.values())
        for n_rounds in sorted(round_counts):
            label = f"{n_rounds} round{'s' if n_rounds != 1 else ''}"
            count = round_counts[n_rounds]
            bar = "█" * int(count / max_count * 40)
            lines.append(f"  {label:<35} {count:>4}  {dim(bar)}")
    
    # Round distribution (incomplete disproof sessions)
    if incomplete_disproofs:
        lines.append("")
        lines.append(bold("  Rounds used (incomplete disproof sessions)"))
        lines.append(hline("─"))
        round_counts: Dict[int, int] = defaultdict(int)
        for s in incomplete_disproofs:
            round_counts[s.rounds_used or 0] += 1
        max_count = max(reason_counts.values())
        for n_rounds in sorted(round_counts):
            label = f"{n_rounds} round{'s' if n_rounds != 1 else ''}"
            count = round_counts[n_rounds]
            bar = "█" * int(count / max_count * 40)
            lines.append(f"  {label:<35} {count:>4}  {dim(bar)}")

    # Timing
    times = [s.elapsed_ms for s in complete if s.elapsed_ms is not None]
    if times:
        avg_ms = sum(times) / len(times)
        median_ms = np.median(times)
        max_ms = max(times)
        min_ms = min(times)
        lines.append("")
        lines.append(bold("  Session timing (complete sessions)"))
        lines.append(hline("─"))
        lines.append(f"  {'Average session time:':<40} {_fmt_ms(int(avg_ms))}")
        lines.append(f"  {'Median session time:':<40} {_fmt_ms(int(median_ms))}")
        lines.append(f"  {'Fastest session:':<40} {_fmt_ms(min_ms)}")
        lines.append(f"  {'Slowest session:':<40} {_fmt_ms(max_ms)}")

    # Lean compilation outcomes across all rounds
    all_rounds = [r for s in complete for r in s.rounds]
    if all_rounds:
        n_ok       = sum(1 for r in all_rounds if r.lean_ok is True)
        n_fail     = sum(1 for r in all_rounds if r.lean_ok is False and not r.lean_timed_out and not r.lean_oom)
        n_timeout  = sum(1 for r in all_rounds if r.lean_timed_out)
        n_oom      = sum(1 for r in all_rounds if r.lean_oom)
        n_nocode   = sum(1 for r in all_rounds if r.termination_reason == "no_code_block")
        n_ctx      = sum(1 for r in all_rounds if r.termination_reason == "context_exceeded")
        lines.append("")
        lines.append(bold("  Lean compilation results (all rounds, complete sessions)"))
        lines.append(hline("─"))
        lines.append(f"  {'Total rounds:':<40} {len(all_rounds)}")
        lines.append(f"  {'  compiled OK:':<40} {green(str(n_ok))}")
        lines.append(f"  {'  compiled with errors:':<40} {red(str(n_fail))}")
        lines.append(f"  {'  timed out:':<40} {yellow(str(n_timeout))}")
        lines.append(f"  {'  out of memory:':<40} {yellow(str(n_oom))}")
        lines.append(f"  {'  no code block in output:':<40} {dim(str(n_nocode))}")
        lines.append(f"  {'  context window exceeded:':<40} {dim(str(n_ctx))}")

    proven_goedel_results = [
        r for r in results.values()
        if r.get("method") in ("goedel_prover", "goedel_disprover")
    ]

    lines.append("")
    lines.append(bold("  check_results.jsonl"))
    lines.append(hline("─"))
    lines.append(f"  {'Total records:':<40} {len(results)}")
    lines.append(f"  {'Decided by Goedel (proof/disproof):':<40} {len(proven_goedel_results)}")
    lines.append(f"  {'  From previous runs:':<40} {dim(len(all_goedel_results) - len(sessions))}")
    n_eq_true  = sum(1 for r in results.values() if r.get("equivalent") is True)
    n_eq_false = sum(1 for r in results.values() if r.get("equivalent") is False)
    n_eq_none  = sum(1 for r in results.values() if r.get("equivalent") is None)
    lines.append(f"  {'equivalent=True:':<40} {green(str(n_eq_true))}")
    lines.append(f"  {'equivalent=False (disproved):':<40} {red(str(n_eq_false))}")
    lines.append(f"  {'equivalent=None (inconclusive):':<40} {yellow(str(n_eq_none))}")
    by_method: Dict[str, int] = defaultdict(int)
    for r in results.values():
        if r.get("equivalent") is True:
            by_method[r.get("method", "?")] += 1
    if by_method:
        lines.append("")
        lines.append(dim("  Methods that yielded equivalent=True:"))
        for method, count in sorted(by_method.items(), key=lambda x: -x[1]):
            lines.append(f"    {method:<35} {count}")

    lines.append("")
    lines.append("═" * WIDTH)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

_FILTER_CHOICES = ("proof", "disproof", "proved", "disproved", "failed", "incomplete")


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
    if filter_val is not None:
        raise ValueError(f"--filter must be one of {_FILTER_CHOICES}, got {filter_val}")
    return sessions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _USE_COLOR

    p = argparse.ArgumentParser(
        description="Inspect Goedel prover conversations from a check_formalizations run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--goedel-events",
        required=True,
        help="Path to goedel_events.jsonl.",
    )
    p.add_argument(
        "--results",
        required=True,
        help="Path to check_results.jsonl (optional, adds cross-reference info).",
    )
    p.add_argument(
        "--filter",
        choices=_FILTER_CHOICES,
        default=None,
        help=(
            "Show only a subset of sessions: "
            "proof / disproof / proved / disproved / failed / incomplete."
            "\n- proof/disproof include unsuccessful sessions"
            "\n- proved/disproved include only successful sessions"
            "\n- incomplete are the ones w/o an end event (timeout exception, keyboardinterrupt, etc.)"
            "\n- failed are the ones that had issues with context/rounds/splice error"
        ),
    )
    p.add_argument(
        "--problem-id",
        default="",
        help="Show only sessions for this problem ID.",
    )
    p.add_argument(
        "--max-sessions",
        type=int,
        default=20,
        help="Maximum number of sessions to display (0 = all).",
    )
    p.add_argument(
        "--stats-only",
        action="store_true",
        help="Skip conversation display; show only summary statistics.",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour output.",
    )
    p.add_argument(
        "--no-truncate",
        action="store_true",
        help="Print full prompt and model output without truncation.",
    )
    p.add_argument(
        "--max-output-chars",
        type=int,
        default=3000,
        help="Truncate model outputs / prompts to this many characters (default: 3000).",
    )
    args = p.parse_args()

    if args.no_color or not sys.stdout.isatty():
        _USE_COLOR = False

    sessions = load_goedel_sessions(args.goedel_events)
    results = load_check_results(args.results)

    # Filter
    filtered_sessions = sessions
    if args.problem_id:
        filtered_sessions = [s for s in filtered_sessions if s.problem_id == args.problem_id]
    filtered_sessions = apply_filter(filtered_sessions, args.filter)

    to_display = filtered_sessions
    if args.max_sessions > 0:
        to_display = to_display[:args.max_sessions]

    # Display conversations
    if not args.stats_only:
        print(f"\n{bold('Goedel Conversation Inspector')}")
        print(dim(f"Events file : {args.goedel_events}"))
        print(dim(f"Results file: {args.results}"))
        filter_desc = []
        if args.problem_id:
            filter_desc.append(f"problem_id={args.problem_id}")
        if args.filter:
            filter_desc.append(f"filter={args.filter}")
        if filter_desc:
            print(dim(f"Filter      : {', '.join(filter_desc)}"))
        print(dim(f"Showing {len(to_display)} of {len(filtered_sessions)} filtered sessions "
                  f"({len(sessions)} total)"))

        for sess in to_display:
            cr = results.get((sess.problem_id, sess.attempt))
            print(render_session(
                sess,
                cr,
                max_output_chars=args.max_output_chars,
                no_truncate=args.no_truncate,
            ))

    print(render_stats(sessions, results))


if __name__ == "__main__":
    main()
