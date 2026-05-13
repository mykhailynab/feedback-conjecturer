#!/usr/bin/env python3
"""
Detailed comparison of TIR prover runs (v1 and v2 schemas).

Produces:
  1. A comprehensive text report with all extractable metrics.
  2. LaTeX table snippets (consolidated for the paper).
  3. An analysis LaTeX section with comparative conclusions.

Usage:
    PYTHONPATH=. python analysis_and_inspection/for_paper/compare_tir_runs.py

    # Override run paths:
    PYTHONPATH=. python analysis_and_inspection/for_paper/compare_tir_runs.py \
        --run logs/full_20mins_tir_pass1_strip  "TIR strip"          v1 \
        --run logs/full_20mins_tir_pass1_no_strip "TIR no-strip"     v1 \
        --run logs/short_6x4090                 "TIR strip+informal 6xGPU" v2 \
        --run logs/short_4x5000                 "TIR strip+informal 4xGPU" v2
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


# ════════════════════════════════════════════════════════════════
#  Schema-aware record accessors
# ════════════════════════════════════════════════════════════════

def _status(r: dict) -> str:
    return r.get("conjecture_formalization_status", r.get("status", ""))


def _is_token_limited(r: dict) -> bool:
    return bool(r.get("token_limit_triggered", r.get("incomplete", False)))


def _get_all_proof_results(r: dict) -> list[dict]:
    """Return list of proof result dicts (works for v1 and v2)."""
    apr = r.get("all_proof_results")
    if apr:
        return apr
    pr = r.get("proof_result")
    if pr:
        return [pr]
    return []


def _get_all_disproof_results(r: dict) -> list[dict]:
    apr = r.get("all_disproof_results")
    if apr:
        return apr
    dr = r.get("disproof_result")
    if dr:
        return [dr]
    return []


def _get_conversation_history(pr: dict) -> list[dict]:
    """Get conversation history from a proof_result (v1 or v2)."""
    sr = pr.get("session_result") or {}
    ch = sr.get("conversation_history")
    if ch:
        return ch
    return pr.get("conversation_history") or []


def _get_turns(pr: dict) -> list[dict]:
    """Get v1-style turns from a proof_result."""
    return pr.get("turns") or []


def _get_partial_turn(pr: dict) -> dict | None:
    sr = pr.get("session_result") or {}
    return sr.get("partial_assistant_turn") or pr.get("partial_assistant_turn")


def _get_session_elapsed_ms(pr: dict) -> int | None:
    sr = pr.get("session_result") or {}
    return sr.get("elapsed_ms") or pr.get("elapsed_ms")


def _get_session_total_output_tokens(pr: dict) -> int | None:
    sr = pr.get("session_result") or {}
    return sr.get("total_output_tokens")


def _get_session_total_generation_ms(pr: dict) -> int | None:
    sr = pr.get("session_result") or {}
    return sr.get("total_generation_ms")


def _get_session_exception(pr: dict) -> str | None:
    sr = pr.get("session_result") or {}
    return sr.get("exception") or pr.get("exception")


def _get_session_token_limit_triggered(pr: dict) -> bool:
    sr = pr.get("session_result") or {}
    return bool(sr.get("token_limit_triggered") or pr.get("token_limit_triggered")
                or pr.get("incomplete"))


# ════════════════════════════════════════════════════════════════
#  Tool call extraction (v1 turns + v2 conversation_history)
# ════════════════════════════════════════════════════════════════

@dataclass
class ToolCallStats:
    lean_calls: int = 0
    lean_final_calls: int = 0
    python_calls: int = 0
    other_calls: int = 0

    @property
    def total(self) -> int:
        return self.lean_calls + self.lean_final_calls + self.python_calls + self.other_calls


def _extract_tool_calls_v2(ch: list[dict]) -> ToolCallStats:
    s = ToolCallStats()
    for msg in ch:
        for tc in msg.get("tool_calls", []):
            name = tc.get("function", {}).get("name", "")
            if name == "lean":
                s.lean_calls += 1
            elif name == "lean_final":
                s.lean_final_calls += 1
            elif name == "python":
                s.python_calls += 1
            else:
                s.other_calls += 1
    return s


def _extract_tool_calls_v1(turns: list[dict]) -> ToolCallStats:
    s = ToolCallStats()
    for t in turns:
        for tc in t.get("tool_calls", []):
            name = tc.get("name", "")
            if name == "lean":
                s.lean_calls += 1
            elif name == "lean_final":
                s.lean_final_calls += 1
            elif name == "python":
                s.python_calls += 1
            else:
                s.other_calls += 1
    return s


def _count_turns(pr: dict) -> int:
    """Count assistant turns in a proof result (including partial turn)."""
    ch = _get_conversation_history(pr)
    if ch:
        n = sum(1 for m in ch if m.get("role") == "assistant")
        pat = _get_partial_turn(pr)
        if pat and pat.get("role") == "assistant":
            n += 1
        return n
    turns = _get_turns(pr)
    return len(turns)


def _has_thinking(pr: dict) -> bool:
    """Check if any turn/message has reasoning_content/thinking (including partial)."""
    ch = _get_conversation_history(pr)
    if ch:
        if any(m.get("reasoning_content") for m in ch if m.get("role") == "assistant"):
            return True
        pat = _get_partial_turn(pr)
        if pat and pat.get("reasoning_content"):
            return True
        return False
    turns = _get_turns(pr)
    return any(t.get("thinking") for t in turns)


def _count_lean_errors_from_tool_results(pr: dict) -> int:
    """Count lean tool calls that returned errors (heuristic: 'error' in result)."""
    errors = 0
    ch = _get_conversation_history(pr)
    if ch:
        # Include partial assistant turn for tool call IDs
        pat = _get_partial_turn(pr)
        if pat:
            ch = list(ch) + [pat]
        lean_call_ids = set()
        for msg in ch:
            for tc in msg.get("tool_calls", []):
                name = tc.get("function", {}).get("name", "")
                if name in ("lean", "lean_final"):
                    lean_call_ids.add(tc.get("id"))
        for msg in ch:
            if msg.get("role") == "tool" and msg.get("tool_call_id") in lean_call_ids:
                content = msg.get("content", "")
                if "error" in content.lower() or "sorry" in content.lower():
                    errors += 1
        return errors
    # v1: check tool_calls result field
    turns = _get_turns(pr)
    for t in turns:
        for tc in t.get("tool_calls", []):
            if tc.get("name") in ("lean", "lean_final"):
                result = tc.get("result", "")
                if "error" in result.lower():
                    errors += 1
    return errors


# ════════════════════════════════════════════════════════════════
#  Per-run statistics
# ════════════════════════════════════════════════════════════════

@dataclass
class RunStats:
    label: str
    schema: str
    path: str

    # Record-level counts
    total_records: int = 0
    n_success: int = 0        # status=success (ran the prover)
    n_skipped: int = 0
    n_failed: int = 0         # status=failed (formalization error)
    n_proved: int = 0
    n_disproved: int = 0
    n_token_limited: int = 0  # token-limited and not proved/disproved
    n_inconclusive: int = 0

    # Attempt-level (per proof_result)
    n_proof_attempts: int = 0
    n_proof_attempts_proved: int = 0

    # Tool usage (aggregated across all proof results)
    total_lean_calls: int = 0
    total_lean_final_calls: int = 0
    total_python_calls: int = 0
    total_lean_errors: int = 0

    # Turns
    turns_per_session: list[int] = field(default_factory=list)

    # Timing
    elapsed_ms_list: list[int] = field(default_factory=list)
    elapsed_ms_proved: list[int] = field(default_factory=list)
    elapsed_ms_failed: list[int] = field(default_factory=list)

    # Output tokens (v2 only, from session_result)
    output_tokens_list: list[int] = field(default_factory=list)
    generation_ms_list: list[int] = field(default_factory=list)

    # Termination reasons
    termination_reasons: Counter = field(default_factory=Counter)

    # Thinking/CoT presence
    n_with_thinking: int = 0
    n_without_thinking: int = 0

    # Informal proof (v2 only)
    n_informal_used: int = 0
    n_informal_not_used: int = 0

    # Token limit triggered
    n_session_token_limited: int = 0

    # Exceptions
    n_exceptions: int = 0

    # Per-problem aggregation
    problems: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))

    # Retries per record
    retries_per_record: list[int] = field(default_factory=list)


def load_run(path: str, label: str, schema: str) -> RunStats:
    rs = RunStats(label=label, schema=schema, path=path)
    results_path = Path(path) / "prove_results.jsonl"

    with results_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rs.total_records += 1
            rs.problems[d["problem_id"]].append(d)

            status = _status(d)
            if d.get("skipped"):
                rs.n_skipped += 1
                continue
            if status == "failed":
                rs.n_failed += 1
                continue
            if status == "success":
                rs.n_success += 1
            else:
                # some v1 records might not have status at all but have proof_result
                pass

            if d.get("proved"):
                rs.n_proved += 1
            if d.get("disproved"):
                rs.n_disproved += 1
            if _is_token_limited(d) and not d.get("proved") and not d.get("disproved"):
                rs.n_token_limited += 1

            # Informal proof (v2)
            if d.get("informal_proof_used"):
                rs.n_informal_used += 1
            elif schema == "v2" and status == "success":
                rs.n_informal_not_used += 1

            # Process proof results
            all_pr = _get_all_proof_results(d)
            rs.retries_per_record.append(len(all_pr))

            for pr in all_pr:
                rs.n_proof_attempts += 1
                if pr.get("proved"):
                    rs.n_proof_attempts_proved += 1

                # Termination reason
                tr = pr.get("termination_reason", "unknown")
                rs.termination_reasons[tr] += 1

                # Tool calls (include partial assistant turn for v2)
                ch = _get_conversation_history(pr)
                if ch:
                    ch_with_partial = list(ch)
                    pat = _get_partial_turn(pr)
                    if pat:
                        ch_with_partial.append(pat)
                    tc_stats = _extract_tool_calls_v2(ch_with_partial)
                else:
                    tc_stats = _extract_tool_calls_v1(_get_turns(pr))
                rs.total_lean_calls += tc_stats.lean_calls
                rs.total_lean_final_calls += tc_stats.lean_final_calls
                rs.total_python_calls += tc_stats.python_calls

                # Lean errors (uses same ch which already includes partial)
                rs.total_lean_errors += _count_lean_errors_from_tool_results(pr)

                # Turns
                n_turns = _count_turns(pr)
                rs.turns_per_session.append(n_turns)

                # Thinking
                if _has_thinking(pr):
                    rs.n_with_thinking += 1
                else:
                    rs.n_without_thinking += 1

                # Timing
                elapsed = _get_session_elapsed_ms(pr)
                if elapsed is not None:
                    rs.elapsed_ms_list.append(elapsed)
                    if pr.get("proved"):
                        rs.elapsed_ms_proved.append(elapsed)
                    else:
                        rs.elapsed_ms_failed.append(elapsed)

                # Output tokens (v2)
                ot = _get_session_total_output_tokens(pr)
                if ot is not None:
                    rs.output_tokens_list.append(ot)
                gms = _get_session_total_generation_ms(pr)
                if gms is not None:
                    rs.generation_ms_list.append(gms)

                # Session token limit
                if _get_session_token_limit_triggered(pr):
                    rs.n_session_token_limited += 1

                # Exceptions
                if _get_session_exception(pr):
                    rs.n_exceptions += 1

    # Inconclusive = success - proved - disproved - token_limited
    rs.n_inconclusive = rs.n_success - rs.n_proved - rs.n_disproved - rs.n_token_limited

    return rs


# ════════════════════════════════════════════════════════════════
#  pass@k computation
# ════════════════════════════════════════════════════════════════

def _pass_at_k(n: int, c: int, k: int) -> float:
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_pass_at_k(rs: RunStats) -> dict:
    n_total = rs.total_records
    n_proved = rs.n_proved
    n_disproved = rs.n_disproved
    n_problems = len(rs.problems)

    p1_lo = n_proved / n_total if n_total else 0.0
    p1_hi = 1.0 - n_disproved / n_total if n_total else 0.0

    # Infer k
    att_counts = [len(v) for v in rs.problems.values()]
    k = max(set(att_counts), key=att_counts.count) if att_counts else 1

    pk_lo_values, pk_hi_values = [], []
    n_problems_proved = 0
    n_problems_proved_upper = 0
    for entries in rs.problems.values():
        n = len(entries)
        c_lo = sum(1 for r in entries if r.get("proved") is True)
        c_hi = n - sum(1 for r in entries if r.get("disproved") is True)
        pk_lo_values.append(_pass_at_k(n, c_lo, k))
        pk_hi_values.append(_pass_at_k(n, c_hi, k))
        if c_lo > 0:
            n_problems_proved += 1
        if c_hi > 0:
            n_problems_proved_upper += 1

    pk_lo = sum(pk_lo_values) / n_problems if n_problems else 0.0
    pk_hi = sum(pk_hi_values) / n_problems if n_problems else 0.0

    return {
        "p1_lo": p1_lo, "p1_hi": p1_hi,
        "pk_lo": pk_lo, "pk_hi": pk_hi,
        "k": k,
        "n_problems": n_problems,
        "n_total_attempts": n_total,
        "n_proved": n_proved,
        "n_disproved": n_disproved,
        "n_problems_proved": n_problems_proved,
        "n_problems_proved_upper": n_problems_proved_upper,
    }


# ════════════════════════════════════════════════════════════════
#  Overlap analysis
# ════════════════════════════════════════════════════════════════

def compute_overlap(runs: list[RunStats]) -> tuple[dict, dict[str, set[str]]]:
    """Compute which problems are proved by which runs.

    Returns (summary_dict, proved_sets_by_label).
    """
    proved_sets: dict[str, set[str]] = {}
    for rs in runs:
        proved_pids = set()
        for pid, entries in rs.problems.items():
            if any(r.get("proved") for r in entries):
                proved_pids.add(pid)
        proved_sets[rs.label] = proved_pids

    result = {"per_run": {label: len(pids) for label, pids in proved_sets.items()}}

    # Pairwise overlaps
    labels = list(proved_sets.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            inter = proved_sets[a] & proved_sets[b]
            only_a = proved_sets[a] - proved_sets[b]
            only_b = proved_sets[b] - proved_sets[a]
            result[f"{a} ∩ {b}"] = len(inter)
            result[f"{a} only vs {b}"] = len(only_a)
            result[f"{b} only vs {a}"] = len(only_b)

    # All runs
    if len(labels) >= 3:
        all_inter = proved_sets[labels[0]]
        all_union = proved_sets[labels[0]]
        for l in labels[1:]:
            all_inter = all_inter & proved_sets[l]
            all_union = all_union | proved_sets[l]
        result["all_intersection"] = len(all_inter)
        result["union"] = len(all_union)

    return result, proved_sets


# ════════════════════════════════════════════════════════════════
#  Text report
# ════════════════════════════════════════════════════════════════

def _fmt_ms(ms: int | float) -> str:
    s = ms / 1000
    if s < 60:
        return f"{s:.2f}s"
    m = s / 60
    return f"{m:.2f}min"


def _safe_stats(vals: list[int | float]) -> dict:
    if not vals:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "sum": 0, "n": 0}
    return {
        "min": min(vals), "max": max(vals),
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "sum": sum(vals), "n": len(vals),
    }


def render_text_report(all_runs: list[RunStats]) -> str:
    lines = []
    W = 80

    for rs in all_runs:
        lines.append("=" * W)
        lines.append(f"  {rs.label}  (schema={rs.schema}, path={rs.path})")
        lines.append("=" * W)

        lines.append("")
        lines.append("── Record-level overview ──")
        lines.append(f"  Total records:                  {rs.total_records}")
        lines.append(f"    status=success:               {rs.n_success}")
        lines.append(f"    skipped:                      {rs.n_skipped}")
        lines.append(f"    status=failed:                {rs.n_failed}")
        lines.append(f"  Among success records:")
        lines.append(f"    proved:                       {rs.n_proved}  ({rs.n_proved/max(rs.n_success,1):.2%})")
        lines.append(f"    disproved:                    {rs.n_disproved}  ({rs.n_disproved/max(rs.n_success,1):.2%})")
        lines.append(f"    token-limited (not resolved): {rs.n_token_limited}")
        lines.append(f"    inconclusive:                 {rs.n_inconclusive}")

        lines.append("")
        lines.append("── Proof attempts (retries) ──")
        lines.append(f"  Total proof attempts:           {rs.n_proof_attempts}")
        lines.append(f"  Proved attempts:                {rs.n_proof_attempts_proved}")
        rs_retries = _safe_stats(rs.retries_per_record)
        lines.append(f"  Retries per record:             min={rs_retries['min']}, mean={rs_retries['mean']:.2f}, "
                      f"median={rs_retries['median']:.0f}, max={rs_retries['max']}")

        lines.append("")
        lines.append("── Termination reasons ──")
        for reason, count in sorted(rs.termination_reasons.items(), key=lambda x: -x[1]):
            pct = count / max(rs.n_proof_attempts, 1) * 100
            lines.append(f"  {reason:<35} {count:>5}  ({pct:.2f}%)")

        lines.append("")
        lines.append("── Tool usage ──")
        lines.append(f"  Lean calls (total):             {rs.total_lean_calls}")
        lines.append(f"  Lean final calls (total):       {rs.total_lean_final_calls}")
        lines.append(f"  Python calls (total):           {rs.total_python_calls}")
        total_tool = rs.total_lean_calls + rs.total_lean_final_calls + rs.total_python_calls
        lines.append(f"  Total tool calls:               {total_tool}")
        lines.append(f"  Lean errors (heuristic):        {rs.total_lean_errors}"
                      + (f"  ({rs.total_lean_errors/max(rs.total_lean_calls+rs.total_lean_final_calls,1):.2%} of lean calls)"
                         if rs.total_lean_calls + rs.total_lean_final_calls > 0 else ""))
        if rs.n_proof_attempts:
            lines.append(f"  Tool calls per proof attempt:   {total_tool/rs.n_proof_attempts:.2f}")
            lines.append(f"  Lean calls per proof attempt:   {(rs.total_lean_calls+rs.total_lean_final_calls)/rs.n_proof_attempts:.2f}")
            lines.append(f"  Python calls per proof attempt: {rs.total_python_calls/rs.n_proof_attempts:.2f}")

        lines.append("")
        lines.append("── Turns per session ──")
        ts = _safe_stats(rs.turns_per_session)
        lines.append(f"  min={ts['min']}, mean={ts['mean']:.2f}, median={ts['median']:.0f}, max={ts['max']}")

        lines.append("")
        lines.append("── Thinking/CoT presence ──")
        lines.append(f"  Sessions with thinking:         {rs.n_with_thinking}")
        lines.append(f"  Sessions without thinking:      {rs.n_without_thinking}")

        if rs.schema == "v2":
            lines.append("")
            lines.append("── Informal proof usage (v2) ──")
            lines.append(f"  Used:                           {rs.n_informal_used}")
            lines.append(f"  Not used:                       {rs.n_informal_not_used}")

        lines.append("")
        lines.append("── Session timing ──")
        ts_all = _safe_stats(rs.elapsed_ms_list)
        lines.append(f"  All sessions (n={ts_all['n']}):")
        lines.append(f"    min={_fmt_ms(ts_all['min'])}, mean={_fmt_ms(ts_all['mean'])}, "
                      f"median={_fmt_ms(ts_all['median'])}, max={_fmt_ms(ts_all['max'])}")
        ts_proved = _safe_stats(rs.elapsed_ms_proved)
        if ts_proved["n"]:
            lines.append(f"  Proved sessions (n={ts_proved['n']}):")
            lines.append(f"    min={_fmt_ms(ts_proved['min'])}, mean={_fmt_ms(ts_proved['mean'])}, "
                          f"median={_fmt_ms(ts_proved['median'])}, max={_fmt_ms(ts_proved['max'])}")
        ts_failed = _safe_stats(rs.elapsed_ms_failed)
        if ts_failed["n"]:
            lines.append(f"  Failed sessions (n={ts_failed['n']}):")
            lines.append(f"    min={_fmt_ms(ts_failed['min'])}, mean={_fmt_ms(ts_failed['mean'])}, "
                          f"median={_fmt_ms(ts_failed['median'])}, max={_fmt_ms(ts_failed['max'])}")

        if rs.output_tokens_list:
            lines.append("")
            lines.append("── Output tokens (v2) ──")
            ot = _safe_stats(rs.output_tokens_list)
            lines.append(f"  min={ot['min']}, mean={ot['mean']:.0f}, median={ot['median']:.0f}, max={ot['max']}")
            lines.append(f"  total={ot['sum']}")

        if rs.generation_ms_list:
            lines.append("")
            lines.append("── Generation time (v2) ──")
            gt = _safe_stats(rs.generation_ms_list)
            lines.append(f"  min={_fmt_ms(gt['min'])}, mean={_fmt_ms(gt['mean'])}, "
                          f"median={_fmt_ms(gt['median'])}, max={_fmt_ms(gt['max'])}")
            lines.append(f"  total={_fmt_ms(gt['sum'])}")
            if gt["sum"] > 0 and rs.output_tokens_list:
                tok_per_s = sum(rs.output_tokens_list) / (gt["sum"] / 1000)
                lines.append(f"  Throughput: {tok_per_s:.2f} tok/s")

        lines.append("")
        lines.append("── Token limit & exceptions ──")
        lines.append(f"  Session-level token limited:    {rs.n_session_token_limited}")
        lines.append(f"  Exceptions:                     {rs.n_exceptions}")

        # pass@k
        pk = compute_pass_at_k(rs)
        lines.append("")
        lines.append("── Proven accuracy (pass@k) ──")
        lines.append(f"  Universe: {pk['n_total_attempts']} attempts across {pk['n_problems']} problems, k={pk['k']}")
        lines.append(f"  pass@1 lower (proved):          {pk['p1_lo']:.2%}  ({pk['n_proved']}/{pk['n_total_attempts']})")
        lines.append(f"  pass@1 upper (1-disproved):     {pk['p1_hi']:.2%}")
        lines.append(f"  pass@{pk['k']} lower:               {pk['pk_lo']:.2%}  (~{pk['pk_lo']*pk['n_problems']:.2f}/{pk['n_problems']} problems)")
        lines.append(f"  pass@{pk['k']} upper:               {pk['pk_hi']:.2%}  (~{pk['pk_hi']*pk['n_problems']:.2f}/{pk['n_problems']} problems)")
        lines.append(f"  Problems w/ ≥1 proof:           {pk['n_problems_proved']}/{pk['n_problems']}")

        lines.append("")

    # ── Overlap analysis ──
    lines.append("=" * W)
    lines.append("  CROSS-RUN OVERLAP ANALYSIS")
    lines.append("=" * W)
    overlap, _ = compute_overlap(all_runs)
    lines.append("")
    for k, v in overlap.items():
        lines.append(f"  {k:<40} {v}")

    lines.append("")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
#  LaTeX tables
# ════════════════════════════════════════════════════════════════

def render_latex_tables(all_runs: list[RunStats]) -> str:
    parts = []
    n_runs = len(all_runs)
    labels = [rs.label for rs in all_runs]

    # ── Table 1: Overview & accuracy ──
    parts.append(r"""
\begin{table}[htbp]
\centering
\caption{TIR prover run comparison: overview and accuracy.}
\label{tab:tir-comparison-overview}
\small
\begin{tabular}{l """ + " ".join(["r"] * n_runs) + r"""}
\toprule""")
    parts.append(" & ".join(["Metric"] + [_latex_escape(l) for l in labels]) + r" \\")
    parts.append(r"\midrule")

    def _row(name: str, vals: list[str]) -> str:
        return " & ".join([name] + vals) + r" \\"

    parts.append(_row("Total records", [f"{rs.total_records:,}" for rs in all_runs]))
    parts.append(_row(r"\quad success (ran prover)", [f"{rs.n_success:,}" for rs in all_runs]))
    parts.append(_row(r"\quad skipped", [f"{rs.n_skipped:,}" for rs in all_runs]))
    parts.append(_row(r"\quad failed (formalization)", [f"{rs.n_failed:,}" for rs in all_runs]))
    parts.append(r"\midrule")

    parts.append(_row("Proved", [f"{rs.n_proved}" for rs in all_runs]))
    parts.append(_row("Disproved", [f"{rs.n_disproved}" for rs in all_runs]))
    parts.append(_row("Token-limited", [f"{rs.n_token_limited}" for rs in all_runs]))
    parts.append(_row("Inconclusive", [f"{rs.n_inconclusive}" for rs in all_runs]))
    parts.append(r"\midrule")

    pks = [compute_pass_at_k(rs) for rs in all_runs]
    parts.append(_row(r"pass@1 lower", [f"{pk['p1_lo']:.2%}" for pk in pks]))
    k_vals = [pk['k'] for pk in pks]
    for i, pk in enumerate(pks):
        k = pk['k']
    parts.append(_row(
        r"pass@$k$ lower",
        [f"{pk['pk_lo']:.2%} ($k$={pk['k']})" for pk in pks],
    ))
    parts.append(_row(
        "Problems proved",
        [f"{pk['n_problems_proved']}/{pk['n_problems']}" for pk in pks],
    ))

    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append(r"\end{table}")

    # ── Table 2: Tool usage & sessions ──
    parts.append("")
    parts.append(r"""
\begin{table}[htbp]
\centering
\caption{TIR prover run comparison: tool usage and session statistics.}
\label{tab:tir-comparison-tools}
\small
\begin{tabular}{l """ + " ".join(["r"] * n_runs) + r"""}
\toprule""")
    parts.append(" & ".join(["Metric"] + [_latex_escape(l) for l in labels]) + r" \\")
    parts.append(r"\midrule")

    parts.append(r"\multicolumn{" + str(n_runs + 1) + r"}{l}{\textit{Tool calls}} \\")
    parts.append(_row(r"\quad Lean calls", [f"{rs.total_lean_calls:,}" for rs in all_runs]))
    parts.append(_row(r"\quad Lean final calls", [f"{rs.total_lean_final_calls:,}" for rs in all_runs]))
    parts.append(_row(r"\quad Python calls", [f"{rs.total_python_calls:,}" for rs in all_runs]))
    total_tools = [rs.total_lean_calls + rs.total_lean_final_calls + rs.total_python_calls for rs in all_runs]
    parts.append(_row(r"\quad Total tool calls", [f"{t:,}" for t in total_tools]))
    parts.append(_row(r"\quad Lean error rate",
        [f"{rs.total_lean_errors/max(rs.total_lean_calls+rs.total_lean_final_calls,1):.2%}" for rs in all_runs]))
    parts.append(r"\midrule")

    parts.append(r"\multicolumn{" + str(n_runs + 1) + r"}{l}{\textit{Turns per session}} \\")
    for stat_name, stat_fn in [("min", min), ("mean", statistics.mean), ("median", statistics.median), ("max", max)]:
        parts.append(_row(
            rf"\quad {stat_name}",
            [f"{stat_fn(rs.turns_per_session):.2f}" if rs.turns_per_session else "---" for rs in all_runs],
        ))
    parts.append(r"\midrule")

    parts.append(r"\multicolumn{" + str(n_runs + 1) + r"}{l}{\textit{CoT / Thinking}} \\")
    parts.append(_row(r"\quad w/ thinking", [f"{rs.n_with_thinking:,}" for rs in all_runs]))
    parts.append(_row(r"\quad w/o thinking", [f"{rs.n_without_thinking:,}" for rs in all_runs]))
    parts.append(r"\midrule")

    parts.append(r"\multicolumn{" + str(n_runs + 1) + r"}{l}{\textit{Termination reasons}} \\")
    all_reasons = sorted(
        set().union(*(rs.termination_reasons.keys() for rs in all_runs)),
        key=lambda r: -sum(rs.termination_reasons.get(r, 0) for rs in all_runs),
    )
    for reason in all_reasons:
        parts.append(_row(
            rf"\quad {_latex_escape(reason)}",
            [f"{rs.termination_reasons.get(reason, 0):,}" for rs in all_runs],
        ))

    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append(r"\end{table}")

    # ── Table 3: Timing ──
    parts.append("")
    parts.append(r"""
\begin{table}[htbp]
\centering
\caption{TIR prover run comparison: session timing (seconds).}
\label{tab:tir-comparison-timing}
\small
\begin{tabular}{l """ + " ".join(["r"] * n_runs) + r"""}
\toprule""")
    parts.append(" & ".join(["Metric"] + [_latex_escape(l) for l in labels]) + r" \\")
    parts.append(r"\midrule")

    def _time_row(name: str, vals_lists: list[list[int]]) -> str:
        return _row(name, [_fmt_s(statistics.mean(v)) if v else "---" for v in vals_lists])

    def _fmt_s(ms: float) -> str:
        return f"{ms/1000:.2f}"

    parts.append(r"\multicolumn{" + str(n_runs + 1) + r"}{l}{\textit{All sessions}} \\")
    for stat_name, stat_fn in [("min", min), ("mean", statistics.mean), ("median", statistics.median), ("max", max)]:
        parts.append(_row(
            rf"\quad {stat_name}",
            [f"{stat_fn(rs.elapsed_ms_list)/1000:.2f}" if rs.elapsed_ms_list else "---" for rs in all_runs],
        ))
    parts.append(r"\midrule")
    parts.append(r"\multicolumn{" + str(n_runs + 1) + r"}{l}{\textit{Proved sessions}} \\")
    for stat_name, stat_fn in [("min", min), ("mean", statistics.mean), ("median", statistics.median), ("max", max)]:
        parts.append(_row(
            rf"\quad {stat_name}",
            [f"{stat_fn(rs.elapsed_ms_proved)/1000:.2f}" if rs.elapsed_ms_proved else "---" for rs in all_runs],
        ))
    parts.append(r"\midrule")
    parts.append(r"\multicolumn{" + str(n_runs + 1) + r"}{l}{\textit{Failed sessions}} \\")
    for stat_name, stat_fn in [("min", min), ("mean", statistics.mean), ("median", statistics.median), ("max", max)]:
        parts.append(_row(
            rf"\quad {stat_name}",
            [f"{stat_fn(rs.elapsed_ms_failed)/1000:.2f}" if rs.elapsed_ms_failed else "---" for rs in all_runs],
        ))

    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append(r"\end{table}")

    return "\n".join(parts)


def render_analysis_latex(all_runs: list[RunStats]) -> str:
    """Generate a comparative analysis LaTeX section with combined tables."""
    parts = []
    n_runs = len(all_runs)
    labels = [rs.label for rs in all_runs]
    pks = [compute_pass_at_k(rs) for rs in all_runs]
    _, proved_sets = compute_overlap(all_runs)

    def _row(name: str, vals: list[str]) -> str:
        return " & ".join([name] + vals) + r" \\"

    # ── Table: Combined comparison (the main paper table) ──
    parts.append(r"""
\begin{table}[htbp]
\centering
\caption{TIR prover comparison across configurations. ``Strip'' = chain-of-thought tokens removed
from the context after generation. ``Infr'' = informal proof injected into the system prompt.
All runs use Qwen3.6-35B-A3B (6-bit) with temperature 0.6, top-p 0.95.}
\label{tab:tir-comparison}
\small
\begin{tabular}{l """ + " ".join(["r"] * n_runs) + r"""}
\toprule""")
    parts.append(" & ".join(["\\textbf{Run Name}"] + [_latex_escape(l) for l in labels]) + r" \\")
    parts.append(r"\midrule")

    # Configuration
    parts.append(r"\multicolumn{" + str(n_runs + 1) + r"}{l}{\textit{Configuration}} \\")
    parts.append(_row(r"\quad CoT stripping", [
        "yes" if "strip" in rs.label.lower() and "no-strip" not in rs.label.lower() else "no"
        for rs in all_runs
    ]))
    parts.append(_row(r"\quad Informal proof", [
        "yes" if ("informal" in rs.label.lower() or "infr" in rs.label.lower()) else "no"
        for rs in all_runs
    ]))
    parts.append(r"\midrule")

    # Accuracy
    parts.append(r"\multicolumn{" + str(n_runs + 1) + r"}{l}{\textit{Accuracy}} \\")
    parts.append(_row(r"\quad Attempts proved",
        [f"{rs.n_proved}/{rs.n_success} ({rs.n_proved/max(rs.n_success,1):.2%})".replace("%", "\\%") for rs in all_runs]))
    parts.append(_row(r"\quad Problems proved (pass@$k$)",
        [f"{pk['n_problems_proved']}/{pk['n_problems']}" for pk in pks]))
    parts.append(_row(r"\quad pass@1",
        [f"{pk['p1_lo']:.2%}".replace("%", "\\%") for pk in pks]))
    parts.append(_row(r"\quad pass@$k$ ($k$={})".format(pks[0]['k']),
        [f"{pk['pk_lo']:.2%}".replace("%", "\\%") for pk in pks]))
    parts.append(r"\midrule")

    # Session behaviour
    parts.append(r"\multicolumn{" + str(n_runs + 1) + r"}{l}{\textit{Session behaviour}} \\")
    parts.append(_row(r"\quad Mean turns / session",
        [f"{statistics.mean(rs.turns_per_session):.2f}" if rs.turns_per_session else "---" for rs in all_runs]))
    total_tools = [rs.total_lean_calls + rs.total_lean_final_calls + rs.total_python_calls for rs in all_runs]
    parts.append(_row(r"\quad Mean tool calls / session",
        [f"{t/max(rs.n_proof_attempts,1):.2f}" for t, rs in zip(total_tools, all_runs)]))
    parts.append(_row(r"\quad Lean calls / session",
        [f"{(rs.total_lean_calls+rs.total_lean_final_calls)/max(rs.n_proof_attempts,1):.2f}" for rs in all_runs]))
    parts.append(_row(r"\quad Python calls / session",
        [f"{rs.total_python_calls/max(rs.n_proof_attempts,1):.2f}" for rs in all_runs]))
    parts.append(_row(r"\quad Lean error rate",
        [f"{rs.total_lean_errors/max(rs.total_lean_calls+rs.total_lean_final_calls,1):.2%}".replace("%", "\\%") for rs in all_runs]))
    parts.append(r"\midrule")

    # Thinking
    parts.append(r"\multicolumn{" + str(n_runs + 1) + r"}{l}{\textit{CoT / Thinking}} \\")
    parts.append(_row(r"\quad Sessions w/ thinking",
        [f"{rs.n_with_thinking}/{rs.n_proof_attempts} ({rs.n_with_thinking/max(rs.n_proof_attempts,1):.0%})".replace("%", "\\%") for rs in all_runs]))
    parts.append(r"\midrule")

    # Termination
    parts.append(r"\multicolumn{" + str(n_runs + 1) + r"}{l}{\textit{Termination reasons}} \\")
    all_reasons = sorted(
        set().union(*(rs.termination_reasons.keys() for rs in all_runs)),
        key=lambda r: -sum(rs.termination_reasons.get(r, 0) for rs in all_runs),
    )
    for reason in all_reasons:
        counts = [rs.termination_reasons.get(reason, 0) for rs in all_runs]
        pcts = [f"{c/max(rs.n_proof_attempts,1):.2%}".replace("%", "\\%") for c, rs in zip(counts, all_runs)]
        parts.append(_row(
            rf"\quad {_latex_escape(reason)}",
            [f"{c:,} ({p})" for c, p in zip(counts, pcts)],
        ))
    parts.append(r"\midrule")

    # Timing
    parts.append(r"\multicolumn{" + str(n_runs + 1) + r"}{l}{\textit{Timing (seconds)}} \\")
    parts.append(_row(r"\quad Mean session time",
        [f"{statistics.mean(rs.elapsed_ms_list)/1000:.0f}" if rs.elapsed_ms_list else "---" for rs in all_runs]))
    parts.append(_row(r"\quad Mean proved session time",
        [f"{statistics.mean(rs.elapsed_ms_proved)/1000:.0f}" if rs.elapsed_ms_proved else "---" for rs in all_runs]))
    parts.append(_row(r"\quad Mean failed session time",
        [f"{statistics.mean(rs.elapsed_ms_failed)/1000:.0f}" if rs.elapsed_ms_failed else "---" for rs in all_runs]))

    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append(r"\end{table}")

    # ── Table: Problem overlap matrix ──
    parts.append("")
    parts.append(r"""
\begin{table}[htbp]
\centering
\caption{Problem-level overlap between TIR prover configurations. Each cell shows the number
of problems proved by the row configuration that are also proved by the column configuration.
Diagonal entries show total problems proved per configuration.}
\label{tab:tir-overlap}
\small""")
    parts.append(r"\begin{tabular}{l " + " ".join(["r"] * n_runs) + r"}")
    parts.append(r"\toprule")
    parts.append(" & ".join([""] + [_latex_escape(l) for l in labels]) + r" \\")
    parts.append(r"\midrule")
    for rs_i in all_runs:
        row_vals = []
        for rs_j in all_runs:
            if rs_i.label == rs_j.label:
                row_vals.append(f"\\textbf{{{len(proved_sets[rs_i.label])}}}")
            else:
                inter = proved_sets[rs_i.label] & proved_sets[rs_j.label]
                row_vals.append(str(len(inter)))
        parts.append(_row(_latex_escape(rs_i.label), row_vals))
    # Union row
    all_union = set()
    for s in proved_sets.values():
        all_union |= s
    parts.append(r"\midrule")
    parts.append(_row(r"\textit{Union}", [str(len(all_union))] * n_runs))
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append(r"\end{table}")

    # ── Table: Unique contributions ──
    parts.append("")
    parts.append(r"""
\begin{table}[htbp]
\centering
\caption{Unique contributions per configuration: problems proved exclusively by one
configuration that no other configuration solved.}
\label{tab:tir-unique}
\small
\begin{tabular}{l r l}
\toprule
Configuration & Unique problems & Problem IDs \\
\midrule""")
    for rs in all_runs:
        others = set()
        for rs2 in all_runs:
            if rs2.label != rs.label:
                others |= proved_sets[rs2.label]
        unique = proved_sets[rs.label] - others
        ids_str = ", ".join(sorted(unique)[:8])  # show up to 8
        if len(unique) > 8:
            ids_str += ", ..."
        ids_str = _latex_escape(ids_str) if ids_str else "---"
        parts.append(_row(_latex_escape(rs.label), [str(len(unique)), ids_str]))
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append(r"\end{table}")

    # ── Table: Effect of CoT stripping (strip vs no-strip) ──
    strip_runs = [rs for rs in all_runs if "strip" in rs.label.lower() and "no-strip" not in rs.label.lower()
                  and "informal" not in rs.label.lower() and "infr" not in rs.label.lower()]
    no_strip_runs = [rs for rs in all_runs if "no-strip" in rs.label.lower()]
    if strip_runs and no_strip_runs:
        sr, nsr = strip_runs[0], no_strip_runs[0]
        sr_pk, nsr_pk = compute_pass_at_k(sr), compute_pass_at_k(nsr)
        sr_total_tool = sr.total_lean_calls + sr.total_lean_final_calls + sr.total_python_calls
        nsr_total_tool = nsr.total_lean_calls + nsr.total_lean_final_calls + nsr.total_python_calls

        parts.append("")
        parts.append(r"""
\begin{table}[htbp]
\centering
\caption{Effect of chain-of-thought stripping on TIR prover performance.
Both runs use identical settings except for CoT handling.}
\label{tab:tir-cot-effect}
\small
\begin{tabular}{l r r r}
\toprule
Metric & Strip & No-strip & $\Delta$ \\
\midrule""")
        def _delta_row(name: str, v1: float, v2: float, fmt: str = ".0f", pct: bool = False) -> str:
            d = v2 - v1
            if pct:
                return f"{name} & {v1:{fmt}} & {v2:{fmt}} & {'+' if d>=0 else ''}{d:{fmt}} \\\\"
            else:
                sign = "+" if d >= 0 else ""
                return f"{name} & {v1:{fmt}} & {v2:{fmt}} & {sign}{d:{fmt}} \\\\"

        parts.append(_delta_row("Proved attempts", sr.n_proved, nsr.n_proved))
        parts.append(_delta_row("Problems proved", sr_pk['n_problems_proved'], nsr_pk['n_problems_proved']))
        parts.append(_delta_row(r"pass@1", sr_pk['p1_lo']*100, nsr_pk['p1_lo']*100, ".2f"))
        parts.append(_delta_row(r"pass@4", sr_pk['pk_lo']*100, nsr_pk['pk_lo']*100, ".2f"))
        parts.append(_delta_row("Mean turns/session",
            statistics.mean(sr.turns_per_session), statistics.mean(nsr.turns_per_session), ".2f"))
        parts.append(_delta_row("Tool calls/session",
            sr_total_tool/max(sr.n_proof_attempts,1), nsr_total_tool/max(nsr.n_proof_attempts,1), ".2f"))
        parts.append(_delta_row(r"Lean error rate (\%)",
            sr.total_lean_errors/max(sr.total_lean_calls+sr.total_lean_final_calls,1)*100,
            nsr.total_lean_errors/max(nsr.total_lean_calls+nsr.total_lean_final_calls,1)*100, ".2f"))
        parts.append(_delta_row("Mean session time (s)",
            statistics.mean(sr.elapsed_ms_list)/1000 if sr.elapsed_ms_list else 0,
            statistics.mean(nsr.elapsed_ms_list)/1000 if nsr.elapsed_ms_list else 0, ".0f"))
        parts.append(_delta_row("Sessions w/ thinking (\\%)",
            sr.n_with_thinking/max(sr.n_proof_attempts,1)*100,
            nsr.n_with_thinking/max(nsr.n_proof_attempts,1)*100, ".0f"))

        parts.append(r"\bottomrule")
        parts.append(r"\end{tabular}")
        parts.append(r"\end{table}")

    # ── Table: Effect of informal proof (strip vs strip+informal) ──
    informal_runs = [rs for rs in all_runs if ("informal" in rs.label.lower() or "infr" in rs.label.lower())]
    if strip_runs and informal_runs:
        sr = strip_runs[0]
        sr_pk = compute_pass_at_k(sr)
        sr_total_tool = sr.total_lean_calls + sr.total_lean_final_calls + sr.total_python_calls

        parts.append("")
        parts.append(r"""
\begin{table}[htbp]
\centering
\caption{Effect of informal proof injection on TIR prover performance (CoT stripping
enabled in all cases). Each informal run is compared against the baseline strip-only run.}
\label{tab:tir-informal-effect}
\small
\begin{tabular}{l r """ + " ".join(["r r"] * len(informal_runs)) + r"""}
\toprule""")
        sub_headers = ["Baseline"]
        for irs in informal_runs:
            sub_headers.extend([_latex_escape(irs.label), r"$\Delta$"])
        parts.append(" & ".join([""] + sub_headers) + r" \\")
        parts.append(r"\midrule")

        def _multi_delta_row(name: str, base_val: float, other_vals: list[float], fmt: str = ".0f") -> str:
            cells = [f"{base_val:{fmt}}"]
            for v in other_vals:
                d = v - base_val
                sign = "+" if d >= 0 else ""
                cells.extend([f"{v:{fmt}}", f"{sign}{d:{fmt}}"])
            return " & ".join([name] + cells) + r" \\"

        inf_pks = [compute_pass_at_k(irs) for irs in informal_runs]
        inf_total_tools = [irs.total_lean_calls + irs.total_lean_final_calls + irs.total_python_calls
                           for irs in informal_runs]

        parts.append(_multi_delta_row("Proved attempts", sr.n_proved,
            [irs.n_proved for irs in informal_runs]))
        parts.append(_multi_delta_row("Problems proved", sr_pk['n_problems_proved'],
            [ipk['n_problems_proved'] for ipk in inf_pks]))
        parts.append(_multi_delta_row(r"pass@1 (\%)", sr_pk['p1_lo']*100,
            [ipk['p1_lo']*100 for ipk in inf_pks], ".2f"))
        parts.append(_multi_delta_row(r"pass@4 (\%)", sr_pk['pk_lo']*100,
            [ipk['pk_lo']*100 for ipk in inf_pks], ".2f"))
        parts.append(_multi_delta_row("Mean turns/session",
            statistics.mean(sr.turns_per_session) if sr.turns_per_session else 0,
            [statistics.mean(irs.turns_per_session) if irs.turns_per_session else 0 for irs in informal_runs], ".2f"))
        parts.append(_multi_delta_row("Lean calls/session",
            (sr.total_lean_calls+sr.total_lean_final_calls)/max(sr.n_proof_attempts,1),
            [(irs.total_lean_calls+irs.total_lean_final_calls)/max(irs.n_proof_attempts,1) for irs in informal_runs], ".2f"))
        parts.append(_multi_delta_row(r"Lean error rate (\%)",
            sr.total_lean_errors/max(sr.total_lean_calls+sr.total_lean_final_calls,1)*100,
            [irs.total_lean_errors/max(irs.total_lean_calls+irs.total_lean_final_calls,1)*100 for irs in informal_runs], ".2f"))
        parts.append(_multi_delta_row("Mean session time (s)",
            statistics.mean(sr.elapsed_ms_list)/1000 if sr.elapsed_ms_list else 0,
            [statistics.mean(irs.elapsed_ms_list)/1000 if irs.elapsed_ms_list else 0 for irs in informal_runs], ".0f"))
        parts.append(_multi_delta_row("Informal proof used",
            0, [irs.n_informal_used for irs in informal_runs]))

        parts.append(r"\bottomrule")
        parts.append(r"\end{tabular}")
        parts.append(r"\end{table}")

    # ── Table: Reproducibility (informal 6xGPU vs 4xGPU) ──
    if len(informal_runs) >= 2:
        parts.append("")
        parts.append(r"""
\begin{table}[htbp]
\centering
\caption{Reproducibility check: two informal proof runs on different hardware
(6$\times$A6000 vs 4$\times$A5000). Both use identical hyperparameters; differences
arise from non-deterministic scheduling and hardware-dependent throughput.}
\label{tab:tir-reproducibility}
\small
\begin{tabular}{l r r}
\toprule""")
        ir0, ir1 = informal_runs[0], informal_runs[1]
        pk0, pk1 = compute_pass_at_k(ir0), compute_pass_at_k(ir1)
        parts.append(f" & {_latex_escape(ir0.label)} & {_latex_escape(ir1.label)} \\\\")
        parts.append(r"\midrule")
        parts.append(_row("Proved attempts", [str(ir0.n_proved), str(ir1.n_proved)]))
        parts.append(_row("Problems proved", [
            f"{pk0['n_problems_proved']}/{pk0['n_problems']}",
            f"{pk1['n_problems_proved']}/{pk1['n_problems']}",
        ]))
        parts.append(_row(r"pass@1", [f"{pk0['p1_lo']:.2%}".replace("%", "\\%"), f"{pk1['p1_lo']:.2%}".replace("%", "\\%")]))
        parts.append(_row(r"pass@4", [f"{pk0['pk_lo']:.2%}".replace("%", "\\%"), f"{pk1['pk_lo']:.2%}".replace("%", "\\%")]))
        inter = proved_sets[ir0.label] & proved_sets[ir1.label]
        only0 = proved_sets[ir0.label] - proved_sets[ir1.label]
        only1 = proved_sets[ir1.label] - proved_sets[ir0.label]
        parts.append(_row("Shared proved problems", [str(len(inter)), str(len(inter))]))
        parts.append(_row("Exclusive to this run", [str(len(only0)), str(len(only1))]))
        parts.append(_row(r"Mean session time (s)", [
            f"{statistics.mean(ir0.elapsed_ms_list)/1000:.0f}" if ir0.elapsed_ms_list else "---",
            f"{statistics.mean(ir1.elapsed_ms_list)/1000:.0f}" if ir1.elapsed_ms_list else "---",
        ]))
        parts.append(_row(r"deadline\_exceeded", [
            str(ir0.termination_reasons.get("deadline_exceeded", 0)),
            str(ir1.termination_reasons.get("deadline_exceeded", 0)),
        ]))
        parts.append(r"\bottomrule")
        parts.append(r"\end{tabular}")
        parts.append(r"\end{table}")

    return "\n".join(parts)


def _latex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detailed comparison of TIR prover runs.",
    )
    p.add_argument(
        "--run", nargs=3, action="append", metavar=("PATH", "LABEL", "SCHEMA"),
        help='Run: path to log dir, display label, schema version (v1|v2). Repeatable.',
    )
    p.add_argument(
        "--latex-output", type=str, default=None,
        help="Write LaTeX tables to this file.",
    )
    args = p.parse_args()

    if args.run is None:
        args.run = [
            [str(ROOT / "logs" / "full_20mins_tir_pass1_strip"), "Strip", "v1"],
            [str(ROOT / "logs" / "full_20mins_tir_pass1_no_strip"), "no-Strip", "v1"],
            # [str(ROOT / "logs" / "short_6x4090"), "Strip+Infr 6xGPU", "v2"],
            [str(ROOT / "logs" / "short_4x5000"), "Strip+Infr 4xGPU", "v2"],
        ]

    for run_arg in args.run:
        if run_arg[2] not in ("v1", "v2"):
            p.error(f"SCHEMA must be 'v1' or 'v2', got '{run_arg[2]}'")

    return args


def main():
    args = parse_args()

    all_runs = []
    for path, label, schema in args.run:
        print(f"Loading {label} ({path})...")
        rs = load_run(path, label, schema)
        all_runs.append(rs)
        print(f"  → {rs.total_records} records, {rs.n_proved} proved")

    # Text report
    report = render_text_report(all_runs)
    print(report)

    # LaTeX — raw data tables
    latex = render_latex_tables(all_runs)
    print("\n" + "=" * 80)
    print("  LaTeX TABLES (raw data)")
    print("=" * 80)
    print(latex)

    # LaTeX — analysis / comparison tables
    analysis = render_analysis_latex(all_runs)
    print("\n" + "=" * 80)
    print("  LaTeX ANALYSIS TABLES (for paper)")
    print("=" * 80)
    print(analysis)

    if args.latex_output:
        out = Path(args.latex_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(latex + "\n\n" + analysis)
        print(f"\nLaTeX tables written to: {out}")


if __name__ == "__main__":
    main()
