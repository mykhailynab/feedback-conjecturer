"""Renders summary statistics for a prove_formalizations run."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from analysis_and_inspection.display_utils import (
    WIDTH,
    bold, dim, green, red, yellow,
    hline, fmt_ms,
)
from analysis_and_inspection.goedel_data import ProverSession


def render_stats(sessions: List[ProverSession], results: Dict[Tuple, Dict]) -> str:
    lines: List[str] = []

    complete   = [s for s in sessions if s.complete]
    incomplete = [s for s in sessions if not s.complete]

    proved   = [s for s in complete if s.theorem_proved and s.checking == "proof"]
    disproved = [s for s in complete if s.theorem_proved and s.checking == "disproof"]
    failed   = [s for s in complete if not s.theorem_proved]

    _pct = lambda n, d: f"{100*n//max(d,1)}%" if d else "n/a"

    lines.append("")
    lines.append("═" * WIDTH)
    lines.append(bold("  SUMMARY STATISTICS  ") + dim("(prove_formalizations)"))
    lines.append("═" * WIDTH)

    # ---- Records overview (from results file) ----
    n_total    = len(results)
    n_success  = sum(1 for r in results.values() if r.get("status") == "success")
    n_skipped  = sum(1 for r in results.values() if r.get("skipped"))
    n_status_failed = sum(1 for r in results.values() if r.get("status") == "failed")
    n_proved   = sum(1 for r in results.values() if r.get("proved"))
    n_disproved = sum(1 for r in results.values() if r.get("disproved"))
    n_incomplete = sum(
        1 for r in results.values()
        if r.get("incomplete") and not r.get("proved") and not r.get("disproved")
    )
    n_inconclusive = n_success - n_proved - n_disproved - n_incomplete

    lines.append(f"  {'Total records in results file:':<44} {n_total}")
    lines.append(f"  {'  status=success (ran the prover):':<44} {n_success}")
    lines.append(f"  {'  skipped (missing fields):':<44} {dim(str(n_skipped))}")
    lines.append(f"  {'  status=failed (formalization error):':<44} {dim(str(n_status_failed))}")
    lines.append("")
    lines.append(f"  {'Of success records:':<44}")
    lines.append(f"  {'  proved:':<44} {green(str(n_proved))}  {dim(_pct(n_proved, n_success))}")
    lines.append(f"  {'  disproved:':<44} {red(str(n_disproved))}  {dim(_pct(n_disproved, n_success))}")
    lines.append(f"  {'  incomplete (token-limited, resumable):':<44} "
                 f"{yellow(str(n_incomplete))}  {dim(_pct(n_incomplete, n_success))}")
    lines.append(f"  {'  inconclusive (rounds exhausted/etc):':<44} "
                 f"{dim(str(n_inconclusive))}  {dim(_pct(n_inconclusive, n_success))}")

    # ---- Prover sessions (from events file) ----
    lines.append("")
    lines.append(bold("  Prover sessions  ") + dim("(from events log)"))
    lines.append(hline("─"))
    lines.append(f"  {'Total sessions tracked:':<44} {len(sessions)}")
    lines.append(f"  {'  complete (session_done recorded):':<44} {len(complete)}")
    lines.append(f"  {'  incomplete (no session_done):':<44} {dim(str(len(incomplete)))}")
    if complete:
        lines.append(f"  {'  proved:':<44} {green(str(len(proved)))}  {dim(_pct(len(proved), len(complete)))}")
        lines.append(f"  {'  disproved:':<44} {red(str(len(disproved)))}  {dim(_pct(len(disproved), len(complete)))}")
        lines.append(f"  {'  failed (inconclusive):':<44} "
                     f"{yellow(str(len(failed)))}  {dim(_pct(len(failed), len(complete)))}")

    # ---- Termination reason breakdown ----
    lines.append("")
    lines.append(bold("  Termination reasons (complete sessions)"))
    lines.append(hline("─"))
    reason_counts: Dict[str, int] = defaultdict(int)
    for s in complete:
        reason_counts[s.termination_reason or "unknown"] += 1
    if reason_counts:
        max_count = max(reason_counts.values())
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            bar = "█" * int(count / max_count * 40)
            lines.append(f"  {reason:<35} {count:>4}  {dim(bar)}")

    # ---- Rounds used ----
    def _round_dist(title: str, sess_list: List[ProverSession]) -> None:
        if not sess_list:
            return
        lines.append("")
        lines.append(bold(f"  {title}"))
        lines.append(hline("─"))
        round_counts: Dict[int, int] = defaultdict(int)
        for s in sess_list:
            round_counts[s.rounds_used or 0] += 1
        rc_max = max(round_counts.values()) if round_counts else 1
        for n_rounds in sorted(round_counts):
            label = f"{n_rounds} round{'s' if n_rounds != 1 else ''}"
            count = round_counts[n_rounds]
            bar = "█" * int(count / rc_max * 40)
            lines.append(f"  {label:<35} {count:>4}  {dim(bar)}")

    _round_dist("Rounds used (complete proof sessions)", [s for s in complete if s.checking == "proof"])
    _round_dist("Rounds used (complete disproof sessions)", [s for s in complete if s.checking == "disproof"])

    # ---- Token budget (incomplete sessions) ----
    token_limited = [
        r for r in results.values()
        if r.get("incomplete") and not r.get("proved") and not r.get("disproved")
    ]
    if token_limited:
        ctx_tokens = [
            (r.get("proof_result") or {}).get("total_context_tokens", 0)
            for r in token_limited
            if (r.get("proof_result") or {}).get("total_context_tokens")
        ]
        if ctx_tokens:
            lines.append("")
            lines.append(bold("  Token budget (incomplete / token-limited sessions)"))
            lines.append(hline("─"))
            lines.append(f"  {'Incomplete sessions:':<44} {len(token_limited)}")
            lines.append(f"  {'  with context token info:':<44} {len(ctx_tokens)}")
            lines.append(f"  {'  max total_context_tokens:':<44} {max(ctx_tokens)}")
            lines.append(f"  {'  median total_context_tokens:':<44} {sorted(ctx_tokens)[len(ctx_tokens)//2]}")
            token_limits = [s.token_limit for s in sessions if s.token_limit > 0]
            if token_limits:
                lines.append(f"  {'  token_limit used (from events):':<44} {token_limits[0]}"
                              + (dim("  (all sessions)") if len(set(token_limits)) == 1 else ""))

    # ---- Lean compilation results ----
    all_rounds = [r for s in complete for r in s.rounds]
    if all_rounds:
        n_ok      = sum(1 for r in all_rounds if r.lean_ok is True)
        n_fail    = sum(1 for r in all_rounds if r.lean_ok is False and not r.lean_timed_out and not r.lean_oom)
        n_timeout = sum(1 for r in all_rounds if r.lean_timed_out)
        n_oom     = sum(1 for r in all_rounds if r.lean_oom)
        n_nocode  = sum(1 for r in all_rounds if r.termination_reason == "no_code_block")
        n_ctx     = sum(1 for r in all_rounds if r.termination_reason == "context_exceeded")
        n_toklim  = sum(1 for r in all_rounds if r.termination_reason == "token_limit")
        lines.append("")
        lines.append(bold("  Lean compilation results (all rounds, complete sessions)"))
        lines.append(hline("─"))
        lines.append(f"  {'Total rounds:':<44} {len(all_rounds)}")
        lines.append(f"  {'  compiled OK:':<44} {green(str(n_ok))}")
        lines.append(f"  {'  compiled with errors:':<44} {red(str(n_fail))}")
        lines.append(f"  {'  timed out:':<44} {yellow(str(n_timeout))}")
        lines.append(f"  {'  out of memory:':<44} {yellow(str(n_oom))}")
        lines.append(f"  {'  no code block in output:':<44} {dim(str(n_nocode))}")
        lines.append(f"  {'  context window exceeded:':<44} {dim(str(n_ctx))}")
        lines.append(f"  {'  stopped by token limit:':<44} {dim(str(n_toklim))}")

    # ---- Session timing ----
    times = [s.elapsed_ms for s in complete if s.elapsed_ms is not None]
    if times:
        lines.append("")
        lines.append(bold("  Session timing (complete sessions)"))
        lines.append(hline("─"))
        lines.append(f"  {'Average session time:':<44} {fmt_ms(int(sum(times)/len(times)))}")
        lines.append(f"  {'Median session time:':<44} {fmt_ms(sorted(times)[len(times)//2])}")
        lines.append(f"  {'Fastest session:':<44} {fmt_ms(min(times))}")
        lines.append(f"  {'Slowest session:':<44} {fmt_ms(max(times))}")

    lines.append("")
    lines.append("═" * WIDTH)
    return "\n".join(lines)
