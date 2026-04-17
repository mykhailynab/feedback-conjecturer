"""Renders summary statistics for a prove_formalizations run."""
from __future__ import annotations

import math
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

    # ---- Proven accuracy (full pipeline: conjecturing + formalization + proving) ----
    # Universe is ALL records including skipped (formalization or conjecture failed).
    # proved=True → answer is confirmed correct; disproved=True → confirmed wrong.
    by_problem: Dict[str, List] = defaultdict(list)
    for r in results.values():
        pid = r.get("problem_id")
        if pid is not None:
            by_problem[pid].append(r)

    n_problems = len(by_problem)
    n_total_attempts = len(results)
    n_proved_total   = sum(1 for r in results.values() if r.get("proved") is True)
    n_disproved_total = sum(1 for r in results.values() if r.get("disproved") is True)

    p1_lo = n_proved_total   / n_total_attempts if n_total_attempts else 0.0
    p1_hi = 1.0 - n_disproved_total / n_total_attempts if n_total_attempts else 0.0

    attempts_per_problem = [len(v) for v in by_problem.values()]
    k_inferred = max(set(attempts_per_problem), key=attempts_per_problem.count) if attempts_per_problem else 1

    def _pass_at_k(n: int, c: int, k: int) -> float:
        """Exact pass@k: P(at least 1 correct in k draws without replacement)."""
        if c <= 0:
            return 0.0
        if n - c < k:
            return 1.0
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)

    pk_lo_values: List[float] = []
    pk_hi_values: List[float] = []
    for entries in by_problem.values():
        n = len(entries)
        c_lo = sum(1 for r in entries if r.get("proved") is True)
        c_hi = n - sum(1 for r in entries if r.get("disproved") is True)
        pk_lo_values.append(_pass_at_k(n, c_lo, k_inferred))
        pk_hi_values.append(_pass_at_k(n, c_hi, k_inferred))

    pk_lo = sum(pk_lo_values) / n_problems if n_problems else 0.0
    pk_hi = sum(pk_hi_values) / n_problems if n_problems else 0.0

    lines.append("")
    lines.append("═" * WIDTH)
    lines.append(bold("  PROVEN ACCURACY  ") + dim("(full pipeline: conjecturing + formalization + proving)"))
    lines.append("═" * WIDTH)
    lines.append(dim(
        "  Bounds derived from prove_results.jsonl: lower = proved / total,"
        " upper = 1 − disproved / total."
    ))
    lines.append(dim("  Universe includes all records: proved, inconclusive, skipped."))
    lines.append(dim(f"  {n_total_attempts} attempts across {n_problems} problems."))
    lines.append(dim(f"  Inferred k = {k_inferred} (most common attempts-per-problem)."))
    lines.append("")
    n_p1_hi = n_total_attempts - n_disproved_total
    lines.append(f"  {'pass@1  lower bound (proved):':<44} {green(f'{p1_lo:.2%}')}  {dim(f'({n_proved_total}/{n_total_attempts} attempts)')}")
    lines.append(f"  {'pass@1  upper bound (1 − disproved):':<44} {yellow(f'{p1_hi:.2%}')}  {dim(f'({n_p1_hi}/{n_total_attempts} attempts)')}")
    lines.append("")
    pk_lo_n = pk_lo * n_problems
    pk_hi_n = pk_hi * n_problems
    lines.append(f"  {'pass@' + str(k_inferred) + '  lower bound:':<44} {green(f'{pk_lo:.2%}')}  {dim(f'(~{pk_lo_n:.1f}/{n_problems} problems)')}")
    lines.append(f"  {'pass@' + str(k_inferred) + '  upper bound:':<44} {yellow(f'{pk_hi:.2%}')}  {dim(f'(~{pk_hi_n:.1f}/{n_problems} problems)')}")

    lines.append("")
    lines.append("═" * WIDTH)
    return "\n".join(lines)
