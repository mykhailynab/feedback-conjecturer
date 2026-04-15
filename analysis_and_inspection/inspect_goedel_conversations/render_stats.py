"""Renders summary statistics for a collection of ProverSessions."""
from __future__ import annotations

import numpy as np
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

    _pct = lambda n, d: f"{100*n//max(d,1)}%" if d else "n/a"

    lines.append(bold("  Proof attempts  ") + dim("(theorem: proposed = gt)"))
    lines.append(hline("─"))
    lines.append(f"  {'Total proof sessions:':<40} {len(all_proofs)}")
    lines.append(f"  {'  proved (equivalent=True):':<40} "
                 f"{green(str(len(proved)))}  {dim(_pct(len(proved), len(all_proofs)))}")
    lines.append(f"  {'  failed (inconclusive):':<40} "
                 f"{yellow(str(len(failed_p)))}  {dim(_pct(len(failed_p), len(all_proofs)))}")
    if incomplete_proofs:
        lines.append(f"  {'  incomplete (killed mid-run):':<40} "
                     f"{dim(str(len(incomplete_proofs)))}  "
                     f"{dim(_pct(len(incomplete_proofs), len(all_proofs)))}")

    lines.append("")
    lines.append(bold("  Disproof attempts  ") + dim("(theorem: proposed ≠ gt)"))
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
        sess_for = [s for s in sessions if s.problem_id == pid and s.attempt == att and s.checking == checking]
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
        otherwise_eq  = results[(pid, att)].get("equivalent") is True
        otherwise_neq = results[(pid, att)].get("equivalent") is False
        if otherwise_eq or otherwise_neq:
            n_resolved_otherwise += 1
            continue
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

    # Failure reason breakdown (complete proof sessions)
    lines.append("")
    lines.append(bold("  Failure reasons (complete proof sessions)"))
    lines.append(hline("─"))
    reason_counts: Dict[str, int] = defaultdict(int)
    for s in proofs:
        reason_counts[s.termination_reason or "unknown"] += 1
    max_count = max(reason_counts.values()) if reason_counts else 1
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(count / max_count * 40)
        lines.append(f"  {reason:<35} {count:>4}  {dim(bar)}")

    all_goedel_results = [
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
    if goedel_killed_reason_counts:
        max_count = max(goedel_killed_reason_counts.values())
        for reason, count in sorted(goedel_killed_reason_counts.items(), key=lambda x: -x[1]):
            bar = "█" * int(count / max_count * 40)
            lines.append(f"  {'  ' + reason:<40} {count:>4}  {dim(bar)}")

    def _round_dist_section(title: str, sess_list: List[ProverSession]) -> None:
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

    _round_dist_section("Rounds used (complete proof sessions)",      proofs)
    _round_dist_section("Rounds used (complete disproof sessions)",   disproofs)
    _round_dist_section("Rounds used (incomplete proof sessions)",    incomplete_proofs)
    _round_dist_section("Rounds used (incomplete disproof sessions)", incomplete_disproofs)

    # Timing
    times = [s.elapsed_ms for s in complete if s.elapsed_ms is not None]
    if times:
        avg_ms    = sum(times) / len(times)
        median_ms = np.median(times)
        max_ms    = max(times)
        min_ms    = min(times)
        lines.append("")
        lines.append(bold("  Session timing (complete sessions)"))
        lines.append(hline("─"))
        lines.append(f"  {'Average session time:':<40} {fmt_ms(int(avg_ms))}")
        lines.append(f"  {'Median session time:':<40} {fmt_ms(int(median_ms))}")
        lines.append(f"  {'Fastest session:':<40} {fmt_ms(min_ms)}")
        lines.append(f"  {'Slowest session:':<40} {fmt_ms(max_ms)}")

    # Lean compilation outcomes
    all_rounds = [r for s in complete for r in s.rounds]
    if all_rounds:
        n_ok      = sum(1 for r in all_rounds if r.lean_ok is True)
        n_fail    = sum(1 for r in all_rounds if r.lean_ok is False and not r.lean_timed_out and not r.lean_oom)
        n_timeout = sum(1 for r in all_rounds if r.lean_timed_out)
        n_oom     = sum(1 for r in all_rounds if r.lean_oom)
        n_nocode  = sum(1 for r in all_rounds if r.termination_reason == "no_code_block")
        n_ctx     = sum(1 for r in all_rounds if r.termination_reason == "context_exceeded")
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
