#!/usr/bin/env python3
"""
Parse solution logs and report accuracy + useful stats.

Inputs:
  - attempts.jsonl : one record per attempt (RunLogger.log_attempt)
  - solutions.csv  : one row per problem with checker_summary JSON (RunLogger.log_solution_row)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

def safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if not s:
            return float("nan")
        return float(s)
    except Exception:
        return float("nan")


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, int):
            return x
        s = str(x).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


def is_nonempty_answer(ans: Any) -> bool:
    if ans is None:
        return False
    s = str(ans).strip()
    return s != "" and s.lower() != "none"


def json_loads_maybe(s: Any) -> Any:
    if s is None:
        return None
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s)
    except Exception:
        return None


def quantiles(xs: List[float], qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> Dict[float, float]:
    xs2 = [x for x in xs if x is not None and not math.isnan(x) and math.isfinite(x)]
    if not xs2:
        return {q: float("nan") for q in qs}
    xs2.sort()
    out = {}
    n = len(xs2)
    for q in qs:
        # linear interpolation
        pos = (n - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            out[q] = xs2[lo]
        else:
            w = pos - lo
            out[q] = xs2[lo] * (1 - w) + xs2[hi] * w
    return out


# ----------------------------
# Loaders
# ----------------------------

def load_attempts_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # skip malformed line
                continue

            rows.append({
                "id": str(obj.get("problem_id", obj.get("id", ""))),
                "attempt": safe_int(obj.get("attempt")),
                "attempt_answer": obj.get("attempt_answer"),
                "entropy": safe_float(obj.get("entropy")),
                "response_length": safe_int(obj.get("response_length")),
                "python_calls": safe_int(obj.get("python_calls")) or 0,
                "python_errors": safe_int(obj.get("python_errors")) or 0,
                "termination_reason": str(obj.get("termination_reason", "")),
                "attempt_elapsed_ms": safe_int(obj.get("attempt_elapsed_ms")),
                "attempt_started_ts": obj.get("attempt_started_ts"),
                "attempt_finished_ts": obj.get("attempt_finished_ts"),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # basic cleanup
    df["id"] = df["id"].astype(str)
    df["has_answer"] = df["attempt_answer"].apply(is_nonempty_answer)
    return df


def load_solutions_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # ensure expected columns exist
    if "id" not in df.columns or "checker_summary" not in df.columns:
        raise ValueError("solutions.csv must have columns: id, checker_summary")
    df["id"] = df["id"].astype(str)
    df["checker_summary_obj"] = df["checker_summary"].apply(json_loads_maybe)
    return df


def extract_truth_map_from_solutions(df_solutions: pd.DataFrame) -> Dict[Tuple[str, int], Optional[bool]]:
    """
    Returns mapping: (problem_id, attempt_idx) -> is_correct (True/False/None)
    None means skipped / unavailable.
    """
    truth_map: Dict[Tuple[str, int], Optional[bool]] = {}
    for _, row in df_solutions.iterrows():
        pid = str(row["id"])
        cs = row.get("checker_summary_obj") or {}
        per_attempt = ((cs or {}).get("per_attempt_truth") or {})
        for k, v in per_attempt.items():
            try:
                aidx = int(k)
            except Exception:
                continue
            if not isinstance(v, dict):
                truth_map[(pid, aidx)] = None
                continue
            is_correct = v.get("is_correct", None)
            truth_map[(pid, aidx)] = (bool(is_correct) if is_correct is not None else None)
    return truth_map


# ----------------------------
# Metrics / reports
# ----------------------------

@dataclass
class Report:
    problem_accuracy_any_correct: float
    problem_accuracy_selected: float
    attempt_accuracy_all: float
    attempt_accuracy_among_answered: float
    answer_rate: float


def choose_selected_attempt(df_attempts_pid: pd.DataFrame) -> Optional[int]:
    """
    selected attempt = lowest-entropy attempt with non-empty Answer, else None.
    """
    sub = df_attempts_pid[df_attempts_pid["has_answer"]].copy()
    if sub.empty:
        return None
    # sub = sub.replace([math.inf, -math.inf], math.nan)
    # If entropy is NaN for some answered attempts, push them to the bottom
    sub["entropy_sort"] = sub["entropy"].fillna(float("inf"))
    best = sub.sort_values(["entropy_sort", "attempt"]).head(1)
    if best.empty:
        return None
    aidx = best.iloc[0]["attempt"]
    return int(aidx) if pd.notna(aidx) else None


def bucketize_entropy(entropy: float, bins: List[float]) -> str:
    if entropy is None or math.isnan(entropy) or not math.isfinite(entropy):
        return "NaN/Inf"
    for b in bins:
        if entropy <= b:
            return f"<= {b:g}"
    return f"> {bins[-1]:g}" if bins else "all"


def fix_has_answer(df):
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=str, default=".", help="Directory containing attempts.jsonl and solutions.csv")
    ap.add_argument("--attempts", type=str, default="attempts.jsonl")
    ap.add_argument("--solutions", type=str, default="solutions.csv")
    ap.add_argument("--entropy-bins", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--topk-termination", type=int, default=15, help="How many termination reasons to print")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    attempts_path = log_dir / args.attempts
    solutions_path = log_dir / args.solutions

    if not attempts_path.exists():
        raise FileNotFoundError(f"Missing attempts file: {attempts_path}")
    if not solutions_path.exists():
        raise FileNotFoundError(f"Missing solutions file: {solutions_path}")

    df_attempts = load_attempts_jsonl(attempts_path)
    df_solutions = load_solutions_csv(solutions_path)

    if df_attempts.empty:
        print("No attempt records found.")
        return

    df_attempts = fix_has_answer(df_attempts)

    truth_map = extract_truth_map_from_solutions(df_solutions)

    # join truth into attempts
    def lookup_truth(row) -> Optional[bool]:
        pid = str(row["id"])
        aidx = row["attempt"]
        if aidx is None or pd.isna(aidx):
            return None
        return truth_map.get((pid, int(aidx)), None)

    df_attempts["is_correct"] = df_attempts.apply(lookup_truth, axis=1)

    # Attempt-level metrics
    n_attempts = len(df_attempts)
    n_answered = int(df_attempts["has_answer"].sum())
    n_with_truth = int(df_attempts["is_correct"].notna().sum())
    n_correct = int((df_attempts["is_correct"] == True).sum())

    answer_rate = n_answered / n_attempts if n_attempts else float("nan")
    attempt_acc_all = n_correct / n_attempts if n_attempts else float("nan")
    attempt_acc_answered = n_correct / n_answered if n_answered else float("nan")

    # Problem-level metrics
    problem_ids = sorted(df_solutions["id"].unique().tolist())
    # Use solutions.csv as canonical set of problems
    any_correct = 0
    maj_correct = 0
    selected_correct = 0
    selected_has_answer = 0

    for pid in problem_ids:
        dfp = df_attempts[df_attempts["id"] == pid]
        # any correct among attempts (truth known)
        any_ok = bool((dfp["is_correct"] == True).any())
        if any_ok:
            any_correct += 1

        maj_count = len(dfp) // 2 + len(dfp) % 2 + 1
        maj_ok = bool((dfp["is_correct"]).sum() >= maj_count)
        if maj_ok:
            maj_correct += 1

        # selected attempt by entropy among answered
        sel = choose_selected_attempt(dfp)
        if sel is not None:
            selected_has_answer += 1
            sel_ok = truth_map.get((pid, sel), None)
            if sel_ok is True:
                selected_correct += 1

    n_problems = len(problem_ids)
    prob_acc_any = any_correct / n_problems if n_problems else float("nan")
    prob_acc_maj = maj_correct / n_problems if n_problems else float("nan")
    prob_acc_selected = selected_correct / n_problems if n_problems else float("nan")

    # Distributions
    ent_all = df_attempts["entropy"].tolist()
    ent_ans = df_attempts[df_attempts["has_answer"]]["entropy"].tolist()
    ms_all = [safe_int(x) for x in df_attempts["attempt_elapsed_ms"].tolist()]
    ms_all = [x for x in ms_all if x is not None]

    term_counts = Counter(df_attempts["termination_reason"].fillna("").astype(str).tolist())
    term_counts_has_answer = Counter(df_attempts["termination_reason"][df_attempts["has_answer"]].fillna("").astype(str).tolist())
    print(term_counts_has_answer)
    print(df_attempts[df_attempts["has_answer"] & (df_attempts["termination_reason"] != "boxed_detected_in_stream")]["attempt_answer"].isna().iloc[0])
    print(df_attempts[~df_attempts["has_answer"]]["attempt_answer"].iloc[0])
    print((~df_attempts["has_answer"]).sum())
    print((df_attempts["termination_reason"] != "boxed_detected_in_stream").sum())
    raise SystemExit(0)
    py_calls = int(df_attempts["python_calls"].sum())
    py_errs = int(df_attempts["python_errors"].sum())
    py_err_rate = (py_errs / py_calls) if py_calls else float("nan")

    # Correctness vs entropy bucket (attempt-level)
    bins = list(args.entropy_bins)
    bucket_stats = defaultdict(lambda: {"n": 0, "n_truth": 0, "n_correct": 0, "n_answered": 0})
    for _, r in df_attempts.iterrows():
        b = bucketize_entropy(safe_float(r["entropy"]), bins)
        bucket_stats[b]["n"] += 1
        if bool(r["has_answer"]):
            bucket_stats[b]["n_answered"] += 1
        if r["is_correct"] is not None:
            bucket_stats[b]["n_truth"] += 1
            if r["is_correct"] is True:
                bucket_stats[b]["n_correct"] += 1

    # ----------------------------
    # Print report
    # ----------------------------
    print("\n==================== Evaluation Report ====================")
    print(f"Attempts file : {attempts_path}")
    print(f"Solutions file: {solutions_path}")
    print("------------------------------------------------------------------")
    print(f"Problems: {n_problems}")
    print(f"Attempts: {n_attempts}  (avg {n_attempts / n_problems:.2f} per problem)")
    print("------------------------------------------------------------------")
    print("Attempt-level stats")
    print(f"  Accuracy (all attempts):           {attempt_acc_all:.2%}  ({n_correct}/{n_attempts}) (if we treat failed equivalence checks and no-answer attempts as equivalent=False)")
    print(f"  Answer rate:                       {answer_rate:.2%}  ({n_answered}/{n_attempts})")
    print(f"  w/ equiv. result (among answered): {n_with_truth/n_answered:.2%}  ({n_with_truth}/{n_answered}) (here, answer equivalence checker produced an output)")
    print(f"  Accuracy (among w/ equiv. result): {n_correct/n_with_truth:.2%}  ({n_correct}/{n_with_truth})")
    print(f"  Accuracy:                          {n_correct/n_answered:.2%}  ({n_correct}/{n_answered}) (if we treat failed equivalence checks as equivalent=False)")
    print("------------------------------------------------------------------")
    print("Problem-level accuracy")
    print(f"  Any-attempt correct (pass@{int(n_attempts / n_problems)}):           {prob_acc_any:.2%}  ({any_correct}/{n_problems})")
    print(f"  Majority of attempts ({maj_count}/{int(n_attempts / n_problems)}) correct:     {prob_acc_maj:.2%}  ({maj_correct}/{n_problems})")
    print(f"  Selected-attempt correct (lowest ent.): {prob_acc_selected:.2%}  ({selected_correct}/{n_problems})")
    print(f"  Selected attempt had an answer:         {selected_has_answer/n_problems:.2%}  ({selected_has_answer}/{n_problems})")
    print("------------------------------------------------------------------")

    q_all = quantiles([safe_float(x) for x in ent_all])
    q_ans = quantiles([safe_float(x) for x in ent_ans])
    print("Entropy")
    print(f"  All attempts quantiles:      " +
          ", ".join([f"p{int(k*100):02d}={v:.3f}" for k, v in q_all.items()]))
    print(f"  Answered-only quantiles:     " +
          ", ".join([f"p{int(k*100):02d}={v:.3f}" for k, v in q_ans.items()]))

    q_ms = quantiles([float(x) for x in ms_all]) if ms_all else {0.5: float("nan")}
    if ms_all:
        print("Latency (attempt_elapsed_ms)")
        print("  Quantiles:                   " +
              ", ".join([f"p{int(k*100):02d}={v:.0f}ms" for k, v in q_ms.items()]))

    print("------------------------------------------------------------------")
    print("Python tool usage")
    print(f"  python_calls: {py_calls}")
    print(f"  python_errors: {py_errs}")
    print(f"  python_error_rate (errs/calls): {py_err_rate:.4f}" if py_calls else "  python_error_rate: n/a (no calls)")
    print("------------------------------------------------------------------")
    print("Termination reasons (top)")
    for reason, cnt in term_counts.most_common(args.topk_termination):
        print(f"  {cnt:7d}  {reason}")

    print("------------------------------------------------------------------")
    print("Correctness vs entropy bucket (attempt-level; uses truth when available)")
    # stable ordering: NaN/Inf, then bins, then tail
    def bucket_key(b: str) -> Tuple[int, float]:
        if b == "NaN/Inf":
            return (0, -1.0)
        if b.startswith("<= "):
            try:
                return (1, float(b.replace("<= ", "")))
            except Exception:
                return (1, 1e9)
        if b.startswith("> "):
            try:
                return (2, float(b.replace("> ", "")))
            except Exception:
                return (2, 1e9)
        return (3, 1e9)

    for b in sorted(bucket_stats.keys(), key=bucket_key):
        st = bucket_stats[b]
        n = st["n"]
        n_truth = st["n_truth"]
        n_corr = st["n_correct"]
        acc = (n_corr / n_truth) if n_truth else float("nan")
        ans_rate_b = st["n_answered"] / n if n else float("nan")
        print(f"  {b:8s}  n={n:6d}  answered={ans_rate_b:5.2%}  truth={n_truth:6d}  acc={acc:6.2%}" if n_truth else
              f"  {b:8s}  n={n:6d}  answered={ans_rate_b:5.2%}  truth={n_truth:6d}  acc=  n/a")

    print("==================================================================\n")


if __name__ == "__main__":
    main()