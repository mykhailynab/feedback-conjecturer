import os
import json
import math
import re
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import polars as pl

LOG_DIR = Path("aimo3_logs")
REF_PATH = Path("reference.csv")

ATTEMPTS_PATH = LOG_DIR / "attempts.jsonl"
SOLUTIONS_PATH = LOG_DIR / "solutions.csv"
EVENTS_PATH = LOG_DIR / "events.jsonl"

OUT_DIR = LOG_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # salvage: skip malformed line
                continue
    return rows

def _safe_json_loads(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return {}
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except Exception:
        return {}

def _coerce_int(x):
    if x is None:
        return None
    try:
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return None
        return int(x)
    except Exception:
        return None

def _coerce_float(x):
    if x is None:
        return None
    try:
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return None
        return float(x)
    except Exception:
        return None

def print_section(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def main():

    # ---------- load ----------
    print_section("Loading reference.csv")
    ref = pl.read_csv(str(REF_PATH))
    ref_cols = set(ref.columns)
    id_col = "id"
    q_col = "problem" if "problem" in ref_cols else ("question" if "question" in ref_cols else None)
    a_col = "answer" if "answer" in ref_cols else None

    if q_col is None or a_col is None:
        raise ValueError(f"reference.csv must have answer and (problem/question). Columns: {ref.columns}")

    ref_pd = ref.select([id_col, q_col, a_col]).to_pandas()
    ref_pd.rename(columns={q_col: "question", a_col: "true_answer"}, inplace=True)
    ref_pd["true_answer"] = ref_pd["true_answer"].apply(_coerce_int)

    print(f"Reference rows: {len(ref_pd):,}")
    print(f"Unique ids:     {ref_pd['id'].nunique():,}")

    print_section("Loading logs")
    attempt_rows = _read_jsonl(ATTEMPTS_PATH)
    events_rows = _read_jsonl(EVENTS_PATH)

    solutions_pd = None
    if SOLUTIONS_PATH.exists():
        solutions_pd = pd.read_csv(SOLUTIONS_PATH)
    else:
        solutions_pd = pd.DataFrame()

    attempts_pd = pd.DataFrame(attempt_rows)
    events_pd = pd.DataFrame(events_rows)

    print(f"attempts.jsonl rows: {len(attempts_pd):,}")
    print(f"solutions.csv rows:  {len(solutions_pd):,}")
    print(f"events.jsonl rows:   {len(events_pd):,}")

    # ---------- normalize attempts ----------
    if len(attempts_pd) > 0:
        # normalize common columns
        for col in ["id", "attempt", "status", "reject_reason", "pred_final_answer", "attempt_answer",
                    "entropy", "response_length", "python_calls", "python_errors", "finish_reason", "ts", "summary"]:
            if col not in attempts_pd.columns:
                attempts_pd[col] = None

        attempts_pd["attempt"] = attempts_pd["attempt"].apply(_coerce_int)
        attempts_pd["pred_final_answer"] = attempts_pd["pred_final_answer"].apply(_coerce_int)
        attempts_pd["attempt_answer"] = attempts_pd["attempt_answer"].apply(_coerce_int)
        attempts_pd["entropy"] = attempts_pd["entropy"].apply(_coerce_float)
        attempts_pd["response_length"] = attempts_pd["response_length"].apply(_coerce_int)
        attempts_pd["python_calls"] = attempts_pd["python_calls"].apply(_coerce_int)
        attempts_pd["python_errors"] = attempts_pd["python_errors"].apply(_coerce_int)

    # ---------- normalize solutions ----------
    if len(solutions_pd) > 0:
        # your newer logger writes many columns; handle both old/new
        if "true_answer" not in solutions_pd.columns and "true" in solutions_pd.columns:
            solutions_pd.rename(columns={"true": "true_answer"}, inplace=True)

        for col in ["id", "pred_answer", "true_answer", "is_correct",
                    "selected_attempt", "selected_entropy",
                    "budget_seconds", "deadline_ts", "solve_elapsed_ms",
                    "finish_reason_counts", "vote_summary"]:
            if col not in solutions_pd.columns:
                solutions_pd[col] = None

        solutions_pd["pred_answer"] = solutions_pd["pred_answer"].apply(_coerce_int)
        solutions_pd["true_answer"] = solutions_pd["true_answer"].apply(_coerce_int)
        solutions_pd["selected_attempt"] = solutions_pd["selected_attempt"].apply(_coerce_int)
        solutions_pd["selected_entropy"] = solutions_pd["selected_entropy"].apply(_coerce_float)
        solutions_pd["budget_seconds"] = solutions_pd["budget_seconds"].apply(_coerce_float)
        solutions_pd["solve_elapsed_ms"] = solutions_pd["solve_elapsed_ms"].apply(_coerce_int)

        # recompute correctness reliably
        solutions_pd["is_correct_recomputed"] = (
            (solutions_pd["pred_answer"].notna()) &
            (solutions_pd["true_answer"].notna()) &
            (solutions_pd["pred_answer"] == solutions_pd["true_answer"])
        )

    # ---------- join solutions with reference ----------
    print_section("Coverage and accuracy (solutions vs reference)")
    if len(solutions_pd) > 0:
        sol_join = ref_pd.merge(solutions_pd, on="id", how="left", suffixes=("_ref", "_sol"))
        covered = sol_join["pred_answer"].notna().sum()
        total = len(sol_join)
        acc = (sol_join["pred_answer"] == sol_join["true_answer_sol"]).mean()

        print(f"Solved/covered ids: {covered:,} / {total:,} ({covered/total:.1%})")
        print(f"Exact-match accuracy: {acc:.2%}")

        # save per-id report
        per_id_cols = ["id", "true_answer_sol", "pred_answer", "is_correct_recomputed",
                    "selected_attempt", "selected_entropy", "budget_seconds", "solve_elapsed_ms"]
        per_id = sol_join[per_id_cols].copy()
        per_id.to_csv(OUT_DIR / "per_id_summary.csv", index=False, encoding="utf-8")
        print(f"Saved: {OUT_DIR/'per_id_summary.csv'}")

        # worst cases: missing / wrong
        missing = sol_join[sol_join["pred_answer"].isna()][["id", "true_answer_sol"]].copy()
        wrong = sol_join[(sol_join["pred_answer"].notna()) & (sol_join["pred_answer"] != sol_join["true_answer_sol"])][
            ["id", "true_answer_sol", "pred_answer"]
        ].copy()

        missing.to_csv(OUT_DIR / "missing_predictions.csv", index=False, encoding="utf-8")
        wrong.to_csv(OUT_DIR / "wrong_predictions.csv", index=False, encoding="utf-8")
        print(f"Missing predictions: {len(missing):,} (saved missing_predictions.csv)")
        print(f"Wrong predictions:   {len(wrong):,} (saved wrong_predictions.csv)")
    else:
        print("solutions.csv not found or empty; skipping accuracy summary.")

    # ---------- attempt-level stats ----------
    print_section("Attempt-level statistics")
    if len(attempts_pd) == 0:
        print("attempts.jsonl not found or empty.")
    else:
        # basic volumes
        n_attempts = len(attempts_pd)
        n_ids = attempts_pd["id"].nunique()
        print(f"Total attempts: {n_attempts:,}")
        print(f"Unique ids:     {n_ids:,}")
        print(f"Attempts per id (mean): {n_attempts/max(n_ids,1):.2f}")

        # status distribution
        print("\nStatus distribution:")
        print(attempts_pd["status"].fillna("NA").value_counts(dropna=False).to_string())

        # finish reasons
        print("\nFinish reason distribution (top 30):")
        fr_counts = attempts_pd["finish_reason"].fillna("NA").value_counts().head(30)
        print(fr_counts.to_string())

        # errors / python usage
        print("\nPython tool usage:")
        print(f"Total python_calls:  {attempts_pd['python_calls'].fillna(0).sum():,}")
        print(f"Total python_errors: {attempts_pd['python_errors'].fillna(0).sum():,}")
        py_used = (attempts_pd["python_calls"].fillna(0) > 0).mean()
        print(f"Share of attempts using python: {py_used:.2%}")

        # entropy / length stats (for attempts with answers vs without)
        with_ans = attempts_pd[attempts_pd["attempt_answer"].notna()].copy()
        no_ans = attempts_pd[attempts_pd["attempt_answer"].isna()].copy()

        def describe_num(df, col):
            s = df[col].dropna()
            if len(s) == 0:
                return None
            return {
                "count": int(s.count()),
                "mean": float(s.mean()),
                "p50": float(s.median()),
                "p90": float(s.quantile(0.90)),
                "p99": float(s.quantile(0.99)),
                "min": float(s.min()),
                "max": float(s.max()),
            }

        ent_with = describe_num(with_ans, "entropy")
        ent_no = describe_num(no_ans, "entropy")
        len_with = describe_num(with_ans, "response_length")
        len_no = describe_num(no_ans, "response_length")

        print("\n\nEntropy stats (attempts WITH parsed answer):", json.dumps(ent_with, indent=4))
        print("Entropy stats (attempts WITHOUT parsed answer):", json.dumps(ent_no, indent=4))
        print("\n\nResponse length stats (attempts WITH parsed answer):", json.dumps(len_with, indent=4))
        print("Response length stats (attempts WITHOUT parsed answer):", json.dumps(len_no, indent=4))

        # answer diversity per id
        print("\nAnswer diversity per id:")
        ans_div = attempts_pd.groupby("id")["attempt_answer"].apply(lambda s: len(set([x for x in s.dropna().tolist()])))
        print(ans_div.describe().to_string())

        # save attempt pivot
        attempts_pd.to_csv(OUT_DIR / "attempts_flat.csv", index=False, encoding="utf-8")
        print(f"\nSaved: {OUT_DIR/'attempts_flat.csv'}")

    # ---------- per-id attempt aggregation ----------
    print_section("Per-id attempt aggregation")
    if len(attempts_pd) > 0:
        grp = attempts_pd.groupby("id", as_index=False)

        per_id_attempts = grp.agg(
            attempts_total=("attempt", "count"),
            attempts_with_answer=("attempt_answer", lambda s: s.notna().sum()),
            attempts_selected=("status", lambda s: (s == "selected").sum()),
            attempts_rejected=("status", lambda s: (s == "rejected").sum()),
            python_calls_total=("python_calls", lambda s: pd.Series(s).fillna(0).sum()),
            python_errors_total=("python_errors", lambda s: pd.Series(s).fillna(0).sum()),
            mean_entropy=("entropy", "mean"),
            min_entropy=("entropy", "min"),
            mean_resp_len=("response_length", "mean"),
        )

        # most common finish reason per id
        def top_finish_reason(s):
            c = Counter([x if x is not None else "NA" for x in s.tolist()])
            return c.most_common(1)[0][0] if c else "NA"

        per_id_attempts["top_finish_reason"] = grp["finish_reason"].apply(top_finish_reason)["finish_reason"].values

        # attach final pred/gt if available
        if len(solutions_pd) > 0:
            per_id_attempts = per_id_attempts.merge(
                solutions_pd[["id", "pred_answer", "true_answer", "is_correct_recomputed"]],
                on="id", how="left"
            )
        else:
            per_id_attempts = per_id_attempts.merge(ref_pd[["id", "true_answer"]], on="id", how="left")

        per_id_attempts.to_csv(OUT_DIR / "per_id_attempt_stats.csv", index=False, encoding="utf-8")
        print(f"Saved: {OUT_DIR/'per_id_attempt_stats.csv'}")

        # show worst ids by failure mode
        print("\nTop ids by no parsed answers:")
        worst_no_ans = per_id_attempts.sort_values(["attempts_with_answer", "attempts_total"]).head(20)
        print(worst_no_ans[["id","attempts_total","attempts_with_answer","top_finish_reason","pred_answer","true_answer"]].to_string(index=False))

    # ---------- correctness vs entropy / votes (if solutions has vote_summary) ----------
    print_section("Correctness correlations (if available)")
    if len(solutions_pd) > 0:
        df = solutions_pd.copy()
        df["correct"] = df["is_correct_recomputed"].fillna(False)

        # selected entropy correlation (note: lower entropy -> more confident)
        if df["selected_entropy"].notna().any():
            corr = df[["selected_entropy", "correct"]].dropna().corr().iloc[0,1]
            print(f"Correlation(selected_entropy, correct): {corr:.3f}")

            # bucketed accuracy by entropy quantiles
            tmp = df[df["selected_entropy"].notna()].copy()
            tmp["entropy_bucket"] = pd.qcut(tmp["selected_entropy"], q=5, duplicates="drop")
            bucket = tmp.groupby("entropy_bucket", observed=False)["correct"].mean()
            print("\nAccuracy by selected_entropy bucket:")
            print(bucket.to_string())

        # budget/time
        if df["solve_elapsed_ms"].notna().any():
            tmp = df[df["solve_elapsed_ms"].notna()].copy()
            tmp["time_bucket"] = pd.qcut(tmp["solve_elapsed_ms"], q=5, duplicates="drop")
            bucket = tmp.groupby("time_bucket", observed=False)["correct"].mean()
            print("\nAccuracy by solve_elapsed_ms bucket:")
            print(bucket.to_string())
    else:
        print("solutions.csv empty; skipping.")

    # ---------- events timeline (optional) ----------
    print_section("Events timeline (optional)")
    if len(events_pd) == 0:
        print("events.jsonl not found or empty.")
    else:
        if "event" not in events_pd.columns:
            print("events.jsonl missing 'event' column.")
        else:
            print("Event counts:")
            print(events_pd["event"].value_counts().to_string())

            # estimate durations if problem_start/problem_end exist
            if {"problem_start", "problem_end"} <= set(events_pd["event"].unique()):
                # build per-id start/end
                starts = events_pd[events_pd["event"] == "problem_start"][["ts","id"]].copy()
                ends = events_pd[events_pd["event"] == "problem_end"][["ts","id"]].copy()

                # parse ts
                starts["ts"] = pd.to_datetime(starts["ts"], errors="coerce", utc=True)
                ends["ts"] = pd.to_datetime(ends["ts"], errors="coerce", utc=True)

                per = starts.merge(ends, on="id", how="inner", suffixes=("_start","_end"))
                per["duration_s"] = (per["ts_end"] - per["ts_start"]).dt.total_seconds()
                print("\nPer-problem durations (from events):")
                print(per["duration_s"].describe().to_string())

                per.to_csv(OUT_DIR / "per_id_event_durations.csv", index=False, encoding="utf-8")
                print(f"Saved: {OUT_DIR/'per_id_event_durations.csv'}")

    print_section("Done")
    print(f"Artifacts written to: {OUT_DIR}")

if __name__ == "__main__":
    main()
