#!/usr/bin/env python3
"""
Given a folder containing:
  - attempts.jsonl
  - events.jsonl
  - solutions.csv

This script:
  1) Re-parses checker JSON for agent_* events using agent_call.raw_output (robustly),
     fixing cases where result.reason == "json_parse_error:JSONDecodeError" (or other parse failures)
     even though the final JSON is present.
  2) Writes:
       - attempts_fixed.jsonl   (verbatim copy; attempts don't contain truth-check results in your pipeline)
       - events_fixed.jsonl     (updated agent_call.result where fixable)
       - solutions_fixed.csv    (updates checker_summary.per_attempt_truth[*].parsed_result using corrected events)
     into the SAME folder.
  3) Prints stats: how many agent events had parse failures, how many were fixed, etc.

"""

import re
import argparse
import csv
import json
import shutil
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List, Union


# ----------------------------
# Robust JSON extraction
# ----------------------------

def _normalize_checker_obj(obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        raise ValueError("obj is not a dict")
    if "equivalent" not in obj:
        raise ValueError("\"equivalent\" not in obj")
    try:
        eq = bool(obj["equivalent"])
        conf = float(obj.get("confidence", 0.0))
        reason = str(obj.get("reason", ""))
        conf = max(0.0, min(1.0, conf))
        return {"equivalent": eq, "confidence": conf, "reason": reason}
    except Exception as e:
        raise e

def extract_checker_json_from_text(text: str) -> Dict[str, Any]:
    """
    Checker MUST emit a single JSON line.
    Robustly locate a JSON object, preferring ones that contain the key "equivalent".
    """
    if not text:
        return {"equivalent": False, "confidence": 0.0, "reason": "Error: empty_checker_output"}

    text = text.strip()

    last_err: Optional[BaseException] = None

    # Fast path: try parsing the entire string as JSON (works if checker truly outputs only JSON)
    try:
        obj = json.loads(text)
    except Exception:
        obj = None  # ignore exception if not a json.
    try:
        if obj is not None:
            return _normalize_checker_obj(obj)
    except Exception as e:
        last_err = e

    candidates = list(re.finditer(r"\{.*?\}", text, flags=re.DOTALL))

    for m in reversed(candidates):
        chunk = m.group(0)
        # if do_debug:
        #     print(f"Candidate 1: {chunk}")
        if '"equivalent"' not in chunk:
            continue
        try:
            obj = json.loads(chunk)
        except Exception as e:
            continue  # ignore exception if not a json.
        try:
            return _normalize_checker_obj(obj)
        except Exception as e:
            last_err = e
            continue

    decoder = json.JSONDecoder()
    for i in reversed([i for i, ch in enumerate(text) if ch == "{"]):
        try:
            obj, end = decoder.raw_decode(text[i:])  # can have extra text at the end
        except Exception:
            continue  # ignore exception if not a json.
        try:
            return _normalize_checker_obj(obj)
        except Exception as e:
            last_err = e
            continue
    
    pattern = re.compile(r'\{\s*"equivalent"\s*:\s*(true|false)\s*,\s*"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)', flags=re.DOTALL)
    candidates_noreason = list(pattern.finditer(text))
    for c in candidates_noreason:
        m_text = c.group(0) + '}'
        try:
            obj = json.loads(m_text)
        except Exception as e:
            continue  # ignore exception if not a json.
        try:
            return _normalize_checker_obj(obj)
        except Exception as e:
            last_err = e
            continue

    pattern = re.compile(r'\{\s*"equivalent"\s*:\s*(true|false)', flags=re.DOTALL)
    candidates_onlyeq = list(pattern.finditer(text))
    for c in candidates_onlyeq:
        m_text = c.group(0) + '}'
        try:
            obj = json.loads(m_text)
        except Exception as e:
            continue  # ignore exception if not a json.
        try:
            return _normalize_checker_obj(obj)
        except Exception as e:
            last_err = e
            continue

    return {
        "equivalent": False,
        "confidence": 0.0,
        "reason": f"Error: {type(last_err).__name__}: {last_err}" if last_err is not None else "Error: no_json_found"
    }


def is_parse_failure_reason(reason: Any) -> bool:
    r = (str(reason or "")).strip()
    if not r:
        return False
    return (
        r.startswith("json_parse_error:")
        or r == "no_json_found"
        or r == "empty_checker_output"
    )


# ----------------------------
# IO helpers
# ----------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # Keep malformed lines as raw text? Here we drop, but you can change policy if needed.
                continue
    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_solutions_csv(path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = [dict(r) for r in reader]
    return header, rows


def save_solutions_csv(path: Path, header: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------
# Main fix logic
# ----------------------------

@dataclass
class Stats:
    agent_events_total: int = 0
    agent_events_with_parse_failure: int = 0
    agent_events_fixed: int = 0
    agent_events_still_failed: int = 0

    still_failed_reason_counter: Counter[str] = field(default_factory=Counter)

    solutions_attempt_entries_total: int = 0
    solutions_attempt_entries_needing_fix: int = 0
    solutions_attempt_entries_fixed: int = 0
    solutions_attempt_entries_still_failed: int = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Folder containing attempts.jsonl, events.jsonl, solutions.csv")
    ap.add_argument("--attempts", default="attempts.jsonl")
    ap.add_argument("--events", default="events.jsonl")
    ap.add_argument("--solutions", default="solutions.csv")
    args = ap.parse_args()

    d = Path(args.dir)
    attempts_path = d / args.attempts
    events_path = d / args.events
    solutions_path = d / args.solutions

    if not attempts_path.exists():
        raise FileNotFoundError(f"Missing {attempts_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Missing {events_path}")
    if not solutions_path.exists():
        raise FileNotFoundError(f"Missing {solutions_path}")

    attempts_fixed_path = d / "attempts_fixed.jsonl"
    events_fixed_path = d / "events_fixed.jsonl"
    solutions_fixed_path = d / "solutions_fixed.csv"

    stats = Stats()

    # 0) attempts: verbatim copy (attempts.jsonl has no truth-check fields in your pipeline)
    shutil.copyfile(attempts_path, attempts_fixed_path)

    # 1) Fix events.jsonl agent_* entries
    events = read_jsonl(events_path)

    # Map corrected results by (problem_id, attempt) for later patching solutions.csv
    corrected_truth: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for ev in events:
        ev_type = str(ev.get("event", ""))
        if not ev_type.startswith("agent_"):
            continue

        stats.agent_events_total += 1

        agent_call = ev.get("agent_call") or {}
        res = (agent_call.get("result") or {})
        reason = res.get("reason")
        if not is_parse_failure_reason(reason):
            # still store if it looks well-formed (helps solutions even if solutions had issues)
            pid = ev.get("problem_id")
            att = ev.get("attempt")
            if pid is not None and att is not None:
                try:
                    corrected_truth[(str(pid), int(att))] = res
                except Exception:
                    pass
            continue

        stats.agent_events_with_parse_failure += 1

        raw_output = agent_call.get("raw_output") or ""
        fixed_output = extract_checker_json_from_text(raw_output)

        if not fixed_output["reason"].startswith("Error: "):
            # overwrite agent_call.result with corrected parse
            agent_call["result"] = fixed_output
            ev["agent_call"] = agent_call

            pid = ev.get("problem_id")
            att = ev.get("attempt")
            if pid is not None and att is not None:
                try:
                    corrected_truth[(str(pid), int(att))] = fixed_output
                except Exception:
                    pass

            stats.agent_events_fixed += 1
        else:
            print(f"\n\n\n================\nFailed to restore: {fixed_output["reason"]}\n")
            if len(raw_output) > 1000:
                print(raw_output[:500])
                print("    (...)    ")
                print(raw_output[-500:])
            else:
                print(raw_output)
            stats.still_failed_reason_counter[fixed_output["reason"]] += 1
            stats.agent_events_still_failed += 1
            print(f"================\n\n\n")

    write_jsonl(events_fixed_path, events)

    # 2) Fix solutions.csv checker_summary using corrected events map
    header, sol_rows = load_solutions_csv(solutions_path)

    if "checker_summary" not in header:
        raise ValueError("solutions.csv missing 'checker_summary' column")

    for row in sol_rows:
        pid = str(row.get("id", ""))

        cs_raw = row.get("checker_summary", "")
        try:
            cs = json.loads(cs_raw) if cs_raw else {}
        except Exception:
            cs = {}

        per_attempt = (cs.get("per_attempt_truth") or {})
        if not isinstance(per_attempt, dict):
            per_attempt = {}

        changed = False
        for k, v in per_attempt.items():
            stats.solutions_attempt_entries_total += 1
            try:
                att = int(k)
            except Exception:
                continue
            if not isinstance(v, dict):
                continue

            parsed_result = v.get("parsed_result") or {}
            reason = parsed_result.get("reason")
            needs = is_parse_failure_reason(reason)

            if needs:
                stats.solutions_attempt_entries_needing_fix += 1

            # If we have corrected truth from events, apply it (even if solutions didn't flag)
            corr = corrected_truth.get((pid, att))
            if corr is not None and (needs or parsed_result != corr):
                v["parsed_result"] = corr
                # keep is_correct consistent if possible
                if "equivalent" in corr:
                    v["is_correct"] = bool(corr["equivalent"])
                per_attempt[str(att)] = v
                changed = True
                if needs:
                    stats.solutions_attempt_entries_fixed += 1

            elif needs and corr is None:
                stats.solutions_attempt_entries_still_failed += 1

        if changed:
            cs["per_attempt_truth"] = per_attempt
            row["checker_summary"] = json.dumps(cs, ensure_ascii=False)

    save_solutions_csv(solutions_fixed_path, header, sol_rows)

    # ----------------------------
    # Print stats
    # ----------------------------
    print("\n=== Fix checker JSON parse: stats ===")
    print(f"Input folder: {d}")
    print(f"Wrote: {attempts_fixed_path.name}, {events_fixed_path.name}, {solutions_fixed_path.name}")
    print("")
    print("Events (agent_*)")
    print(f"  total agent events:                 {stats.agent_events_total}")
    print(f"  agent events w/ parse-failure:      {stats.agent_events_with_parse_failure}")
    print(f"  fixed:                              {stats.agent_events_fixed}")
    print(f"  still failed:                       {stats.agent_events_still_failed}")
    if stats.agent_events_with_parse_failure:
        print(f"  fix rate:                           {stats.agent_events_fixed / stats.agent_events_with_parse_failure:.2%}")
    print("")
    print("Solutions checker_summary.per_attempt_truth")
    print(f"  total attempt entries seen:         {stats.solutions_attempt_entries_total}")
    print(f"  entries needing fix:                {stats.solutions_attempt_entries_needing_fix}")
    print(f"  fixed (via events mapping):         {stats.solutions_attempt_entries_fixed}")
    print(f"  still failed (no event fix found):  {stats.solutions_attempt_entries_still_failed}")
    if stats.solutions_attempt_entries_needing_fix:
        print(f"  fix rate:                           {stats.solutions_attempt_entries_fixed / stats.solutions_attempt_entries_needing_fix:.2%}")
    print("Fix failed reasons:")
    for reason, count in stats.still_failed_reason_counter.items():
        print(f"    - #{count}: \"{reason}\"")
    print("=====================================\n")


if __name__ == "__main__":
    main()
