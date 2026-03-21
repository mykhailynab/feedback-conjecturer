#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ============================================================
# Helpers
# ============================================================

def parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def short(x: Any, max_len: int = 160) -> str:
    if x is None:
        return "None"
    s = str(x).replace("\n", "\\n")
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"...[+{len(s) - max_len} chars]"


def fmt_dt_delta_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ============================================================
# Event loading
# ============================================================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                print(f"[warn] Failed to parse JSON at line {line_num}: {exc}")
                continue
            if not isinstance(obj, dict):
                print(f"[warn] Non-dict JSON object at line {line_num}; skipping.")
                continue
            records.append(obj)
    return records


# ============================================================
# Logical task grouping
# ============================================================

@dataclass(frozen=True)
class TaskKey:
    problem_id: Optional[str]
    attempt: Optional[int]
    # round: Optional[int]
    # agent_name: Optional[str]
    # task_kind: Optional[str]

    def pretty(self) -> str:
        parts: List[str] = []
        if self.problem_id is not None:
            parts.append(f"id={self.problem_id}")
        if self.attempt is not None:
            parts.append(f"attempt={self.attempt}")
        # if self.round is not None:
        #     parts.append(f"round={self.round}")
        # if self.agent_name:
        #     parts.append(f"agent={self.agent_name}")
        # if self.task_kind:
        #     parts.append(f"task_kind={self.task_kind}")
        if not parts:
            return "<ungrouped>"
        return " ".join(parts)


@dataclass
class TaskTrace:
    key: TaskKey
    events: List[Dict[str, Any]]

    @property
    def first_event(self) -> Optional[Dict[str, Any]]:
        return self.events[0] if self.events else None

    @property
    def last_event(self) -> Optional[Dict[str, Any]]:
        return self.events[-1] if self.events else None


def extract_task_key(ev: Dict[str, Any]) -> Optional[TaskKey]:
    # Strongest identifiers first
    problem_id = ev.get("problem_id")
    attempt_raw = ev.get("attempt")
    # round_raw = ev.get("round")
    # agent_name = ev.get("agent_name")
    # task_kind = ev.get("task_kind")

    attempt: Optional[int]
    # round_: Optional[int]

    try:
        attempt = int(attempt_raw) if attempt_raw is not None else None
    except Exception:
        attempt = None

    # try:
    #     round_ = int(round_raw) if round_raw is not None else None
    # except Exception:
    #     round_ = None

    if (
        problem_id is None
        and attempt is None
        # and round_ is None
        # and agent_name is None
        # and task_kind is None
    ):
        return None

    return TaskKey(
        problem_id=str(problem_id) if problem_id is not None else None,
        attempt=attempt,
        # round=round_,
        # agent_name=str(agent_name) if agent_name is not None else None,
        # task_kind=str(task_kind) if task_kind is not None else None,
    )


def group_events_by_task(events: Iterable[Dict[str, Any]]) -> Tuple[Dict[TaskKey, TaskTrace], List[Dict[str, Any]]]:
    grouped: Dict[TaskKey, List[Dict[str, Any]]] = defaultdict(list)
    ungrouped: List[Dict[str, Any]] = []

    for ev in events:
        key = extract_task_key(ev)
        if key is None:
            ungrouped.append(ev)
        else:
            grouped[key].append(ev)

    traces = {k: TaskTrace(key=k, events=v) for k, v in grouped.items()}
    return traces, ungrouped


# ============================================================
# Stuck / finished classification
# ============================================================


def is_task_finished(trace: TaskTrace) -> bool:
    names = {str(ev.get("event")) for ev in trace.events}

    if "formalization_done" in names:
        return True
    
    # warn_events = {
    #     str(ev.get("event")): ev.get("warn")
    #     for ev in trace.events
    #     if str(ev.get("event")) == "warn"
    # }
    # "event": "warn", "msg": "Failed to remove abbrev answer comment ..."
                           #  "Could not extract target abbrev name ..."
    if "warn" in names:
        print("\n\n\n\n\ntest\n\n\n\n\n")
        return True

    return False


def last_meaningful_event(trace: TaskTrace) -> Optional[Dict[str, Any]]:
    if not trace.events:
        return None
    return trace.events[-1]


def summarize_last_event(ev: Dict[str, Any]) -> str:
    name = str(ev.get("event", "<missing-event>"))

    fields = []
    for key in [
        "termination_reason",
        "reason",
        "recipient",
        "turn",
        "round",
        "ok",
        "timed_out",
        "returncode",
        "relative_path",
        "exception_type",
        "exception",
        "message_channel",
        "message_recipient",
    ]:
        if key in ev:
            fields.append(f"{key}={short(ev.get(key), 80)}")

    if not fields and "msg" in ev:
        fields.append(f"msg={short(ev.get('msg'), 120)}")

    if fields:
        return f"{name} ({', '.join(fields)})"
    return name


# ============================================================
# Reporting
# ============================================================

def print_overview(events: List[Dict[str, Any]], traces: Dict[TaskKey, TaskTrace]) -> None:
    print("=" * 80)
    print("EVENT OVERVIEW")
    print("=" * 80)

    event_counts = Counter(str(ev.get("event")) for ev in events)
    print("Event counts:")
    for name, count in event_counts.most_common():
        print(f"  {count:>6}  {name}")
    print()

    print(f"Grouped logical tasks: {len(traces)}")
    finished = sum(1 for tr in traces.values() if is_task_finished(tr))
    active = len(traces) - finished
    print(f"Finished tasks: {finished}")
    print(f"Active/stuck tasks: {active}")
    print()


def print_stuck_tasks(
    traces: Dict[TaskKey, TaskTrace],
    *,
    min_age_seconds: float = 0.0,
    limit: int = 200,
) -> None:
    print("=" * 80)
    print("TASKS THAT LOOK STUCK")
    print("=" * 80)

    active: List[Tuple[float, TaskTrace]] = []
    now = max([parse_iso(str(trace.last_event.get("ts", ""))) for trace in traces.values()])

    for trace in traces.values():
        if is_task_finished(trace):
            continue

        last_ev = trace.last_event
        if last_ev is None:
            continue

        ts = parse_iso(str(last_ev.get("ts", "")))
        if ts is None:
            age_s = float("inf")
        else:
            age_s = max(0.0, (now - ts).total_seconds())

        if age_s >= min_age_seconds:
            active.append((age_s, trace))

    active.sort(key=lambda x: x[0], reverse=True)

    if not active:
        print("No active/stuck tasks found.")
        print()
        return

    print(f"Found {len(active)} active/stuck tasks.")
    print()

    for idx, (age_s, trace) in enumerate(active[:limit], start=1):
        first_ev = trace.first_event
        last_ev = trace.last_event
        if first_ev is None or last_ev is None:
            continue

        first_ts = parse_iso(str(first_ev.get("ts", "")))
        last_ts = parse_iso(str(last_ev.get("ts", "")))

        runtime_s: Optional[float] = None
        if first_ts is not None and last_ts is not None:
            runtime_s = max(0.0, (last_ts - first_ts).total_seconds())

        print(f"[{idx}] {trace.key.pretty()}")
        print(f"    first event: {summarize_last_event(first_ev)}")
        print(f"    last event : {summarize_last_event(last_ev)}")
        print(f"    last seen  : {last_ev.get('ts')}  ({fmt_dt_delta_seconds(age_s)} from last event)")
        if runtime_s is not None:
            print(f"    span       : {fmt_dt_delta_seconds(runtime_s)} from first->last event")
        print(f"    num events : {len(trace.events)}")
        print()


def print_last_event_histogram(traces: Dict[TaskKey, TaskTrace]) -> None:
    print("=" * 80)
    print("WHERE ACTIVE TASKS ARE CURRENTLY STUCK")
    print("=" * 80)

    counter = Counter()

    for trace in traces.values():
        if is_task_finished(trace):
            continue
        last_ev = trace.last_event
        if last_ev is None:
            counter["<no-events>"] += 1
        else:
            counter[str(last_ev.get("event", "<missing-event>"))] += 1

    if not counter:
        print("(none)")
        print()
        return

    for name, count in counter.most_common():
        print(f"  {count:>6}  {name}")
    print()


def print_candidate_hang_diagnosis(traces: Dict[TaskKey, TaskTrace]) -> None:
    print("=" * 80)
    print("LIKELY BLOCKING PHASES")
    print("=" * 80)

    explanations = {
        "backend_llm_request_start": "likely blocked before/inside completions.create(...)",
        "backend_llm_request_created": "likely blocked waiting for first streamed chunk",
        "backend_llm_first_chunk": "likely blocked mid-stream waiting for later chunk(s)",
        "backend_tool_dispatch_start": "likely blocked inside a tool backend",
        "formalization_agent_call_start": "likely blocked in the LLM/backend call before it returned",
        "formalization_external_compile_start": "likely blocked in external Lean compilation",
        "formalization_close_start": "likely blocked during backend/tool teardown",
        "backend_turn_start": "likely blocked very early in a backend turn",
        "formalization_round_start": "likely blocked early in the round before agent return",
    }

    max_examples = 100

    counter = Counter()
    examples: Dict[str, List[str]] = defaultdict(list)

    for trace in traces.values():
        if is_task_finished(trace):
            continue
        last_ev = trace.last_event
        if last_ev is None:
            continue

        name = str(last_ev.get("event", "<missing-event>"))
        counter[name] += 1
        if len(examples[name]) < max_examples:
            examples[name].append(trace.key.pretty())

    for name, count in counter.most_common():
        explanation = explanations.get(name, "no heuristic explanation available")
        print(f"{name}: {count}  ->  {explanation}")
        for ex in examples[name]:
            print(f"    - {ex}")
        if len(examples[name]) > max_examples:
            print(f"... {len(examples[name]) - max_examples} hidden ...")
    print()


def print_backend_only_orphans(traces: Dict[TaskKey, TaskTrace]) -> None:
    """
    Useful when a backend session exists without a matching outer formalization_done.
    """
    print("=" * 80)
    print("BACKEND-ONLY ORPHAN TRACES")
    print("=" * 80)

    found = 0
    for trace in traces.values():
        names = {str(ev.get("event")) for ev in trace.events}
        if "backend_session_start" in names and "formalization_done" not in names:
            found += 1
            last_ev = trace.last_event
            print(f"- {trace.key.pretty()}")
            if last_ev is not None:
                print(f"  last event: {summarize_last_event(last_ev)}")
            if "backend_session_done" in names:
                print(f"  backend_session_done is present")
            if "backend_session_exception" in names:
                print(f"  backend_session_exception")
    if found == 0:
        print("(none)")
    print()


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse events.jsonl and report which logical tasks appear stuck on which log message."
    )
    p.add_argument("events_jsonl", help="Path to events.jsonl")
    p.add_argument(
        "--min-age-seconds",
        type=float,
        default=0.0,
        help="Only report active tasks whose last event is at least this old",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max number of active tasks to print in detail",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    events_path = Path(args.events_jsonl)
    if not events_path.exists():
        raise FileNotFoundError(f"Missing file: {events_path}")

    events = load_jsonl(events_path)
    traces, ungrouped = group_events_by_task(events)

    print(f"Loaded events: {len(events)}")
    print(f"Grouped task traces: {len(traces)}")
    print(f"Ungrouped events: {len(ungrouped)}")
    print()

    print_overview(events, traces)
    print_last_event_histogram(traces)
    print_candidate_hang_diagnosis(traces)
    print_stuck_tasks(
        traces,
        min_age_seconds=args.min_age_seconds,
        limit=args.limit,
    )
    print_backend_only_orphans(traces)

    if ungrouped:
        print("=" * 80)
        print("UNGROUPED EVENTS (first 20)")
        print("=" * 80)
        for ev in ungrouped[:20]:
            print(short(ev, 240))


if __name__ == "__main__":
    main()