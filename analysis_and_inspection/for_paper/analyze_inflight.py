#!/usr/bin/env python3
"""
Analyze prover_events.jsonl to identify in-flight (unfinished) processes.

Matches start/done event pairs by proof_id, reports completed durations and
in-flight elapsed times.  Relies solely on events — does not read prove_results.jsonl.

Detects "orphaned" sub-events (e.g. chat_stream) where the parent session ended
(deadline_exceeded, stop_event, etc.) but the child's done event was never emitted.
These are shown separately from truly in-flight processes.

Usage:
    PYTHONPATH=. python analysis_and_inspection/for_paper/analyze_inflight.py \
        --events logs/.../prover_events.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Event pairs to track: (start_event, done_event, label, parent_done_event)
# parent_done_event: if set, unmatched starts whose parent session has ended
# are classified as "orphaned" rather than "in-flight".
EVENT_PAIRS = [
    ("tir_prover_session_start", "tir_prover_session_done", "formal_prover", None),
    ("informal_prover_session_start", "informal_prover_session_done", "informal_prover", None),
    ("tir_session_start", "tir_session_done", "tir_session", None),
    ("tir_chat_stream_start", "tir_chat_stream_done", "chat_stream", "tir_session_done"),
    ("tir_lean_compile_start", "tir_lean_compile_done", "lean_compile", "tir_session_done"),
    ("tir_python_start", "tir_python_done", "python_tool", "tir_session_done"),
]


def parse_ts(ts_str: str) -> datetime:
    # Handle both timezone-aware and naive ISO formats
    ts_str = ts_str.replace("+00:00", "+0000").replace("Z", "+0000")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(ts_str)


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:04.1f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze in-flight processes from prover events")
    parser.add_argument("--events", required=True, help="Path to prover_events.jsonl")
    args = parser.parse_args()

    events: List[Dict[str, Any]] = []
    with open(args.events, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    if not events:
        print("No events found.")
        return

    # Determine the latest timestamp as the "now" reference for in-flight durations
    all_ts = [parse_ts(e["ts"]) for e in events]
    t_last = max(all_ts)
    t_first = min(all_ts)
    print(f"Events: {len(events)}")
    print(f"Time range: {t_first.isoformat()} → {t_last.isoformat()} "
          f"({fmt_duration((t_last - t_first).total_seconds())})")
    print()

    # Build parent-done lookup: proof_id → latest parent done timestamp + reason
    # Used to detect orphaned sub-events (child start exists, no child done,
    # but parent session already ended).
    parent_done_map: Dict[str, Dict[str, Any]] = {}
    for e in events:
        if e["event"] == "tir_session_done":
            pid = e.get("proof_id", "")
            parent_done_map[pid] = e

    for start_evt, done_evt, label, parent_done_evt in EVENT_PAIRS:
        starts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        dones: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for e in events:
            pid = e.get("proof_id", "")
            if e["event"] == start_evt:
                starts[pid].append(e)
            elif e["event"] == done_evt:
                dones[pid].append(e)

        n_starts = sum(len(v) for v in starts.values())
        n_dones = sum(len(v) for v in dones.values())
        n_unmatched = n_starts - n_dones

        # Match start→done by proof_id, FIFO within each proof_id
        completed_durations: List[Tuple[str, float, Dict[str, Any]]] = []
        inflight: List[Tuple[str, float, Dict[str, Any]]] = []
        # Orphaned: parent session ended but child done event was never emitted
        orphaned: List[Tuple[str, float, Dict[str, Any], str]] = []  # (pid, dur, evt, reason)

        for pid in sorted(set(list(starts.keys()) + list(dones.keys()))):
            s_list = sorted(starts.get(pid, []), key=lambda e: e["ts"])
            d_list = sorted(dones.get(pid, []), key=lambda e: e["ts"])

            for i, s in enumerate(s_list):
                s_ts = parse_ts(s["ts"])
                if i < len(d_list):
                    d_ts = parse_ts(d_list[i]["ts"])
                    dur = (d_ts - s_ts).total_seconds()
                    completed_durations.append((pid, dur, {**s, **d_list[i]}))
                else:
                    # Unmatched start — check if parent session ended
                    parent = parent_done_map.get(pid) if parent_done_evt else None
                    if parent is not None:
                        parent_ts = parse_ts(parent["ts"])
                        if parent_ts >= s_ts:
                            dur = (parent_ts - s_ts).total_seconds()
                            reason = parent.get("termination_reason", "unknown")
                            orphaned.append((pid, dur, s, reason))
                            continue
                    elapsed = (t_last - s_ts).total_seconds()
                    inflight.append((pid, elapsed, s))

        n_orphaned = len(orphaned)
        n_inflight = len(inflight)

        print(f"{'=' * 60}")
        parts = [f"{n_starts} started", f"{n_dones} done"]
        if n_orphaned:
            parts.append(f"{n_orphaned} orphaned")
        parts.append(f"{n_inflight} in-flight")
        print(f"{label}: {', '.join(parts)}")
        print(f"{'=' * 60}")

        if n_inflight == 0 and n_orphaned == 0 and n_starts == 0:
            print()
            continue

        # Print orphaned processes (parent session ended, child done never emitted)
        if orphaned:
            orphaned.sort(key=lambda x: -x[1])  # longest first
            print(f"\n  ORPHANED ({n_orphaned}) — parent session ended, done event missing:")
            for pid, dur, evt, reason in orphaned:
                problem = evt.get("problem_id", "?")
                attempt = evt.get("attempt", "?")
                extra = f"  reason={reason}"
                if "base_url" in evt:
                    extra += f"  backend={evt['base_url']}"
                if "turn" in evt:
                    extra += f"  turn={evt['turn']}"
                print(f"    {pid}  ({problem}/a{attempt})  duration={fmt_duration(dur)}{extra}")

        # Print truly in-flight processes
        if inflight:
            inflight.sort(key=lambda x: -x[1])  # longest first
            print(f"\n  IN-FLIGHT ({n_inflight}):")
            for pid, elapsed, evt in inflight:
                problem = evt.get("problem_id", "?")
                attempt = evt.get("attempt", "?")
                extra = ""
                if "base_url" in evt:
                    extra += f"  backend={evt['base_url']}"
                if "turn" in evt:
                    extra += f"  turn={evt['turn']}"
                print(f"    {pid}  ({problem}/a{attempt})  elapsed={fmt_duration(elapsed)}{extra}")

        # Print completed summary
        if completed_durations:
            durations = [d for _, d, _ in completed_durations]
            avg = sum(durations) / len(durations)
            mn, mx = min(durations), max(durations)
            print(f"\n  COMPLETED ({len(completed_durations)}): "
                  f"avg={fmt_duration(avg)}, min={fmt_duration(mn)}, max={fmt_duration(mx)}")

        print()

    # Backend distribution
    print(f"{'=' * 60}")
    print("Backend URL distribution (from tir_chat_stream_start)")
    print(f"{'=' * 60}")
    url_counts: Dict[str, int] = defaultdict(int)
    for e in events:
        if e["event"] == "tir_chat_stream_start" and "base_url" in e:
            url_counts[e["base_url"]] += 1
    if url_counts:
        for url, count in sorted(url_counts.items(), key=lambda x: -x[1]):
            print(f"  {url}: {count} requests")
    else:
        # Try host field (Ollama backend)
        for e in events:
            if e["event"] == "tir_chat_stream_start" and "host" in e:
                url_counts[e["host"]] += 1
        if url_counts:
            for url, count in sorted(url_counts.items(), key=lambda x: -x[1]):
                print(f"  {url}: {count} requests")
        else:
            print("  (no base_url/host found in events)")
    print()


if __name__ == "__main__":
    main()
