#!/usr/bin/env python3
from __future__ import annotations

import os
import csv
import json
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# ============================================================
# Logging
# ============================================================

class RunLogger:
    """
    Writes:
      - attempts.jsonl
      - solutions.csv
      - events.jsonl
    """

    def __init__(
        self,
        *,
        attempts_path: str,
        solutions_path: str,
        log_dir: str,
        verbose: bool = True,
        log_attempt_progress: bool = True,
    ):
        self.attempts_path = attempts_path
        self.solutions_path = solutions_path
        self.events_path = str(Path(log_dir) / "events.jsonl")
        self.verbose = verbose
        self.log_attempt_progress = log_attempt_progress
        self._lock = threading.Lock()
        self._init_solutions_csv()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _append_jsonl(self, path: str, records: List[Dict[str, Any]]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @staticmethod
    def _safe_one_line(x: Any, max_len: int = 200) -> str:
        if x is None:
            return "None"
        s = str(x).replace("\n", "\\n")
        if len(s) > max_len:
            return s[:max_len] + f"...[truncated:{len(s) - max_len}]"
        return s

    def _init_solutions_csv(self) -> None:
        if os.path.exists(self.solutions_path):
            return

        header = [
            "id",
            "true_answer_text",
            "solve_started_ts",
            "solve_finished_ts",
            "solve_elapsed_ms",
            "attempts_total",
            "attempts_with_answer",
            "selected_attempt",
            "selected_answer_text",
            "selected_entropy",
            "selected_is_correct",
            "checker_summary",
        ]
        with open(self.solutions_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        rec = {"ts": self._now_iso(), "event": event_type, **payload}
        with self._lock:
            self._append_jsonl(self.events_path, [rec])

        if self.verbose:
            msg = payload.get("msg") or ""
            print(f"[LOG:{event_type}] {msg}".rstrip())

    def log_attempt(self, attempt_record: Dict[str, Any]) -> None:
        ts = self._now_iso()
        summary = (
            f"problem_id={attempt_record.get('problem_id')} attempt={attempt_record.get('attempt')} "
            f"ans={self._safe_one_line(attempt_record.get('attempt_answer'))} "
            f"ent={attempt_record.get('entropy')} "
            f"py={attempt_record.get('python_calls')}/{attempt_record.get('python_errors')} "
            f"termination={attempt_record.get('termination_reason')} "
            f"elapsed_ms={attempt_record.get('attempt_elapsed_ms')}"
        )

        record = dict(attempt_record)
        record["ts"] = ts
        record["summary"] = summary

        with self._lock:
            self._append_jsonl(self.attempts_path, [record])

        if self.log_attempt_progress:
            print(f"[LOG:attempts] Attempt summary: {summary}")

    def log_agent_call(self, *, kind: str, payload: Dict[str, Any]) -> None:
        rec = {"ts": self._now_iso(), "event": f"agent_{kind}", **payload}
        with self._lock:
            self._append_jsonl(self.events_path, [rec])

        if self.log_attempt_progress:
            rid = payload.get("problem_id")
            att = payload.get("attempt")
            eq = payload.get("agent_call", {}).get("result", {}).get("equivalent", None)
            print(f"[LOG:agent_{kind}] id={rid} attempt={att} equivalent={eq}")

    def log_solution_row(
        self,
        *,
        id_value: str,
        true_answer_text: str,
        solve_started_ts: Optional[str],
        solve_finished_ts: Optional[str],
        solve_elapsed_ms: Optional[int],
        attempts_total: int,
        attempts_with_answer: int,
        selected_attempt: Optional[int],
        selected_answer_text: str,
        selected_entropy: Optional[float],
        selected_is_correct: Optional[bool],
        checker_summary: Dict[str, Any],
    ) -> None:
        row = [
            id_value,
            true_answer_text,
            solve_started_ts,
            solve_finished_ts,
            solve_elapsed_ms,
            attempts_total,
            attempts_with_answer,
            selected_attempt,
            selected_answer_text,
            selected_entropy,
            selected_is_correct,
            json.dumps(checker_summary or {}, ensure_ascii=False),
        ]

        with self._lock:
            with open(self.solutions_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

        if self.log_attempt_progress:
            print(f"[LOG:attempts_finished] id={id_value}")
