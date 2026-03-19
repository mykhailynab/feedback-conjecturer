from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunLogger:
    """
    Writes:
      - formalizations.jsonl : one record per processed attempt
      - events.jsonl         : lifecycle / warning events
    """

    def __init__(
        self,
        *,
        formalizations_path: str,
        log_dir: str,
        verbose: bool = True,
    ):
        self.formalizations_path = formalizations_path
        self.events_path = str(Path(log_dir) / "events.jsonl")
        self.verbose = verbose
        self._lock = threading.Lock()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _append_jsonl(self, path: str, records: List[Dict[str, Any]]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @staticmethod
    def _safe_one_line(x: Any, max_len: int = 220) -> str:
        if x is None:
            return "None"
        s = str(x).replace("\n", "\\n")
        if len(s) > max_len:
            return s[:max_len] + f"...[truncated:{len(s) - max_len}]"
        return s

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        rec = {"ts": self._now_iso(), "event": event_type, **payload}
        with self._lock:
            self._append_jsonl(self.events_path, [rec])

        if self.verbose:
            msg = payload.get("msg") or ""
            print(f"[LOG:{event_type}] {msg}".rstrip())

    def log_warn(self, msg: str, **payload: Any) -> None:
        rec = {"ts": self._now_iso(), "event": "warn", "msg": msg, **payload}
        with self._lock:
            self._append_jsonl(self.events_path, [rec])
        print(f"[warn] {msg}")

    def log_formalization_result(self, result_record: Dict[str, Any]) -> None:
        ts = self._now_iso()
        summary = (
            f"id={result_record.get('problem_id')} "
            f"attempt={result_record.get('attempt')} "
            f"status={result_record.get('status')} "
            f"rounds={result_record.get('rounds_used')} "
            f"lean_ok={result_record.get('final_compile_ok')} "
            f"abbrev={self._safe_one_line(result_record.get('final_abbrev_declaration'))}"
        )

        record = dict(result_record)
        record["ts"] = ts
        record["summary"] = summary

        with self._lock:
            self._append_jsonl(self.formalizations_path, [record])

        if self.verbose:
            print(f"[LOG:formalizations] {summary}")