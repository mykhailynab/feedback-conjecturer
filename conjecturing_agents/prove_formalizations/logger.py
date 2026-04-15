"""
Thread-safe JSONL loggers for prove_formalizations runs.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProveFormalizationsLogger:
    """
    Thread-safe JSONL event logger.

    Writes one record per line to the given file.  Each record has the form::

        {"ts": "<iso>", "event": "<type>", ...payload...}
    """

    def __init__(self, events_path: str) -> None:
        self.events_path = events_path
        self._lock = threading.Lock()
        Path(events_path).parent.mkdir(parents=True, exist_ok=True)
        Path(events_path).write_text("", encoding="utf-8")

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        rec = {"ts": _now_iso(), "event": event_type, **payload}
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with self._lock:
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(line)

    def close(self) -> None:
        pass  # File is opened/closed per write.
