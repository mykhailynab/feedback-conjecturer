"""
Thread-safe JSONL loggers for prove_formalizations runs.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from conjecturing_agents.inference_backends.raw_base import EventLogger


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProveFormalizationsLogger(EventLogger):
    """
    Thread-safe JSONL event logger.

    Writes one record per line to the given file.  Each record has the form::

        {"ts": "<iso>", "event": "<type>", ...payload...}

    Inherits per-thread metadata support from ``EventLogger``: callers (e.g.
    the scheduler) can call ``set_thread_metadata({"proof_id": ...})`` so
    that backend-level events automatically carry the proof_id without
    requiring changes to the backend's ``_log_event`` calls.
    """

    def __init__(self, events_path: str) -> None:
        super().__init__()
        self.events_path = events_path
        Path(events_path).parent.mkdir(parents=True, exist_ok=True)
        Path(events_path).write_text("", encoding="utf-8")

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        merged = self._merge_thread_metadata(payload)
        rec = {"ts": _now_iso(), "event": event_type, **merged}
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with self._lock:
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(line)

    def close(self) -> None:
        pass  # File is opened/closed per write.
