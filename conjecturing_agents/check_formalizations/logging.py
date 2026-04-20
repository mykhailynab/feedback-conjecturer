"""
Thread-safe JSONL event logger for check_formalizations runs.

Writes one record per line to ``check_goedel_events.jsonl`` (alongside
``check_results.jsonl``).  Each record has the form::

    {"ts": "<iso>", "event": "<type>", ...payload...}

Events come from two sources:

* ``GoedelProverAgent`` — session/round-level events including full prompt
  and output text (``prover_session_start``, ``prover_round_done``,
  ``prover_session_done``).
* ``VLLMRawBackend`` / ``OllamaBackend`` — lightweight generation timing
  events (``raw_generation_start``, ``raw_generation_first_chunk``,
  ``raw_generation_done``).
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from conjecturing_agents.inference_backends.raw_base import EventLoggerFn  # re-export


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CheckFormalizationsLogger:
    """
    Thread-safe JSONL event logger.

    The file is truncated on construction and appended to on each
    ``log_event`` call.  Safe to share across threads.
    """

    def __init__(self, events_path: str) -> None:
        self.events_path = events_path
        self._lock = threading.Lock()
        Path(events_path).parent.mkdir(parents=True, exist_ok=True)
        # Truncate / create the file
        Path(events_path).write_text("", encoding="utf-8")

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        rec = {"ts": _now_iso(), "event": event_type, **payload}
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with self._lock:
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(line)

    def close(self) -> None:
        pass  # File is opened/closed per write; nothing to flush.
