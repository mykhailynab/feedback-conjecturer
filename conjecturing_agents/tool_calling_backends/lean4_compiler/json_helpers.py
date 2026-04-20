from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence

def _safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _is_sorry_warning(msg: Dict[str, Any]) -> bool:
    """
    Lean warning for sorry may vary slightly across versions/plugins.
    """
    if msg.get("severity") != "warning":
        return False

    data = msg.get("data")
    if data == "declaration uses 'sorry'":
        return True
    if isinstance(data, str) and "declaration uses 'sorry'" in data:
        return True
    return False


def parse_lean_json_stdout(stdout: str) -> Dict[str, Any]:
    """
    Parse line-delimited JSON emitted by `lake env lean --json`.
    """
    json_messages: List[Dict[str, Any]] = []
    json_errors: List[Dict[str, Any]] = []
    json_warnings: List[Dict[str, Any]] = []
    sorry_warnings: List[Dict[str, Any]] = []
    non_json_stdout_lines: List[str] = []

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        msg = _safe_json_loads(line)
        if msg is None:
            non_json_stdout_lines.append(raw_line)
            continue

        json_messages.append(msg)

        severity = msg.get("severity")
        if severity == "error":
            json_errors.append(msg)
        elif severity == "warning":
            json_warnings.append(msg)
            if _is_sorry_warning(msg):
                sorry_warnings.append(msg)

    return {
        "json_messages": json_messages,
        "json_errors": json_errors,
        "json_warnings": json_warnings,
        "sorry_warnings": sorry_warnings,
        "non_json_stdout_lines": non_json_stdout_lines,
    }

__all__ = [
    "_safe_json_loads",
    "_is_sorry_warning",
    "parse_lean_json_stdout",
]