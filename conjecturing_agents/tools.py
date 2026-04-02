import json
import math
from pathlib import Path
from typing import Any, Dict, List, Union


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a JSONL file, skipping blank lines and non-dict objects with a warning."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                print(f"[warn] Failed to parse JSON at {path}:{line_num}: {exc}")
                continue
            if not isinstance(obj, dict):
                print(f"[warn] Non-dict JSON object at {path}:{line_num}; skipping.")
                continue
            records.append(obj)
    return records


def write_jsonl(path: Union[str, Path], records: List[Dict[str, Any]]) -> None:
    """Write records as newline-delimited JSON, creating parent directories as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def to_float_or_inf(x: Any) -> float:
    try:
        value = float(x)
    except Exception:
        return float("inf")
    if math.isnan(value):
        return float("inf")
    return value