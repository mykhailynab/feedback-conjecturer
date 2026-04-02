from __future__ import annotations

import re
from typing import Optional

from .result import CheckResult


def _normalize(s: str) -> str:
    """Collapse whitespace and strip; preserves mathematical content."""
    return re.sub(r"\s+", " ", s.strip())


def check_string_match(proposed_rhs: Optional[str], gt_rhs: Optional[str]) -> CheckResult:
    """
    Heuristic 1: exact string match after whitespace normalization.

    Returns equivalent=True on match, equivalent=None otherwise
    (never returns False — a non-match here is just inconclusive).
    """
    if not proposed_rhs or not gt_rhs:
        return CheckResult(
            equivalent=None,
            method="string_match",
            details={"reason": "empty_input"},
        )

    p = _normalize(proposed_rhs)
    g = _normalize(gt_rhs)
    match = p == g

    return CheckResult(
        equivalent=True if match else None,
        method="string_match",
        details={"proposed_normalized": p, "gt_normalized": g},
    )


__all__ = ["check_string_match"]
