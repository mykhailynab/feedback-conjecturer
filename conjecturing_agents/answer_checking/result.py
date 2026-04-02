from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CheckResult:
    """
    Result from a single answer-checking heuristic.

    equivalent=True  : the check concluded the answers are equivalent
    equivalent=False : the check concluded they are NOT equivalent
    equivalent=None  : inconclusive (proof not found / method not applicable)
    """

    equivalent: Optional[bool]
    method: str  # "string_match" | "lean_equiv" | "goedel_prover" | "skipped" | "error"
    details: Dict[str, Any] = field(default_factory=dict)


__all__ = ["CheckResult"]
