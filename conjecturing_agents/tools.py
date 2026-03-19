import math
from typing import Any

def to_float_or_inf(x: Any) -> float:
    try:
        value = float(x)
    except Exception:
        return float("inf")
    if math.isnan(value):
        return float("inf")
    return value