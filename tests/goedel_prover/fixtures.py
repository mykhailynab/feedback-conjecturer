"""
Shared test fixtures: Lean 4 theorem statements, Lean JSON error dicts, model outputs.

All statements follow the format the Goedel prover receives: a full Lean file
ending with ``theorem ... := by sorry``.
"""

# ---------------------------------------------------------------------------
# Lean theorem statements
# ---------------------------------------------------------------------------

LEAN_STMT_SIMPLE = """\
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

theorem lean_workbook_42 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    a + b ≥ 2 * Real.sqrt (a * b) := by sorry"""

LEAN_STMT_MULTILINE_SIG = """\
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

theorem lean_workbook_99
    (n : ℕ)
    (hn : 0 < n) :
    n + 1 > n := by sorry"""

# ---------------------------------------------------------------------------
# Code string whose 1-based line numbers match the error fixtures below.
#
#  1: import Mathlib
#  2: import Aesop
#  3: (blank)
#  4: set_option maxHeartbeats 0
#  5: (blank)
#  6: open BigOperators Real Nat Topology Rat
#  7: (blank)
#  8: theorem foo (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
#  9:     a + b ≥ 2 * Real.sqrt (a * b) := by
# 10:     linarith [sq_nonneg (Real.sqrt a - Real.sqrt b),
# 11:               Real.sq_sqrt ha.le,
# 12:               Real.sq_sqrt hb.le,
# 13:               Real.sqrt_nonneg a,
# 14:               Real.sqrt_nonneg b,
# 15:               Real.sqrt_mul_self ha.le]
# ---------------------------------------------------------------------------

CODE_15_LINES = """\
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

theorem foo (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    a + b ≥ 2 * Real.sqrt (a * b) := by
    linarith [sq_nonneg (Real.sqrt a - Real.sqrt b),
              Real.sq_sqrt ha.le,
              Real.sq_sqrt hb.le,
              Real.sqrt_nonneg a,
              Real.sqrt_nonneg b,
              Real.sqrt_mul_self ha.le]"""

# ---------------------------------------------------------------------------
# Lean JSON error dicts (same schema as LeanCompileResult.json_errors)
# ---------------------------------------------------------------------------

# Single-line error span (start_line == end_line)
ERROR_SINGLE_LINE = {
    "severity": "error",
    "pos": {"line": 10, "column": 4},
    "endPos": {"line": 10, "column": 11},
    "data": "unknown tactic 'linarith'",
}

# Multi-line error span within show_line=6 (no truncation)
ERROR_MULTILINE_SHORT = {
    "severity": "error",
    "pos": {"line": 10, "column": 4},
    "endPos": {"line": 12, "column": 6},
    "data": "type mismatch",
}

# Multi-line error span exceeding show_line=6 (truncation fires)
ERROR_MULTILINE_LONG = {
    "severity": "error",
    "pos": {"line": 3, "column": 0},
    "endPos": {"line": 15, "column": 5},
    "data": "function expected",
}

# Error with endPos=None
ERROR_NO_ENDPOS = {
    "severity": "error",
    "pos": {"line": 8, "column": 2},
    "endPos": None,
    "data": "failed to synthesize instance Inhabited ℝ",
}

# Nine errors — exercises the 8-error cap and the omission footer
ERRORS_NINE = [
    {
        "severity": "error",
        "pos": {"line": i + 1, "column": 0},
        "endPos": {"line": i + 1, "column": 4},
        "data": f"error number {i + 1}",
    }
    for i in range(9)
]

# ---------------------------------------------------------------------------
# Model outputs
# ---------------------------------------------------------------------------

MODEL_OUTPUT_WITH_PROOF = """\
Let me prove this using the AM-GM inequality.

The key insight is (√a - √b)² ≥ 0.

```lean4
theorem foo (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    a + b ≥ 2 * Real.sqrt (a * b) := by
  nlinarith [sq_nonneg (Real.sqrt a - Real.sqrt b),
             Real.mul_self_sqrt ha.le,
             Real.mul_self_sqrt hb.le]
```"""
