"""
Token-for-token equivalence tests for ``format_lean_errors`` vs the reference
``get_error_str`` from ``goedel_original_scripts/src/utils.py``.

Both functions accept:
  - ``code``: the full Lean source string
  - ``errors``: a list of Lean JSON message dicts with ``pos``, ``endPos``, ``data``

The reference always passes ``error_thres=True`` (the ``inference.py`` default),
which caps error output at 8 and applies the 6-line truncation for long spans.
Our ``format_lean_errors`` matches that via ``truncate=True`` (the default).
"""
import pytest

from conjecturing_agents.agents.goedel_prover.lean_utils import format_lean_errors
from tests.goedel_prover.reference import reference_get_error_str
from tests.goedel_prover.fixtures import (
    CODE_15_LINES,
    ERROR_SINGLE_LINE,
    ERROR_MULTILINE_SHORT,
    ERROR_MULTILINE_LONG,
    ERROR_NO_ENDPOS,
    ERRORS_NINE,
)


def _ref(code, errors):
    return reference_get_error_str(code, errors, error_thres=True)


def _ours(code, errors):
    return format_lean_errors(code, errors, truncate=True)


# ---------------------------------------------------------------------------
# Single-line error span
# ---------------------------------------------------------------------------

def test_single_line_error():
    errors = [ERROR_SINGLE_LINE]
    assert _ours(CODE_15_LINES, errors) == _ref(CODE_15_LINES, errors)


# ---------------------------------------------------------------------------
# Multi-line error span — short (no truncation)
# ---------------------------------------------------------------------------

def test_multiline_error_no_truncation():
    errors = [ERROR_MULTILINE_SHORT]
    assert _ours(CODE_15_LINES, errors) == _ref(CODE_15_LINES, errors)


# ---------------------------------------------------------------------------
# Multi-line error span — long (truncation fires after show_line=6 lines)
# ---------------------------------------------------------------------------

def test_multiline_error_with_truncation():
    errors = [ERROR_MULTILINE_LONG]
    assert _ours(CODE_15_LINES, errors) == _ref(CODE_15_LINES, errors)


# ---------------------------------------------------------------------------
# Error with endPos=None
# ---------------------------------------------------------------------------

def test_error_no_endpos():
    errors = [ERROR_NO_ENDPOS]
    assert _ours(CODE_15_LINES, errors) == _ref(CODE_15_LINES, errors)


# ---------------------------------------------------------------------------
# Multiple errors — exactly at the cap
# ---------------------------------------------------------------------------

def test_eight_errors_no_omission_footer():
    errors = ERRORS_NINE[:8]
    assert _ours(CODE_15_LINES, errors) == _ref(CODE_15_LINES, errors)


# ---------------------------------------------------------------------------
# Nine errors — exercises the omission footer
# ---------------------------------------------------------------------------

def test_nine_errors_omission_footer():
    assert _ours(CODE_15_LINES, ERRORS_NINE) == _ref(CODE_15_LINES, ERRORS_NINE)


# ---------------------------------------------------------------------------
# Empty error list
# ---------------------------------------------------------------------------

def test_empty_errors():
    assert _ours(CODE_15_LINES, []) == _ref(CODE_15_LINES, [])
