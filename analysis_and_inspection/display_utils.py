"""
Shared terminal display utilities for analysis scripts.

Provides ANSI colour helpers, box-drawing primitives, and small formatting
functions that any analysis script can import.
"""
from __future__ import annotations

import textwrap
from typing import Tuple, Optional

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------

_USE_COLOR = True  # call set_color(False) to disable


def set_color(enabled: bool) -> None:
    global _USE_COLOR
    _USE_COLOR = enabled


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def green(t: str)   -> str: return _c("32", t)
def red(t: str)     -> str: return _c("31", t)
def yellow(t: str)  -> str: return _c("33", t)
def cyan(t: str)    -> str: return _c("36", t)
def bold(t: str)    -> str: return _c("1",  t)
def dim(t: str)     -> str: return _c("2",  t)
def blue(t: str)    -> str: return _c("34", t)
def magenta(t: str) -> str: return _c("35", t)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

WIDTH = 100


def hline(char: str = "─", width: int = WIDTH) -> str:
    return char * width


def box_top(title: str = "", width: int = WIDTH, char: str = "─") -> str:
    if title:
        pad = width - len(title) - 4
        left = pad // 2
        right = pad - left
        return f"┌─ {title} " + "─" * right + "┐" if pad >= 0 else f"┌ {title} ┐"
    return "┌" + char * (width - 2) + "┐"


def box_bottom(width: int = WIDTH, char: str = "─") -> str:
    return "└" + char * (width - 2) + "┘"


def box_line(text: str, width: int = WIDTH) -> str:
    inner = width - 4
    lines = []
    for raw in text.split("\n"):
        if len(raw) <= inner:
            lines.append("│ " + raw + " " * (inner - len(raw)) + " │")
        else:
            for chunk in textwrap.wrap(raw, inner) or [""]:
                lines.append("│ " + chunk + " " * (inner - len(chunk)) + " │")
    return "\n".join(lines)


def section_header(title: str, width: int = WIDTH) -> str:
    pad = width - len(title) - 4
    return f"  {bold(title)}  " + dim("─" * max(pad, 0))

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_ms(ms: Optional[int]) -> str:
    if ms is None:
        return "?"
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms/1000:.1f}s"


def truncate(text: str, max_chars: int) -> Tuple[str, bool]:
    """Return (possibly truncated text, was_truncated)."""
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars], True
    return text, False
