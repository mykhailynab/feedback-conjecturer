from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from openai_harmony import ToolNamespaceConfig

from conjecturering_agents.inference_backends.vllm_harmony import (
    ToolDispatchResult,
    ToolInvocation,
    make_tool_message,
)


# ============================================================
# Result / config types
# ============================================================

@dataclass
class LeanCompileResult:
    ok: bool
    stdout: str
    stderr: str

    # Parsed Lean --json messages
    json_messages: List[Dict[str, Any]] = field(default_factory=list)
    json_errors: List[Dict[str, Any]] = field(default_factory=list)
    json_warnings: List[Dict[str, Any]] = field(default_factory=list)
    sorry_warnings: List[Dict[str, Any]] = field(default_factory=list)

    # Non-JSON lines from stdout, if any
    non_json_stdout_lines: List[str] = field(default_factory=list)

    # Execution metadata
    timed_out: bool = False
    elapsed_ms: int = 0
    returncode: Optional[int] = None
    command: List[str] = field(default_factory=list)

    # File metadata
    source_path: Optional[str] = None
    relative_path: Optional[str] = None

    # Helpful prompt-facing formatting
    formatted_diagnostics: str = ""


@dataclass
class LeanCompilerConfig:
    """
    Configuration for compiling Lean 4 snippets inside an existing Lean project.
    """

    # Lean project root (must contain lakefile / lake-manifest / .lake etc.)
    project_dir: str

    # Where temporary source files are created inside project_dir
    workspace_subdir: str = ".conjecturering_agents/lean_tool_runs"

    # Compilation command settings
    timeout_seconds: int = 120
    lean_jobs: int = 12

    # Tool wiring
    recipient_name: str = "lean"
    tool_name: str = "lean"

    # Policy
    treat_sorry_warning_as_failure: bool = True
    treat_any_warning_as_failure: bool = False
    auto_extract_code_block: bool = True
    cleanup_source_file: bool = False

    # Optional prefix added to temp files
    filename_prefix: str = "lean_tool_"

    # Prompt-facing formatting
    max_formatted_messages: int = 8
    include_stdout_in_tool_response: bool = False
    include_stderr_in_tool_response: bool = True
    include_non_json_stdout_lines: bool = True


# ============================================================
# Lean code extraction helpers
# ============================================================

def extract_lean_code_block(model_text: str) -> Optional[str]:
    """
    Extract the last fenced Lean code block.
    """
    if not model_text:
        return None

    patterns = [
        r"```lean4\s*\n(.*?)\n```",
        r"```lean4\s*\n(.*?)```",
        r"```lean\s*\n(.*?)\n```",
        r"```lean\s*\n(.*?)```",
    ]

    for pat in patterns:
        matches = re.findall(pat, model_text, re.DOTALL)
        if matches:
            return matches[-1].strip()

    return None


def extract_lean_code_block_or_text(text: str) -> str:
    code = extract_lean_code_block(text)
    if code is not None:
        return code
    return (text or "").strip()


# ============================================================
# Lean JSON parsing helpers
# ============================================================

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


# ============================================================
# Lean diagnostic formatting
# ============================================================

def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(n, hi))


def format_lean_messages(
    code: str,
    messages: Sequence[Dict[str, Any]],
    *,
    max_messages: int = 8,
    truncate_middle_lines: bool = True,
) -> str:
    """
    Prompt-friendly formatting for Lean diagnostics.

    It highlights the approximate error span using <error>...</error> markers,
    following the same spirit as your previous theorem-proving utilities.
    """
    if not messages:
        return ""

    code_lines = code.split("\n")
    out_parts: List[str] = []
    shown = list(messages[:max_messages])

    for i, msg in enumerate(shown, start=1):
        severity = str(msg.get("severity", "unknown")).upper()
        message_text = str(msg.get("data", "")).strip()

        out_parts.append(f"\n{severity} {i}:\n")

        pos = msg.get("pos")
        if not isinstance(pos, dict) or "line" not in pos or "column" not in pos:
            out_parts.append("Message:\n")
            out_parts.append(f"{message_text}\n")
            continue

        start_line = int(pos["line"]) - 1
        start_col = int(pos["column"])

        end_pos = msg.get("endPos")
        if isinstance(end_pos, dict) and "line" in end_pos and "column" in end_pos:
            end_line = int(end_pos["line"]) - 1
            end_col = int(end_pos["column"])
        else:
            end_line = start_line
            if 0 <= start_line < len(code_lines):
                end_col = len(code_lines[start_line])
            else:
                end_col = start_col

        if not code_lines:
            out_parts.append("Corresponding Code:\n```lean4\n")
            out_parts.append("\n```\n")
            out_parts.append(f"Message:\n{message_text}\n")
            continue

        start_line = _clamp(start_line, 0, len(code_lines) - 1)
        end_line = _clamp(end_line, 0, len(code_lines) - 1)
        start_col = _clamp(start_col, 0, len(code_lines[start_line]))
        end_col = _clamp(end_col, 0, len(code_lines[end_line]))

        out_parts.append("Corresponding Code:\n```lean4\n")

        # Context before
        for j in range(max(0, start_line - 4), start_line):
            out_parts.append(code_lines[j] + "\n")

        if start_line != end_line:
            out_parts.append(
                code_lines[start_line][:start_col]
                + "<error>"
                + code_lines[start_line][start_col:]
                + "\n"
            )

            if truncate_middle_lines:
                show_line_budget = 6
                upper = min(end_line, start_line + show_line_budget)
                for j in range(start_line + 1, upper):
                    out_parts.append(code_lines[j] + "\n")

                if end_line > start_line + show_line_budget:
                    last_visible = upper - 1 if upper > start_line + 1 else start_line
                    leading_spaces = 0
                    if 0 <= last_visible < len(code_lines):
                        line = code_lines[last_visible]
                        leading_spaces = len(line) - len(line.lstrip(" "))
                    out_parts.append(" " * leading_spaces + "... --[Truncated]-- ...\n")
            else:
                for j in range(start_line + 1, end_line):
                    out_parts.append(code_lines[j] + "\n")

            out_parts.append(
                code_lines[end_line][:end_col]
                + "</error>"
                + code_lines[end_line][end_col:]
                + "\n"
            )
        else:
            out_parts.append(
                code_lines[start_line][:start_col]
                + "<error>"
                + code_lines[start_line][start_col:end_col]
                + "</error>"
                + code_lines[start_line][end_col:]
                + "\n"
            )

        if end_line + 1 < len(code_lines):
            out_parts.append(code_lines[end_line + 1] + "\n")

        out_parts.append("```\n")
        out_parts.append(f"Message:\n{message_text}\n")

    if len(messages) > max_messages:
        out_parts.append(f"\n... [Omitted {len(messages) - max_messages} more messages] ...\n")

    return "".join(out_parts).strip()


def build_tool_facing_feedback(
    result: LeanCompileResult,
    *,
    cfg: LeanCompilerConfig,
) -> str:
    """
    Build the textual response fed back to the agent/tool-calling model.
    """
    parts: List[str] = []

    status = "[OK]" if result.ok else "[ERROR]"
    parts.append(f"{status} Lean compilation {'succeeded' if result.ok else 'failed'}.")

    if result.relative_path:
        parts.append(f"File: {result.relative_path}")
    if result.returncode is not None:
        parts.append(f"Return code: {result.returncode}")
    parts.append(f"Elapsed ms: {result.elapsed_ms}{' [TIMED OUT]' if result.timed_out else ''}")

    if result.json_errors:
        parts.append(f"Errors: {len(result.json_errors)}")
    if result.json_warnings:
        parts.append(f"Warnings: {len(result.json_warnings)}")
    if result.sorry_warnings:
        parts.append(f"Sorry warnings: {len(result.sorry_warnings)}")

    if result.formatted_diagnostics:
        parts.append("")
        parts.append(result.formatted_diagnostics)

    if cfg.include_non_json_stdout_lines and result.non_json_stdout_lines:
        parts.append("")
        parts.append("Non-JSON stdout:")
        parts.append("\n".join(result.non_json_stdout_lines))

    if cfg.include_stdout_in_tool_response and result.stdout.strip():
        parts.append("")
        parts.append("Raw stdout:")
        parts.append(result.stdout)

    if cfg.include_stderr_in_tool_response and result.stderr.strip():
        parts.append("")
        parts.append("Raw stderr:")
        parts.append(result.stderr)

    return "\n".join(parts).strip()


# ============================================================
# Low-level compiler backend
# ============================================================

class Lean4CompilerBackend:
    """
    Low-level Lean 4 compiler runner.

    Responsibility:
      - take Lean code as a string
      - write it into a temp file inside a Lean project
      - run `lake env lean --json <file>`
      - parse JSON diagnostics
      - return a structured result

    This backend is intentionally generic: it does not assume whether the code
    is an abbrev, theorem proof, or a whole file assembled by another agent.
    """

    def __init__(self, cfg: LeanCompilerConfig):
        self.cfg = cfg
        self.project_dir = Path(cfg.project_dir).resolve()
        self.workspace_dir = self.project_dir / cfg.workspace_subdir
        self._execution_lock = threading.Lock()
        self._counter = 0

        if not self.project_dir.exists():
            raise FileNotFoundError(f"Lean project directory does not exist: {self.project_dir}")

        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # File helpers
    # --------------------------------------------------------

    def _next_relpath(self) -> Path:
        """
        Produce a unique relative path inside the Lean project.
        """
        with self._execution_lock:
            self._counter += 1
            counter = self._counter

        stamp = int(time.time() * 1000)
        token = uuid.uuid4().hex[:8]
        filename = f"{self.cfg.filename_prefix}{stamp}_{counter}_{token}.lean"
        return Path(self.cfg.workspace_subdir) / filename

    def _write_source_file(self, code: str, relpath: Path) -> Path:
        abspath = self.project_dir / relpath
        abspath.parent.mkdir(parents=True, exist_ok=True)
        abspath.write_text(code, encoding="utf-8")
        return abspath

    def _remove_source_file(self, abspath: Path) -> None:
        try:
            if abspath.exists():
                abspath.unlink()
        except Exception:
            pass

    # --------------------------------------------------------
    # Compilation
    # --------------------------------------------------------

    def compile_code(
        self,
        code: str,
        *,
        timeout_seconds: Optional[int] = None,
        relative_path: Optional[str] = None,
    ) -> LeanCompileResult:
        timeout_s = int(timeout_seconds or self.cfg.timeout_seconds)

        relpath = Path(relative_path) if relative_path is not None else self._next_relpath()
        abspath = self._write_source_file(code, relpath)

        cmd = [
            "lake",
            "env",
            "lean",
            "-j",
            str(self.cfg.lean_jobs),
            "--json",
            relpath.as_posix(),
        ]

        t0 = time.time()

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.project_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
            )
            timed_out = False
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = str(exc.stdout or "")
            stderr = str(exc.stderr or "") + "\n[TIMEOUT]\n"
            returncode = None

        elapsed_ms = int((time.time() - t0) * 1000)

        parsed = parse_lean_json_stdout(stdout)
        json_errors = parsed["json_errors"]
        json_warnings = parsed["json_warnings"]
        sorry_warnings = parsed["sorry_warnings"]

        hard_failure = False
        if timed_out:
            hard_failure = True
        elif json_errors:
            hard_failure = True
        elif self.cfg.treat_sorry_warning_as_failure and sorry_warnings:
            hard_failure = True
        elif self.cfg.treat_any_warning_as_failure and json_warnings:
            hard_failure = True
        elif returncode not in (0, None):
            hard_failure = True

        relevant_messages: List[Dict[str, Any]] = []
        if json_errors:
            relevant_messages.extend(json_errors)
        elif self.cfg.treat_sorry_warning_as_failure and sorry_warnings:
            relevant_messages.extend(sorry_warnings)
        elif self.cfg.treat_any_warning_as_failure and json_warnings:
            relevant_messages.extend(json_warnings)

        formatted = format_lean_messages(
            code,
            relevant_messages,
            max_messages=self.cfg.max_formatted_messages,
        )

        result = LeanCompileResult(
            ok=not hard_failure,
            stdout=stdout,
            stderr=stderr,
            json_messages=parsed["json_messages"],
            json_errors=json_errors,
            json_warnings=json_warnings,
            sorry_warnings=sorry_warnings,
            non_json_stdout_lines=parsed["non_json_stdout_lines"],
            timed_out=timed_out,
            elapsed_ms=elapsed_ms,
            returncode=returncode,
            command=cmd,
            source_path=str(abspath),
            relative_path=relpath.as_posix(),
            formatted_diagnostics=formatted,
        )

        if self.cfg.cleanup_source_file:
            self._remove_source_file(abspath)

        return result

    def close(self) -> None:
        """
        Present for symmetry with other tool backends.
        """
        return None


# ============================================================
# Harmony adapter
# ============================================================

class Lean4CompilerToolBackend:
    """
    Harmony tool adapter for Lean compilation.

    Typical usage:

        lean_tool = Lean4CompilerToolBackend(
            description="Use this tool to compile Lean 4 code..."
            cfg=LeanCompilerConfig(project_dir="/path/to/project"),
        )

        agent.tool_configs = [lean_tool.tool_config]
        agent.tool_handlers = {"lean": lean_tool.handle_invocation}

    The handler expects the model to send Lean code directly, optionally inside
    a fenced ```lean4 ... ``` block.
    """

    def __init__(
        self,
        description: str,
        *,
        cfg: LeanCompilerConfig,
        request_extractor: Optional[Callable[[ToolInvocation], str]] = None,
        record_extra: Optional[Callable[[ToolInvocation, LeanCompileResult], Dict[str, Any]]] = None,
    ):
        self.description = description
        self.cfg = cfg
        self.backend = Lean4CompilerBackend(cfg)
        self.request_extractor = request_extractor
        self.record_extra = record_extra

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.cfg.recipient_name,
            description=self.description,
            tools=[],
        )

    # --------------------------------------------------------
    # Direct compile helper
    # --------------------------------------------------------

    def compile_code(
        self,
        code: str,
        *,
        timeout_seconds: Optional[int] = None,
        relative_path: Optional[str] = None,
    ) -> LeanCompileResult:
        return self.backend.compile_code(
            code,
            timeout_seconds=timeout_seconds,
            relative_path=relative_path,
        )

    def close(self) -> None:
        self.backend.close()

    def __enter__(self) -> "Lean4CompilerToolBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # --------------------------------------------------------
    # Harmony tool handler
    # --------------------------------------------------------

    def handle_invocation(self, invocation: ToolInvocation) -> ToolDispatchResult:
        if invocation.recipient != self.cfg.recipient_name:
            raise ValueError(
                f"Lean4CompilerToolBackend received recipient={invocation.recipient!r}, "
                f"expected {self.cfg.recipient_name!r}"
            )

        request_text = self._extract_request_text(invocation)
        code = (
            extract_lean_code_block_or_text(request_text)
            if self.cfg.auto_extract_code_block
            else request_text.strip()
        )

        result = self.backend.compile_code(code)
        tool_feedback = build_tool_facing_feedback(result, cfg=self.cfg)

        response_message = make_tool_message(
            tool_name=self.cfg.tool_name,
            output=tool_feedback,
            channel=invocation.message.channel,
            recipient="assistant",
        )

        record: Dict[str, Any] = {
            "recipient": invocation.recipient,
            "request_text": request_text,
            "compiled_code": code,
            "ok": result.ok,
            "timed_out": result.timed_out,
            "elapsed_ms": result.elapsed_ms,
            "returncode": result.returncode,
            "relative_path": result.relative_path,
            "source_path": result.source_path,
            "json_error_count": len(result.json_errors),
            "json_warning_count": len(result.json_warnings),
            "sorry_warning_count": len(result.sorry_warnings),
            "formatted_diagnostics": result.formatted_diagnostics,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "json_errors": result.json_errors,
            "json_warnings": result.json_warnings,
            "sorry_warnings": result.sorry_warnings,
            "non_json_stdout_lines": result.non_json_stdout_lines,
        }

        if self.record_extra is not None:
            extra = self.record_extra(invocation, result) or {}
            record.update(extra)

        return ToolDispatchResult(messages=[response_message], record=record)

    def _extract_request_text(self, invocation: ToolInvocation) -> str:
        if self.request_extractor is not None:
            return self.request_extractor(invocation)

        if not invocation.message.content:
            return ""

        parts: List[str] = []
        for item in invocation.message.content:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(text)
        return "\n".join(parts)


# ============================================================
# Optional cleanup helper
# ============================================================

def remove_workspace_dir(project_dir: str, workspace_subdir: str) -> None:
    """
    Utility for scripts/tests that want to clean temp Lean files created by the tool.
    """
    path = Path(project_dir).resolve() / workspace_subdir
    try:
        if path.exists():
            shutil.rmtree(path)
    except Exception:
        pass


__all__ = [
    "LeanCompileResult",
    "LeanCompilerConfig",
    "Lean4CompilerBackend",
    "Lean4CompilerToolBackend",
    "extract_lean_code_block",
    "extract_lean_code_block_or_text",
    "parse_lean_json_stdout",
    "format_lean_messages",
    "build_tool_facing_feedback",
    "remove_workspace_dir",
]