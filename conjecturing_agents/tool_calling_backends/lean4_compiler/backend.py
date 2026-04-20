from __future__ import annotations

import time
import uuid
import threading
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .data_model import (
    LeanCompilerConfig, LeanCompileResult
)

from json_helpers import parse_lean_json_stdout

from prompt_formatting import format_lean_messages


class LeanCompilerBackend:
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
            *([
                "--memory",
                f"{self.cfg.max_memory_megabytes}"
            ] if self.cfg.max_memory_megabytes > 0 else []),
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

        oom = not timed_out and "excessive memory" in stderr

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
            oom=oom,
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

__all__ = [
    "LeanCompilerBackend"
]