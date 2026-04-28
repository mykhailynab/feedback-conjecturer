"""
TODO: NOTE: IMPORTANT: The Jupyter tool allows to import new libraries.
For example, one of the agents imported subprocess and started grep'ping the contents of the file system.
This behavior should be prohibited in the future.
"""
from __future__ import annotations

import os
import re
import ast
import time
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from jupyter_client import KernelManager
from openai_harmony import ToolNamespaceConfig, TextContent

from conjecturing_agents.inference_backends.vllm_harmony import (
    ToolDispatchResult,
    ToolInvocation,
    make_tool_message,
)


# ============================================================
# Config / result types
# ============================================================

DEFAULT_PRELOAD_CODE = (
    "import math\n"
    "import numpy\n"
    "import sympy\n"
    "import itertools\n"
    "import collections\n"
    "import mpmath\n"
    "mpmath.mp.dps = 64\n"
)

DEFAULT_RESET_CODE = (
    "%reset -f\n"
    "import math\n"
    "import numpy\n"
    "import sympy\n"
    "import itertools\n"
    "import collections\n"
    "import mpmath\n"
    "mpmath.mp.dps = 64\n"
)


@dataclass
class JupyterKernelConfig:
    """
    Low-level kernel/session settings for a single stateful Jupyter runtime.
    """
    timeout_seconds: float = 6.0
    preload_code: str = DEFAULT_PRELOAD_CODE
    reset_code: str = DEFAULT_RESET_CODE

    recipient_name: str = "python"
    tool_name: str = "python"

    ensure_last_print: bool = True
    init_on_create: bool = True
    interrupt_on_timeout: bool = True

    kernel_extra_arguments: List[str] = field(
        default_factory=lambda: ["--Application.log_level=CRITICAL"]
    )
    env_overrides: Dict[str, str] = field(
        default_factory=lambda: {
            "PYDEVD_DISABLE_FILE_VALIDATION": "1",
            "PYDEVD_WARN_EVALUATION_TIMEOUT": "0",
            "JUPYTER_PLATFORM_DIRS": "1",
            "PYTHONWARNINGS": "ignore",
            "MPLBACKEND": "Agg",
        }
    )


@dataclass
class JupyterExecutionResult:
    output: str
    success: bool
    timed_out: bool
    had_error: bool
    elapsed_ms: int
    executed_code: str


# ============================================================
# Kernel session
# ============================================================

class _PortAllocator:
    _lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def reserve_ports(cls, count: int = 5) -> List[int]:
        with cls._lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports


class JupyterKernelSession:
    """
    Owns one persistent stateful Jupyter kernel.

    This class is intentionally tool-backend agnostic. It just executes Python
    code and returns normalized textual output.
    """

    def __init__(self, cfg: Optional[JupyterKernelConfig] = None):
        self.cfg = cfg or JupyterKernelConfig()

        self._km: Optional[KernelManager] = None
        self._client = None
        self._owns_kernel = False

        self._init_lock = threading.Lock()
        self._execution_lock = threading.Lock()

        if self.cfg.init_on_create:
            self.start()

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def start(self) -> None:
        with self._init_lock:
            if self._client is not None:
                return

            ports = _PortAllocator.reserve_ports(5)

            env = os.environ.copy()
            env.update(self.cfg.env_overrides)

            km = KernelManager()
            km.shell_port = ports[0]
            km.iopub_port = ports[1]
            km.stdin_port = ports[2]
            km.hb_port = ports[3]
            km.control_port = ports[4]

            km.start_kernel(
                env=env,
                extra_arguments=self.cfg.kernel_extra_arguments,
            )

            client = km.blocking_client()
            client.start_channels()
            client.wait_for_ready(timeout=self.cfg.timeout_seconds)

            self._km = km
            self._client = client
            self._owns_kernel = True

            if self.cfg.preload_code.strip():
                self._execute_after_start(
                    self.cfg.preload_code,
                    timeout=self.cfg.timeout_seconds,
                )

    def close(self) -> None:
        with self._init_lock:
            try:
                if self._client is not None:
                    self._client.stop_channels()
            except Exception:
                pass
            finally:
                self._client = None

            if self._owns_kernel and self._km is not None:
                try:
                    self._km.shutdown_kernel(now=True)
                except Exception:
                    pass
                try:
                    self._km.cleanup_resources()
                except Exception:
                    pass
                finally:
                    self._km = None
                    self._owns_kernel = False

    def reset(self) -> None:
        self.start()
        if self.cfg.reset_code.strip():
            self._execute_after_start(
                self.cfg.reset_code,
                timeout=self.cfg.timeout_seconds,
            )

    def __enter__(self) -> "JupyterKernelSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _format_error(self, traceback_list: List[str]) -> str:
        clean_lines: List[str] = []
        for frame in traceback_list:
            clean_frame = re.sub(r"\x1b\[[0-9;]*m", "", frame)
            if 'File "' in clean_frame and "ipython-input" not in clean_frame:
                continue
            clean_lines.append(clean_frame)
        return "".join(clean_lines)

    def ensure_last_print(self, code: str) -> str:
        """
        If the final top-level statement is a bare expression, wrap that expression
        in print(...). Otherwise, leave the code unchanged.

        Examples:
        - "2 + 3"                   -> "print(2 + 3)"
        - "x = 3"                   -> unchanged
        - "while True:\\n    pass"  -> unchanged
        - "import math"             -> unchanged
        - "print(2 + 3)"            -> unchanged
        """
        if not code or not code.strip():
            return code

        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError:
            # If the snippet is syntactically invalid/incomplete, do nothing.
            return code

        if not tree.body:
            return code

        last_stmt = tree.body[-1]

        # Only wrap a final top-level bare expression.
        if not isinstance(last_stmt, ast.Expr):
            return code

        expr = last_stmt.value

        # Don't wrap an existing print(...) call.
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id == "print"
        ):
            return code

        # Avoid wrapping a trailing module docstring / bare string literal.
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return code

        # We rely on end positions, available in modern Python versions.
        if not all(
            hasattr(last_stmt, attr)
            for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset")
        ):
            return code

        lines = code.splitlines(keepends=True)

        def to_offset(lineno: int, col: int) -> int:
            # ast line numbers are 1-based
            return sum(len(lines[i]) for i in range(lineno - 1)) + col

        start = to_offset(last_stmt.lineno, last_stmt.col_offset)
        end = to_offset(last_stmt.end_lineno, last_stmt.end_col_offset)

        original_expr_text = code[start:end]
        wrapped_expr_text = f"print({original_expr_text})"

        return code[:start] + wrapped_expr_text + code[end:]

    # --------------------------------------------------------
    # Execution
    # --------------------------------------------------------

    def _execute_after_start(self, code: str, timeout: Optional[float] = None) -> JupyterExecutionResult:
        """
        Execute code assuming the kernel has already been started.
        It is used both by public execute() and by start() during preload,
        which avoids recursive acquisition of self._init_lock.
        """
        assert self._client is not None
        assert self._km is not None

        effective_timeout = timeout if timeout is not None else self.cfg.timeout_seconds
        final_code = self.ensure_last_print(code) if self.cfg.ensure_last_print else code

        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
        timed_out = False
        had_error = False

        t0 = time.time()

        with self._execution_lock:
            msg_id = self._client.execute(
                final_code,
                store_history=True,
                allow_stdin=False,
                stop_on_error=False,
            )

            while True:
                elapsed = time.time() - t0
                if elapsed > effective_timeout:
                    timed_out = True
                    had_error = True
                    if self.cfg.interrupt_on_timeout:
                        try:
                            self._km.interrupt_kernel()
                        except Exception:
                            pass
                    output = f"[ERROR] Execution timed out after {effective_timeout} seconds"
                    return JupyterExecutionResult(
                        output=output,
                        success=False,
                        timed_out=True,
                        had_error=True,
                        elapsed_ms=int((time.time() - t0) * 1000),
                        executed_code=final_code,
                    )

                try:
                    msg = self._client.get_iopub_msg(timeout=1.0)
                except queue.Empty:
                    continue

                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue

                msg_type = msg.get("msg_type")
                content = msg.get("content", {})

                if msg_type == "stream":
                    text = content.get("text", "")
                    if content.get("name") == "stdout":
                        stdout_parts.append(text)
                    else:
                        stderr_parts.append(text)

                elif msg_type == "error":
                    had_error = True
                    stderr_parts.append(self._format_error(content.get("traceback", [])))

                elif msg_type in {"execute_result", "display_data"}:
                    data = content.get("data", {})
                    text = data.get("text/plain")
                    if text:
                        stdout_parts.append(text if text.endswith("\n") else f"{text}\n")

                elif msg_type == "status":
                    if content.get("execution_state") == "idle":
                        break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)

        if stderr:
            output = f"{stdout.rstrip()}\n{stderr}" if stdout else stderr
        else:
            output = stdout if stdout.strip() else "[WARN] No output. Use print() to see results."

        return JupyterExecutionResult(
            output=output,
            success=not had_error and not timed_out,
            timed_out=timed_out,
            had_error=had_error,
            elapsed_ms=int((time.time() - t0) * 1000),
            executed_code=final_code,
        )

    def execute(self, code: str, timeout: Optional[float] = None) -> JupyterExecutionResult:
        self.start()
        return self._execute_after_start(code, timeout=timeout)


# ============================================================
# Harmony adapter
# ============================================================

class JupyterToolBackend:
    """
    Adapts a single JupyterKernelSession into a Harmony recipient handler.

    Typical usage from an agent module:

        jupyter = JupyterToolBackend(
            description="Use this tool to run Python code ..."
        )

        agent = HarmonyAgentSpec(
            ...
            tool_configs=[jupyter.tool_config],
            tool_handlers={"python": jupyter.handle_invocation},
        )

    Each JupyterToolBackend instance owns one stateful kernel, so instantiating
    one backend per agent naturally gives each agent its own Python session.
    """

    def __init__(
        self,
        description: str,
        *,
        cfg: Optional[JupyterKernelConfig] = None,
        record_extra: Optional[Callable[[ToolInvocation, JupyterExecutionResult], Dict[str, Any]]] = None,
    ):
        self.description = description
        self.cfg = cfg or JupyterKernelConfig()
        self.session = JupyterKernelSession(self.cfg)
        self.record_extra = record_extra

    # --------------------------------------------------------
    # Tool config exposed to Harmony
    # --------------------------------------------------------

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.cfg.recipient_name,
            description=self.description,
            tools=[],
        )

    # --------------------------------------------------------
    # Direct execution helpers
    # --------------------------------------------------------

    def execute(self, code: str, timeout: Optional[float] = None) -> JupyterExecutionResult:
        return self.session.execute(code, timeout=timeout)

    def reset(self) -> None:
        self.session.reset()

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "JupyterToolBackend":
        self.session.start()
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
                f"JupyterToolBackend received recipient={invocation.recipient!r}, "
                f"expected {self.cfg.recipient_name!r}"
            )

        request_text = self._extract_request_text(invocation)
        result = self.session.execute(request_text)

        response_message = make_tool_message(
            tool_name=self.cfg.tool_name,
            output=result.output,
            channel=invocation.message.channel,
            recipient="assistant",
        )

        record: Dict[str, Any] = {
            "recipient": invocation.recipient,
            "request_text": request_text,
            "executed_code": result.executed_code,
            "output": result.output,
            "success": result.success,
            "timed_out": result.timed_out,
            "had_error": result.had_error,
            "elapsed_ms": result.elapsed_ms,
        }

        if self.record_extra is not None:
            extra = self.record_extra(invocation, result) or {}
            record.update(extra)

        return ToolDispatchResult(messages=[response_message], record=record)

    @staticmethod
    def _extract_request_text(invocation: ToolInvocation) -> str:
        if not invocation.message.content:
            return ""
        chunks = []
        for item in invocation.message.content:
            if not isinstance(item, TextContent):
                continue
            chunks.append(item.text)
        return "\n".join(chunks)


# ============================================================
# TIR adapter (non-Harmony)
# ============================================================

class JupyterTIRToolBackend:
    """
    TIR (non-Harmony) adapter for a stateful Jupyter Python kernel.

    Exposes:
      - tool_def: the OpenAI/Ollama JSON function schema to pass to the model
      - handle_call(tool_name, arguments) -> str: invoked by TIRBackend.run_session()

    Typical usage:

        jupyter_tir = JupyterTIRToolBackend(
            description="Execute Python code and return its output ...",
        )

        result = backend.run_session(
            messages=[...],
            tools=[jupyter_tir.tool_def],
            tool_handlers={jupyter_tir.cfg.tool_name: jupyter_tir.handle_call},
            cfg=TIRGenerationConfig(...),
        )

    Each instance owns one stateful kernel, so create one per agent/session.
    """

    def __init__(
        self,
        description: str,
        *,
        cfg: Optional[JupyterKernelConfig] = None,
    ) -> None:
        self.description = description
        self.cfg = cfg or JupyterKernelConfig()
        self.session = JupyterKernelSession(self.cfg)

    @property
    def tool_def(self) -> Dict[str, Any]:
        """OpenAI/Ollama JSON function schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.cfg.tool_name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "required": ["code"],
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute",
                        }
                    },
                },
            },
        }

    def handle_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a tool call dispatched by TIRBackend.run_session().

        Runs the Python code in arguments["code"] and returns the output
        (stdout / stderr / error) as a string.
        """
        code = arguments.get("code", "")
        result = self.session.execute(code)
        return result.output

    def execute(self, code: str, timeout: Optional[float] = None) -> JupyterExecutionResult:
        """Direct execution bypass."""
        return self.session.execute(code, timeout=timeout)

    def reset(self) -> None:
        self.session.reset()

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "JupyterTIRToolBackend":
        self.session.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "JupyterKernelConfig",
    "JupyterExecutionResult",
    "JupyterKernelSession",
    "JupyterToolBackend",
    "JupyterTIRToolBackend",
]
