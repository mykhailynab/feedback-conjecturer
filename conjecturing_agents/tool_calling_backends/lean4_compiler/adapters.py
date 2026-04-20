from typing import Any, Callable, Dict, List, Optional, Sequence

from openai_harmony import ToolNamespaceConfig

from conjecturing_agents.inference_backends.vllm_harmony import (
    ToolDispatchResult,
    ToolInvocation,
    make_tool_message,
    TextContent
)

from .data_model import (
    LeanCompilerConfig, LeanCompileResult
)

from .backend import LeanCompilerBackend

from conjecturing_agents.lean_regex import (
    extract_lean_code_block_or_text,
)

from .prompt_formatting import build_tool_facing_feedback


# ============================================================
# Harmony adapter
# ============================================================

class LeanCompilerToolHarmonyAdapter:
    """
    Harmony tool adapter for Lean compilation.

    Typical usage:

        lean_tool = LeanCompilerToolHarmonyAdapter(
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
        self.backend = LeanCompilerBackend(cfg)
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

    def __enter__(self) -> "LeanCompilerToolHarmonyAdapter":
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
                f"LeanCompilerToolHarmonyAdapter received recipient={invocation.recipient!r}, "
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

        chunks = []
        for item in invocation.message.content:
            if not isinstance(item, TextContent):
                continue
            chunks.append(item.text)
        return "\n".join(chunks)


# ============================================================
# TIR adapter (non-Harmony)
# ============================================================

class LeanCompilerToolAdapter:
    """
    TIR (non-Harmony) adapter for Lean 4 compilation.

    Exposes:
      - tool_def: the OpenAI/Ollama JSON function schema to pass to the model
      - handle_call(tool_name, arguments) -> str: invoked by TIRBackend.run_session()

    Typical usage:

        lean_tir = LeanCompilerToolAdapter(
            description="Compile and check Lean 4 code ...",
            cfg=LeanCompilerConfig(project_dir="/path/to/project"),
        )

        result = backend.run_session(
            messages=[...],
            tools=[lean_tir.tool_def],
            tool_handlers={lean_tir.cfg.tool_name: lean_tir.handle_call},
            cfg=TIRGenerationConfig(...),
        )
    """

    def __init__(
        self,
        description: str,
        *,
        cfg: LeanCompilerConfig,
    ) -> None:
        self.description = description
        self.cfg = cfg
        self.backend = LeanCompilerBackend(cfg)

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
                            "description": "Lean 4 code to compile and check",
                        }
                    },
                },
            },
        }

    def handle_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a tool call dispatched by TIRBackend.run_session().

        Compiles the Lean code in arguments["code"] and returns formatted
        diagnostic feedback as a string.
        """
        code_raw = arguments.get("code", "")
        code = (
            extract_lean_code_block_or_text(code_raw)
            if self.cfg.auto_extract_code_block
            else code_raw.strip()
        )
        result = self.backend.compile_code(code)
        return build_tool_facing_feedback(result, cfg=self.cfg)

    def compile_code(
        self,
        code: str,
        *,
        timeout_seconds: Optional[int] = None,
        relative_path: Optional[str] = None,
    ) -> LeanCompileResult:
        """Direct compilation bypass to the backend."""
        return self.backend.compile_code(
            code,
            timeout_seconds=timeout_seconds,
            relative_path=relative_path,
        )

    def close(self) -> None:
        self.backend.close()

    def __enter__(self) -> "LeanCompilerToolAdapter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

