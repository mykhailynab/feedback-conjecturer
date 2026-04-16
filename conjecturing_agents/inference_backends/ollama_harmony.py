"""
Ollama chat inference backend for the conjecturing pipeline.

Drives multi-turn solver/checker conversations using Ollama's chat API
(standard system/user/assistant/tool message format) instead of the
Harmony encoding used by VLLMHarmonyBackend.

Compatible interface:
  - run_user_prompt(agent, user_prompt, seed, metadata) → HarmonyRunResult
  - start() / close() / context-manager

Key differences from VLLMHarmonyBackend:
  - No Harmony encoding; uses standard OpenAI-style chat messages.
  - No logprobs → mean_entropy is always float("inf").
  - No vLLM server lifecycle; Ollama must already be running.
  - Tool calling via Ollama's native function-call API; tool handlers
    are called with a synthetic Harmony ToolInvocation for compatibility
    with existing JupyterToolBackend.handle_invocation.
  - Every non-tool-call assistant response is treated as "final"
    (no Harmony channel concept).
"""
from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from openai_harmony import (ToolNamespaceConfig, Conversation)

import ollama
from openai_harmony import Author, Message, Role, TextContent

from conjecturing_agents.inference_backends.vllm_harmony import (
    EventLoggerFn,
    HarmonyAgentSpec,
    HarmonyRunResult,
    HarmonySessionState,
    TerminationSignal,
    ToolDispatchResult,
    ToolInvocation,
    utcnow_iso,
)


# ============================================================
# Config
# ============================================================

@dataclass
class OllamaChatConfig:
    # Model / server
    model: str = "qwen3.5"
    host: str = "http://localhost:11434"
    client_timeout: int = 960

    # Generation defaults (overridden per-agent at inference time)
    top_p: float = 0.9
    num_predict: int = 32768  # default max tokens per turn

    # Context size used only for reference / future budget checks
    context_tokens: int = 262144


# ============================================================
# Backend
# ============================================================

class OllamaChatBackend:
    """
    Ollama-based drop-in replacement for VLLMHarmonyBackend.

    Accepts the same HarmonyAgentSpec used by SolverAgent and
    InformalCorrectnessCheckerAgent and drives the multi-turn loop
    via Ollama's chat API.
    """

    def __init__(
        self,
        cfg: OllamaChatConfig,
        *,
        event_logger: Optional[EventLoggerFn] = None,
    ) -> None:
        self.cfg = cfg
        self.event_logger = event_logger
        self._client: Optional[ollama.Client] = None
        self._lifecycle_lock = threading.Lock()
        self._started = False

    # ------------------------------------------------------------------
    # Client (lazy)
    # ------------------------------------------------------------------

    def _get_client(self) -> ollama.Client:
        if self._client is None:
            self._client = ollama.Client(
                host=self.cfg.host,
                timeout=self.cfg.client_timeout,
            )
        return self._client

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.event_logger is None:
            return
        try:
            self.event_logger(event_type, payload)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                return
            self._get_client().list()  # probe server; raises if unreachable
            self._started = True

    def close(self) -> None:
        with self._lifecycle_lock:
            self._client = None
            self._started = False

    def __enter__(self) -> "OllamaChatBackend":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Tool definition helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tool_defs(agent: HarmonyAgentSpec) -> List[Dict[str, Any]]:
        """Convert HarmonyAgentSpec.tool_configs to Ollama function definitions."""
        if agent.tool_configs is None:
            return []

        configs = (
            [agent.tool_configs]
            if isinstance(agent.tool_configs, ToolNamespaceConfig)
            else list(agent.tool_configs)
        )

        return [
            {
                "type": "function",
                "function": {
                    "name": tc.name,
                    "description": tc.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute",
                            }
                        },
                        "required": ["code"],
                    },
                },
            }
            for tc in configs
        ]

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool_call(
        self,
        *,
        agent: HarmonyAgentSpec,
        state: HarmonySessionState,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> ToolDispatchResult:
        handler = agent.tool_handlers.get(tool_name)
        if handler is None:
            error_msg = f"[ERROR] No handler for tool '{tool_name}'"
            error_message = Message(
                author=Author(role=Role.TOOL, name=tool_name),
                content=[TextContent(text=error_msg)],
            )
            return ToolDispatchResult(
                messages=[error_message],
                record={
                    "recipient": tool_name,
                    "request_text": "",
                    "response_texts": [error_msg],
                },
            )

        # Build synthetic Harmony Message so existing handlers work unchanged
        code = tool_args.get("code") or ""
        synthetic_message = Message(
            author=Author(role=Role.ASSISTANT, name="assistant"),
            content=[TextContent(text=code)],
        ).with_recipient(tool_name)

        invocation = ToolInvocation(
            recipient=tool_name,
            message=synthetic_message,
            state=state,
            backend=self,  # type: ignore[arg-type]
        )

        raw_result = handler(invocation)

        if isinstance(raw_result, ToolDispatchResult):
            return raw_result
        if isinstance(raw_result, Message):
            return ToolDispatchResult(messages=[raw_result])
        if isinstance(raw_result, str):
            msg = Message(
                author=Author(role=Role.TOOL, name=tool_name),
                content=[TextContent(text=raw_result)],
            )
            return ToolDispatchResult(messages=[msg])
        if isinstance(raw_result, (list, tuple)):
            return ToolDispatchResult(messages=list(raw_result))
        raise TypeError(f"Unsupported handler return type: {type(raw_result).__name__}")

    @staticmethod
    def _extract_dispatch_text(dispatch: ToolDispatchResult) -> str:
        parts: List[str] = []
        for msg in dispatch.messages:
            if not msg.content:
                continue
            for item in msg.content:
                if isinstance(item, TextContent):
                    parts.append(item.text)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Main inference entry point
    # ------------------------------------------------------------------

    def run_user_prompt(
        self,
        *,
        agent: HarmonyAgentSpec,
        user_prompt: str,
        seed: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HarmonyRunResult:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._run_loop(
            agent=agent,
            messages=messages,
            seed=seed,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Conversation loop
    # ------------------------------------------------------------------

    def _run_loop(
        self,
        *,
        agent: HarmonyAgentSpec,
        messages: List[Dict[str, Any]],
        seed: int,
        metadata: Optional[Dict[str, Any]],
    ) -> HarmonyRunResult:
        if not self._started:
            self.start()

        started_ts = utcnow_iso()
        t0 = time.time()
        deadline = t0 + float(agent.timeout_seconds)

        # Use HarmonySessionState for compat with existing termination callbacks
        state = HarmonySessionState(
            agent=agent,
            conversation=Conversation.from_messages([]),
            backend=self,  # type: ignore[arg-type]
            metadata=dict(metadata or {}),
            started_ts=started_ts,
        )

        tool_defs = self._build_tool_defs(agent)
        client = self._get_client()

        exception_text: Optional[str] = None
        termination_signal: Optional[TerminationSignal] = None

        self._log_event("backend_session_start", {
            "agent_name": agent.name,
            **dict(metadata or {}),
            "max_turns": agent.max_turns,
            "timeout_seconds": agent.timeout_seconds,
        })

        try:
            for turn_idx in range(agent.max_turns):
                if time.time() > deadline:
                    state.termination_reason = "deadline_exceeded"
                    break

                self._log_event("backend_turn_start", {
                    "agent_name": agent.name,
                    "turn": turn_idx,
                })

                completion_text_parts: List[str] = []
                tool_calls_from_stream: List[Any] = []
                stream_interrupted = False

                try:
                    options: Dict[str, Any] = {
                        "temperature": agent.temperature,
                        "top_p": self.cfg.top_p,
                        "seed": (seed + turn_idx) & 0x7FFFFFFF,
                        "num_predict": self.cfg.num_predict,
                    }
                    stream = client.chat(
                        model=self.cfg.model,
                        messages=messages,
                        tools=tool_defs if tool_defs else None,
                        stream=True,
                        options=options,
                    )

                    for chunk in stream:
                        if time.time() > deadline:
                            state.termination_reason = "deadline_exceeded"
                            stream_interrupted = True
                            break

                        new_text = chunk.message.content or ""
                        if new_text:
                            completion_text_parts.append(new_text)

                        # Tool calls appear (complete) on the final chunk
                        if chunk.message.tool_calls:
                            tool_calls_from_stream = chunk.message.tool_calls

                        # Call chunk termination callback during text streaming
                        if agent.terminate_on_chunk is not None and new_text:
                            recent_text = "".join(
                                completion_text_parts[-agent.stream_text_window:]
                            )
                            maybe_signal = agent.terminate_on_chunk(
                                state, recent_text, new_text
                            )
                            if maybe_signal is not None:
                                termination_signal = maybe_signal
                                state.parsed_output = maybe_signal.parsed_output
                                state.termination_reason = maybe_signal.reason
                                stream_interrupted = True
                                break

                except Exception as exc:
                    exception_text = f"{type(exc).__name__}: {exc}"
                    state.termination_reason = f"exception:{type(exc).__name__}"
                    break

                completion_text = "".join(completion_text_parts)
                # Rough token estimate: ~4 chars per token
                state.total_tokens += max(1, len(completion_text) // 4)
                state.turns.append({
                    "turn": turn_idx,
                    "completion_token_ids": [],
                    "completion_text": completion_text,
                })

                self._log_event("backend_llm_stream_done", {
                    "agent_name": agent.name,
                    "turn": turn_idx,
                    "completion_chars": len(completion_text),
                    "has_tool_calls": bool(tool_calls_from_stream),
                })

                if termination_signal is not None:
                    break

                if stream_interrupted:
                    break

                if not completion_text and not tool_calls_from_stream:
                    state.termination_reason = "no_tokens"
                    break

                # ---- Tool calls ----
                if tool_calls_from_stream:
                    # Add assistant turn (may include partial reasoning text)
                    messages.append({
                        "role": "assistant",
                        "content": completion_text or None,
                        "tool_calls": [
                            {
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                }
                            }
                            for tc in tool_calls_from_stream
                        ],
                    })

                    for tc in tool_calls_from_stream:
                        tool_name = tc.function.name
                        args = tc.function.arguments
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {"code": args}

                        self._log_event("backend_tool_dispatch_start", {
                            "agent_name": agent.name,
                            "turn": turn_idx,
                            "recipient": tool_name,
                        })

                        dispatch = self._dispatch_tool_call(
                            agent=agent,
                            state=state,
                            tool_name=tool_name,
                            tool_args=args,
                        )

                        tool_output = self._extract_dispatch_text(dispatch)
                        record = dispatch.record or {
                            "recipient": tool_name,
                            "request_text": args.get("code", ""),
                            "response_texts": [tool_output],
                        }
                        state.tool_calls.append(record)

                        messages.append({
                            "role": "tool",
                            "content": tool_output,
                            "name": tool_name,
                        })

                        self._log_event("backend_tool_dispatch_done", {
                            "agent_name": agent.name,
                            "turn": turn_idx,
                            "recipient": tool_name,
                        })

                    continue  # next inference turn

                # ---- Non-tool assistant response ----
                # No Harmony channel concept; treat every non-tool response as final.
                messages.append({"role": "assistant", "content": completion_text})

                # Create synthetic Harmony Message with channel="final" for callbacks
                synthetic_msg = Message(
                    author=Author(role=Role.ASSISTANT, name="assistant"),
                    content=[TextContent(text=completion_text)],
                ).with_channel("final")
                state.last_assistant_message = synthetic_msg

                if agent.terminate_on_message is not None:
                    maybe_signal = agent.terminate_on_message(
                        state, synthetic_msg, completion_text
                    )
                    if maybe_signal is not None:
                        termination_signal = maybe_signal
                        state.parsed_output = maybe_signal.parsed_output
                        state.termination_reason = maybe_signal.reason
                        break

                # stop_on_final_channel: stop after any non-tool response
                if agent.stop_on_final_channel:
                    state.termination_reason = "assistant_final"
                    break

            if state.termination_reason == "unknown":
                state.termination_reason = "max_turns_exhausted"

            # Session-end fallback (e.g. post-hoc boxed scan for solver)
            if state.parsed_output is None and agent.terminate_on_session_end is not None:
                maybe_signal = agent.terminate_on_session_end(state)
                if maybe_signal is not None:
                    state.parsed_output = maybe_signal.parsed_output
                    state.termination_reason = maybe_signal.reason
                    self._log_event("backend_termination_signal_session_end", {
                        "agent_name": agent.name,
                        "reason": state.termination_reason,
                    })

        except Exception as exc:
            exception_text = f"{type(exc).__name__}: {exc}"
            state.termination_reason = f"exception:{type(exc).__name__}"

        finished_ts = utcnow_iso()
        elapsed_ms = int((time.time() - t0) * 1000)
        state.finished_ts = finished_ts
        state.elapsed_ms = elapsed_ms
        state.raw_output = state.combined_completion_text()

        self._log_event("backend_session_done", {
            "agent_name": agent.name,
            "termination_reason": state.termination_reason,
            "elapsed_ms": elapsed_ms,
            "total_tokens": state.total_tokens,
            "turn_count": len(state.turns),
            "exception": exception_text,
        })

        prompt_text_initial = ""
        if len(messages) > 1:
            user_msg = messages[1]
            prompt_text_initial = user_msg.get("content") or ""

        return HarmonyRunResult(
            agent_name=agent.name,
            started_ts=started_ts,
            finished_ts=finished_ts,
            elapsed_ms=elapsed_ms,
            termination_reason=state.termination_reason,
            parsed_output=state.parsed_output,
            total_tokens=state.total_tokens,
            # No logprobs from Ollama; inf means attempt index is used as tiebreaker
            mean_entropy=float("inf"),
            prompt_token_ids_initial=[],
            prompt_text_initial=prompt_text_initial,
            turns=state.turns,
            tool_calls=state.tool_calls,
            full_completion_token_ids=[],
            full_conversation_token_ids=[],
            raw_output=state.raw_output,
            last_assistant_channel=None,
            last_assistant_recipient=None,
            exception=exception_text,
        )


__all__ = ["OllamaChatConfig", "OllamaChatBackend"]
