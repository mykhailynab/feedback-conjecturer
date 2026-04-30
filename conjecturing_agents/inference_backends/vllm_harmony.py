from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import threading
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from openai import OpenAI
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    SystemContent,
    ReasoningEffort,
    ToolNamespaceConfig,
    Author,
    Message,
    Role,
    TextContent,
    Conversation,
)

# ============================================================
# Types
# ============================================================

ToolConfigLike = Optional[Union[ToolNamespaceConfig, Sequence[ToolNamespaceConfig]]]


@dataclass
class TerminationSignal:
    """
    Returned by agent-supplied termination callbacks.
    """
    reason: str
    parsed_output: Any = None


@dataclass
class ToolInvocation:
    """
    Information passed to a recipient handler, e.g. for recipient='python'.
    """
    recipient: str
    message: Message
    state: HarmonySessionState
    backend: VLLMHarmonyBackend


@dataclass
class ToolDispatchResult:
    """
    Normalized result of a tool recipient handler.

    messages:
        Messages to append back into the Harmony conversation.

    record:
        Optional structured record to include in run_result.tool_calls.
    """
    messages: List[Message]
    record: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Generic helper utilities
# ============================================================

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_last_boxed_content(text: str) -> Optional[str]:
    """
    Extract the content of the last \\boxed{...} in text using brace balancing.
    """
    if not text:
        return None

    idx = text.rfind("\\boxed")
    while idx != -1:
        brace_start = text.find("{", idx)
        if brace_start == -1:
            idx = text.rfind("\\boxed", 0, idx)
            continue

        depth = 0
        i = brace_start
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[brace_start + 1 : i].strip()
            i += 1

        idx = text.rfind("\\boxed", 0, idx)

    return None


def extract_json_object_from_text(
    text: str,
    *,
    required_key: Optional[str] = None,
    validator: Optional[Callable[[Any], Any]] = None,
) -> Optional[Any]:
    """
    Robustly locate a JSON object inside arbitrary text.

    If required_key is provided, only candidates containing that key are concidered.
    If validator is provided, it should either return a normalized object or raise.
    """
    if not text:
        return None

    text = text.strip()

    def _accept(obj: Any) -> Optional[Any]:
        if required_key is not None:
            if not isinstance(obj, dict) or required_key not in obj:
                return None
        if validator is None:
            return obj
        try:
            return validator(obj)
        except Exception:
            return None

    # Fast path: full text is JSON
    try:
        obj = json.loads(text)
        accepted = _accept(obj)
        if accepted is not None:
            return accepted
    except Exception:
        pass

    # Regex candidates
    candidates = list(re.finditer(r"\{.*?\}", text, flags=re.DOTALL))
    for m in reversed(candidates):
        chunk = m.group(0)
        if required_key is not None and f'"{required_key}"' not in chunk:
            continue
        try:
            obj = json.loads(chunk)
        except Exception:
            continue
        accepted = _accept(obj)
        if accepted is not None:
            return accepted

    # Decoder-based search
    decoder = json.JSONDecoder()
    for i in reversed([i for i, ch in enumerate(text) if ch == "{"]):
        try:
            obj, _end = decoder.raw_decode(text[i:])
        except Exception:
            continue
        accepted = _accept(obj)
        if accepted is not None:
            return accepted

    return None


def compute_mean_entropy(logprobs_buffer: List[dict]) -> float:
    if not logprobs_buffer:
        return float("inf")

    total_entropy = 0.0
    token_count = 0

    for top_logprobs_dict in logprobs_buffer:
        if not isinstance(top_logprobs_dict, dict):
            continue
        if not top_logprobs_dict:
            continue

        token_entropy = 0.0
        for _token_str, log_prob in top_logprobs_dict.items():
            prob = math.exp(log_prob)
            if prob > 0:
                token_entropy -= prob * math.log2(prob)

        total_entropy += token_entropy
        token_count += 1

    if token_count == 0:
        return float("inf")

    return total_entropy / token_count


def make_tool_message(
    *,
    tool_name: str,
    output: str,
    channel: Optional[str] = None,
    recipient: str = "assistant",
) -> Message:
    """
    Convenience helper for recipient handlers.
    """
    content = TextContent(text=output)
    author = Author(role=Role.TOOL, name=tool_name)
    message = Message(author=author, content=[content]).with_recipient(recipient)
    if channel:
        message = message.with_channel(channel)
    return message


# ============================================================
# Configs
# ============================================================

@dataclass
class VLLMHarmonyBackendConfig:
    # Server / client
    served_model_name: str
    model_path: Optional[str] = None
    port: int = 8000
    api_key: str = "sk-local"
    host: str = "0.0.0.0"
    client_host: str = "0.0.0.0"
    session_timeout: int = 960
    server_timeout: int = 180

    # vLLM model serving
    stream_interval: int = 200
    context_tokens: int = 65536
    batch_size: int = 256
    dtype: str = "auto"
    kv_cache_dtype: str = "fp8_e4m3"
    gpu_memory_utilization: float = 0.96
    preload_workers: int = 8
    preload_model_weights: bool = True

    # Lifecycle
    manage_server: bool = True
    server_log_path: str = "vllm_server.log"
    extra_server_args: List[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.client_host}:{self.port}/v1"


from conjecturing_agents.inference_backends.raw_base import EventLogger
ChunkTerminationFn = Callable[["HarmonySessionState", str, str], Optional[TerminationSignal]]
MessageTerminationFn = Callable[["HarmonySessionState", Message, str], Optional[TerminationSignal]]
SessionTerminationFn = Callable[["HarmonySessionState"], Optional[TerminationSignal]]
ToolHandler = Callable[[ToolInvocation], Union[ToolDispatchResult, Message, Sequence[Message], str]]


@dataclass
class HarmonyAgentSpec:
    """
    All per-agent configuration lives here.

    The idea is that each concrete agent module creates one of these and passes
    it into VLLMHarmonyBackend.run_user_prompt(...) or run_conversation(...).
    """
    name: str
    system_prompt: str

    # Reasoning / decoding
    reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH
    temperature: float = 0.0
    min_p: float = 0.0
    top_logprobs: Optional[int] = None
    seed_offset: int = 0

    # Session limits
    max_turns: int = 64
    timeout_seconds: int = 120
    buffer_tokens: int = 512
    stream_text_window: int = 32

    # Tooling
    tool_configs: ToolConfigLike = None
    tool_handlers: Dict[str, ToolHandler] = field(default_factory=dict)

    # Agent-defined termination behavior
    terminate_on_chunk: Optional[ChunkTerminationFn] = None
    terminate_on_message: Optional[MessageTerminationFn] = None
    terminate_on_session_end: Optional[SessionTerminationFn] = None

    # Whether a final assistant message should end the session even if the
    # agent-specific message callback did not parse a result.
    stop_on_final_channel: bool = True


@dataclass
class HarmonySessionState:
    """
    Mutable state visible to tool handlers and termination callbacks.
    """
    agent: HarmonyAgentSpec
    conversation: Conversation
    backend: VLLMHarmonyBackend
    metadata: Dict[str, Any] = field(default_factory=dict)

    prompt_token_ids_initial: List[int] = field(default_factory=list)
    prompt_text_initial: str = ""

    turns: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    full_completion_token_ids: List[int] = field(default_factory=list)
    total_tokens: int = 0
    logprobs_buffer: List[dict] = field(default_factory=list)

    started_ts: str = ""
    finished_ts: str = ""
    elapsed_ms: int = 0

    termination_reason: str = "unknown"
    parsed_output: Any = None
    last_assistant_message: Optional[Message] = None
    raw_output: str = ""

    def combined_completion_text(self) -> str:
        return "\n".join(turn.get("completion_text", "") for turn in self.turns if turn.get("completion_text"))


@dataclass
class HarmonyRunResult:
    agent_name: str

    started_ts: str
    finished_ts: str
    elapsed_ms: int

    termination_reason: str
    parsed_output: Any

    total_tokens: int
    mean_entropy: float

    prompt_token_ids_initial: List[int]
    prompt_text_initial: str

    turns: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]

    full_completion_token_ids: List[int]
    full_conversation_token_ids: List[int]
    raw_output: str

    last_assistant_channel: Optional[str] = None
    last_assistant_recipient: Optional[str] = None
    exception: Optional[str] = None


# ============================================================
# Backend
# ============================================================

class VLLMHarmonyBackend:
    """
    Reusable backend for:
      - vLLM server lifecycle
      - OpenAI-compatible completion calls
      - Harmony message rendering/parsing
      - agent-configurable tool dispatch
      - agent-configurable termination logic

    Important design points:
      - No solver-specific logic
      - No checker-specific logic
      - No built-in Python or Lean execution assumptions
      - recipient handlers are injected by the agent
      - termination conditions are injected by the agent
    """

    def __init__(
            self,
            cfg: VLLMHarmonyBackendConfig,
            *,
            event_logger: Optional[EventLogger] = None,
        ):
        self.cfg = cfg
        self.event_logger = event_logger
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        self.server_process: Optional[subprocess.Popen] = None
        self.log_file = None
        self.client = OpenAI(
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
            timeout=self.cfg.session_timeout,
        )

        self._lifecycle_lock = threading.Lock()
        self._started = False

    def _log_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
    ) -> None:
        if self.event_logger is None:
            return
        try:
            self.event_logger.log_event(event_type, payload)
        except Exception:
            pass

    def _session_log_payload(
        self,
        *,
        agent: HarmonyAgentSpec,
        state: Optional[HarmonySessionState] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "agent_name": agent.name,
        }

        if state is not None:
            payload.update({
                "problem_id": state.metadata.get("problem_id"),
                "attempt": state.metadata.get("attempt"),
                "attempt_index": state.metadata.get("attempt_index"),
                "round": state.metadata.get("round"),
                "task_kind": state.metadata.get("task_kind"),
            })

        if extra:
            payload.update(extra)

        return payload

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                return

            if self.cfg.manage_server:
                if self.cfg.model_path is None:
                    raise ValueError("cfg.model_path must be set when manage_server=True")

                if self.cfg.preload_model_weights:
                    self._preload_model_weights()

                self.server_process = self._start_server()

            self._wait_for_server()
            self._started = True

    def close(self) -> None:
        with self._lifecycle_lock:
            if self.server_process is not None:
                try:
                    self.server_process.terminate()
                    self.server_process.wait(timeout=10)
                except Exception:
                    pass
                finally:
                    self.server_process = None

            if self.log_file is not None:
                try:
                    self.log_file.flush()
                    self.log_file.close()
                except Exception:
                    pass
                finally:
                    self.log_file = None

            self._started = False

    def __enter__(self) -> VLLMHarmonyBackend:
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
    # Server management
    # --------------------------------------------------------

    def _preload_model_weights(self) -> None:
        if self.cfg.model_path is None:
            return

        model_path = Path(self.cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        files_to_load: List[str] = []
        total_size = 0

        for root, _, files in os.walk(model_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    files_to_load.append(file_path)
                    total_size += os.path.getsize(file_path)

        if not files_to_load:
            return

        def _read_file(path: str) -> None:
            with open(path, "rb") as file_object:
                while file_object.read(1024 * 1024 * 1024):
                    pass

        from concurrent.futures import ThreadPoolExecutor

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.cfg.preload_workers) as executor:
            list(executor.map(_read_file, files_to_load))
        _elapsed = time.time() - start_time

    def _start_server(self) -> subprocess.Popen:
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(self.cfg.model_path),
            "--served-model-name",
            self.cfg.served_model_name,
            "--tensor-parallel-size",
            "1",
            "--max-num-seqs",
            str(self.cfg.batch_size),
            "--gpu-memory-utilization",
            str(self.cfg.gpu_memory_utilization),
            "--host",
            str(self.cfg.host),
            "--port",
            str(self.cfg.port),
            "--dtype",
            self.cfg.dtype,
            "--kv-cache-dtype",
            self.cfg.kv_cache_dtype,
            "--max-model-len",
            str(self.cfg.context_tokens),
            "--stream-interval",
            str(self.cfg.stream_interval),
            "--async-scheduling",
            "--disable-log-stats",
            "--enable-prefix-caching",
        ]
        cmd.extend(self.cfg.extra_server_args)

        self.log_file = open(self.cfg.server_log_path, "w", encoding="utf-8")
        return subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _wait_for_server(self) -> None:
        start_time = time.time()

        for _ in range(self.cfg.server_timeout):
            if self.server_process is not None:
                return_code = self.server_process.poll()
                if return_code is not None:
                    self.log_file.flush()
                    with open(self.cfg.server_log_path, "r", encoding="utf-8") as f:
                        logs = f.read()
                    raise RuntimeError(f"vLLM server died with code {return_code}. Logs:\n{logs}")

            try:
                self.client.models.list()
                return
            except Exception:
                time.sleep(1)

        raise RuntimeError(
            f"Timed out waiting for server readiness after {time.time() - start_time:.2f}s"
        )

    # --------------------------------------------------------
    # Conversation helpers
    # --------------------------------------------------------

    def get_system_content(
        self,
        *,
        system_prompt: str,
        reasoning_effort: ReasoningEffort,
        tool_configs: ToolConfigLike,
    ) -> SystemContent:
        system_content = (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=reasoning_effort)
        )

        if tool_configs is None:
            return system_content

        if isinstance(tool_configs, ToolNamespaceConfig):
            return system_content.with_tools(tool_configs)

        for tool_cfg in tool_configs:
            system_content = system_content.with_tools(tool_cfg)

        return system_content

    def apply_chat_template(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        reasoning_effort: ReasoningEffort,
        tool_configs: ToolConfigLike,
    ) -> List[Message]:
        system_content = self.get_system_content(
            system_prompt=system_prompt,
            reasoning_effort=reasoning_effort,
            tool_configs=tool_configs,
        )
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
        user_message = Message.from_role_and_content(Role.USER, user_prompt)
        return [system_message, user_message]

    def build_conversation_from_user_prompt(
        self,
        *,
        agent: HarmonyAgentSpec,
        user_prompt: str,
    ) -> Conversation:
        messages = self.apply_chat_template(
            system_prompt=agent.system_prompt,
            user_prompt=user_prompt,
            reasoning_effort=agent.reasoning_effort,
            tool_configs=agent.tool_configs,
        )
        return Conversation.from_messages(messages)

    def _decode_ids(self, ids: List[int]) -> str:
        return self.encoding.decode_utf8(ids)

    # --------------------------------------------------------
    # Tool dispatch
    # --------------------------------------------------------

    def _normalize_tool_dispatch_output(
        self,
        *,
        recipient: str,
        original_message: Message,
        raw_result: Union[ToolDispatchResult, Message, Sequence[Message], str],
    ) -> ToolDispatchResult:
        if isinstance(raw_result, ToolDispatchResult):
            return raw_result

        if isinstance(raw_result, Message):
            return ToolDispatchResult(messages=[raw_result])

        if isinstance(raw_result, str):
            return ToolDispatchResult(
                messages=[
                    make_tool_message(
                        tool_name=recipient,
                        output=raw_result,
                        channel=original_message.channel,
                    )
                ]
            )

        if isinstance(raw_result, Sequence):
            messages = list(raw_result)
            if not all(isinstance(m, Message) for m in messages):
                raise TypeError("Tool handler returned a Sequence, but not all elements are Message")
            return ToolDispatchResult(messages=messages)

        raise TypeError(
            f"Unsupported tool handler return type for recipient={recipient!r}: {type(raw_result).__name__}"
        )

    def _dispatch_tool_message(
        self,
        *,
        agent: HarmonyAgentSpec,
        state: HarmonySessionState,
        message: Message,
    ) -> ToolDispatchResult:
        recipient = message.recipient
        if not recipient:
            raise ValueError("Tried to dispatch a tool message without recipient")

        handler = agent.tool_handlers.get(recipient)
        if handler is None:
            raise KeyError(f"No tool handler registered for recipient={recipient!r}")

        invocation = ToolInvocation(
            recipient=recipient,
            message=message,
            state=state,
            backend=self,
        )
        raw_result = handler(invocation)
        dispatch = self._normalize_tool_dispatch_output(
            recipient=recipient,
            original_message=message,
            raw_result=raw_result,
        )

        if not dispatch.record:
            dispatch.record = {
                "recipient": recipient,
                "request_text": self.get_message_text(message),
                "response_texts": [self.get_message_text(m) for m in dispatch.messages],
            }

        return dispatch

    @staticmethod
    def get_message_text(message: Message) -> str:
        if not message.content:
            return ""
        chunks = []
        for item in message.content:
            if not isinstance(item, TextContent):
                continue
            chunks.append(item.text)
        return "\n".join(chunks)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    def run_user_prompt(
        self,
        *,
        agent: HarmonyAgentSpec,
        user_prompt: str,
        seed: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HarmonyRunResult:
        conversation = self.build_conversation_from_user_prompt(agent=agent, user_prompt=user_prompt)
        return self.run_conversation(
            agent=agent,
            conversation=conversation,
            seed=seed,
            metadata=metadata,
        )

    def run_conversation(
        self,
        *,
        agent: HarmonyAgentSpec,
        conversation: Conversation,
        seed: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HarmonyRunResult:
        if not self._started:
            self.start()

        started_ts = utcnow_iso()
        t0 = time.time()
        deadline = t0 + float(agent.timeout_seconds)

        state = HarmonySessionState(
            agent=agent,
            conversation=conversation,
            backend=self,
            metadata=dict(metadata or {}),
            started_ts=started_ts,
        )

        self._log_event("backend_session_start",
            self._session_log_payload(agent=agent, state=state, extra={
                "max_turns": agent.max_turns,
                "timeout_seconds": agent.timeout_seconds,
            })
        )

        exception_text: Optional[str] = None

        def _log_event_and_extra(event_type: str, extra: Dict[str, Any]):
            self._log_event(event_type, self._session_log_payload(agent=agent, state=state, extra=extra))

        def _log_turn_stop(extra: Dict[str, Any]):
            extra.update({"turn": turn_idx})
            _log_event_and_extra("backend_turn_stop", extra)

        try:
            prompt_token_ids_initial = list(
                self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
            )
            state.prompt_token_ids_initial = prompt_token_ids_initial
            state.prompt_text_initial = (
                self._decode_ids(prompt_token_ids_initial) if prompt_token_ids_initial else ""
            )

            termination_signal: Optional[TerminationSignal] = None

            for turn_idx in range(agent.max_turns):
                _log_event_and_extra("backend_turn_start", extra={"turn": turn_idx})
                if time.time() > deadline:
                    state.termination_reason = "deadline_exceeded"
                    _log_turn_stop({"reason": state.termination_reason})
                    break

                prompt_ids = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)

                if max_tokens < agent.buffer_tokens:
                    state.termination_reason = "context_exhausted"
                    _log_turn_stop({
                        "reason": state.termination_reason,
                        "remaining_tokens": max_tokens,
                    })
                    break

                token_buffer: List[int] = []
                completion_text_parts: List[str] = []
                stream_interrupted = False
                stream = None

                try:
                    _log_event_and_extra("backend_llm_request_start", extra={
                        "turn": turn_idx,
                        "prompt_tokens": len(prompt_ids),
                        "max_tokens": max_tokens,
                        "seed": int((seed + agent.seed_offset) ** 2),
                    })
                    stream = self.client.completions.create(
                        model=self.cfg.served_model_name,
                        temperature=agent.temperature,
                        logprobs=agent.top_logprobs,
                        max_tokens=max_tokens,
                        prompt=prompt_ids,
                        seed=int((seed + agent.seed_offset) ** 2),
                        stream=True,
                        extra_body={
                            "min_p": agent.min_p,
                            "stop_token_ids": self.stop_token_ids,
                            "return_token_ids": True,
                        },
                    )
                    _log_event_and_extra("backend_llm_request_created", extra={"turn": turn_idx})

                    first_chunk_seen = False

                    for chunk in stream:
                        if not first_chunk_seen:
                            first_chunk_seen = True
                            _log_event_and_extra("backend_llm_first_chunk", extra={"turn": turn_idx})

                        if time.time() > deadline:
                            state.termination_reason = "deadline_exceeded"
                            stream_interrupted = True
                            _log_event_and_extra("backend_llm_stream_stop", extra={
                                "turn": turn_idx,
                                "reason": "deadline_exceeded_during_stream",
                                "tokens_so_far": len(token_buffer),
                            })
                            break

                        choice = chunk.choices[0]
                        new_tokens = choice.token_ids or []
                        new_text = choice.text or ""

                        if new_tokens:
                            token_buffer.extend(new_tokens)
                            state.full_completion_token_ids.extend(new_tokens)
                            state.total_tokens += len(new_tokens)

                        if new_text:
                            completion_text_parts.append(new_text)

                        chunk_logprobs = choice.logprobs
                        if chunk_logprobs is not None and chunk_logprobs.top_logprobs:
                            state.logprobs_buffer.extend(chunk_logprobs.top_logprobs)

                        if agent.terminate_on_chunk is not None and (new_text or new_tokens):
                            recent_text = "".join(completion_text_parts[-agent.stream_text_window :])
                            maybe_signal = agent.terminate_on_chunk(state, recent_text, new_text)
                            if maybe_signal is not None:
                                termination_signal = maybe_signal
                                state.parsed_output = maybe_signal.parsed_output
                                state.termination_reason = maybe_signal.reason
                                _log_event_and_extra("backend_llm_stream_stop", extra={
                                    "turn": turn_idx,
                                    "reason": state.termination_reason,
                                    "tokens_so_far": len(token_buffer),
                                })
                                break

                finally:
                    try:
                        if stream is not None:
                            stream.close()
                    except Exception:
                        pass

                _log_event_and_extra("backend_llm_stream_done", extra={
                    "turn": turn_idx,
                    "first_chunk_seen": first_chunk_seen,
                    "completion_tokens": len(token_buffer),
                })

                completion_text = self._decode_ids(token_buffer) if token_buffer else ""
                state.turns.append(
                    {
                        "turn": turn_idx,
                        "completion_token_ids": token_buffer,
                        "completion_text": completion_text,
                    }
                )

                if termination_signal is not None:
                    _log_turn_stop({"reason": state.termination_reason})
                    break

                if stream_interrupted:
                    _log_turn_stop({"reason": state.termination_reason})
                    break

                if not token_buffer:
                    if state.termination_reason == "unknown":
                        state.termination_reason = "no_tokens"
                    _log_turn_stop({"reason": state.termination_reason})
                    break

                new_messages = self.encoding.parse_messages_from_completion_tokens(
                    token_buffer, Role.ASSISTANT
                )
                conversation.messages.extend(new_messages)
                last_message = new_messages[-1]
                state.last_assistant_message = last_message

                _log_event_and_extra("backend_turn_parsed", extra={
                    "turn": turn_idx,
                    "message_channel": last_message.channel,
                    "message_recipient": last_message.recipient,
                    "message_text_len": len(self.get_message_text(last_message)),
                })

                if last_message.recipient:
                    _log_event_and_extra("backend_tool_dispatch_start", extra={
                        "turn": turn_idx,
                        "recipient": last_message.recipient,
                    })
                    dispatch = self._dispatch_tool_message(
                        agent=agent,
                        state=state,
                        message=last_message,
                    )
                    state.tool_calls.append(dispatch.record)
                    conversation.messages.extend(dispatch.messages)
                    _log_event_and_extra("backend_tool_dispatch_done", extra={
                        "turn": turn_idx,
                        "recipient": last_message.recipient,
                        "tool_response_count": len(dispatch.messages),
                    })
                    continue

                if agent.terminate_on_message is not None:
                    maybe_signal = agent.terminate_on_message(state, last_message, completion_text)
                    if maybe_signal is not None:
                        termination_signal = maybe_signal
                        state.parsed_output = maybe_signal.parsed_output
                        state.termination_reason = maybe_signal.reason
                        _log_event_and_extra("backend_termination_signal_message", extra={
                            "turn": turn_idx,
                            "reason": state.termination_reason,
                        })
                        break

                if agent.stop_on_final_channel and last_message.channel == "final":
                    state.termination_reason = "assistant_final"
                    _log_turn_stop({"reason": state.termination_reason})
                    break

            if state.termination_reason == "unknown":
                state.termination_reason = "max_turns_exhausted"

            if state.parsed_output is None and agent.terminate_on_session_end is not None:
                maybe_signal = agent.terminate_on_session_end(state)
                if maybe_signal is not None:
                    state.parsed_output = maybe_signal.parsed_output
                    state.termination_reason = maybe_signal.reason
                    _log_event_and_extra("backend_termination_signal_session_end", extra={
                        "reason": state.termination_reason,
                    })

        except Exception as exc:
            exception_text = f"{type(exc).__name__}: {exc}"
            state.termination_reason = f"exception:{type(exc).__name__}"
            _log_event_and_extra("backend_session_exception", extra={
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            })

        finished_ts = utcnow_iso()
        elapsed_ms = int((time.time() - t0) * 1000)

        state.finished_ts = finished_ts
        state.elapsed_ms = elapsed_ms
        state.raw_output = state.combined_completion_text()

        last_channel = None
        last_recipient = None
        if state.last_assistant_message is not None:
            last_channel = state.last_assistant_message.channel
            last_recipient = state.last_assistant_message.recipient

        _log_event_and_extra("backend_session_done", extra={
            "termination_reason": state.termination_reason,
            "elapsed_ms": elapsed_ms,
            "total_tokens": state.total_tokens,
            "turn_count": len(state.turns),
            "exception": exception_text,
        })

        return HarmonyRunResult(
            agent_name=agent.name,
            started_ts=started_ts,
            finished_ts=finished_ts,
            elapsed_ms=elapsed_ms,
            termination_reason=state.termination_reason,
            parsed_output=state.parsed_output,
            total_tokens=state.total_tokens,
            mean_entropy=compute_mean_entropy(state.logprobs_buffer),
            prompt_token_ids_initial=state.prompt_token_ids_initial,
            prompt_text_initial=state.prompt_text_initial,
            turns=state.turns,
            tool_calls=state.tool_calls,
            full_completion_token_ids=state.full_completion_token_ids,
            full_conversation_token_ids=(
                self.encoding.render_conversation(conversation) if conversation is not None else []
            ),
            raw_output=state.raw_output,
            last_assistant_channel=last_channel,
            last_assistant_recipient=last_recipient,
            exception=exception_text,
        )
