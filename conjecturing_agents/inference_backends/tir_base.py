"""
Common types and abstract base for Tool-Integrated Reasoning (TIR) backends.

TIR backends differ from RawBackend in that they:
  - Accept OpenAI-style message lists (system/user/assistant/tool) directly
    rather than pre-rendered prompt strings.
  - Support structured tool call dispatch via tool definitions and handlers.
  - Manage the multi-turn conversation loop internally via run_session().

Design contract
---------------
- chat_streaming() is the single abstract method that subclasses must implement.
  It streams one assistant turn and yields TIRStreamChunk objects.
- run_session() is a concrete method in this base class that implements the
  full multi-turn tool dispatch loop on top of chat_streaming().
- Tool handlers are plain callables: (tool_name: str, arguments: dict) -> str.
  No Harmony types are required.
"""
from __future__ import annotations

import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterator, List, Optional

from .raw_backend import EventLoggerFn


# ============================================================
# Config / result types
# ============================================================

@dataclass
class TIRGenerationConfig:
    """Generation parameters for a single TIR session."""

    max_tokens: int = 8192
    temperature: float = 0.6
    top_p: float = 0.95
    seed: int = 0
    max_turns: int = 32
    timeout_seconds: float = 300.0
    # Stop the session (marking it incomplete) when the rendered prompt reaches
    # this many tokens.  0 = unlimited.  Use with --continue for progressive
    # budget runs (same semantics as GoedelProverAgent token_limit).
    token_limit: int = 0


@dataclass
class TIRToolCallSpec:
    """A single tool call requested by the model in one streaming turn."""

    name: str
    arguments: Dict[str, Any]


@dataclass
class TIRStreamChunk:
    """
    A single chunk yielded by chat_streaming().

    Fields are additive — the caller accumulates them across chunks:
      - thinking: partial thinking text (empty if model doesn't think aloud)
      - content: partial response text
      - tool_calls: tool calls from this chunk (typically only in the final chunk)
    """

    thinking: str = ""
    content: str = ""
    tool_calls: List[TIRToolCallSpec] = field(default_factory=list)


@dataclass
class TIRTurnRecord:
    """Record of one assistant turn (thinking, response text, tool calls dispatched)."""

    turn: int
    thinking: str
    content: str
    # Each entry: {"name": ..., "arguments": ..., "result": ...}
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TIRSessionResult:
    """Result returned by run_session()."""

    final_text: str
    turns: List[TIRTurnRecord]
    termination_reason: str
    elapsed_ms: int
    exception: Optional[str] = None
    # Optional throughput fields (populated by backends that track tokens)
    total_output_tokens: Optional[int] = None
    total_generation_ms: Optional[int] = None
    # Set to True when the session was stopped by the token limit.
    # messages_at_cutoff holds the full message list at the point of cutoff
    # so the caller can resume the session in a later run.
    incomplete: bool = False
    messages_at_cutoff: List[Dict[str, Any]] = field(default_factory=list)
    # Partial assistant turn in progress when the stream was interrupted
    # (thinking + content accumulated before the cutoff fired).  Separate from
    # messages_at_cutoff so callers that build a resume conversation do not
    # need to strip it out themselves.
    partial_assistant_turn: Optional[Dict[str, Any]] = None


# Callable type for a tool handler: receives tool name + arguments, returns string.
TIRToolHandler = Callable[[str, Dict[str, Any]], str]


# ============================================================
# Abstract base
# ============================================================

class TIRBackend(ABC):
    """
    Abstract base for Tool-Integrated Reasoning inference backends.

    Subclasses implement chat_streaming(); the multi-turn tool dispatch loop
    (run_session) is provided here and can be used without override.

    Implementors:
      - OllamaTIRBackend  (ollama_tir.py)
    """

    _verbose: bool = False
    _event_logger: Optional[EventLoggerFn] = None
    _token_counter_fn: Optional[Callable[..., int]] = None
    _text_counter_fn: Optional[Callable[[str], int]] = None

    def set_verbose(self, enabled: bool) -> None:
        """Enable real-time prompt+token printing to stdout."""
        self._verbose = enabled

    def set_event_logger(self, fn: EventLoggerFn) -> None:
        """Register a callable that receives (event_type, payload) dicts."""
        self._event_logger = fn

    def set_token_counter(
        self,
        fn: Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], int],
    ) -> None:
        """Register a callable for counting tokens in a (messages, tools) pair."""
        self._token_counter_fn = fn

    def set_text_counter(self, fn: Callable[[str], int]) -> None:
        """Register a callable for counting tokens in a raw text string.

        Used for per-chunk budget checks during streaming.
        """
        self._text_counter_fn = fn

    def count_tokens_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> int:
        """Return token count of the rendered prompt.  Returns 0 if no counter is set."""
        if self._token_counter_fn is None:
            return 0
        return self._token_counter_fn(messages, tools)

    def count_tokens_text(self, text: str) -> int:
        """Return token count of a raw text string.  Returns 0 if no counter is set."""
        if self._text_counter_fn is None:
            return 0
        return self._text_counter_fn(text)

    def _log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self._event_logger is not None:
            self._event_logger(event_type, payload)

    # ------------------------------------------------------------------
    # Abstract: one streaming turn
    # ------------------------------------------------------------------

    @abstractmethod
    def chat_streaming(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        cfg: TIRGenerationConfig,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[TIRStreamChunk]:
        """
        Stream one assistant turn given a message history and tool definitions.

        Yields TIRStreamChunk objects as content/thinking arrive.  Tool calls,
        when present, appear in TIRStreamChunk.tool_calls (typically emitted
        in a final chunk after all text content has been streamed).

        Implementations should check stop_event between chunks and stop
        iterating as soon as it is set.
        """

    # ------------------------------------------------------------------
    # Concrete: full multi-turn session loop
    # ------------------------------------------------------------------

    def run_session(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_handlers: Dict[str, TIRToolHandler],
        cfg: TIRGenerationConfig,
        stop_event: Optional[threading.Event] = None,
    ) -> TIRSessionResult:
        """
        Run a full multi-turn tool-integrated reasoning session.

        Loop:
          1. Call chat_streaming() for the current message history.
          2. Accumulate thinking / content / tool_calls from the stream.
          3. If the model requested tool calls, dispatch each via tool_handlers,
             append results to messages, and continue to the next turn.
          4. If no tool calls in the response, treat it as the final answer.
          5. Respect stop_event (checked between chunks) and cfg.timeout_seconds.

        The seed is incremented per turn (like existing Ollama backends) so
        each turn gets a different but deterministic random state.
        """
        messages = list(messages)   # copy; don't mutate the caller's list
        t0 = time.time()
        deadline = t0 + cfg.timeout_seconds

        turns: List[TIRTurnRecord] = []
        termination_reason = "unknown"
        exception_text: Optional[str] = None
        final_text = ""
        incomplete = False
        messages_at_cutoff: List[Dict[str, Any]] = []
        partial_assistant_turn: Optional[Dict[str, Any]] = None

        self._log_event("tir_session_start", {
            "max_turns": cfg.max_turns,
            "timeout_seconds": cfg.timeout_seconds,
            "tool_count": len(tools),
        })

        try:
            for turn_idx in range(cfg.max_turns):
                if time.time() > deadline:
                    termination_reason = "deadline_exceeded"
                    break

                if stop_event is not None and stop_event.is_set():
                    termination_reason = "stop_event"
                    break

                # Token limit check — before submitting the next LLM call.
                if cfg.token_limit > 0:
                    current_tokens = self.count_tokens_messages(messages, tools)
                    self._log_event("tir_token_count", {
                        "turn": turn_idx,
                        "tokens": current_tokens,
                        "limit": cfg.token_limit,
                    })
                    if current_tokens >= cfg.token_limit:
                        termination_reason = "token_limit"
                        incomplete = True
                        messages_at_cutoff = list(messages)
                        break

                # Increment seed per turn for diversity across turns.
                turn_cfg = replace(cfg, seed=(cfg.seed + turn_idx) & 0x7FFFFFFF)

                self._log_event("tir_turn_start", {"turn": turn_idx})

                # Snapshot prompt token count once per turn for per-chunk budget checks.
                turn_prompt_tokens = (
                    self.count_tokens_messages(messages, tools)
                    if cfg.token_limit > 0
                    else 0
                )

                thinking = ""
                content = ""
                tool_call_specs: List[TIRToolCallSpec] = []
                stream_interrupted = False
                chars_since_recount = 0

                try:
                    for chunk in self.chat_streaming(
                        messages, tools, turn_cfg, stop_event=stop_event
                    ):
                        if time.time() > deadline:
                            termination_reason = "deadline_exceeded"
                            stream_interrupted = True
                            break
                        if stop_event is not None and stop_event.is_set():
                            termination_reason = "stop_event"
                            stream_interrupted = True
                            break
                        thinking += chunk.thinking
                        content += chunk.content
                        tool_call_specs.extend(chunk.tool_calls)
                        # Per-chunk token budget check (every ~200 chars).
                        if cfg.token_limit > 0:
                            chars_since_recount += len(chunk.thinking) + len(chunk.content)
                            if chars_since_recount >= 200:
                                chars_since_recount = 0
                                gen_tokens = self.count_tokens_text(thinking + content)
                                if turn_prompt_tokens + gen_tokens >= cfg.token_limit:
                                    termination_reason = "token_limit"
                                    incomplete = True
                                    messages_at_cutoff = list(messages)
                                    stream_interrupted = True
                                    break

                except Exception as exc:
                    exception_text = f"{type(exc).__name__}: {exc}"
                    termination_reason = f"exception:{type(exc).__name__}"
                    break

                if stream_interrupted and termination_reason == "token_limit":
                    # Record the partial assistant turn for logging/debugging.
                    # Stored separately from messages_at_cutoff so callers that
                    # build a resume conversation do not need to strip it out.
                    if thinking or content:
                        partial_assistant_turn = {
                            "role": "assistant",
                            "content": content or None,
                        }
                        if thinking:
                            partial_assistant_turn["thinking"] = thinking
                    break

                if not content and not tool_call_specs:
                    termination_reason = "no_tokens"
                    break

                turn_record = TIRTurnRecord(
                    turn=turn_idx,
                    thinking=thinking,
                    content=content,
                )

                self._log_event("tir_turn_done", {
                    "turn": turn_idx,
                    "content_chars": len(content),
                    "thinking_chars": len(thinking),
                    "tool_call_count": len(tool_call_specs),
                })

                if tool_call_specs:
                    # Build assistant message with tool calls.
                    # Include thinking if present (Ollama preserves it across turns).
                    assistant_msg: Dict[str, Any] = {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": [
                            {
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments,
                                }
                            }
                            for tc in tool_call_specs
                        ],
                    }
                    if thinking:
                        assistant_msg["thinking"] = thinking
                    messages.append(assistant_msg)

                    # Dispatch each tool call.
                    for tc in tool_call_specs:
                        handler = tool_handlers.get(tc.name)
                        if handler is None:
                            result_text = f"[ERROR] No handler registered for tool '{tc.name}'"
                        else:
                            try:
                                result_text = handler(tc.name, tc.arguments)
                            except Exception as exc:
                                result_text = (
                                    f"[ERROR] Tool '{tc.name}' raised "
                                    f"{type(exc).__name__}: {exc}"
                                )

                        self._log_event("tir_tool_call", {
                            "turn": turn_idx,
                            "tool_name": tc.name,
                            "result_chars": len(result_text),
                        })

                        turn_record.tool_calls.append({
                            "name": tc.name,
                            "arguments": tc.arguments,
                            "result": result_text,
                        })

                        messages.append({
                            "role": "tool",
                            "name": tc.name,
                            "content": result_text,
                        })

                    turns.append(turn_record)
                    continue  # next inference turn

                # No tool calls → final answer.
                turns.append(turn_record)
                final_text = content
                termination_reason = "final_answer"
                break

            if termination_reason == "unknown":
                termination_reason = "max_turns_exhausted"
                if turns:
                    final_text = turns[-1].content

        except Exception as exc:
            exception_text = f"{type(exc).__name__}: {exc}"
            termination_reason = f"exception:{type(exc).__name__}"

        elapsed_ms = int((time.time() - t0) * 1000)
        self._log_event("tir_session_done", {
            "termination_reason": termination_reason,
            "elapsed_ms": elapsed_ms,
            "turns": len(turns),
            "exception": exception_text,
        })

        return TIRSessionResult(
            final_text=final_text,
            turns=turns,
            termination_reason=termination_reason,
            elapsed_ms=elapsed_ms,
            exception=exception_text,
            incomplete=incomplete,
            messages_at_cutoff=messages_at_cutoff,
            partial_assistant_turn=partial_assistant_turn,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release resources. Override as needed."""

    def __enter__(self) -> "TIRBackend":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ============================================================
# Token counting helpers
# ============================================================

class TIRTokenCounter:
    """Counts tokens using a HuggingFace tokenizer's built-in chat template.

    Uses ``tokenizer.apply_chat_template(messages, tools=tools,
    tokenize=True, add_generation_prompt=True)`` — renders and tokenizes the
    conversation in one call.  No separate Jinja2 template file is needed.

    The ``count`` method is thread-safe (the tokenizer is read-only after
    construction).

    Usage::

        counter = TIRTokenCounter(tokenizer_path="tokenizers/Qwen3.5-27B")
        backend.set_token_counter(counter.count)
        backend.set_text_counter(counter.count_text)
    """

    def __init__(self, tokenizer_path: str) -> None:
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def count(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> int:
        """Return the number of tokens in the rendered prompt."""
        ids = self._tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(ids)

    def count_text(self, text: str) -> int:
        """Return the token count for a raw text string."""
        return len(self._tokenizer.encode(text, add_special_tokens=False))


__all__ = [
    "EventLoggerFn",
    "TIRGenerationConfig",
    "TIRToolCallSpec",
    "TIRStreamChunk",
    "TIRTurnRecord",
    "TIRSessionResult",
    "TIRToolHandler",
    "TIRBackend",
    "TIRTokenCounter",
]
