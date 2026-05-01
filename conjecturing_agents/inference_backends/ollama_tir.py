"""
Ollama TIR (Tool-Integrated Reasoning) inference backend.

Implements TIRBackend.chat_streaming() using Ollama's native chat API with:
  - Structured tool/function-call support (JSON schema definitions)
  - Extended thinking via think=True (for models that support it, e.g. qwen3)
  - Streaming with stop_event support, matching the pattern in OllamaBackend

The multi-turn tool dispatch loop is inherited from TIRBackend.run_session().
"""
from __future__ import annotations

import json
import time
import threading
from argparse import ArgumentParser
from dataclasses import dataclass
from transformers import AutoTokenizer
from typing import Any, Dict, Iterator, List, Optional

import ollama

from .tir_base import TIRBackend, TIRGenerationConfig, TIRStreamChunk, TIRToolCallSpec, TIRTokenCounter


@dataclass
class OllamaTIRConfig:
    # ------------------------------------------------------------------ #
    # Model / server
    # ------------------------------------------------------------------ #
    model: str = "qwen3.5"
    host: str = "http://localhost:11434"
    client_timeout: int = 960
    # If no streaming chunk is received within this many seconds, forcibly
    # close the HTTP stream.  Guards against half-open TCP connections where
    # the server stops sending but the client never times out.
    stream_inactivity_timeout: float = 60.0

    # ------------------------------------------------------------------ #
    # Enable extended thinking for models that support it (e.g. qwen3).
    # When True, the backend passes think=True to client.chat() and
    # thinking text is included in TIRStreamChunk.thinking.
    # ------------------------------------------------------------------ #
    think: bool = True

    # ------------------------------------------------------------------ #
    # Sampling parameters (passed to Ollama options).
    # ------------------------------------------------------------------ #
    top_k: int = -1           # -1 = disabled
    min_p: float = 0.0        # 0.0 = disabled
    presence_penalty: float = 0.0
    repeat_penalty: float = 1.0   # 1.0 = disabled

    # ------------------------------------------------------------------ #
    # Token counting (required for --limit-prover-tokens support).
    # When set, the backend uses the tokenizer's built-in chat template via
    # apply_chat_template(tokenize=True) — no separate Jinja template needed.
    # ------------------------------------------------------------------ #
    # Path to the HuggingFace tokenizer directory for this model
    # (e.g. "tokenizers/Qwen3.5-27B").
    tokenizer_path: str = ""

    # ------------------------------------------------------------------
    # CLI integration
    # ------------------------------------------------------------------

    @classmethod
    def add_cli_args(
        cls,
        parser: ArgumentParser,
        prefix: str = "tir",
        defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register CLI args for OllamaTIRConfig fields.

        Uses a SPLIT prefix convention matching the existing CLI:
        - Server connection fields use ``--{prefix}-ollama-{field}``
        - Model/sampling fields use ``--{prefix}-{field}``
        """
        d = defaults or {}
        pre = prefix
        dst = prefix.replace("-", "_")
        defs = cls()

        # Server connection fields → --{prefix}-ollama-{field}
        parser.add_argument(
            f"--{pre}-ollama-model",
            dest=f"{dst}_ollama_model",
            default=d.get("model", defs.model),
            help=f"Ollama model name for the {prefix} backend. Default: %(default)s.",
        )
        parser.add_argument(
            f"--{pre}-ollama-host",
            dest=f"{dst}_ollama_host",
            default=d.get("host", defs.host),
            help=f"Ollama server URL for the {prefix} backend. Default: %(default)s.",
        )
        parser.add_argument(
            f"--{pre}-ollama-client-timeout",
            dest=f"{dst}_ollama_client_timeout",
            type=int,
            default=d.get("client_timeout", defs.client_timeout),
            help=f"HTTP client timeout (seconds) for the {prefix} Ollama backend. Default: %(default)s.",
        )

        parser.add_argument(
            f"--{pre}-ollama-stream-inactivity-timeout",
            dest=f"{dst}_ollama_stream_inactivity_timeout",
            type=float,
            default=d.get("stream_inactivity_timeout", defs.stream_inactivity_timeout),
            help=(
                "Close the HTTP stream if no chunk arrives within this many "
                "seconds. Guards against hung connections. Default: %(default)s."
            ),
        )

        # Model/sampling fields → --{prefix}-{field}
        parser.add_argument(
            f"--{pre}-no-think",
            dest=f"{dst}_think",
            action="store_false",
            default=d.get("think", defs.think),
            help="Disable extended thinking for the model.",
        )
        parser.add_argument(
            f"--{pre}-top-k",
            dest=f"{dst}_top_k",
            type=int,
            default=d.get("top_k", defs.top_k),
            help="Top-k sampling. -1 = disabled. Default: %(default)s.",
        )
        parser.add_argument(
            f"--{pre}-min-p",
            dest=f"{dst}_min_p",
            type=float,
            default=d.get("min_p", defs.min_p),
            help="Min-p sampling. 0.0 = disabled. Default: %(default)s.",
        )
        parser.add_argument(
            f"--{pre}-presence-penalty",
            dest=f"{dst}_presence_penalty",
            type=float,
            default=d.get("presence_penalty", defs.presence_penalty),
            help="Presence penalty. 0.0 = no penalty. Default: %(default)s.",
        )
        parser.add_argument(
            f"--{pre}-repeat-penalty",
            dest=f"{dst}_repeat_penalty",
            type=float,
            default=d.get("repeat_penalty", defs.repeat_penalty),
            help="Repetition penalty. 1.0 = disabled. Default: %(default)s.",
        )
        parser.add_argument(
            f"--{pre}-tokenizer-path",
            dest=f"{dst}_tokenizer_path",
            default=d.get("tokenizer_path", defs.tokenizer_path),
            help=(
                "Path to the HuggingFace tokenizer directory for the model. "
                "Required when token counting is required."
            ),
        )

    @classmethod
    def from_parsed_args(
        cls,
        args: Any,
        prefix: str = "tir",
        **overrides: Any,
    ) -> OllamaTIRConfig:
        """Construct from an argparse namespace.

        Reads server fields from ``{prefix}_ollama_*`` attrs and model fields
        from ``{prefix}_*`` attrs, matching the split prefix convention.
        """
        dst = prefix.replace("-", "_")
        kwargs: Dict[str, Any] = {
            "model": getattr(args, f"{dst}_ollama_model"),
            "host": getattr(args, f"{dst}_ollama_host"),
            "client_timeout": getattr(args, f"{dst}_ollama_client_timeout"),
            "stream_inactivity_timeout": getattr(args, f"{dst}_ollama_stream_inactivity_timeout"),
            "think": getattr(args, f"{dst}_think"),
            "top_k": getattr(args, f"{dst}_top_k"),
            "min_p": getattr(args, f"{dst}_min_p"),
            "presence_penalty": getattr(args, f"{dst}_presence_penalty"),
            "repeat_penalty": getattr(args, f"{dst}_repeat_penalty"),
            "tokenizer_path": getattr(args, f"{dst}_tokenizer_path"),
        }
        kwargs.update(overrides)
        return cls(**kwargs)


class OllamaTIRBackend(TIRBackend):
    """
    Ollama-backed TIR backend.

    Streams one assistant turn per chat_streaming() call; the multi-turn
    tool dispatch loop is inherited from TIRBackend.run_session().

    Tool definitions must follow the OpenAI/Ollama JSON schema format:

        {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "...",
                "parameters": {
                    "type": "object",
                    "required": ["code"],
                    "properties": {
                        "code": {"type": "string", "description": "..."}
                    }
                }
            }
        }
    """

    def __init__(self, cfg: OllamaTIRConfig) -> None:
        self.cfg = cfg
        self._client: Optional[ollama.Client] = None
        self._tokenizer = None
        if cfg.tokenizer_path:
            counter = TIRTokenCounter(tokenizer_path=cfg.tokenizer_path)
            self.set_token_counter(counter.count)
            self.set_text_counter(counter.count_text)
            self._tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)

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
    # chat_streaming
    # ------------------------------------------------------------------

    def chat_streaming(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        cfg: TIRGenerationConfig,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[TIRStreamChunk]:
        """
        Stream one assistant turn using Ollama's chat API.

        Yields TIRStreamChunk objects as thinking/content arrive.  Any tool
        calls produced by the model are accumulated and emitted in a final
        TIRStreamChunk after the stream ends (tool calls come complete on the
        last stream chunk from Ollama).

        The stop_event is checked between every received chunk.
        """
        client = self._get_client()

        options: Dict[str, Any] = {
            "seed": cfg.seed,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "num_predict": cfg.max_tokens,
            "top_k": self.cfg.top_k,
            "min_p": self.cfg.min_p,
            "presence_penalty": self.cfg.presence_penalty,
            "repeat_penalty": self.cfg.repeat_penalty,
        }

        self._log_event("tir_chat_stream_start", {
            "model": self.cfg.model,
            "host": self.cfg.host,
            "message_count": len(messages),
            "tool_count": len(tools),
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "seed": cfg.seed,
        })
        t0 = time.time()

        if self._verbose:
            print(
                f"\n{'='*60}\n[TIR MESSAGES]\n{'='*60}",
                flush=True,
            )
            if self._tokenizer is not None:
                print(self._tokenizer.apply_chat_template(
                    messages,
                    tools=tools or None,
                    tokenize=False,
                    add_generation_prompt=True,
                ))
            else:
                print("[OllamaTIRBackend: verbose] WARN: No tokenizer availabe. Falling back.")
                for m in messages:
                    role = m.get("role", "?")
                    body = str(m.get("content") or "")
                    print(f"[{role}] {body}", flush=True)
                
            print(f"{'='*60}\n[TIR GENERATION]\n{'='*60}", flush=True)

        chat_kwargs: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "tools": tools if tools else None,
            "stream": True,
            "options": options,
        }
        if self.cfg.think:
            chat_kwargs["think"] = True

        tool_calls_accumulated: List[TIRToolCallSpec] = []
        first_chunk = True

        stream = client.chat(**chat_kwargs)

        # Inactivity watchdog: forcibly close the stream if no chunk arrives
        # within ``stream_inactivity_timeout`` seconds.  Guards against
        # half-open TCP connections where the server stops sending but the
        # client never times out.
        _stream_ref = stream

        def _close_stream() -> None:
            try:
                _stream_ref.close()
            except Exception:
                pass

        inactivity_timeout = self.cfg.stream_inactivity_timeout
        watchdog: Optional[threading.Timer] = None
        if inactivity_timeout > 0:
            watchdog = threading.Timer(inactivity_timeout, _close_stream)
            watchdog.daemon = True
            watchdog.start()

        last_printed_think = False

        try:
            for chunk in stream:
                # Reset the inactivity watchdog on every received chunk.
                if watchdog is not None:
                    watchdog.cancel()
                    watchdog = threading.Timer(inactivity_timeout, _close_stream)
                    watchdog.daemon = True
                    watchdog.start()

                if stop_event is not None and stop_event.is_set():
                    break

                if first_chunk:
                    self._log_event("tir_chat_stream_first_chunk", {
                        "elapsed_ms": int((time.time() - t0) * 1000),
                    })
                    first_chunk = False

                thinking_text = chunk.message.thinking or ""
                content_text = chunk.message.content or ""

                # Tool calls come complete (not streamed token-by-token); accumulate.
                if chunk.message.tool_calls:
                    for tc in chunk.message.tool_calls:
                        args = tc.function.arguments
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {"code": args}
                        # Ollama SDK does not expose tool call IDs; generate synthetic ones.
                        synthetic_id = f"call_{len(tool_calls_accumulated)}"
                        tool_calls_accumulated.append(
                            TIRToolCallSpec(id=synthetic_id, name=tc.function.name, arguments=args)
                        )

                if self._verbose:
                    if thinking_text:
                        if not last_printed_think:
                            last_printed_think = True
                            print('\n[think]\n')
                        print(f"{thinking_text}", end="", flush=True)
                    if content_text:
                        if last_printed_think:
                            last_printed_think = False
                            print('\n[\\think]\n')
                        print(content_text, end="", flush=True)

                if thinking_text or content_text:
                    yield TIRStreamChunk(thinking=thinking_text, content=content_text)
        finally:
            if watchdog is not None:
                watchdog.cancel()
            try:
                stream.close()
            except Exception:
                pass

        # Emit accumulated tool calls (if any) in a final chunk so the caller
        # can distinguish "turn with tool calls" from "turn with only text".
        if tool_calls_accumulated:
            if self._verbose:
                names = [tc.name for tc in tool_calls_accumulated]
                print(f"\n[tool_calls: {names}]", flush=True)
            yield TIRStreamChunk(tool_calls=tool_calls_accumulated)

        self._log_event("tir_chat_stream_done", {
            "elapsed_ms": int((time.time() - t0) * 1000),
            "tool_call_count": len(tool_calls_accumulated),
        })

        if self._verbose:
            print(f"\n{'='*60}", flush=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._client = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["OllamaTIRConfig", "OllamaTIRBackend"]
