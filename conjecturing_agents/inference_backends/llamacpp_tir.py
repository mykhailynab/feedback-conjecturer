"""
llama.cpp TIR (Tool-Integrated Reasoning) inference backend.

Implements TIRBackend.chat_streaming() using llama.cpp's OpenAI-compatible
/v1/chat/completions endpoint with:
  - Streaming tool call support (index-based delta accumulation)
  - Native reasoning_content support (like Deepseek API)
  - stop_event support for cancellation

Sampling parameters (top_k, min_p, repeat_penalty) are configured at
llama-server startup time, not per-request.  Only standard OpenAI params
(temperature, top_p, presence_penalty, max_tokens, seed) are sent per-request.

The multi-turn tool dispatch loop is inherited from TIRBackend.run_session().
"""
from __future__ import annotations

import json
import time
import threading
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import httpx
import openai

from .tir_base import TIRBackend, TIRGenerationConfig, TIRStreamChunk, TIRToolCallSpec, TIRTokenCounter


@dataclass
class LlamaCppTIRConfig:
    # ------------------------------------------------------------------ #
    # Server
    # ------------------------------------------------------------------ #
    base_url: str = "http://localhost:8001"  # without /v1
    model: str = ""
    api_key: str = "sk-no-key-required"
    client_timeout: int = 960

    # ------------------------------------------------------------------ #
    # Per-request sampling (standard OpenAI params).
    # Non-standard params (top_k, min_p, repeat_penalty) must be set at
    # llama-server startup and are NOT sent per-request.
    # ------------------------------------------------------------------ #
    presence_penalty: float = 0.0

    # ------------------------------------------------------------------ #
    # Token counting (required for --limit-prover-tokens support).
    # ------------------------------------------------------------------ #
    tokenizer_path: str = ""

    # ------------------------------------------------------------------
    # CLI integration
    # ------------------------------------------------------------------

    @classmethod
    def add_cli_args(
        cls,
        parser: ArgumentParser,
        prefix: str = "llamacpp",
        defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register CLI args for LlamaCppTIRConfig fields."""
        d = defaults or {}
        pre = prefix
        dst = prefix.replace("-", "_")
        defs = cls()

        parser.add_argument(
            f"--{pre}-base-url",
            dest=f"{dst}_base_url",
            default=d.get("base_url", defs.base_url),
            help="llama.cpp server URL (without /v1). Default: %(default)s.",
        )
        parser.add_argument(
            f"--{pre}-model",
            dest=f"{dst}_model",
            default=d.get("model", defs.model),
            help="Model name for llama.cpp requests.",
        )
        parser.add_argument(
            f"--{pre}-api-key",
            dest=f"{dst}_api_key",
            default=d.get("api_key", defs.api_key),
            help="API key for the llama.cpp endpoint.",
        )
        parser.add_argument(
            f"--{pre}-client-timeout",
            dest=f"{dst}_client_timeout",
            type=int,
            default=d.get("client_timeout", defs.client_timeout),
            help="HTTP client timeout in seconds. Default: %(default)s.",
        )
        parser.add_argument(
            f"--{pre}-presence-penalty",
            dest=f"{dst}_presence_penalty",
            type=float,
            default=d.get("presence_penalty", defs.presence_penalty),
            help="Presence penalty (per-request). Default: %(default)s.",
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
        prefix: str = "llamacpp",
        **overrides: Any,
    ) -> "LlamaCppTIRConfig":
        """Construct from an argparse namespace."""
        dst = prefix.replace("-", "_")
        kwargs: Dict[str, Any] = {
            "base_url": getattr(args, f"{dst}_base_url"),
            "model": getattr(args, f"{dst}_model"),
            "api_key": getattr(args, f"{dst}_api_key"),
            "client_timeout": getattr(args, f"{dst}_client_timeout"),
            "presence_penalty": getattr(args, f"{dst}_presence_penalty"),
            "tokenizer_path": getattr(args, f"{dst}_tokenizer_path"),
        }
        kwargs.update(overrides)
        return cls(**kwargs)


class LlamaCppTIRBackend(TIRBackend):
    """
    llama.cpp-backed TIR backend using the OpenAI-compatible chat endpoint.

    Streams one assistant turn per chat_streaming() call; the multi-turn
    tool dispatch loop is inherited from TIRBackend.run_session().

    Tool definitions must follow the OpenAI JSON schema format (same as
    OllamaTIRBackend).
    """

    def __init__(self, cfg: LlamaCppTIRConfig) -> None:
        self.cfg = cfg
        self._client: Optional[openai.OpenAI] = None
        self._tokenizer = None
        if cfg.tokenizer_path:
            from transformers import AutoTokenizer
            counter = TIRTokenCounter(tokenizer_path=cfg.tokenizer_path)
            self.set_token_counter(counter.count)
            self.set_text_counter(counter.count_text)
            self._tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)

    # ------------------------------------------------------------------
    # Client (lazy)
    # ------------------------------------------------------------------

    def _get_client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(
                base_url=self.cfg.base_url.rstrip("/") + "/v1",
                api_key=self.cfg.api_key,
                timeout=httpx.Timeout(self.cfg.client_timeout),
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
        Stream one assistant turn using llama.cpp's OpenAI-compatible endpoint.

        Yields TIRStreamChunk objects as reasoning/content arrive.  Tool calls
        are accumulated from streaming deltas (index-based) and emitted in a
        final TIRStreamChunk after the stream ends.

        The stop_event is checked between every received chunk.
        """
        client = self._get_client()

        self._log_event("tir_chat_stream_start", {
            "model": self.cfg.model,
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
                for m in messages:
                    role = m.get("role", "?")
                    body = str(m.get("content") or "")
                    print(f"[{role}] {body}", flush=True)
            print(f"{'='*60}\n[TIR GENERATION]\n{'='*60}", flush=True)

        create_kwargs: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": True,
            "max_tokens": cfg.max_tokens,
            "seed": cfg.seed,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "presence_penalty": self.cfg.presence_penalty,
        }
        if tools:
            create_kwargs["tools"] = tools

        # Tool call delta accumulation: {index: (id, name, arguments_str)}
        tc_accum: Dict[int, List] = {}  # index -> [id, name, args_str]
        first_chunk = True
        last_printed_think = False

        stream = client.chat.completions.create(**create_kwargs)

        for chunk in stream:
            if stop_event is not None and stop_event.is_set():
                break

            if first_chunk:
                self._log_event("tir_chat_stream_first_chunk", {
                    "elapsed_ms": int((time.time() - t0) * 1000),
                })
                first_chunk = False

            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                continue
            delta = choice.delta

            thinking_text = delta.model_extra.get("reasoning_content") or ""
            content_text = delta.content or ""

            # Accumulate tool call deltas by index.
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_accum:
                        tc_accum[idx] = ["", "", ""]  # [id, name, args_str]
                    if tc_delta.id:
                        tc_accum[idx][0] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc_accum[idx][1] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc_accum[idx][2] += tc_delta.function.arguments

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

        # Build final tool call specs from accumulated deltas.
        tool_calls_final: List[TIRToolCallSpec] = []
        for idx in sorted(tc_accum):
            tc_id, tc_name, tc_args_str = tc_accum[idx]
            try:
                args = json.loads(tc_args_str)
            except Exception:
                args = {"code": tc_args_str}
            tool_calls_final.append(
                TIRToolCallSpec(id=tc_id, name=tc_name, arguments=args)
            )

        if tool_calls_final:
            if self._verbose:
                names = [tc.name for tc in tool_calls_final]
                print(f"\n[tool_calls: {names}]", flush=True)
            yield TIRStreamChunk(tool_calls=tool_calls_final)

        self._log_event("tir_chat_stream_done", {
            "elapsed_ms": int((time.time() - t0) * 1000),
            "tool_call_count": len(tool_calls_final),
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


__all__ = ["LlamaCppTIRConfig", "LlamaCppTIRBackend"]
