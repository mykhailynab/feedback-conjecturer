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
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import ollama

from .tir_base import TIRBackend, TIRGenerationConfig, TIRStreamChunk, TIRToolCallSpec


@dataclass
class OllamaTIRConfig:
    # ------------------------------------------------------------------ #
    # Model / server
    # ------------------------------------------------------------------ #
    model: str = "qwen3.5"
    host: str = "http://localhost:11434"
    client_timeout: int = 960

    # ------------------------------------------------------------------ #
    # Enable extended thinking for models that support it (e.g. qwen3).
    # When True, the backend passes think=True to client.chat() and
    # thinking text is included in TIRStreamChunk.thinking.
    # ------------------------------------------------------------------ #
    think: bool = True


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
        }

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
            for m in messages:
                role = m.get("role", "?")
                body = str(m.get("content") or "")[:300]
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

        for chunk in stream:
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
                    tool_calls_accumulated.append(
                        TIRToolCallSpec(name=tc.function.name, arguments=args)
                    )

            if self._verbose:
                if thinking_text:
                    print(f"[think]{thinking_text}", end="", flush=True)
                if content_text:
                    print(content_text, end="", flush=True)

            if thinking_text or content_text:
                yield TIRStreamChunk(thinking=thinking_text, content=content_text)

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
