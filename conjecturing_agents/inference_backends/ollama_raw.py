"""
Ollama inference backend for raw rendered prompt strings.

Wraps the ``ollama`` Python client (https://github.com/ollama/ollama-python).
Like VLLMRawBackend, it accepts a pre-rendered prompt string — no message-
format abstraction is applied here.

Token counting requires a HF tokenizer (``tokenizer_path``).  Ollama does
report ``prompt_eval_count`` / ``eval_count`` in its responses, but only
after generation, making it unsuitable for pre-flight context budget checks.
"""
from __future__ import annotations

import time
import ollama
import threading
from argparse import ArgumentParser
from typing import Any, Dict, Iterator, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer

from .raw_base import RawBackend, RawGenerationConfig, RawGenerationResult


@dataclass
class OllamaConfig:
    # ------------------------------------------------------------------ #
    # Model / server
    # ------------------------------------------------------------------ #
    model: str = "goedel-v2"
    host: str = "http://localhost:11434"
    client_timeout: int = 240

    # ------------------------------------------------------------------ #
    # HF tokenizer path for exact pre-flight token counting.
    # Required for context-budget enforcement before generation.
    # ------------------------------------------------------------------ #
    tokenizer_path: str = ""

    # ------------------------------------------------------------------
    # CLI integration
    # ------------------------------------------------------------------

    @classmethod
    def add_cli_args(
        cls,
        parser: ArgumentParser,
        prefix: str = "ollama",
        defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register CLI args for OllamaConfig fields.

        ``tokenizer_path`` is excluded — it is typically wired from the agent
        config (e.g. GoedelProverConfig.tokenizer_path) in factory helpers.
        """
        d = defaults or {}
        pre = prefix
        dst = prefix.replace("-", "_")
        defs = cls()

        parser.add_argument(
            f"--{pre}-model",
            dest=f"{dst}_model",
            default=d.get("model", defs.model),
            help="Ollama model name.",
        )
        parser.add_argument(
            f"--{pre}-host",
            dest=f"{dst}_host",
            default=d.get("host", defs.host),
            help="Ollama server URL.",
        )
        parser.add_argument(
            f"--{pre}-client-timeout",
            dest=f"{dst}_client_timeout",
            type=int,
            default=d.get("client_timeout", defs.client_timeout),
            help="HTTP client timeout in seconds for Ollama requests.",
        )

    @classmethod
    def from_parsed_args(
        cls,
        args: Any,
        prefix: str = "ollama",
        **overrides: Any,
    ) -> "OllamaConfig":
        """Construct from an argparse namespace.

        ``tokenizer_path`` defaults to "" unless provided via ``overrides``.
        """
        dst = prefix.replace("-", "_")
        kwargs: Dict[str, Any] = {
            "model": getattr(args, f"{dst}_model"),
            "host": getattr(args, f"{dst}_host"),
            "client_timeout": getattr(args, f"{dst}_client_timeout"),
        }
        kwargs.update(overrides)
        return cls(**kwargs)


class OllamaBackend(RawBackend):
    """
    Inference backend using the Ollama Python client.

    - Accepts a raw rendered prompt string.
    - Streams via ``ollama.Client.generate(..., stream=True)``.
    - Token counting via a HF tokenizer (``tokenizer_path``).
    """

    def __init__(self, cfg: OllamaConfig) -> None:
        self.cfg = cfg
        self._client = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Ollama client (lazy)
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            self._client = ollama.Client(
                host=self.cfg.host,
                timeout=self.cfg.client_timeout,
            )
        return self._client

    # ------------------------------------------------------------------
    # Tokenizer (lazy)
    # ------------------------------------------------------------------

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        if not self.cfg.tokenizer_path:
            raise ValueError(
                "OllamaBackend: set tokenizer_path for exact token counting."
            )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.tokenizer_path, use_fast=True
        )
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        # add_special_tokens=False because text is already pre-rendered, so no
        # need to add new special tokens.
        tok = self._get_tokenizer()
        n = len(tok.encode(text, add_special_tokens=False))
        if n > tok.model_max_length:
            self._log_event("count_tokens_overflow", {
                "token_count": n,
                "model_max_length": tok.model_max_length,
                "text_chars": len(text),
            })
        return n

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def _build_options(self, cfg: RawGenerationConfig) -> dict:
        opts: dict = {
            "seed": cfg.seed,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "num_predict": cfg.max_tokens,
        }
        if cfg.repeat_penalty is not None:
            opts["repeat_penalty"] = cfg.repeat_penalty
        return opts

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, cfg: RawGenerationConfig) -> RawGenerationResult:
        if self._verbose:
            # generate_streaming handles event logging in this path
            t0 = time.time()
            chunks = list(self.generate_streaming(prompt, cfg))
            return RawGenerationResult(
                text="".join(chunks),
                elapsed_ms=int((time.time() - t0) * 1000),
            )

        client = self._get_client()
        options = self._build_options(cfg)

        self._log_event("raw_generation_start", {
            "prompt_chars": len(prompt),
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "seed": cfg.seed,
        })
        t0 = time.time()
        resp = client.generate(
            model=self.cfg.model,
            prompt=prompt,
            options=options,
            stream=False,
        )
        elapsed_ms = int((time.time() - t0) * 1000)

        text = resp.response
        prompt_tokens = resp.prompt_eval_count
        generated_tokens = resp.eval_count

        self._log_event("raw_generation_done", {
            "elapsed_ms": elapsed_ms,
            "output_chars": len(text),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "timed_out": False,
        })
        return RawGenerationResult(
            text=text,
            elapsed_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
        )

    def generate_streaming(
        self,
        prompt: str,
        cfg: RawGenerationConfig,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        client = self._get_client()
        options = self._build_options(cfg)

        self._log_event("raw_generation_start", {
            "prompt_chars": len(prompt),
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "seed": cfg.seed,
        })
        t0 = time.time()

        if self._verbose:
            print(f"\n{'='*60}\n[PROMPT]\n{'='*60}\n{prompt}\n{'='*60}\n[GENERATION]\n{'='*60}", flush=True)

        first_chunk = True
        output_chars = 0
        last_part = None
        stopped_early = False
        for part in client.generate(
            model=self.cfg.model,
            prompt=prompt,
            options=options,
            stream=True,
        ):
            last_part = part
            if stop_event is not None and stop_event.is_set():
                stopped_early = True
                break
            chunk = part.response
            if chunk:
                if first_chunk:
                    self._log_event("raw_generation_first_chunk", {
                        "elapsed_ms": int((time.time() - t0) * 1000),
                    })
                    first_chunk = False
                output_chars += len(chunk)
                if self._verbose:
                    print(chunk, end="", flush=True)
                yield chunk

        # Token counts are only accurate when the stream ran to completion.
        prompt_tokens = last_part.prompt_eval_count if last_part is not None and not stopped_early else None
        generated_tokens = last_part.eval_count if last_part is not None and not stopped_early else None

        self._log_event("raw_generation_done", {
            "elapsed_ms": int((time.time() - t0) * 1000),
            "output_chars": output_chars,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "timed_out": False,
        })
        if self._verbose:
            print(f"\n{'='*60}", flush=True)

    def close(self) -> None:
        self._client = None


__all__ = ["OllamaConfig", "OllamaBackend"]
