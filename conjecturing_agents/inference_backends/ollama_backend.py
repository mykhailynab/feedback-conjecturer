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
from typing import Iterator
from dataclasses import dataclass
from transformers import AutoTokenizer

from .raw_backend import RawBackend, RawGenerationConfig, RawGenerationResult


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
        return len(self._get_tokenizer().encode(text, add_special_tokens=False))

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

    def generate_streaming(self, prompt: str, cfg: RawGenerationConfig) -> Iterator[str]:
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
        for part in client.generate(
            model=self.cfg.model,
            prompt=prompt,
            options=options,
            stream=True,
        ):
            last_part = part
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

        prompt_tokens = last_part.prompt_eval_count
        generated_tokens = last_part.eval_count

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
