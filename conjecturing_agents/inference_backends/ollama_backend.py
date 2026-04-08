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
from dataclasses import dataclass
from typing import Iterator, Optional

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


def _get_response_attr(resp, key: str, default):
    """Access ``resp.key`` or ``resp[key]`` gracefully across ollama versions."""
    val = getattr(resp, key, None)
    if val is not None:
        return val
    if hasattr(resp, "get"):
        return resp.get(key, default)
    return default


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
            import ollama
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
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.tokenizer_path, use_fast=True
        )
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
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
        client = self._get_client()
        options = self._build_options(cfg)

        t0 = time.time()
        resp = client.generate(
            model=self.cfg.model,
            prompt=prompt,
            options=options,
            stream=False,
        )
        elapsed_ms = int((time.time() - t0) * 1000)

        text = _get_response_attr(resp, "response", "")
        prompt_tokens = _get_response_attr(resp, "prompt_eval_count", -1)
        generated_tokens = _get_response_attr(resp, "eval_count", -1)

        return RawGenerationResult(
            text=text,
            elapsed_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
        )

    def generate_streaming(self, prompt: str, cfg: RawGenerationConfig) -> Iterator[str]:
        client = self._get_client()
        options = self._build_options(cfg)

        for part in client.generate(
            model=self.cfg.model,
            prompt=prompt,
            options=options,
            stream=True,
        ):
            chunk = _get_response_attr(part, "response", "")
            if chunk:
                yield chunk

    def close(self) -> None:
        self._client = None


__all__ = ["OllamaConfig", "OllamaBackend"]
