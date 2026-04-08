"""
Common types and abstract base for raw-prompt (non-Harmony) inference backends.

Both VLLMRawBackend and OllamaBackend implement RawBackend.

Design contract
---------------
- Agents are responsible for rendering their own prompt strings (e.g. via a
  chat template). Backends accept a pre-rendered ``str`` — they have no
  knowledge of message formats.
- ``count_tokens`` must be callable before ``generate`` to let callers enforce
  context budgets without relying on server-side rejection.
- Streaming is available via ``generate_streaming``; it yields text chunks as
  they arrive, leaving collection/printing to the caller.
- ``close`` / context manager are optional lifecycle hooks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class RawGenerationConfig:
    """Generation parameters for a single raw-prompt completion request."""

    max_tokens: int = 8192
    temperature: float = 0.6
    top_p: float = 0.95
    seed: int = 0
    # Repetition penalty — primarily for Ollama; ignored by vLLM (for now)
    repeat_penalty: Optional[float] = None


@dataclass
class RawGenerationResult:
    """Result of a single ``generate()`` call."""

    text: str
    elapsed_ms: int

    # Token counts; None means the backend did not report them.
    prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None

    timed_out: bool = False


class RawBackend(ABC):
    """
    Abstract base for raw-prompt text generation backends.

    Implementors:
      - ``VLLMRawBackend``  (vllm_raw.py)
      - ``OllamaBackend``   (ollama_backend.py)
    """

    # Set to True via set_verbose() to print prompts and tokens to stdout.
    _verbose: bool = False

    def set_verbose(self, enabled: bool) -> None:
        """Enable real-time prompt+token printing to stdout."""
        self._verbose = enabled

    @abstractmethod
    def generate(
        self,
        prompt: str,
        cfg: RawGenerationConfig,
    ) -> RawGenerationResult:
        """Blocking single-shot generation. Returns the complete output text."""

    @abstractmethod
    def generate_streaming(
        self,
        prompt: str,
        cfg: RawGenerationConfig,
    ) -> Iterator[str]:
        """Yield text chunks as they arrive from the model."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in ``text`` using the backend's tokenizer."""

    def close(self) -> None:
        """Release resources (server process, connections, …). Override as needed."""

    def __enter__(self) -> RawBackend:
        return self

    def __exit__(self, *_) -> None:
        self.close()


__all__ = [
    "RawGenerationConfig",
    "RawGenerationResult",
    "RawBackend",
]
