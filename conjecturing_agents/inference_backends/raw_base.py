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

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

class EventLogger:
    """Base class for structured event loggers.

    Subclass and override ``log_event()`` to write events to JSONL, intercept
    for metrics, etc.

    Supports per-thread metadata via ``set_thread_metadata()`` /
    ``clear_thread_metadata()``.  When set, the metadata dict is merged into
    every ``log_event()`` call from that thread — e.g. the scheduler sets
    ``proof_id`` so that backend-level events automatically carry it.
    """

    def __init__(self) -> None:
        self._tls = threading.local()
        self._lock = threading.Lock()

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Write a single event.  Override in subclasses."""

    def set_thread_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set metadata merged into every ``log_event`` from this thread."""
        self._tls.metadata = metadata

    def clear_thread_metadata(self) -> None:
        """Remove per-thread metadata."""
        self._tls.metadata = None

    def _merge_thread_metadata(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return *payload* with per-thread metadata prepended (if any)."""
        meta = getattr(self._tls, "metadata", None)
        if meta:
            return {**meta, **payload}
        return payload


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
      - ``OllamaBackend``   (ollama_raw.py)
    """

    # Set to True via set_verbose() to print prompts and tokens to stdout.
    _verbose: bool = False
    _event_logger: Optional[EventLogger] = None

    def set_verbose(self, enabled: bool) -> None:
        """Enable real-time prompt+token printing to stdout."""
        self._verbose = enabled

    def set_event_logger(self, logger: EventLogger) -> None:
        """Register an EventLogger whose log_event() receives structured events."""
        self._event_logger = logger

    def _log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self._event_logger is not None:
            self._event_logger.log_event(event_type, payload)

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
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """Yield text chunks as they arrive from the model.

        If ``stop_event`` is provided, the implementation should check it
        between chunks and stop yielding as soon as it is set.
        """

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
    "EventLogger",
    "RawGenerationConfig",
    "RawGenerationResult",
    "RawBackend",
]
