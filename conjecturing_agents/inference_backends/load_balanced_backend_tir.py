"""
Load-balanced TIR backend that distributes chat_streaming() calls across
multiple sub-backends, enforcing a per-backend concurrency limit.

Intended use: two Ollama instances on separate GPUs, each configured with
``OLLAMA_NUM_PARALLEL=N``.  Wrap them in a single ``LoadBalancedTIRBackend``
with ``max_concurrent=N`` per sub-backend, then share that one object across
all worker threads.

Dispatch strategy
-----------------
- Pick the first sub-backend whose in-flight count is below its limit.
- If all sub-backends are at capacity, block (via ``threading.Condition``)
  until one releases.  No busy loop.
- On release, ``notify()`` wakes exactly one waiting caller.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .tir_base import (
    TIRBackend,
    TIRGenerationConfig,
    TIRSessionResult,
    TIRStreamChunk,
    TIRToolHandler,
)


class LoadBalancedTIRBackend(TIRBackend):
    """
    Routes ``chat_streaming`` / ``run_session`` calls across a list of
    ``TIRBackend`` instances, honouring per-backend concurrency limits.

    Parameters
    ----------
    backends_and_limits:
        List of ``(backend, max_concurrent)`` pairs.  ``max_concurrent``
        should match the ``OLLAMA_NUM_PARALLEL`` (or equivalent) setting on
        the corresponding server.

    Example
    -------
    ::

        backend = LoadBalancedTIRBackend([
            (OllamaTIRBackend(OllamaTIRConfig(host="http://localhost:11434", ...)), 6),
            (OllamaTIRBackend(OllamaTIRConfig(host="http://localhost:11435", ...)), 6),
        ])
    """

    def __init__(
        self,
        backends_and_limits: List[Tuple[TIRBackend, int]],
    ) -> None:
        if not backends_and_limits:
            raise ValueError("LoadBalancedTIRBackend requires at least one backend.")
        self._backends: List[TIRBackend] = [b for b, _ in backends_and_limits]
        self._limits: List[int] = [n for _, n in backends_and_limits]
        self._inflight: List[int] = [0] * len(self._backends)
        # Condition guards both _inflight reads and blocking waits.
        self._cond = threading.Condition(threading.Lock())

    # ------------------------------------------------------------------
    # Slot management
    # ------------------------------------------------------------------

    def _acquire(self) -> int:
        """Block until a sub-backend has a free slot; return its index."""
        with self._cond:
            while True:
                for i in range(len(self._backends)):
                    if self._inflight[i] < self._limits[i]:
                        self._inflight[i] += 1
                        return i
                # All sub-backends at capacity — wait for a release signal.
                self._cond.wait()

    def _release(self, idx: int) -> None:
        with self._cond:
            self._inflight[idx] -= 1
            self._cond.notify()

    # ------------------------------------------------------------------
    # TIRBackend interface
    # ------------------------------------------------------------------

    def chat_streaming(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        cfg: TIRGenerationConfig,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[TIRStreamChunk]:
        idx = self._acquire()
        try:
            yield from self._backends[idx].chat_streaming(
                messages, tools, cfg, stop_event
            )
        finally:
            self._release(idx)

    def run_session(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_handlers: Dict[str, TIRToolHandler],
        cfg: TIRGenerationConfig,
        stop_event: Optional[threading.Event] = None,
    ) -> TIRSessionResult:
        """Acquire a slot for the entire multi-turn session.

        All turns within a session must go to the same server (no shared
        state across llama.cpp instances), so we hold the slot for the
        full duration rather than acquiring per-turn in chat_streaming().
        """
        idx = self._acquire()
        try:
            return self._backends[idx].run_session(
                messages, tools, tool_handlers, cfg, stop_event
            )
        finally:
            self._release(idx)

    # ------------------------------------------------------------------
    # Verbose / event logger / token counters — propagate to all sub-backends
    # ------------------------------------------------------------------

    def set_verbose(self, enabled: bool) -> None:
        super().set_verbose(enabled)
        for b in self._backends:
            b.set_verbose(enabled)

    def set_event_logger(self, fn) -> None:
        super().set_event_logger(fn)
        for b in self._backends:
            b.set_event_logger(fn)

    def set_token_counter(
        self,
        fn: Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], int],
    ) -> None:
        super().set_token_counter(fn)
        for b in self._backends:
            b.set_token_counter(fn)

    def set_text_counter(self, fn: Callable[[str], int]) -> None:
        super().set_text_counter(fn)
        for b in self._backends:
            b.set_text_counter(fn)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        for b in self._backends:
            b.close()


__all__ = ["LoadBalancedTIRBackend"]
