"""
Load-balanced raw backend that distributes generate() calls across multiple
sub-backends, enforcing a per-backend concurrency limit.

Intended use: two Ollama instances on separate GPUs, each configured with
``OLLAMA_NUM_PARALLEL=N``.  Wrap them in a single ``LoadBalancedRawBackend``
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
from typing import Iterator, List, Optional, Tuple

from .raw_backend import RawBackend, RawGenerationConfig, RawGenerationResult


class LoadBalancedRawBackend(RawBackend):
    """
    Routes ``generate`` / ``generate_streaming`` calls across a list of
    ``RawBackend`` instances, honouring per-backend concurrency limits.

    Parameters
    ----------
    backends_and_limits:
        List of ``(backend, max_concurrent)`` pairs.  ``max_concurrent``
        should match the ``OLLAMA_NUM_PARALLEL`` (or equivalent) setting on
        the corresponding server.

    Example
    -------
    ::

        backend = LoadBalancedRawBackend([
            (OllamaBackend(OllamaConfig(host="http://localhost:11434", ...)), 6),
            (OllamaBackend(OllamaConfig(host="http://localhost:11435", ...)), 6),
        ])
    """

    def __init__(
        self,
        backends_and_limits: List[Tuple[RawBackend, int]],
    ) -> None:
        if not backends_and_limits:
            raise ValueError("LoadBalancedRawBackend requires at least one backend.")
        self._backends: List[RawBackend] = [b for b, _ in backends_and_limits]
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
    # RawBackend interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, cfg: RawGenerationConfig) -> RawGenerationResult:
        idx = self._acquire()
        try:
            return self._backends[idx].generate(prompt, cfg)
        finally:
            self._release(idx)

    def generate_streaming(
        self,
        prompt: str,
        cfg: RawGenerationConfig,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        idx = self._acquire()
        try:
            yield from self._backends[idx].generate_streaming(prompt, cfg, stop_event)
        finally:
            self._release(idx)

    def count_tokens(self, text: str) -> int:
        # All sub-backends share the same tokenizer, so any one will do.
        return self._backends[0].count_tokens(text)

    # ------------------------------------------------------------------
    # Verbose / event logger — propagate to all sub-backends
    # ------------------------------------------------------------------

    def set_verbose(self, enabled: bool) -> None:
        super().set_verbose(enabled)
        for b in self._backends:
            b.set_verbose(enabled)

    def set_event_logger(self, fn) -> None:
        super().set_event_logger(fn)
        for b in self._backends:
            b.set_event_logger(fn)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        for b in self._backends:
            b.close()


__all__ = ["LoadBalancedRawBackend"]
