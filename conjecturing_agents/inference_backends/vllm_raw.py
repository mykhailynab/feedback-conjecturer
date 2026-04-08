"""
vLLM inference backend for raw rendered prompt strings.

Uses the OpenAI-compatible ``POST /v1/completions`` endpoint — NOT
``/v1/chat/completions`` and NOT openai_harmony encoding.  The caller
is expected to have already rendered the full prompt string (e.g. via
a Jinja2 chat template).

Optionally manages the vLLM server subprocess (``manage_server=True``).
"""
from __future__ import annotations

import sys
import time
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from openai import OpenAI

from .raw_backend import RawBackend, RawGenerationConfig, RawGenerationResult


@dataclass
class VLLMRawConfig:
    # ------------------------------------------------------------------ #
    # Endpoint
    # ------------------------------------------------------------------ #
    base_url: str = "http://0.0.0.0:8001/v1"
    served_model_name: str = "goedel"
    api_key: str = "sk-local"
    client_timeout: int = 240

    # ------------------------------------------------------------------ #
    # Tokenizer (HF) for exact token counting
    # Falls back to ``model_path`` when ``manage_server=True`` and left empty.
    # ------------------------------------------------------------------ #
    tokenizer_path: str = ""

    # ------------------------------------------------------------------ #
    # Server lifecycle
    # ------------------------------------------------------------------ #
    manage_server: bool = False
    model_path: str = ""
    port: int = 8001
    host: str = "0.0.0.0"
    server_timeout: int = 240
    server_log_path: str = "vllm_goedel_server.log"

    # ------------------------------------------------------------------ #
    # vLLM server knobs (only used when manage_server=True)
    # ------------------------------------------------------------------ #
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "fp8_e4m3"
    context_tokens: int = 40960
    gpu_memory_utilization: float = 0.96
    max_num_seqs: int = 32
    stream_interval: int = 200
    enable_prefix_caching: bool = True
    extra_server_args: List[str] = field(default_factory=list)


class VLLMRawBackend(RawBackend):
    """
    vLLM backend that operates on raw rendered prompt strings.

    - Uses ``/v1/completions`` (not ``/v1/chat/completions``).
    - Does NOT depend on openai_harmony.
    - Token counting via a HF AutoTokenizer loaded from ``tokenizer_path``.
    - Optionally starts and stops a vLLM server subprocess.
    """

    def __init__(self, cfg: VLLMRawConfig) -> None:
        self.cfg = cfg
        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.client_timeout,
        )
        self._tokenizer = None
        self._server_process: Optional[subprocess.Popen] = None
        self._log_file = None
        self._lifecycle_lock = threading.Lock()
        self._started = False

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        tok_path = self.cfg.tokenizer_path or self.cfg.model_path
        if not tok_path:
            raise ValueError(
                "VLLMRawBackend: set tokenizer_path (or model_path) for token counting."
            )
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self._get_tokenizer().encode(text, add_special_tokens=False))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                return
            if self.cfg.manage_server:
                if not self.cfg.model_path:
                    raise ValueError("model_path must be set when manage_server=True")
                self._server_process = self._start_server()
            self._wait_for_server()
            self._started = True

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._server_process is not None:
                try:
                    self._server_process.terminate()
                    self._server_process.wait(timeout=10)
                except Exception:
                    pass
                finally:
                    self._server_process = None
            if self._log_file is not None:
                try:
                    self._log_file.flush()
                    self._log_file.close()
                except Exception:
                    pass
                finally:
                    self._log_file = None
            self._started = False

    def __enter__(self) -> "VLLMRawBackend":
        self.start()
        return self

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _start_server(self) -> subprocess.Popen:
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.cfg.model_path,
            "--served-model-name", self.cfg.served_model_name,
            "--tensor-parallel-size", "1",
            "--max-num-seqs", str(self.cfg.max_num_seqs),
            "--gpu-memory-utilization", str(self.cfg.gpu_memory_utilization),
            "--host", self.cfg.host,
            "--port", str(self.cfg.port),
            "--dtype", self.cfg.dtype,
            "--kv-cache-dtype", self.cfg.kv_cache_dtype,
            "--max-model-len", str(self.cfg.context_tokens),
            "--stream-interval", str(self.cfg.stream_interval),
            # "--async-scheduling",
            "--disable-log-stats",
        ]
        if self.cfg.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")
        cmd.extend(self.cfg.extra_server_args)

        Path(self.cfg.server_log_path).parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.cfg.server_log_path, "w", encoding="utf-8")
        return subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _wait_for_server(self) -> None:
        start = time.time()
        for _ in range(self.cfg.server_timeout):
            if self._server_process is not None:
                rc = self._server_process.poll()
                if rc is not None:
                    if self._log_file:
                        self._log_file.flush()
                    logs = Path(self.cfg.server_log_path).read_text(encoding="utf-8", errors="ignore")
                    raise RuntimeError(f"vLLM server exited with code {rc}. Logs:\n{logs}")
            try:
                self.client.models.list()
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError(f"Timed out waiting for vLLM server after {time.time() - start:.1f}s")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, cfg: RawGenerationConfig) -> RawGenerationResult:
        if self._verbose:
            t0 = time.time()
            chunks = list(self.generate_streaming(prompt, cfg))
            return RawGenerationResult(
                text="".join(chunks),
                elapsed_ms=int((time.time() - t0) * 1000),
            )

        if not self._started:
            self.start()

        t0 = time.time()
        resp = self.client.completions.create(
            model=self.cfg.served_model_name,
            prompt=prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            seed=cfg.seed,
        )
        elapsed_ms = int((time.time() - t0) * 1000)

        text = resp.choices[0].text or ""
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage is not None else None
        generated_tokens = usage.completion_tokens if usage is not None else None

        return RawGenerationResult(
            text=text,
            elapsed_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
        )

    def generate_streaming(self, prompt: str, cfg: RawGenerationConfig) -> Iterator[str]:
        if not self._started:
            self.start()

        if self._verbose:
            print(f"\n{'='*60}\n[PROMPT]\n{'='*60}\n{prompt}\n{'='*60}\n[GENERATION]\n{'='*60}", flush=True)

        stream = self.client.completions.create(
            model=self.cfg.served_model_name,
            prompt=prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            seed=cfg.seed,
            stream=True,
        )
        try:
            for chunk in stream:
                text = chunk.choices[0].text or ""
                if text:
                    if self._verbose:
                        print(text, end="", flush=True)
                    yield text
        finally:
            if self._verbose:
                print(f"\n{'='*60}", flush=True)
            try:
                stream.close()
            except Exception:
                pass


__all__ = ["VLLMRawConfig", "VLLMRawBackend"]
