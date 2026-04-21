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
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

from openai import OpenAI

from .raw_base import RawBackend, RawGenerationConfig, RawGenerationResult


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

    # ------------------------------------------------------------------
    # CLI integration
    # ------------------------------------------------------------------

    @classmethod
    def add_cli_args(
        cls,
        parser: ArgumentParser,
        prefix: str = "vllm",
        defaults: Optional[Dict[str, Any]] = None,
        exclude: Optional[Set[str]] = None,
    ) -> None:
        """Register CLI args for VLLMRawConfig fields.

        ``exclude`` omits fields that are wired from elsewhere (e.g.
        ``tokenizer_path`` and ``context_tokens`` when used with the Goedel
        prover — those come from GoedelProverConfig).
        """
        d = defaults or {}
        ex = exclude or set()
        pre = prefix
        dst = prefix.replace("-", "_")
        defs = cls()

        if "base_url" not in ex:
            parser.add_argument(
                f"--{pre}-base-url",
                dest=f"{dst}_base_url",
                default=d.get("base_url", defs.base_url),
                help="vLLM OpenAI-compatible base URL.",
            )
        if "served_model_name" not in ex:
            parser.add_argument(
                f"--{pre}-model-name",
                dest=f"{dst}_model_name",
                default=d.get("served_model_name", defs.served_model_name),
                help="Served model name for the vLLM endpoint.",
            )
        if "api_key" not in ex:
            parser.add_argument(
                f"--{pre}-api-key",
                dest=f"{dst}_api_key",
                default=d.get("api_key", defs.api_key),
                help="API key for the vLLM endpoint.",
            )
        if "client_timeout" not in ex:
            parser.add_argument(
                f"--{pre}-client-timeout",
                dest=f"{dst}_client_timeout",
                type=int,
                default=d.get("client_timeout", defs.client_timeout),
                help="HTTP client timeout in seconds for vLLM requests.",
            )
        if "manage_server" not in ex:
            parser.add_argument(
                f"--{pre}-manage-server",
                dest=f"{dst}_manage_server",
                action="store_true",
                default=d.get("manage_server", defs.manage_server),
                help="Start and manage a vLLM server subprocess.",
            )
        if "model_path" not in ex:
            parser.add_argument(
                f"--{pre}-model-path",
                dest=f"{dst}_model_path",
                default=d.get("model_path", defs.model_path),
                help="Path to model weights (required when manage-server is set).",
            )
        if "port" not in ex:
            parser.add_argument(
                f"--{pre}-port",
                dest=f"{dst}_port",
                type=int,
                default=d.get("port", defs.port),
                help="Port for the managed vLLM server.",
            )
        if "host" not in ex:
            parser.add_argument(
                f"--{pre}-host",
                dest=f"{dst}_host",
                default=d.get("host", defs.host),
                help="Host for the managed vLLM server.",
            )
        if "server_timeout" not in ex:
            parser.add_argument(
                f"--{pre}-server-timeout",
                dest=f"{dst}_server_timeout",
                type=int,
                default=d.get("server_timeout", defs.server_timeout),
                help="Seconds to wait for the vLLM server to become ready.",
            )
        if "server_log_path" not in ex:
            parser.add_argument(
                f"--{pre}-server-log-path",
                dest=f"{dst}_server_log_path",
                default=d.get("server_log_path", defs.server_log_path),
                help="File path for vLLM server stdout/stderr logs.",
            )
        if "dtype" not in ex:
            parser.add_argument(
                f"--{pre}-dtype",
                dest=f"{dst}_dtype",
                default=d.get("dtype", defs.dtype),
                help="Model weight dtype passed to vLLM.",
            )
        if "kv_cache_dtype" not in ex:
            parser.add_argument(
                f"--{pre}-kv-cache-dtype",
                dest=f"{dst}_kv_cache_dtype",
                default=d.get("kv_cache_dtype", defs.kv_cache_dtype),
                help="KV-cache dtype passed to vLLM.",
            )
        if "context_tokens" not in ex:
            parser.add_argument(
                f"--{pre}-context-tokens",
                dest=f"{dst}_context_tokens",
                type=int,
                default=d.get("context_tokens", defs.context_tokens),
                help="Max model context length for vLLM server.",
            )
        if "gpu_memory_utilization" not in ex:
            parser.add_argument(
                f"--{pre}-gpu-memory-utilization",
                dest=f"{dst}_gpu_memory_utilization",
                type=float,
                default=d.get("gpu_memory_utilization", defs.gpu_memory_utilization),
                help="GPU memory utilization fraction for vLLM.",
            )
        if "max_num_seqs" not in ex:
            parser.add_argument(
                f"--{pre}-max-num-seqs",
                dest=f"{dst}_max_num_seqs",
                type=int,
                default=d.get("max_num_seqs", defs.max_num_seqs),
                help="Maximum number of concurrent sequences for vLLM.",
            )
        if "stream_interval" not in ex:
            parser.add_argument(
                f"--{pre}-stream-interval",
                dest=f"{dst}_stream_interval",
                type=int,
                default=d.get("stream_interval", defs.stream_interval),
                help="Token streaming interval for vLLM.",
            )
        if "enable_prefix_caching" not in ex:
            parser.add_argument(
                f"--{pre}-no-prefix-caching",
                dest=f"{dst}_enable_prefix_caching",
                action="store_false",
                default=d.get("enable_prefix_caching", defs.enable_prefix_caching),
                help="Disable prefix caching in vLLM.",
            )
        if "extra_server_args" not in ex:
            parser.add_argument(
                f"--{pre}-extra-server-args",
                dest=f"{dst}_extra_server_args",
                nargs="*",
                default=d.get("extra_server_args", []),
                help="Extra CLI arguments forwarded verbatim to the vLLM server.",
            )

    @classmethod
    def from_parsed_args(
        cls,
        args: Any,
        prefix: str = "vllm",
        exclude: Optional[Set[str]] = None,
        **overrides: Any,
    ) -> "VLLMRawConfig":
        """Construct from an argparse namespace.

        Fields in ``exclude`` or ``overrides`` are handled accordingly;
        everything else is read from the namespace using the prefix.
        """
        ex = exclude or set()
        dst = prefix.replace("-", "_")
        defs = cls()

        field_map = {
            "base_url": f"{dst}_base_url",
            "served_model_name": f"{dst}_model_name",
            "api_key": f"{dst}_api_key",
            "client_timeout": f"{dst}_client_timeout",
            "manage_server": f"{dst}_manage_server",
            "model_path": f"{dst}_model_path",
            "port": f"{dst}_port",
            "host": f"{dst}_host",
            "server_timeout": f"{dst}_server_timeout",
            "server_log_path": f"{dst}_server_log_path",
            "dtype": f"{dst}_dtype",
            "kv_cache_dtype": f"{dst}_kv_cache_dtype",
            "context_tokens": f"{dst}_context_tokens",
            "gpu_memory_utilization": f"{dst}_gpu_memory_utilization",
            "max_num_seqs": f"{dst}_max_num_seqs",
            "stream_interval": f"{dst}_stream_interval",
            "enable_prefix_caching": f"{dst}_enable_prefix_caching",
            "extra_server_args": f"{dst}_extra_server_args",
        }

        kwargs: Dict[str, Any] = {}
        for field_name, attr_name in field_map.items():
            if field_name in overrides:
                kwargs[field_name] = overrides[field_name]
            elif field_name in ex:
                kwargs[field_name] = getattr(defs, field_name)
            else:
                val = getattr(args, attr_name, None)
                if val is not None:
                    kwargs[field_name] = val
                else:
                    kwargs[field_name] = getattr(defs, field_name)

        # Apply remaining overrides not in field_map
        for k, v in overrides.items():
            if k not in field_map:
                kwargs[k] = v

        return cls(**kwargs)


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
            # generate_streaming handles event logging in this path
            t0 = time.time()
            chunks = list(self.generate_streaming(prompt, cfg))
            return RawGenerationResult(
                text="".join(chunks),
                elapsed_ms=int((time.time() - t0) * 1000),
            )

        if not self._started:
            self.start()

        self._log_event("raw_generation_start", {
            "prompt_chars": len(prompt),
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "seed": cfg.seed,
        })
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
        if not self._started:
            self.start()

        self._log_event("raw_generation_start", {
            "prompt_chars": len(prompt),
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "seed": cfg.seed,
        })
        t0 = time.time()

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
        first_chunk = True
        output_chars = 0
        try:
            for chunk in stream:
                if stop_event is not None and stop_event.is_set():
                    break
                text = chunk.choices[0].text or ""
                if text:
                    if first_chunk:
                        self._log_event("raw_generation_first_chunk", {
                            "elapsed_ms": int((time.time() - t0) * 1000),
                        })
                        first_chunk = False
                    output_chars += len(text)
                    if self._verbose:
                        print(text, end="", flush=True)
                    yield text
        finally:
            self._log_event("raw_generation_done", {
                "elapsed_ms": int((time.time() - t0) * 1000),
                "output_chars": output_chars,
                "prompt_tokens": None,
                "generated_tokens": None,
                "timed_out": False,
            })
            if self._verbose:
                print(f"\n{'='*60}", flush=True)
            try:
                stream.close()
            except Exception:
                pass


__all__ = ["VLLMRawConfig", "VLLMRawBackend"]
