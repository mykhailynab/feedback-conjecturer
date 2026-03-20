from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from argparse import ArgumentParser

from conjecturing_agents.agents.solver import SolverAgent, SolverAgentConfig
from conjecturing_agents.inference_backends.vllm_harmony import (
    VLLMHarmonyBackendConfig,
)
from conjecturing_agents.tool_calling_backends.jupyter import JupyterKernelConfig

from conjecturing_agents.agents.informal_correctness_checker import (
    InformalCorrectnessCheckerAgent,
    InformalCorrectnessCheckerConfig,
)

# ============================================================
# Script config
# ============================================================

@dataclass
class RunConfig:
    # Paths
    reference_path: str = "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv"
    log_dir: str = "/kaggle/working/aimo3_logs"
    attempts_filename: str = "attempts.jsonl"
    solutions_filename: str = "solutions.csv"
    submission_filename: str = "submission.csv"

    # vLLM / model serving
    served_model_name: str = "gpt-oss"
    model_path: str = "/kaggle/input/models/danielhanchen/gpt-oss-20b/transformers/default/1"
    port: int = 8000
    api_key: str = "sk-local"
    host: str = "0.0.0.0"
    client_host: str = "0.0.0.0"
    dtype: str = "auto"
    kv_cache_dtype: str = "fp8_e4m3"
    gpu_memory_utilization: float = 0.96
    batch_size: int = 256
    context_tokens: int = 65536
    stream_interval: int = 200
    server_timeout: int = 180
    session_timeout: int = 960
    preload_workers: int = 32
    preload_model_weights: bool = True
    manage_server: bool = True
    server_log_path: str = "vllm_server.log"

    # Global orchestration
    agent_parallelism: int = 8
    attempts_per_problem: int = 16
    max_problems: int = 0
    seed: int = 42

    # Shared tool runtime defaults
    jupyter_timeout: float = 6.0

    # Solver agent
    solver_temperature: float = 0.5
    solver_min_p: float = 0.02
    solver_top_logprobs: int = 5
    solver_max_turns: int = 128
    solver_timeout_seconds: int = 600
    solver_buffer_tokens: int = 512
    solver_stream_text_window: int = 32

    # Checker agent
    checker_temperature: float = 0.0
    checker_min_p: float = 0.0
    checker_max_turns: int = 64
    checker_timeout_seconds: int = 120
    checker_buffer_tokens: int = 512
    checker_stream_text_window: int = 32

    # Logging
    verbose: bool = True


def validate_cfg(cfg: RunConfig) -> None:
    errs: List[str] = []

    if cfg.attempts_per_problem <= 0:
        errs.append("attempts_per_problem must be >= 1")
    if cfg.agent_parallelism <= 0:
        errs.append("agent_parallelism must be >= 1")
    if cfg.batch_size <= 0:
        errs.append("batch_size must be >= 1")
    if cfg.context_tokens <= 0:
        errs.append("context_tokens must be >= 1")
    if cfg.preload_workers <= 0:
        errs.append("preload_workers must be >= 1")
    if cfg.jupyter_timeout <= 0:
        errs.append("jupyter_timeout must be > 0")
    if cfg.solver_max_turns <= 0:
        errs.append("solver_max_turns must be >= 1")
    if cfg.checker_max_turns <= 0:
        errs.append("checker_max_turns must be >= 1")
    if cfg.solver_timeout_seconds <= 0:
        errs.append("solver_timeout_seconds must be >= 1")
    if cfg.checker_timeout_seconds <= 0:
        errs.append("checker_timeout_seconds must be >= 1")
    if cfg.solver_buffer_tokens <= 0:
        errs.append("solver_buffer_tokens must be >= 1")
    if cfg.checker_buffer_tokens <= 0:
        errs.append("checker_buffer_tokens must be >= 1")
    if cfg.solver_temperature < 0:
        errs.append("solver_temperature must be >= 0")
    if cfg.checker_temperature < 0:
        errs.append("checker_temperature must be >= 0")
    if not (0.0 <= cfg.solver_min_p <= 1.0):
        errs.append("solver_min_p must be in [0, 1]")
    if not (0.0 <= cfg.checker_min_p <= 1.0):
        errs.append("checker_min_p must be in [0, 1]")
    if cfg.max_problems < 0:
        errs.append("max_problems must be >= 0")

    if errs:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errs))


# ============================================================
# Agent factory helpers
# ============================================================

def make_backend_config(cfg: RunConfig) -> VLLMHarmonyBackendConfig:
    return VLLMHarmonyBackendConfig(
        served_model_name=cfg.served_model_name,
        model_path=cfg.model_path,
        port=cfg.port,
        api_key=cfg.api_key,
        host=cfg.host,
        client_host=cfg.client_host,
        session_timeout=cfg.session_timeout,
        server_timeout=cfg.server_timeout,
        stream_interval=cfg.stream_interval,
        context_tokens=cfg.context_tokens,
        batch_size=cfg.batch_size,
        dtype=cfg.dtype,
        kv_cache_dtype=cfg.kv_cache_dtype,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        preload_workers=cfg.preload_workers,
        preload_model_weights=cfg.preload_model_weights,
        manage_server=cfg.manage_server,
        server_log_path=cfg.server_log_path,
    )

def make_solver_agent(cfg: RunConfig) -> SolverAgent:
    jupyter_cfg = JupyterKernelConfig(
        timeout_seconds=cfg.jupyter_timeout,
        init_on_create=False,
    )
    solver_cfg = SolverAgentConfig(
        temperature=cfg.solver_temperature,
        min_p=cfg.solver_min_p,
        top_logprobs=cfg.solver_top_logprobs,
        max_turns=cfg.solver_max_turns,
        timeout_seconds=cfg.solver_timeout_seconds,
        buffer_tokens=cfg.solver_buffer_tokens,
        stream_text_window=cfg.solver_stream_text_window,
        jupyter=jupyter_cfg,
    )
    return SolverAgent(cfg=solver_cfg)


def make_checker_agent(cfg: RunConfig) -> InformalCorrectnessCheckerAgent:
    jupyter_cfg = JupyterKernelConfig(
        timeout_seconds=cfg.jupyter_timeout,
        init_on_create=False,
    )
    checker_cfg = InformalCorrectnessCheckerConfig(
        temperature=cfg.checker_temperature,
        min_p=cfg.checker_min_p,
        max_turns=cfg.checker_max_turns,
        timeout_seconds=cfg.checker_timeout_seconds,
        buffer_tokens=cfg.checker_buffer_tokens,
        stream_text_window=cfg.checker_stream_text_window,
        jupyter=jupyter_cfg,
    )
    return InformalCorrectnessCheckerAgent(cfg=checker_cfg)



# ============================================================
# CLI
# ============================================================



def parse_args_and_validate() -> RunConfig:
    p = ArgumentParser(
        description="Run conjecturing pipeline: solver attempts plus parallel informal correctness checks."
    )

    # Paths
    p.add_argument("--reference-path", default=RunConfig.reference_path)
    p.add_argument("--log-dir", default=RunConfig.log_dir)
    p.add_argument("--attempts-log", dest="attempts_filename", default=RunConfig.attempts_filename)
    p.add_argument("--solutions-log", dest="solutions_filename", default=RunConfig.solutions_filename)
    p.add_argument("--submission-out", dest="submission_filename", default=RunConfig.submission_filename)

    # Backend / server
    p.add_argument("--served-model-name", default=RunConfig.served_model_name)
    p.add_argument("--model-path", default=RunConfig.model_path)
    p.add_argument("--port", type=int, default=RunConfig.port)
    p.add_argument("--api-key", default=RunConfig.api_key)
    p.add_argument("--host", default=RunConfig.host)
    p.add_argument("--client-host", default=RunConfig.client_host)
    p.add_argument("--dtype", default=RunConfig.dtype)
    p.add_argument("--kv-cache-dtype", default=RunConfig.kv_cache_dtype)
    p.add_argument("--gpu-memory-utilization", type=float, default=RunConfig.gpu_memory_utilization)
    p.add_argument("--max-num-seqs", dest="batch_size", type=int, default=RunConfig.batch_size)
    p.add_argument("--context-tokens", type=int, default=RunConfig.context_tokens)
    p.add_argument("--stream-interval", type=int, default=RunConfig.stream_interval)
    p.add_argument("--server-startup-timeout-seconds", dest="server_timeout", type=int, default=RunConfig.server_timeout)
    p.add_argument("--openai-client-timeout-seconds", dest="session_timeout", type=int, default=RunConfig.session_timeout)
    p.add_argument("--preload-workers", type=int, default=RunConfig.preload_workers)
    p.add_argument("--server-log-path", default=RunConfig.server_log_path)

    p.add_argument(
        "--no-preload-model-weights",
        dest="preload_model_weights",
        action="store_false",
        default=RunConfig.preload_model_weights,
        help="Disable page-caching model weights before server startup.",
    )
    p.add_argument(
        "--use-existing-server",
        dest="manage_server",
        action="store_false",
        default=RunConfig.manage_server,
        help="Do not start vLLM; connect to an already running OpenAI-compatible endpoint.",
    )

    # Global orchestration
    p.add_argument("--agent-parallelism", type=int, default=RunConfig.agent_parallelism)
    p.add_argument("--attempts-per-problem", type=int, default=RunConfig.attempts_per_problem)
    p.add_argument("--max-problems", type=int, default=RunConfig.max_problems)
    p.add_argument("--seed", type=int, default=RunConfig.seed)

    # Shared tool runtime
    p.add_argument("--jupyter-exec-timeout-seconds", dest="jupyter_timeout", type=float, default=RunConfig.jupyter_timeout)

    # Solver
    p.add_argument("--solver-temperature", type=float, default=RunConfig.solver_temperature)
    p.add_argument("--solver-min-p", type=float, default=RunConfig.solver_min_p)
    p.add_argument("--solver-top-logprobs", type=int, default=RunConfig.solver_top_logprobs)
    p.add_argument("--solver-max-turns", type=int, default=RunConfig.solver_max_turns)
    p.add_argument("--solver-timeout-seconds", type=int, default=RunConfig.solver_timeout_seconds)
    p.add_argument("--solver-buffer-tokens", type=int, default=RunConfig.solver_buffer_tokens)
    p.add_argument("--solver-stream-text-window", type=int, default=RunConfig.solver_stream_text_window)

    # Checker
    p.add_argument("--checker-temperature", type=float, default=RunConfig.checker_temperature)
    p.add_argument("--checker-min-p", type=float, default=RunConfig.checker_min_p)
    p.add_argument("--checker-max-turns", type=int, default=RunConfig.checker_max_turns)
    p.add_argument("--checker-timeout-seconds", type=int, default=RunConfig.checker_timeout_seconds)
    p.add_argument("--checker-buffer-tokens", type=int, default=RunConfig.checker_buffer_tokens)
    p.add_argument("--checker-stream-text-window", type=int, default=RunConfig.checker_stream_text_window)

    # Logging
    p.add_argument("--verbose", action="store_true", default=RunConfig.verbose)

    args = p.parse_args()

    config = RunConfig(
        reference_path=args.reference_path,
        log_dir=args.log_dir,
        attempts_filename=args.attempts_filename,
        solutions_filename=args.solutions_filename,
        submission_filename=args.submission_filename,
        served_model_name=args.served_model_name,
        model_path=args.model_path,
        port=args.port,
        api_key=args.api_key,
        host=args.host,
        client_host=args.client_host,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        batch_size=args.batch_size,
        context_tokens=args.context_tokens,
        stream_interval=args.stream_interval,
        server_timeout=args.server_timeout,
        session_timeout=args.session_timeout,
        preload_workers=args.preload_workers,
        preload_model_weights=args.preload_model_weights,
        manage_server=args.manage_server,
        server_log_path=args.server_log_path,
        agent_parallelism=args.agent_parallelism,
        attempts_per_problem=args.attempts_per_problem,
        max_problems=args.max_problems,
        seed=args.seed,
        jupyter_timeout=args.jupyter_timeout,
        solver_temperature=args.solver_temperature,
        solver_min_p=args.solver_min_p,
        solver_top_logprobs=args.solver_top_logprobs,
        solver_max_turns=args.solver_max_turns,
        solver_timeout_seconds=args.solver_timeout_seconds,
        solver_buffer_tokens=args.solver_buffer_tokens,
        solver_stream_text_window=args.solver_stream_text_window,
        checker_temperature=args.checker_temperature,
        checker_min_p=args.checker_min_p,
        checker_max_turns=args.checker_max_turns,
        checker_timeout_seconds=args.checker_timeout_seconds,
        checker_buffer_tokens=args.checker_buffer_tokens,
        checker_stream_text_window=args.checker_stream_text_window,
        verbose=args.verbose,
    )

    validate_cfg(config)

    return config

