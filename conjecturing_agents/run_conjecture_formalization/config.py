from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional

from conjecturing_agents.agents.conjecture_formalizer import (
    ConjectureFormalizerAgent,
    ConjectureFormalizerConfig,
)
from conjecturing_agents.inference_backends.vllm_harmony import (
    VLLMHarmonyBackendConfig,
)
from conjecturing_agents.tool_calling_backends.jupyter import (
    JupyterKernelConfig,
)
from conjecturing_agents.tool_calling_backends.lean4_compiler import (
    LeanCompilerConfig,
)


@dataclass
class RunConjectureFormalizationConfig:
    # -----------------------------
    # Input / output paths
    # -----------------------------
    attempts_path: str = "attempts.jsonl"
    references_putnam_path: str = "references_putnam.csv"
    extracted_putnam_path: str = "extracted_putnam.jsonl"

    log_dir: str = "conjecture_formalization_logs"
    formalizations_filename: str = "formalizations.jsonl"

    # -----------------------------
    # vLLM / model serving
    # -----------------------------
    served_model_name: str = "gpt-oss"
    model_path: str = ""
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
    server_timeout: int = 600
    session_timeout: int = 960
    preload_workers: int = 8
    preload_model_weights: bool = True
    manage_server: bool = True
    server_log_path: str = "vllm_server.log"

    # -----------------------------
    # Parallelism / dataset slicing
    # -----------------------------
    agent_parallelism: int = 8
    max_attempts: int = 0
    skip_empty_answers: bool = True
    seed: int = 42

    # -----------------------------
    # Formalization loop
    # -----------------------------
    solution_tail_chars: int = 1000
    max_correction_rounds: int = 2

    # -----------------------------
    # Formalizer agent controls
    # -----------------------------
    formalizer_temperature: float = 0.2
    formalizer_min_p: float = 0.0
    formalizer_max_turns: int = 48
    formalizer_timeout_seconds: int = 300
    formalizer_buffer_tokens: int = 512
    formalizer_stream_text_window: int = 4096

    use_python_tool: bool = True
    use_lean_tool: bool = True

    # -----------------------------
    # Python tool
    # -----------------------------
    jupyter_timeout: float = 6.0

    # -----------------------------
    # Lean tool / external validation
    # -----------------------------
    lean_project_dir: str = "."
    lean_timeout_seconds: int = 120
    lean_jobs: int = 4
    lean_workspace_subdir: str = ".conjecturing_agents/lean_tool_runs"
    lean_max_memory_megabytes: int = 2 * 1024  # 2 GiB

    # -----------------------------
    # Logging
    # -----------------------------
    verbose: bool = True
    log_formalization_progress: bool = True


def validate_cfg(cfg: RunConjectureFormalizationConfig) -> None:
    errs: list[str] = []

    if cfg.agent_parallelism <= 0:
        errs.append("agent_parallelism must be >= 1")
    if cfg.max_attempts < 0:
        errs.append("max_attempts must be >= 0")
    if cfg.seed < 0:
        errs.append("seed must be >= 0")

    if cfg.batch_size <= 0:
        errs.append("batch_size must be >= 1")
    if cfg.context_tokens <= 0:
        errs.append("context_tokens must be >= 1")
    if cfg.stream_interval <= 0:
        errs.append("stream_interval must be >= 1")
    if cfg.server_timeout <= 0:
        errs.append("server_timeout must be >= 1")
    if cfg.session_timeout <= 0:
        errs.append("session_timeout must be >= 1")
    if cfg.preload_workers <= 0:
        errs.append("preload_workers must be >= 1")

    if cfg.solution_tail_chars <= 0:
        errs.append("solution_tail_chars must be >= 1")
    if cfg.max_correction_rounds < 0:
        errs.append("max_correction_rounds must be >= 0")

    if cfg.formalizer_temperature < 0:
        errs.append("formalizer_temperature must be >= 0")
    if not (0.0 <= cfg.formalizer_min_p <= 1.0):
        errs.append("formalizer_min_p must be in [0, 1]")
    if cfg.formalizer_max_turns <= 0:
        errs.append("formalizer_max_turns must be >= 1")
    if cfg.formalizer_timeout_seconds <= 0:
        errs.append("formalizer_timeout_seconds must be >= 1")
    if cfg.formalizer_buffer_tokens <= 0:
        errs.append("formalizer_buffer_tokens must be >= 1")
    if cfg.formalizer_stream_text_window <= 0:
        errs.append("formalizer_stream_text_window must be >= 1")

    if cfg.jupyter_timeout <= 0:
        errs.append("jupyter_timeout must be > 0")

    if not cfg.lean_project_dir:
        errs.append("lean_project_dir must be non-empty")
    if cfg.lean_timeout_seconds <= 0:
        errs.append("lean_timeout_seconds must be >= 1")
    if cfg.lean_jobs <= 0:
        errs.append("lean_jobs must be >= 1")

    if errs:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errs))


def parse_args_and_validate() -> RunConjectureFormalizationConfig:
    p = ArgumentParser(
        description=(
            "Run conjecture formalization from an existing attempts.jsonl, "
            "using the conjecture formalizer agent and Lean validation."
        )
    )

    # -----------------------------
    # Input / output
    # -----------------------------
    p.add_argument("--attempts-path", default=RunConjectureFormalizationConfig.attempts_path)
    p.add_argument("--references-putnam-path", default=RunConjectureFormalizationConfig.references_putnam_path)
    p.add_argument("--extracted-putnam-path", default=RunConjectureFormalizationConfig.extracted_putnam_path)

    p.add_argument("--log-dir", default=RunConjectureFormalizationConfig.log_dir)
    p.add_argument(
        "--formalizations-log",
        dest="formalizations_filename",
        default=RunConjectureFormalizationConfig.formalizations_filename,
    )

    # -----------------------------
    # Backend / server
    # -----------------------------
    p.add_argument("--served-model-name", default=RunConjectureFormalizationConfig.served_model_name)
    p.add_argument("--model-path", default=RunConjectureFormalizationConfig.model_path)
    p.add_argument("--port", type=int, default=RunConjectureFormalizationConfig.port)
    p.add_argument("--api-key", default=RunConjectureFormalizationConfig.api_key)
    p.add_argument("--host", default=RunConjectureFormalizationConfig.host)
    p.add_argument("--client-host", default=RunConjectureFormalizationConfig.client_host)
    p.add_argument("--dtype", default=RunConjectureFormalizationConfig.dtype)
    p.add_argument("--kv-cache-dtype", default=RunConjectureFormalizationConfig.kv_cache_dtype)
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=RunConjectureFormalizationConfig.gpu_memory_utilization,
    )
    p.add_argument("--max-num-seqs", dest="batch_size", type=int, default=RunConjectureFormalizationConfig.batch_size)
    p.add_argument("--context-tokens", type=int, default=RunConjectureFormalizationConfig.context_tokens)
    p.add_argument("--stream-interval", type=int, default=RunConjectureFormalizationConfig.stream_interval)
    p.add_argument(
        "--server-startup-timeout-seconds",
        dest="server_timeout",
        type=int,
        default=RunConjectureFormalizationConfig.server_timeout,
    )
    p.add_argument(
        "--openai-client-timeout-seconds",
        dest="session_timeout",
        type=int,
        default=RunConjectureFormalizationConfig.session_timeout,
    )
    p.add_argument("--preload-workers", type=int, default=RunConjectureFormalizationConfig.preload_workers)
    p.add_argument("--server-log-path", default=RunConjectureFormalizationConfig.server_log_path)

    p.add_argument(
        "--no-preload-model-weights",
        dest="preload_model_weights",
        action="store_false",
        default=RunConjectureFormalizationConfig.preload_model_weights,
        help="Disable page-caching model weights before server startup.",
    )
    p.add_argument(
        "--use-existing-server",
        dest="manage_server",
        action="store_false",
        default=RunConjectureFormalizationConfig.manage_server,
        help="Do not start vLLM; connect to an already running OpenAI-compatible endpoint.",
    )

    # -----------------------------
    # Parallelism / slicing
    # -----------------------------
    p.add_argument("--agent-parallelism", type=int, default=RunConjectureFormalizationConfig.agent_parallelism)
    p.add_argument("--max-attempts", type=int, default=RunConjectureFormalizationConfig.max_attempts, help="max attempts to consider from the attempts file")
    p.add_argument("--seed", type=int, default=RunConjectureFormalizationConfig.seed)

    p.add_argument(
        "--keep-empty-answers",
        dest="skip_empty_answers",
        action="store_false",
        default=RunConjectureFormalizationConfig.skip_empty_answers,
        help="Do not skip attempts whose extracted boxed answer is empty.",
    )

    # -----------------------------
    # Formalization loop
    # -----------------------------
    p.add_argument("--solution-tail-chars", type=int, default=RunConjectureFormalizationConfig.solution_tail_chars)
    p.add_argument(
        "--max-correction-rounds",
        type=int,
        default=RunConjectureFormalizationConfig.max_correction_rounds,
    )

    # -----------------------------
    # Formalizer agent
    # -----------------------------
    p.add_argument("--formalizer-temperature", type=float, default=RunConjectureFormalizationConfig.formalizer_temperature)
    p.add_argument("--formalizer-min-p", type=float, default=RunConjectureFormalizationConfig.formalizer_min_p)
    p.add_argument("--formalizer-max-turns", type=int, default=RunConjectureFormalizationConfig.formalizer_max_turns)
    p.add_argument(
        "--formalizer-timeout-seconds",
        type=int,
        dest="formalizer_timeout_seconds",
        default=RunConjectureFormalizationConfig.formalizer_timeout_seconds,
    )
    p.add_argument(
        "--formalizer-buffer-tokens",
        type=int,
        default=RunConjectureFormalizationConfig.formalizer_buffer_tokens,
        help="Buffer from context tokens to current input prompt tokens to stop with context_exhausted"
    )
    p.add_argument(
        "--formalizer-stream-text-window",
        type=int,
        default=RunConjectureFormalizationConfig.formalizer_stream_text_window,
    )

    p.add_argument(
        "--no-python-tool",
        dest="use_python_tool",
        action="store_false",
        default=RunConjectureFormalizationConfig.use_python_tool,
    )
    p.add_argument(
        "--no-lean-tool",
        dest="use_lean_tool",
        action="store_false",
        default=RunConjectureFormalizationConfig.use_lean_tool,
    )

    # -----------------------------
    # Python tool
    # -----------------------------
    p.add_argument(
        "--jupyter-exec-timeout-seconds",
        dest="jupyter_timeout",
        type=float,
        default=RunConjectureFormalizationConfig.jupyter_timeout,
    )

    # -----------------------------
    # Lean tool / external validation
    # -----------------------------
    p.add_argument(
        "--lean-project-dir",
        default=RunConjectureFormalizationConfig.lean_project_dir,
        help="Path to a Lean project with Mathlib configured.",
    )
    p.add_argument(
        "--lean-timeout-seconds",
        dest="lean_timeout_seconds",
        type=int,
        default=RunConjectureFormalizationConfig.lean_timeout_seconds,
    )
    p.add_argument("--lean-jobs", type=int, default=RunConjectureFormalizationConfig.lean_jobs)
    p.add_argument(
        "--lean-workspace-subdir",
        default=RunConjectureFormalizationConfig.lean_workspace_subdir,
    )
    p.add_argument(
        "--lean-max-memory-megabytes",
        type=int,
        default=RunConjectureFormalizationConfig.lean_max_memory_megabytes,
        help=(
            "Maximum virtual memory (megabytes) for each lake/lean subprocess. "
            "Example: 2 * 1024 for 2 GiB."
        ),
    )

    # -----------------------------
    # Logging
    # -----------------------------
    p.add_argument("--verbose", action="store_true", default=RunConjectureFormalizationConfig.verbose)
    p.add_argument("--no-verbose", dest="verbose", action="store_false", default=RunConjectureFormalizationConfig.verbose)
    p.add_argument("--no-log-formalization-progress", dest="log_formalization_progress", action="store_false", default=RunConjectureFormalizationConfig.log_formalization_progress)

    args = p.parse_args()

    cfg = RunConjectureFormalizationConfig(
        attempts_path=args.attempts_path,
        references_putnam_path=args.references_putnam_path,
        extracted_putnam_path=args.extracted_putnam_path,
        log_dir=args.log_dir,
        formalizations_filename=args.formalizations_filename,
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
        max_attempts=args.max_attempts,
        skip_empty_answers=args.skip_empty_answers,
        seed=args.seed,
        solution_tail_chars=args.solution_tail_chars,
        max_correction_rounds=args.max_correction_rounds,
        formalizer_temperature=args.formalizer_temperature,
        formalizer_min_p=args.formalizer_min_p,
        formalizer_max_turns=args.formalizer_max_turns,
        formalizer_timeout_seconds=args.formalizer_timeout_seconds,
        formalizer_buffer_tokens=args.formalizer_buffer_tokens,
        formalizer_stream_text_window=args.formalizer_stream_text_window,
        use_python_tool=args.use_python_tool,
        use_lean_tool=args.use_lean_tool,
        jupyter_timeout=args.jupyter_timeout,
        lean_project_dir=args.lean_project_dir,
        lean_timeout_seconds=args.lean_timeout_seconds,
        lean_jobs=args.lean_jobs,
        lean_workspace_subdir=args.lean_workspace_subdir,
        lean_max_memory_megabytes=args.lean_max_memory_megabytes,
        verbose=args.verbose,
        log_formalization_progress=args.log_formalization_progress,
    )
    validate_cfg(cfg)
    return cfg


def make_backend_config(cfg: RunConjectureFormalizationConfig) -> VLLMHarmonyBackendConfig:
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


def make_lean_compiler_config(cfg: RunConjectureFormalizationConfig) -> LeanCompilerConfig:
    return LeanCompilerConfig(
        project_dir=cfg.lean_project_dir,
        workspace_subdir=cfg.lean_workspace_subdir,
        timeout_seconds=cfg.lean_timeout_seconds,
        lean_jobs=cfg.lean_jobs,
        max_memory_megabytes=cfg.lean_max_memory_megabytes,
        recipient_name="lean",
        tool_name="lean",
        treat_sorry_warning_as_failure=False,
        treat_any_warning_as_failure=False,
        auto_extract_code_block=True,
        cleanup_source_file=False,
    )


def make_formalizer_agent(cfg: RunConjectureFormalizationConfig) -> ConjectureFormalizerAgent:
    jupyter_cfg = JupyterKernelConfig(
        timeout_seconds=cfg.jupyter_timeout,
        init_on_create=False,
    )

    lean_cfg: Optional[LeanCompilerConfig] = None
    if cfg.use_lean_tool:
        lean_cfg = make_lean_compiler_config(cfg)

    formalizer_cfg = ConjectureFormalizerConfig(
        solution_tail_chars=cfg.solution_tail_chars,
        temperature=cfg.formalizer_temperature,
        min_p=cfg.formalizer_min_p,
        max_turns=cfg.formalizer_max_turns,
        timeout_seconds=cfg.formalizer_timeout_seconds,
        buffer_tokens=cfg.formalizer_buffer_tokens,
        stream_text_window=cfg.formalizer_stream_text_window,
        use_python_tool=cfg.use_python_tool,
        use_lean_tool=cfg.use_lean_tool,
        jupyter=jupyter_cfg,
        lean=lean_cfg,
    )
    return ConjectureFormalizerAgent(cfg=formalizer_cfg)