from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import List


# ============================================================
# Script config
# ============================================================

@dataclass
class ProveFormalizationsConfig:
    # Input / output
    formalizations_path: str = "logs/conjecture_formalization_logs/formalizations.jsonl"
    output_path: str = ""  # defaults to <formalizations_path parent>/prove_results.jsonl
    events_path: str = ""  # defaults to <output_path parent>/prove_goedel_events.jsonl

    # Lean compiler (shared by proof and disproof agents)
    lean_project_dir: str = "."
    lean_workspace_subdir: str = ".conjecturing_agents/prove_formalizations"
    lean_timeout_seconds: int = 120
    lean_jobs: int = 4
    lean_max_memory_megabytes: int = 8 * 1024  # 0 = no limit

    # Proof / disproof
    # Number of independent proof attempts per record.
    proof_retries: int = 1
    # Run proof and disproof concurrently in separate threads per record.
    # Uses parallelism // 2 outer workers (each spawning 2 sub-threads).
    enable_parallel_disproof: bool = False
    # Run disproof sequentially after a failed proof attempt, within the same
    # worker thread.  Uses parallelism workers (no division).
    # Mutually exclusive with enable_parallel_disproof.
    enable_sequential_disproof: bool = False
    # Number of independent disproof attempts per record (used when either
    # disproof flag is set).
    disproof_retries: int = 1

    # Goedel prover settings
    goedel_chat_template_path: str = "goedel_template.jinja"
    goedel_max_rounds: int = 2
    goedel_max_tokens: int = 16384
    goedel_temperature: float = 0.6
    goedel_top_p: float = 0.95
    goedel_repeat_penalty: float = 1.0
    goedel_context_tokens: int = 40960
    goedel_lean_workspace_subdir: str = ".conjecturing_agents/goedel_lean_runs_prove"
    goedel_backend_type: str = "ollama"  # "ollama" | "vllm"
    goedel_ollama_model: str = "goedel-v2:latest"
    goedel_ollama_host: str = "http://localhost:11434"
    goedel_ollama_client_timeout: int = 3600
    goedel_vllm_base_url: str = "http://0.0.0.0:8001/v1"
    goedel_vllm_model_name: str = "goedel"
    goedel_vllm_api_key: str = "sk-local"
    goedel_vllm_client_timeout: int = 3600
    goedel_vllm_manage_server: bool = False
    goedel_vllm_model_path: str = ""
    goedel_vllm_port: int = 8001
    goedel_vllm_host: str = "0.0.0.0"
    goedel_vllm_server_timeout: int = 3600
    goedel_vllm_server_log_path: str = "vllm_goedel_server.log"
    goedel_vllm_dtype: str = "bfloat16"
    goedel_vllm_kv_cache_dtype: str = "fp8_e4m3"
    goedel_vllm_gpu_memory_utilization: float = 0.96
    goedel_vllm_max_num_seqs: int = 32
    goedel_vllm_stream_interval: int = 200
    goedel_vllm_enable_prefix_caching: bool = True
    goedel_vllm_extra_server_args: List[str] = field(default_factory=list)
    goedel_tokenizer_path: str = "goedel_prover_hf_tokenizer"

    # Progressive token budget.
    # Stop each prover session when its rendered prompt reaches this many
    # tokens.  0 = unlimited.  Use --continue with a higher value to resume
    # sessions that were cut off.
    limit_prover_tokens: int = 0

    # Logging / debug
    print_agent_conv: bool = False

    # Orchestration
    parallelism: int = 4
    max_records: int = 0  # 0 = all
    resume: bool = False  # re-prove only undecided records

    # Logging
    verbose: bool = True


# ============================================================
# Validation
# ============================================================

def validate_cfg(cfg: ProveFormalizationsConfig) -> None:
    errs: List[str] = []

    if not cfg.formalizations_path:
        errs.append("formalizations_path must be non-empty")
    if not cfg.lean_project_dir:
        errs.append("lean_project_dir must be non-empty")
    if cfg.lean_timeout_seconds <= 0:
        errs.append("lean_timeout_seconds must be >= 1")
    if cfg.lean_jobs <= 0:
        errs.append("lean_jobs must be >= 1")
    if cfg.parallelism <= 0:
        errs.append("parallelism must be >= 1")
    if cfg.max_records < 0:
        errs.append("max_records must be >= 0")
    if cfg.proof_retries <= 0:
        errs.append("proof_retries must be >= 1")
    if cfg.disproof_retries <= 0:
        errs.append("disproof_retries must be >= 1")
    if cfg.enable_parallel_disproof and cfg.enable_sequential_disproof:
        errs.append(
            "--enable-parallel-disproof and --enable-sequential-disproof are mutually exclusive"
        )
    if cfg.enable_parallel_disproof and cfg.parallelism % 2 != 0:
        errs.append(
            "when --enable-parallel-disproof is set, --parallelism must be divisible by 2"
        )

    if errs:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errs))


# ============================================================
# Factory helpers
# ============================================================

def make_goedel_prover_config(cfg: ProveFormalizationsConfig, workspace_suffix: str = ""):
    """Build a ``GoedelProverConfig`` from the flat script config."""
    from conjecturing_agents.agents.goedel_prover import GoedelProverConfig
    from conjecturing_agents.tool_calling_backends.lean4_compiler import LeanCompilerConfig

    subdir = cfg.goedel_lean_workspace_subdir
    if workspace_suffix:
        subdir = subdir + "_" + workspace_suffix
    lean_cfg = LeanCompilerConfig(
        project_dir=cfg.lean_project_dir,
        workspace_subdir=subdir,
        timeout_seconds=cfg.lean_timeout_seconds,
        lean_jobs=cfg.lean_jobs,
        max_memory_megabytes=cfg.lean_max_memory_megabytes,
        treat_sorry_warning_as_failure=True,
        treat_any_warning_as_failure=False,
    )
    return GoedelProverConfig(
        chat_template_path=cfg.goedel_chat_template_path,
        max_rounds=cfg.goedel_max_rounds,
        max_tokens=cfg.goedel_max_tokens,
        temperature=cfg.goedel_temperature,
        top_p=cfg.goedel_top_p,
        repeat_penalty=cfg.goedel_repeat_penalty,
        context_tokens=cfg.goedel_context_tokens,
        lean=lean_cfg,
    )


def make_goedel_backend(cfg: ProveFormalizationsConfig):
    """Instantiate the appropriate ``RawBackend`` for the Goedel prover."""
    if cfg.goedel_backend_type == "vllm":
        from conjecturing_agents.inference_backends.vllm_raw import (
            VLLMRawBackend,
            VLLMRawConfig,
        )
        return VLLMRawBackend(VLLMRawConfig(
            base_url=cfg.goedel_vllm_base_url,
            served_model_name=cfg.goedel_vllm_model_name,
            api_key=cfg.goedel_vllm_api_key,
            client_timeout=cfg.goedel_vllm_client_timeout,
            tokenizer_path=cfg.goedel_tokenizer_path,
            manage_server=cfg.goedel_vllm_manage_server,
            model_path=cfg.goedel_vllm_model_path,
            port=cfg.goedel_vllm_port,
            host=cfg.goedel_vllm_host,
            server_timeout=cfg.goedel_vllm_server_timeout,
            server_log_path=cfg.goedel_vllm_server_log_path,
            dtype=cfg.goedel_vllm_dtype,
            kv_cache_dtype=cfg.goedel_vllm_kv_cache_dtype,
            context_tokens=cfg.goedel_context_tokens,
            gpu_memory_utilization=cfg.goedel_vllm_gpu_memory_utilization,
            max_num_seqs=cfg.goedel_vllm_max_num_seqs,
            stream_interval=cfg.goedel_vllm_stream_interval,
            enable_prefix_caching=cfg.goedel_vllm_enable_prefix_caching,
            extra_server_args=cfg.goedel_vllm_extra_server_args,
        ))
    else:
        from conjecturing_agents.inference_backends.ollama_backend import (
            OllamaBackend,
            OllamaConfig,
        )
        return OllamaBackend(OllamaConfig(
            model=cfg.goedel_ollama_model,
            host=cfg.goedel_ollama_host,
            client_timeout=cfg.goedel_ollama_client_timeout,
            tokenizer_path=cfg.goedel_tokenizer_path,
        ))


# ============================================================
# CLI
# ============================================================

def parse_args_and_validate() -> ProveFormalizationsConfig:
    p = ArgumentParser(
        description=(
            "Attempt to prove (and optionally disprove) formalized conjectures "
            "from a formalizations.jsonl file using the Goedel prover. "
            "Writes prove_results.jsonl alongside the input file."
        )
    )

    # Input / output
    p.add_argument(
        "--formalizations-path",
        default=ProveFormalizationsConfig.formalizations_path,
        help="Path to formalizations.jsonl produced by run_conjecture_formalization.py.",
    )
    p.add_argument(
        "--output-path",
        default=ProveFormalizationsConfig.output_path,
        help=(
            "Where to write prove_results.jsonl. "
            "Defaults to <formalizations_path parent>/prove_results.jsonl."
        ),
    )
    p.add_argument(
        "--events-path",
        default=ProveFormalizationsConfig.events_path,
        help=(
            "Where to write the raw generation event log (prove_goedel_events.jsonl). "
            "Defaults to <output_path parent>/prove_goedel_events.jsonl."
        ),
    )

    # Lean compiler
    p.add_argument(
        "--lean-project-dir",
        default=ProveFormalizationsConfig.lean_project_dir,
        help="Path to a Lean project with Mathlib configured.",
    )
    p.add_argument(
        "--lean-workspace-subdir",
        default=ProveFormalizationsConfig.lean_workspace_subdir,
        help="Subdirectory inside lean_project_dir where temporary .lean files are written.",
    )
    p.add_argument(
        "--lean-timeout-seconds",
        type=int,
        default=ProveFormalizationsConfig.lean_timeout_seconds,
        help="Per-file Lean compilation timeout in seconds.",
    )
    p.add_argument(
        "--lean-jobs",
        type=int,
        default=ProveFormalizationsConfig.lean_jobs,
        help="Number of parallel jobs passed to lake env lean -j.",
    )
    p.add_argument(
        "--lean-max-memory-megabytes",
        type=int,
        default=ProveFormalizationsConfig.lean_max_memory_megabytes,
        help=(
            "Maximum virtual memory (megabytes) for each lake/lean subprocess. "
            "0 means no limit."
        ),
    )

    # Proof / disproof
    p.add_argument(
        "--proof-retries",
        type=int,
        default=ProveFormalizationsConfig.proof_retries,
        help="Number of independent proof attempts per record (pass@N). Default: 1.",
    )
    p.add_argument(
        "--enable-parallel-disproof",
        dest="enable_parallel_disproof",
        action="store_true",
        default=ProveFormalizationsConfig.enable_parallel_disproof,
        help=(
            "For each record, run proof and negation-disproof concurrently in "
            "separate threads. Uses parallelism // 2 outer workers. Whichever "
            "succeeds first cancels the other."
        ),
    )
    p.add_argument(
        "--enable-sequential-disproof",
        dest="enable_sequential_disproof",
        action="store_true",
        default=ProveFormalizationsConfig.enable_sequential_disproof,
        help=(
            "After a failed proof attempt, run a disproof of the negated theorem "
            "in the same worker thread. Uses --parallelism workers (no division). "
            "Mutually exclusive with --enable-parallel-disproof."
        ),
    )
    p.add_argument(
        "--disproof-retries",
        type=int,
        default=ProveFormalizationsConfig.disproof_retries,
        help=(
            "Number of independent disproof attempts per record. "
            "Used when --enable-parallel-disproof or --enable-sequential-disproof "
            "is set. Default: 1."
        ),
    )

    # Goedel prover settings
    p.add_argument(
        "--goedel-chat-template-path",
        default=ProveFormalizationsConfig.goedel_chat_template_path,
        help="Path to the Jinja2 chat template for the Goedel model.",
    )
    p.add_argument(
        "--goedel-max-rounds",
        type=int,
        default=ProveFormalizationsConfig.goedel_max_rounds,
        help="Self-correction rounds for Goedel prover (0 = initial attempt only).",
    )
    p.add_argument(
        "--goedel-max-tokens",
        type=int,
        default=ProveFormalizationsConfig.goedel_max_tokens,
        help="Max tokens to generate per Goedel round.",
    )
    p.add_argument(
        "--goedel-temperature",
        type=float,
        default=ProveFormalizationsConfig.goedel_temperature,
        help="Sampling temperature for the Goedel model.",
    )
    p.add_argument(
        "--goedel-top-p",
        type=float,
        default=ProveFormalizationsConfig.goedel_top_p,
        help="Top-p (nucleus) sampling parameter for the Goedel model.",
    )
    p.add_argument(
        "--goedel-repeat-penalty",
        type=float,
        default=ProveFormalizationsConfig.goedel_repeat_penalty,
        help="Repetition penalty for the Goedel model (1.0 = no penalty).",
    )
    p.add_argument(
        "--goedel-context-tokens",
        type=int,
        default=ProveFormalizationsConfig.goedel_context_tokens,
        help="Model max context length for pre-flight token budget checks.",
    )
    p.add_argument(
        "--goedel-backend",
        dest="goedel_backend_type",
        choices=["ollama", "vllm"],
        default=ProveFormalizationsConfig.goedel_backend_type,
        help="Inference backend for the Goedel prover.",
    )
    p.add_argument(
        "--goedel-ollama-model",
        default=ProveFormalizationsConfig.goedel_ollama_model,
        help="Ollama model name (backend=ollama).",
    )
    p.add_argument(
        "--goedel-ollama-host",
        default=ProveFormalizationsConfig.goedel_ollama_host,
        help="Ollama server URL (backend=ollama).",
    )
    p.add_argument(
        "--goedel-ollama-client-timeout",
        type=int,
        default=ProveFormalizationsConfig.goedel_ollama_client_timeout,
        help="HTTP client timeout in seconds for Ollama requests (backend=ollama).",
    )
    p.add_argument(
        "--goedel-vllm-base-url",
        default=ProveFormalizationsConfig.goedel_vllm_base_url,
        help="vLLM OpenAI-compatible base URL (backend=vllm).",
    )
    p.add_argument(
        "--goedel-vllm-model-name",
        default=ProveFormalizationsConfig.goedel_vllm_model_name,
        help="Served model name for the vLLM endpoint (backend=vllm).",
    )
    p.add_argument(
        "--goedel-vllm-api-key",
        default=ProveFormalizationsConfig.goedel_vllm_api_key,
        help="API key for the vLLM endpoint (backend=vllm).",
    )
    p.add_argument(
        "--goedel-vllm-client-timeout",
        type=int,
        default=ProveFormalizationsConfig.goedel_vllm_client_timeout,
        help="HTTP client timeout in seconds for vLLM requests (backend=vllm).",
    )
    p.add_argument(
        "--goedel-vllm-manage-server",
        dest="goedel_vllm_manage_server",
        action="store_true",
        default=ProveFormalizationsConfig.goedel_vllm_manage_server,
        help="Start and manage a vLLM server subprocess (backend=vllm).",
    )
    p.add_argument(
        "--goedel-vllm-model-path",
        default=ProveFormalizationsConfig.goedel_vllm_model_path,
        help="Path to the model weights (required when --goedel-vllm-manage-server is set).",
    )
    p.add_argument(
        "--goedel-vllm-port",
        type=int,
        default=ProveFormalizationsConfig.goedel_vllm_port,
        help="Port for the managed vLLM server (backend=vllm, manage-server=True).",
    )
    p.add_argument(
        "--goedel-vllm-host",
        default=ProveFormalizationsConfig.goedel_vllm_host,
        help="Host for the managed vLLM server (backend=vllm, manage-server=True).",
    )
    p.add_argument(
        "--goedel-vllm-server-timeout",
        type=int,
        default=ProveFormalizationsConfig.goedel_vllm_server_timeout,
        help="Seconds to wait for the vLLM server to become ready (backend=vllm, manage-server=True).",
    )
    p.add_argument(
        "--goedel-vllm-server-log-path",
        default=ProveFormalizationsConfig.goedel_vllm_server_log_path,
        help="File path for vLLM server stdout/stderr logs (backend=vllm, manage-server=True).",
    )
    p.add_argument(
        "--goedel-vllm-dtype",
        default=ProveFormalizationsConfig.goedel_vllm_dtype,
        help="Model weight dtype passed to vLLM (backend=vllm, manage-server=True).",
    )
    p.add_argument(
        "--goedel-vllm-kv-cache-dtype",
        default=ProveFormalizationsConfig.goedel_vllm_kv_cache_dtype,
        help="KV-cache dtype passed to vLLM (backend=vllm, manage-server=True).",
    )
    p.add_argument(
        "--goedel-vllm-gpu-memory-utilization",
        type=float,
        default=ProveFormalizationsConfig.goedel_vllm_gpu_memory_utilization,
        help="GPU memory utilization fraction for vLLM (backend=vllm, manage-server=True).",
    )
    p.add_argument(
        "--goedel-vllm-max-num-seqs",
        type=int,
        default=ProveFormalizationsConfig.goedel_vllm_max_num_seqs,
        help="Maximum number of concurrent sequences for vLLM (backend=vllm, manage-server=True).",
    )
    p.add_argument(
        "--goedel-vllm-stream-interval",
        type=int,
        default=ProveFormalizationsConfig.goedel_vllm_stream_interval,
        help="Token streaming interval for vLLM (backend=vllm, manage-server=True).",
    )
    p.add_argument(
        "--goedel-vllm-no-prefix-caching",
        dest="goedel_vllm_enable_prefix_caching",
        action="store_false",
        default=ProveFormalizationsConfig.goedel_vllm_enable_prefix_caching,
        help="Disable prefix caching in vLLM (backend=vllm, manage-server=True).",
    )
    p.add_argument(
        "--goedel-vllm-extra-server-args",
        nargs="*",
        default=[],
        help="Extra CLI arguments forwarded verbatim to the vLLM server (backend=vllm, manage-server=True).",
    )
    p.add_argument(
        "--goedel-tokenizer-path",
        default=ProveFormalizationsConfig.goedel_tokenizer_path,
        help="HF tokenizer path for exact token counting.",
    )
    p.add_argument(
        "--limit-prover-tokens",
        type=int,
        default=ProveFormalizationsConfig.limit_prover_tokens,
        help=(
            "Stop each prover session when its rendered prompt reaches this many tokens. "
            "0 = unlimited. Use --continue with a higher value to resume sessions that "
            "were cut off, extending the budget progressively (e.g. 4000 → 8000 → 40960)."
        ),
    )
    p.add_argument(
        "--print-agent-conv",
        dest="print_agent_conv",
        action="store_true",
        default=ProveFormalizationsConfig.print_agent_conv,
        help="Print Goedel prover prompts and generated tokens to stdout in real time.",
    )

    # Orchestration
    p.add_argument(
        "--parallelism",
        type=int,
        default=ProveFormalizationsConfig.parallelism,
        help=(
            "Number of concurrent workers. With --enable-parallel-disproof, "
            "parallelism // 2 outer workers are used (each spawning 2 sub-threads)."
        ),
    )
    p.add_argument(
        "--max-records",
        type=int,
        default=ProveFormalizationsConfig.max_records,
        help="Maximum number of records to process (0 = all).",
    )
    p.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        default=ProveFormalizationsConfig.resume,
        help=(
            "Resume from an existing output file. Already-decided records "
            "(proved or disproved) are kept; undecided records are re-run."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=ProveFormalizationsConfig.verbose,
        help="Print progress to stdout.",
    )
    p.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="Suppress progress output.",
    )

    args = p.parse_args()

    cfg = ProveFormalizationsConfig(
        formalizations_path=args.formalizations_path,
        output_path=args.output_path,
        events_path=args.events_path,
        lean_project_dir=args.lean_project_dir,
        lean_workspace_subdir=args.lean_workspace_subdir,
        lean_timeout_seconds=args.lean_timeout_seconds,
        lean_jobs=args.lean_jobs,
        lean_max_memory_megabytes=args.lean_max_memory_megabytes,
        proof_retries=args.proof_retries,
        enable_parallel_disproof=args.enable_parallel_disproof,
        enable_sequential_disproof=args.enable_sequential_disproof,
        disproof_retries=args.disproof_retries,
        goedel_chat_template_path=args.goedel_chat_template_path,
        goedel_max_rounds=args.goedel_max_rounds,
        goedel_max_tokens=args.goedel_max_tokens,
        goedel_temperature=args.goedel_temperature,
        goedel_top_p=args.goedel_top_p,
        goedel_repeat_penalty=args.goedel_repeat_penalty,
        goedel_context_tokens=args.goedel_context_tokens,
        goedel_backend_type=args.goedel_backend_type,
        goedel_ollama_model=args.goedel_ollama_model,
        goedel_ollama_host=args.goedel_ollama_host,
        goedel_ollama_client_timeout=args.goedel_ollama_client_timeout,
        goedel_vllm_base_url=args.goedel_vllm_base_url,
        goedel_vllm_model_name=args.goedel_vllm_model_name,
        goedel_vllm_api_key=args.goedel_vllm_api_key,
        goedel_vllm_client_timeout=args.goedel_vllm_client_timeout,
        goedel_vllm_manage_server=args.goedel_vllm_manage_server,
        goedel_vllm_model_path=args.goedel_vllm_model_path,
        goedel_vllm_port=args.goedel_vllm_port,
        goedel_vllm_host=args.goedel_vllm_host,
        goedel_vllm_server_timeout=args.goedel_vllm_server_timeout,
        goedel_vllm_server_log_path=args.goedel_vllm_server_log_path,
        goedel_vllm_dtype=args.goedel_vllm_dtype,
        goedel_vllm_kv_cache_dtype=args.goedel_vllm_kv_cache_dtype,
        goedel_vllm_gpu_memory_utilization=args.goedel_vllm_gpu_memory_utilization,
        goedel_vllm_max_num_seqs=args.goedel_vllm_max_num_seqs,
        goedel_vllm_stream_interval=args.goedel_vllm_stream_interval,
        goedel_vllm_enable_prefix_caching=args.goedel_vllm_enable_prefix_caching,
        goedel_vllm_extra_server_args=args.goedel_vllm_extra_server_args or [],
        goedel_tokenizer_path=args.goedel_tokenizer_path,
        limit_prover_tokens=args.limit_prover_tokens,
        print_agent_conv=args.print_agent_conv,
        parallelism=args.parallelism,
        max_records=args.max_records,
        resume=args.resume,
        verbose=args.verbose,
    )
    validate_cfg(cfg)
    return cfg
