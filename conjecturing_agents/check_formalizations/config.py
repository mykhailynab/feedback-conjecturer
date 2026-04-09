from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import List

from conjecturing_agents.answer_checking.checker import AnswerCheckerConfig


# ============================================================
# Script config
# ============================================================

@dataclass
class CheckFormalizationsConfig:
    # Input / output
    formalizations_path: str = "logs/conjecture_formalization_logs/formalizations.jsonl"
    output_path: str = ""  # defaults to <formalizations_path parent>/check_results.jsonl

    # Lean compiler
    lean_project_dir: str = "."
    lean_workspace_subdir: str = ".conjecturing_agents/answer_checking"
    lean_timeout_seconds: int = 120
    lean_jobs: int = 4
    lean_max_memory_megabytes: int = 4 * 1024  # 0 = no limit

    # Heuristics
    use_string_match: bool = True
    use_lean_equiv: bool = True
    use_goedel_prover: bool = False
    use_goedel_disprover: bool = False

    # Goedel prover settings
    goedel_chat_template_path: str = "goedel_template.jinja"
    goedel_max_rounds: int = 2
    goedel_max_tokens: int = 16384
    goedel_temperature: float = 0.6
    goedel_top_p: float = 0.95
    goedel_repeat_penalty: float = 1.0
    goedel_context_tokens: int = 40960
    goedel_lean_workspace_subdir: str = ".conjecturing_agents/goedel_lean_runs"
    goedel_backend_type: str = "ollama"  # "ollama" | "vllm"
    goedel_ollama_model: str = "goedel-v2:latest"
    goedel_ollama_host: str = "http://localhost:11434"
    goedel_vllm_base_url: str = "http://0.0.0.0:8001/v1"
    goedel_vllm_model_name: str = "goedel"
    goedel_vllm_api_key: str = "sk-local"
    goedel_vllm_client_timeout: int = 240
    goedel_vllm_manage_server: bool = False
    goedel_vllm_model_path: str = ""
    goedel_vllm_port: int = 8001
    goedel_vllm_host: str = "0.0.0.0"
    goedel_vllm_server_timeout: int = 240
    goedel_vllm_server_log_path: str = "vllm_goedel_server.log"
    goedel_vllm_dtype: str = "bfloat16"
    goedel_vllm_kv_cache_dtype: str = "fp8_e4m3"
    goedel_vllm_gpu_memory_utilization: float = 0.96
    goedel_vllm_max_num_seqs: int = 32
    goedel_vllm_stream_interval: int = 200
    goedel_vllm_enable_prefix_caching: bool = True
    goedel_vllm_extra_server_args: List[str] = field(default_factory=list)
    goedel_tokenizer_path: str = "goedel_prover_hf_tokenizer"  # HF tokenizer for token counting

    # Logging / debug
    print_agent_conv: bool = False

    # Orchestration
    parallelism: int = 4
    max_records: int = 0  # 0 = all
    resume: bool = False  # --continue: re-check only undecided records from existing output

    # Logging
    verbose: bool = True


# ============================================================
# Validation
# ============================================================

def validate_cfg(cfg: CheckFormalizationsConfig) -> None:
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

    if errs:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errs))


# ============================================================
# Factory helpers
# ============================================================

def make_checker_config(cfg: CheckFormalizationsConfig) -> AnswerCheckerConfig:
    return AnswerCheckerConfig(
        lean_project_dir=cfg.lean_project_dir,
        lean_workspace_subdir=cfg.lean_workspace_subdir,
        lean_timeout_seconds=cfg.lean_timeout_seconds,
        lean_jobs=cfg.lean_jobs,
        lean_max_memory_megabytes=cfg.lean_max_memory_megabytes,
        use_string_match=cfg.use_string_match,
        use_lean_equiv=cfg.use_lean_equiv,
        use_goedel_prover=cfg.use_goedel_prover,
        use_goedel_disprover=cfg.use_goedel_disprover,
        goedel_chat_template_path=cfg.goedel_chat_template_path,
        goedel_max_rounds=cfg.goedel_max_rounds,
        goedel_max_tokens=cfg.goedel_max_tokens,
        goedel_temperature=cfg.goedel_temperature,
        goedel_top_p=cfg.goedel_top_p,
        goedel_repeat_penalty=cfg.goedel_repeat_penalty,
        goedel_context_tokens=cfg.goedel_context_tokens,
        goedel_lean_workspace_subdir=cfg.goedel_lean_workspace_subdir,
        goedel_backend_type=cfg.goedel_backend_type,
        goedel_ollama_model=cfg.goedel_ollama_model,
        goedel_ollama_host=cfg.goedel_ollama_host,
        goedel_vllm_base_url=cfg.goedel_vllm_base_url,
        goedel_vllm_model_name=cfg.goedel_vllm_model_name,
        goedel_vllm_api_key=cfg.goedel_vllm_api_key,
        goedel_vllm_client_timeout=cfg.goedel_vllm_client_timeout,
        goedel_vllm_manage_server=cfg.goedel_vllm_manage_server,
        goedel_vllm_model_path=cfg.goedel_vllm_model_path,
        goedel_vllm_port=cfg.goedel_vllm_port,
        goedel_vllm_host=cfg.goedel_vllm_host,
        goedel_vllm_server_timeout=cfg.goedel_vllm_server_timeout,
        goedel_vllm_server_log_path=cfg.goedel_vllm_server_log_path,
        goedel_vllm_dtype=cfg.goedel_vllm_dtype,
        goedel_vllm_kv_cache_dtype=cfg.goedel_vllm_kv_cache_dtype,
        goedel_vllm_gpu_memory_utilization=cfg.goedel_vllm_gpu_memory_utilization,
        goedel_vllm_max_num_seqs=cfg.goedel_vllm_max_num_seqs,
        goedel_vllm_stream_interval=cfg.goedel_vllm_stream_interval,
        goedel_vllm_enable_prefix_caching=cfg.goedel_vllm_enable_prefix_caching,
        goedel_vllm_extra_server_args=cfg.goedel_vllm_extra_server_args,
        goedel_tokenizer_path=cfg.goedel_tokenizer_path,
        goedel_print_agent_conv=cfg.print_agent_conv,
    )


# ============================================================
# CLI
# ============================================================

def parse_args_and_validate() -> CheckFormalizationsConfig:
    p = ArgumentParser(
        description=(
            "Check formalized answers against ground-truth answers from a "
            "formalizations.jsonl file. Applies string-match and Lean equivalence "
            "proof heuristics, writes check_results.jsonl."
        )
    )

    # Input / output
    p.add_argument(
        "--formalizations-path",
        default=CheckFormalizationsConfig.formalizations_path,
        help="Path to formalizations.jsonl produced by run_conjecture_formalization.py.",
    )
    p.add_argument(
        "--output-path",
        default=CheckFormalizationsConfig.output_path,
        help=(
            "Where to write check_results.jsonl. "
            "Defaults to <formalizations_path parent>/check_results.jsonl."
        ),
    )

    # Lean compiler
    p.add_argument(
        "--lean-project-dir",
        default=CheckFormalizationsConfig.lean_project_dir,
        help="Path to a Lean project with Mathlib configured (needed for --lean-equiv).",
    )
    p.add_argument(
        "--lean-workspace-subdir",
        default=CheckFormalizationsConfig.lean_workspace_subdir,
        help="Subdirectory inside lean_project_dir where temporary .lean files are written.",
    )
    p.add_argument(
        "--lean-timeout-seconds",
        type=int,
        default=CheckFormalizationsConfig.lean_timeout_seconds,
        help="Per-file Lean compilation timeout in seconds.",
    )
    p.add_argument(
        "--lean-jobs",
        type=int,
        default=CheckFormalizationsConfig.lean_jobs,
        help="Number of parallel jobs passed to lake env lean -j.",
    )
    p.add_argument(
        "--lean-max-memory-megabytes",
        type=int,
        default=CheckFormalizationsConfig.lean_max_memory_megabytes,
        help=(
            "Maximum virtual memory (megabytes) for each lake/lean subprocess. "
            "0 means no limit. Example: 4 * 1024 for 4 GiB."
        ),
    )

    # Heuristics
    p.add_argument(
        "--no-string-match",
        dest="use_string_match",
        action="store_false",
        default=CheckFormalizationsConfig.use_string_match,
        help="Disable whitespace-normalized string match (heuristic 1).",
    )
    p.add_argument(
        "--no-lean",
        dest="use_lean_equiv",
        action="store_false",
        default=CheckFormalizationsConfig.use_lean_equiv,
        help="Disable Lean equivalence proof check (heuristic 2).",
    )
    p.add_argument(
        "--goedel",
        dest="use_goedel_prover",
        action="store_true",
        default=CheckFormalizationsConfig.use_goedel_prover,
        help="Enable Goedel-prover equivalence check (heuristic 3).",
    )
    p.add_argument(
        "--goedel-disprover",
        dest="use_goedel_disprover",
        action="store_true",
        default=CheckFormalizationsConfig.use_goedel_disprover,
        help=(
            "After a failed Goedel proof, attempt to disprove equivalence (heuristic 4). "
            "Requires --goedel. A successful disproof yields equivalent=False."
        ),
    )
    p.add_argument(
        "--goedel-chat-template-path",
        default=CheckFormalizationsConfig.goedel_chat_template_path,
        help="Path to the Jinja2 chat template for the Goedel model.",
    )
    p.add_argument(
        "--goedel-max-rounds",
        type=int,
        default=CheckFormalizationsConfig.goedel_max_rounds,
        help="Self-correction rounds for Goedel prover (0 = initial attempt only).",
    )
    p.add_argument(
        "--goedel-max-tokens",
        type=int,
        default=CheckFormalizationsConfig.goedel_max_tokens,
        help="Max tokens to generate per Goedel round.",
    )
    p.add_argument(
        "--goedel-temperature",
        type=float,
        default=CheckFormalizationsConfig.goedel_temperature,
        help="Sampling temperature for the Goedel model.",
    )
    p.add_argument(
        "--goedel-top-p",
        type=float,
        default=CheckFormalizationsConfig.goedel_top_p,
        help="Top-p (nucleus) sampling parameter for the Goedel model.",
    )
    p.add_argument(
        "--goedel-repeat-penalty",
        type=float,
        default=CheckFormalizationsConfig.goedel_repeat_penalty,
        help="Repetition penalty for the Goedel model (1.0 = no penalty).",
    )
    p.add_argument(
        "--goedel-context-tokens",
        type=int,
        default=CheckFormalizationsConfig.goedel_context_tokens,
        help="Model max context length for pre-flight token budget checks.",
    )
    p.add_argument(
        "--goedel-backend",
        dest="goedel_backend_type",
        choices=["ollama", "vllm"],
        default=CheckFormalizationsConfig.goedel_backend_type,
        help="Inference backend for the Goedel prover.",
    )
    p.add_argument(
        "--goedel-ollama-model",
        default=CheckFormalizationsConfig.goedel_ollama_model,
        help="Ollama model name (backend=ollama).",
    )
    p.add_argument(
        "--goedel-ollama-host",
        default=CheckFormalizationsConfig.goedel_ollama_host,
        help="Ollama server URL (backend=ollama).",
    )
    p.add_argument(
        "--goedel-vllm-base-url",
        default=CheckFormalizationsConfig.goedel_vllm_base_url,
        help="vLLM OpenAI-compatible base URL (backend=vllm).",
    )
    p.add_argument(
        "--goedel-vllm-model-name",
        default=CheckFormalizationsConfig.goedel_vllm_model_name,
        help="Served model name for the vLLM endpoint (backend=vllm).",
    )
    p.add_argument(
        "--goedel-vllm-api-key",
        default=CheckFormalizationsConfig.goedel_vllm_api_key,
        help="API key for the vLLM endpoint (backend=vllm).",
    )
    p.add_argument(
        "--goedel-vllm-client-timeout",
        type=int,
        default=CheckFormalizationsConfig.goedel_vllm_client_timeout,
        help="HTTP client timeout in seconds for vLLM requests (backend=vllm).",
    )
    p.add_argument(
        "--goedel-vllm-manage-server",
        dest="goedel_vllm_manage_server",
        action="store_true",
        default=CheckFormalizationsConfig.goedel_vllm_manage_server,
        help="Start and manage a vLLM server subprocess (backend=vllm).",
    )
    p.add_argument(
        "--goedel-vllm-model-path",
        default=CheckFormalizationsConfig.goedel_vllm_model_path,
        help="Path to the model weights (required when --goedel-vllm-manage-server is set).",
    )
    p.add_argument(
        "--goedel-vllm-port",
        type=int,
        default=CheckFormalizationsConfig.goedel_vllm_port,
        help="Port for the managed vLLM server (backend=vllm, manage-server=True).",
    )
    p.add_argument(
        "--goedel-vllm-host",
        default=CheckFormalizationsConfig.goedel_vllm_host,
        help="Host for the managed vLLM server (backend=vllm, manage-server=True).",
    )
    p.add_argument(
        "--goedel-vllm-server-timeout",
        type=int,
        default=CheckFormalizationsConfig.goedel_vllm_server_timeout,
        help="Seconds to wait for the vLLM server to become ready (backend=vllm, manage-server=True).",
    )
    p.add_argument(
        "--goedel-vllm-server-log-path",
        default=CheckFormalizationsConfig.goedel_vllm_server_log_path,
        help="File path for vLLM server stdout/stderr logs (backend=vllm, manage-server=True).",
    )
    p.add_argument(
        "--goedel-vllm-dtype",
        default=CheckFormalizationsConfig.goedel_vllm_dtype,
        help="Model weight dtype passed to vLLM (backend=vllm, manage-server=True).",
    )
    p.add_argument(
        "--goedel-vllm-kv-cache-dtype",
        default=CheckFormalizationsConfig.goedel_vllm_kv_cache_dtype,
        help="KV-cache dtype passed to vLLM (backend=vllm, manage-server=True).",
    )
    p.add_argument(
        "--goedel-vllm-gpu-memory-utilization",
        type=float,
        default=CheckFormalizationsConfig.goedel_vllm_gpu_memory_utilization,
        help="GPU memory utilization fraction for vLLM (backend=vllm, manage-server=True).",
    )
    p.add_argument(
        "--goedel-vllm-max-num-seqs",
        type=int,
        default=CheckFormalizationsConfig.goedel_vllm_max_num_seqs,
        help="Maximum number of concurrent sequences for vLLM (backend=vllm, manage-server=True).",
    )
    p.add_argument(
        "--goedel-vllm-stream-interval",
        type=int,
        default=CheckFormalizationsConfig.goedel_vllm_stream_interval,
        help="Token streaming interval for vLLM (backend=vllm, manage-server=True).",
    )
    p.add_argument(
        "--goedel-vllm-no-prefix-caching",
        dest="goedel_vllm_enable_prefix_caching",
        action="store_false",
        default=CheckFormalizationsConfig.goedel_vllm_enable_prefix_caching,
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
        default=CheckFormalizationsConfig.goedel_tokenizer_path,
        help="HF tokenizer path for exact token counting (required when --goedel is set).",
    )
    p.add_argument(
        "--print-agent-conv",
        dest="print_agent_conv",
        action="store_true",
        default=CheckFormalizationsConfig.print_agent_conv,
        help="Print Goedel prover prompts and generated tokens to stdout in real time.",
    )

    # Orchestration
    p.add_argument(
        "--parallelism",
        type=int,
        default=CheckFormalizationsConfig.parallelism,
        help="Number of concurrent checker threads.",
    )
    p.add_argument(
        "--max-records",
        type=int,
        default=CheckFormalizationsConfig.max_records,
        help="Maximum number of records to check (0 = all).",
    )

    # Continue / resume
    p.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        default=CheckFormalizationsConfig.resume,
        help=(
            "Resume from an existing output file. Already-decided records "
            "(equivalent=True or equivalent=False) are kept as-is; undecided records "
            "(equivalent=None) are re-checked."
        ),
    )

    # Logging
    p.add_argument(
        "--verbose",
        action="store_true",
        default=CheckFormalizationsConfig.verbose,
        help="Print progress to stdout.",
    )
    p.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="Suppress progress output.",
    )

    args = p.parse_args()

    cfg = CheckFormalizationsConfig(
        formalizations_path=args.formalizations_path,
        output_path=args.output_path,
        lean_project_dir=args.lean_project_dir,
        lean_workspace_subdir=args.lean_workspace_subdir,
        lean_timeout_seconds=args.lean_timeout_seconds,
        lean_jobs=args.lean_jobs,
        lean_max_memory_megabytes=args.lean_max_memory_megabytes,
        use_string_match=args.use_string_match,
        use_lean_equiv=args.use_lean_equiv,
        use_goedel_prover=args.use_goedel_prover,
        use_goedel_disprover=args.use_goedel_disprover,
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
        print_agent_conv=args.print_agent_conv,
        parallelism=args.parallelism,
        max_records=args.max_records,
        resume=args.resume,
        verbose=args.verbose,
    )
    validate_cfg(cfg)
    return cfg
