from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass, field, replace
from typing import List

from conjecturing_agents.agents.goedel_prover.config import GoedelProverConfig
from conjecturing_agents.agents.tir_prover.config import TIRProverConfig
from conjecturing_agents.inference_backends.ollama_raw import OllamaConfig
from conjecturing_agents.inference_backends.ollama_tir import OllamaTIRConfig
from conjecturing_agents.inference_backends.llamacpp_tir import LlamaCppTIRConfig
from conjecturing_agents.inference_backends.vllm_raw import VLLMRawConfig
from conjecturing_agents.tool_calling_backends.lean4_compiler import LeanCompilerConfig


# ============================================================
# Script config
# ============================================================

@dataclass
class ProveFormalizationsConfig:
    # Input / output
    formalizations_path: str = "logs/conjecture_formalization_logs/formalizations.jsonl"
    output_path: str = ""  # defaults to <formalizations_path parent>/prove_results.jsonl
    events_path: str = ""  # defaults to <output_path parent>/prover_events.jsonl

    # Lean compiler (shared base settings)
    lean: LeanCompilerConfig = field(
        default_factory=lambda: LeanCompilerConfig(
            project_dir=".",
            workspace_subdir=".conjecturing_agents/prove_formalizations",
            max_memory_megabytes=8 * 1024,
        )
    )

    # Proof / disproof
    proof_retries: int = 1
    enable_parallel_disproof: bool = False
    enable_sequential_disproof: bool = False
    disproof_retries: int = 1

    # Prover type selection: "goedel" | "tir"
    prover_type: str = "goedel"

    # Goedel prover (composed sub-configs)
    goedel: GoedelProverConfig = field(default_factory=GoedelProverConfig)
    goedel_backend_type: str = "ollama"  # "ollama" | "vllm"
    goedel_ollama: OllamaConfig = field(default_factory=OllamaConfig)
    goedel_vllm: VLLMRawConfig = field(default_factory=VLLMRawConfig)
    goedel_lean_workspace_subdir: str = ".conjecturing_agents/goedel_lean_runs_prove"
    # Multiple Ollama hosts for load-balanced multi-GPU setups.
    goedel_ollama_hosts: List[str] = field(default_factory=list)
    goedel_ollama_max_concurrent: int = 6

    # TIR prover (composed sub-configs)
    tir: TIRProverConfig = field(default_factory=TIRProverConfig)
    tir_backend_type: str = "ollama"  # "ollama" | "llamacpp"
    tir_ollama: OllamaTIRConfig = field(default_factory=OllamaTIRConfig)
    tir_llamacpp: LlamaCppTIRConfig = field(default_factory=LlamaCppTIRConfig)
    tir_lean_workspace_subdir: str = ".conjecturing_agents/tir_lean_runs_prove"
    # Multiple llama.cpp server URLs for load-balanced multi-GPU setups.
    tir_llamacpp_base_urls: List[str] = field(default_factory=list)
    tir_llamacpp_max_concurrent: int = 1

    # Progressive token budget.
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
    if not cfg.lean.project_dir:
        errs.append("lean_project_dir must be non-empty")
    if cfg.lean.timeout_seconds <= 0:
        errs.append("lean_timeout_seconds must be >= 1")
    if cfg.lean.lean_jobs <= 0:
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
    if cfg.limit_prover_tokens > 0 and cfg.prover_type == "tir" and cfg.tir_backend_type == "ollama" and not cfg.tir_ollama.tokenizer_path:
        errs.append(
            "when --limit-prover-tokens is set with --prover-type=tir, --tir-backend-type=ollama, --tir-ollama-tokenizer-path must be set"
        )
    if cfg.limit_prover_tokens > 0 and cfg.prover_type == "tir" and cfg.tir_backend_type == "llamacpp" and not cfg.tir_llamacpp.tokenizer_path:
        errs.append(
            "when --limit-prover-tokens is set with --prover-type=tir, --tir-backend-type=llamacpp, --tir-llamacpp-tokenizer-path must be set"
        )

    if errs:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errs))


# ============================================================
# Factory helpers
# ============================================================

def make_goedel_prover_config(cfg: ProveFormalizationsConfig, workspace_suffix: str = ""):
    """Build a ``GoedelProverConfig`` from the composed script config."""
    subdir = cfg.goedel_lean_workspace_subdir
    if workspace_suffix:
        subdir = subdir + "_" + workspace_suffix
    lean_cfg = replace(
        cfg.lean,
        workspace_subdir=subdir,
        treat_sorry_warning_as_failure=True,
        treat_any_warning_as_failure=False,
    )
    return replace(cfg.goedel, lean=lean_cfg)


def make_goedel_backend(cfg: ProveFormalizationsConfig):
    """Instantiate the appropriate ``RawBackend`` for the Goedel prover.

    When ``goedel_ollama_hosts`` is non-empty (Ollama backend only), returns a
    ``LoadBalancedRawBackend`` that distributes requests across all listed hosts
    with at most ``goedel_ollama_max_concurrent`` concurrent requests per host.
    """
    if cfg.goedel_backend_type == "vllm":
        from conjecturing_agents.inference_backends.vllm_raw import VLLMRawBackend
        vllm_cfg = replace(
            cfg.goedel_vllm,
            tokenizer_path=cfg.goedel.tokenizer_path,
            context_tokens=cfg.goedel.context_tokens,
        )
        return VLLMRawBackend(vllm_cfg)

    from conjecturing_agents.inference_backends.ollama_raw import OllamaBackend

    ollama_cfg = replace(
        cfg.goedel_ollama,
        tokenizer_path=cfg.goedel.tokenizer_path,
    )

    if cfg.goedel_ollama_hosts:
        from conjecturing_agents.inference_backends.load_balanced_backend_raw import (
            LoadBalancedRawBackend,
        )
        sub_backends = [
            OllamaBackend(replace(ollama_cfg, host=host))
            for host in cfg.goedel_ollama_hosts
        ]
        return LoadBalancedRawBackend(
            [(b, cfg.goedel_ollama_max_concurrent) for b in sub_backends]
        )

    return OllamaBackend(ollama_cfg)


def make_tir_prover_config(cfg: ProveFormalizationsConfig, workspace_suffix: str = ""):
    """Build a ``TIRProverConfig`` from the composed script config."""
    from conjecturing_agents.tool_calling_backends.jupyter import JupyterKernelConfig

    subdir = cfg.tir_lean_workspace_subdir
    if workspace_suffix:
        subdir = subdir + "_" + workspace_suffix
    lean_cfg = replace(
        cfg.lean,
        workspace_subdir=subdir,
        treat_sorry_warning_as_failure=True,
        treat_any_warning_as_failure=False,
    )
    return replace(cfg.tir, lean=lean_cfg, jupyter=JupyterKernelConfig())


def make_tir_backend(cfg: ProveFormalizationsConfig):
    """Instantiate the TIR backend (Ollama or llama.cpp).

    When ``tir_llamacpp_base_urls`` is non-empty (llamacpp backend only),
    returns a ``LoadBalancedTIRBackend`` that distributes sessions across all
    listed servers with at most ``tir_llamacpp_max_concurrent`` concurrent
    sessions per server.
    """
    if cfg.tir_backend_type == "ollama":
        from conjecturing_agents.inference_backends.ollama_tir import OllamaTIRBackend
        return OllamaTIRBackend(cfg.tir_ollama)
    elif cfg.tir_backend_type == "llamacpp":
        from conjecturing_agents.inference_backends.llamacpp_tir import LlamaCppTIRBackend

        if cfg.tir_llamacpp_base_urls:
            from conjecturing_agents.inference_backends.load_balanced_backend_tir import (
                LoadBalancedTIRBackend,
            )
            sub_backends = [
                LlamaCppTIRBackend(replace(cfg.tir_llamacpp, base_url=url))
                for url in cfg.tir_llamacpp_base_urls
            ]
            return LoadBalancedTIRBackend(
                [(b, cfg.tir_llamacpp_max_concurrent) for b in sub_backends]
            )

        return LlamaCppTIRBackend(cfg.tir_llamacpp)
    else:
        raise ValueError(f"Unknown tir_backend_type: {cfg.tir_backend_type!r}")


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
            "Where to write the raw generation event log (prover_events.jsonl). "
            "Defaults to <output_path parent>/prover_events.jsonl."
        ),
    )

    # Lean compiler
    LeanCompilerConfig.add_cli_args(p, "lean", defaults={
        "workspace_subdir": ".conjecturing_agents/prove_formalizations",
        "max_memory_megabytes": 8 * 1024,
    })

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

    # Prover type
    p.add_argument(
        "--prover-type",
        dest="prover_type",
        choices=["goedel", "tir"],
        default=ProveFormalizationsConfig.prover_type,
        help=(
            "Which prover to use. 'goedel' uses GoedelProverAgent with a RawBackend "
            "(vLLM or Ollama raw completion). 'tir' uses TIRProverAgent with a TIR backend "
            "(Ollama or llama.cpp; selected via --tir-backend)."
        ),
    )

    # Goedel prover agent + backend sub-configs
    GoedelProverConfig.add_cli_args(p, "goedel")
    OllamaConfig.add_cli_args(p, "goedel-ollama", defaults={
        "model": "goedel-v2:latest",
        "client_timeout": 3600,
    })
    VLLMRawConfig.add_cli_args(p, "goedel-vllm", defaults={
        "client_timeout": 3600,
        "server_timeout": 3600,
    }, exclude={"tokenizer_path", "context_tokens"})

    # Goedel backend selection
    p.add_argument(
        "--goedel-backend",
        dest="goedel_backend_type",
        choices=["ollama", "vllm"],
        default=ProveFormalizationsConfig.goedel_backend_type,
        help="Inference backend for the Goedel prover.",
    )

    # Goedel lean workspace (separate from main lean workspace)
    p.add_argument(
        "--goedel-lean-workspace-subdir",
        default=ProveFormalizationsConfig.goedel_lean_workspace_subdir,
        help="Lean workspace subdir for Goedel prover runs.",
    )

    # Load-balanced Ollama (script-level)
    p.add_argument(
        "--goedel-ollama-hosts",
        nargs="+",
        default=[],
        metavar="URL",
        help=(
            "Multiple Ollama server URLs for load-balanced multi-GPU inference "
            "(backend=ollama). Requests are distributed across all hosts with at "
            "most --goedel-ollama-max-concurrent requests per host. "
            "Example: --goedel-ollama-hosts http://localhost:11434 http://localhost:11435"
        ),
    )
    p.add_argument(
        "--goedel-ollama-max-concurrent",
        type=int,
        default=ProveFormalizationsConfig.goedel_ollama_max_concurrent,
        help=(
            "Maximum concurrent requests per Ollama host when using "
            "--goedel-ollama-hosts. Should match OLLAMA_NUM_PARALLEL on each server. "
            "Default: %(default)s."
        ),
    )

    # TIR prover agent + backend sub-configs
    TIRProverConfig.add_cli_args(p, "tir")
    # TIR backend selection
    p.add_argument(
        "--tir-backend",
        dest="tir_backend_type",
        choices=["ollama", "llamacpp"],
        default=ProveFormalizationsConfig.tir_backend_type,
        help="Inference backend for the TIR prover.",
    )
    OllamaTIRConfig.add_cli_args(p, "tir-ollama")
    LlamaCppTIRConfig.add_cli_args(p, "tir-llamacpp")

    # Load-balanced llama.cpp (script-level)
    p.add_argument(
        "--tir-llamacpp-base-urls",
        nargs="+",
        default=[],
        metavar="URL",
        help=(
            "Multiple llama.cpp server URLs for load-balanced multi-GPU inference "
            "(backend=llamacpp). Sessions are distributed across all servers with at "
            "most --tir-llamacpp-max-concurrent sessions per server. "
            "Example: --tir-llamacpp-base-urls http://localhost:8001 http://localhost:8002"
        ),
    )
    p.add_argument(
        "--tir-llamacpp-max-concurrent",
        type=int,
        default=ProveFormalizationsConfig.tir_llamacpp_max_concurrent,
        help=(
            "Maximum concurrent sessions per llama.cpp server when using "
            "--tir-llamacpp-base-urls. Should match -np on each llama-server. "
            "Default: %(default)s."
        ),
    )

    # TIR lean workspace (separate from main lean workspace)
    p.add_argument(
        "--tir-lean-workspace-subdir",
        default=ProveFormalizationsConfig.tir_lean_workspace_subdir,
        help="Subdirectory for TIR Lean temp files inside lean_project_dir. Default: %(default)s.",
    )

    # Progressive token budget
    p.add_argument(
        "--limit-prover-tokens",
        type=int,
        default=ProveFormalizationsConfig.limit_prover_tokens,
        help=(
            "Stop each prover session when its rendered prompt reaches this many tokens. "
            "0 = unlimited. Use --continue with a higher value to resume sessions that "
            "were cut off, extending the budget progressively (e.g. 4000 → 8000 → 40960). "
            "For the TIR prover, requires --tir-ollama-tokenizer-path / --tir-llamacpp-tokenizer-path"
        ),
    )
    p.add_argument(
        "--print-agent-conv",
        dest="print_agent_conv",
        action="store_true",
        default=ProveFormalizationsConfig.print_agent_conv,
        help="Print prover prompts and generated tokens to stdout in real time.",
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
        lean=LeanCompilerConfig.from_parsed_args(args, "lean"),
        proof_retries=args.proof_retries,
        enable_parallel_disproof=args.enable_parallel_disproof,
        enable_sequential_disproof=args.enable_sequential_disproof,
        disproof_retries=args.disproof_retries,
        prover_type=args.prover_type,
        goedel=GoedelProverConfig.from_parsed_args(args, "goedel"),
        goedel_backend_type=args.goedel_backend_type,
        goedel_ollama=OllamaConfig.from_parsed_args(args, "goedel-ollama"),
        goedel_vllm=VLLMRawConfig.from_parsed_args(args, "goedel-vllm",
            exclude={"tokenizer_path", "context_tokens"},
        ),
        goedel_lean_workspace_subdir=args.goedel_lean_workspace_subdir,
        goedel_ollama_hosts=args.goedel_ollama_hosts or [],
        goedel_ollama_max_concurrent=args.goedel_ollama_max_concurrent,
        tir=TIRProverConfig.from_parsed_args(args, "tir"),
        tir_backend_type=args.tir_backend_type,
        tir_ollama=OllamaTIRConfig.from_parsed_args(args, "tir-ollama"),
        tir_llamacpp=LlamaCppTIRConfig.from_parsed_args(args, "tir-llamacpp"),
        tir_lean_workspace_subdir=args.tir_lean_workspace_subdir,
        tir_llamacpp_base_urls=args.tir_llamacpp_base_urls or [],
        tir_llamacpp_max_concurrent=args.tir_llamacpp_max_concurrent,
        limit_prover_tokens=args.limit_prover_tokens,
        print_agent_conv=args.print_agent_conv,
        parallelism=args.parallelism,
        max_records=args.max_records,
        resume=args.resume,
        verbose=args.verbose,
    )
    validate_cfg(cfg)
    return cfg
