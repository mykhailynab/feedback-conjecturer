from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import List

from conjecturing_agents.agents.goedel_prover.config import GoedelProverConfig
from conjecturing_agents.answer_checking.checker import AnswerCheckerConfig
from conjecturing_agents.inference_backends.ollama_raw import OllamaConfig
from conjecturing_agents.inference_backends.vllm_raw import VLLMRawConfig
from conjecturing_agents.tool_calling_backends.lean4_compiler import LeanCompilerConfig


# ============================================================
# Script config
# ============================================================

@dataclass
class CheckFormalizationsConfig:
    # Input / output
    formalizations_path: str = "logs/conjecture_formalization_logs/formalizations.jsonl"
    output_path: str = ""  # defaults to <formalizations_path parent>/check_results.jsonl

    # Lean compiler (shared base settings)
    lean: LeanCompilerConfig = field(
        default_factory=lambda: LeanCompilerConfig(
            project_dir=".",
            workspace_subdir=".conjecturing_agents/answer_checking",
        )
    )

    # Heuristics
    use_string_match: bool = True
    use_lean_equiv: bool = True
    use_goedel_prover: bool = False
    use_goedel_disprover: bool = False
    goedel_proof_retries: int = 1
    goedel_disproof_retries: int = 1

    # Goedel prover (composed sub-configs)
    goedel: GoedelProverConfig = field(default_factory=GoedelProverConfig)
    goedel_backend_type: str = "ollama"  # "ollama" | "vllm"
    goedel_ollama: OllamaConfig = field(default_factory=OllamaConfig)
    goedel_vllm: VLLMRawConfig = field(default_factory=VLLMRawConfig)
    goedel_lean_workspace_subdir: str = ".conjecturing_agents/goedel_lean_runs"
    # Multiple Ollama hosts for load-balanced multi-GPU setups.
    goedel_ollama_hosts: List[str] = field(default_factory=list)
    goedel_ollama_max_concurrent: int = 6

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

    if errs:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errs))


# ============================================================
# Factory helpers
# ============================================================

def make_checker_config(cfg: CheckFormalizationsConfig) -> AnswerCheckerConfig:
    return AnswerCheckerConfig(
        lean=cfg.lean,
        use_string_match=cfg.use_string_match,
        use_lean_equiv=cfg.use_lean_equiv,
        use_goedel_prover=cfg.use_goedel_prover,
        use_goedel_disprover=cfg.use_goedel_disprover,
        goedel_proof_retries=cfg.goedel_proof_retries,
        goedel_disproof_retries=cfg.goedel_disproof_retries,
        goedel=cfg.goedel,
        goedel_backend_type=cfg.goedel_backend_type,
        goedel_ollama=cfg.goedel_ollama,
        goedel_vllm=cfg.goedel_vllm,
        goedel_lean_workspace_subdir=cfg.goedel_lean_workspace_subdir,
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
    LeanCompilerConfig.add_cli_args(p, "lean", defaults={
        "workspace_subdir": ".conjecturing_agents/answer_checking",
        "max_memory_megabytes": 4 * 1024,
    })

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
        "--goedel-proof-retries",
        type=int,
        default=CheckFormalizationsConfig.goedel_proof_retries,
        help=(
            "Number of independent proof attempts before giving up (pass@N). "
            "Each retry uses a different seed. Default: 1."
        ),
    )
    p.add_argument(
        "--goedel-disproof-retries",
        type=int,
        default=CheckFormalizationsConfig.goedel_disproof_retries,
        help=(
            "Number of independent disproof attempts before giving up (pass@N). "
            "Each retry uses a different seed. Requires --goedel-disprover. Default: 1."
        ),
    )

    # Goedel prover agent + backend sub-configs
    GoedelProverConfig.add_cli_args(p, "goedel")
    OllamaConfig.add_cli_args(p, "goedel-ollama", defaults={
        "model": "goedel-v2:latest",
        "client_timeout": 1200,
    })
    VLLMRawConfig.add_cli_args(p, "goedel-vllm", defaults={
        "client_timeout": 1200,
        "server_timeout": 1200,
    }, exclude={"tokenizer_path", "context_tokens"})

    # Goedel backend selection
    p.add_argument(
        "--goedel-backend",
        dest="goedel_backend_type",
        choices=["ollama", "vllm"],
        default=CheckFormalizationsConfig.goedel_backend_type,
        help="Inference backend for the Goedel prover.",
    )

    # Goedel lean workspace (separate from main lean workspace)
    p.add_argument(
        "--goedel-lean-workspace-subdir",
        default=CheckFormalizationsConfig.goedel_lean_workspace_subdir,
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
            "(backend=ollama). When set, requests are distributed across all hosts "
            "with at most --goedel-ollama-max-concurrent requests per host. "
            "Example: --goedel-ollama-hosts http://localhost:11434 http://localhost:11435"
        ),
    )
    p.add_argument(
        "--goedel-ollama-max-concurrent",
        type=int,
        default=CheckFormalizationsConfig.goedel_ollama_max_concurrent,
        help=(
            "Maximum concurrent requests per Ollama host when using "
            "--goedel-ollama-hosts. Should match OLLAMA_NUM_PARALLEL on each server. "
            "Default: %(default)s."
        ),
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
        lean=LeanCompilerConfig.from_parsed_args(args, "lean"),
        use_string_match=args.use_string_match,
        use_lean_equiv=args.use_lean_equiv,
        use_goedel_prover=args.use_goedel_prover,
        use_goedel_disprover=args.use_goedel_disprover,
        goedel_proof_retries=args.goedel_proof_retries,
        goedel_disproof_retries=args.goedel_disproof_retries,
        goedel=GoedelProverConfig.from_parsed_args(args, "goedel"),
        goedel_backend_type=args.goedel_backend_type,
        goedel_ollama=OllamaConfig.from_parsed_args(args, "goedel-ollama"),
        goedel_vllm=VLLMRawConfig.from_parsed_args(args, "goedel-vllm",
            exclude={"tokenizer_path", "context_tokens"},
        ),
        goedel_lean_workspace_subdir=args.goedel_lean_workspace_subdir,
        goedel_ollama_hosts=args.goedel_ollama_hosts or [],
        goedel_ollama_max_concurrent=args.goedel_ollama_max_concurrent,
        print_agent_conv=args.print_agent_conv,
        parallelism=args.parallelism,
        max_records=args.max_records,
        resume=args.resume,
        verbose=args.verbose,
    )
    validate_cfg(cfg)
    return cfg
