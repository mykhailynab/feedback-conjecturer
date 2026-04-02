from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
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

    # Heuristics
    use_string_match: bool = True
    use_lean_equiv: bool = True

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
        use_string_match=cfg.use_string_match,
        use_lean_equiv=cfg.use_lean_equiv,
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
            "(equivalent is not None) are kept as-is; undecided records "
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
        use_string_match=args.use_string_match,
        use_lean_equiv=args.use_lean_equiv,
        parallelism=args.parallelism,
        max_records=args.max_records,
        resume=args.resume,
        verbose=args.verbose,
    )
    validate_cfg(cfg)
    return cfg
