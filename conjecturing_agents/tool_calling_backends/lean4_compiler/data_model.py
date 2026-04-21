from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

@dataclass
class LeanCompileResult:
    ok: bool
    stdout: str
    stderr: str

    # Parsed Lean --json messages
    json_messages: List[Dict[str, Any]] = field(default_factory=list)
    json_errors: List[Dict[str, Any]] = field(default_factory=list)
    json_warnings: List[Dict[str, Any]] = field(default_factory=list)
    sorry_warnings: List[Dict[str, Any]] = field(default_factory=list)

    # Non-JSON lines from stdout, if any
    non_json_stdout_lines: List[str] = field(default_factory=list)

    # Execution metadata
    timed_out: bool = False
    oom: bool = False
    elapsed_ms: int = 0
    returncode: Optional[int] = None
    command: List[str] = field(default_factory=list)

    # File metadata
    source_path: Optional[str] = None
    relative_path: Optional[str] = None

    # Helpful prompt-facing formatting
    formatted_diagnostics: str = ""


@dataclass
class LeanCompilerConfig:
    """
    Configuration for compiling Lean 4 snippets inside an existing Lean project.
    """

    # Lean project root (must contain lakefile / lake-manifest / .lake etc.)
    project_dir: str

    # Where temporary source files are created inside project_dir
    workspace_subdir: str = ".conjecturing_agents/lean_tool_runs"

    # Compilation command settings
    timeout_seconds: int = 120
    lean_jobs: int = 4

    # Tool wiring
    recipient_name: str = "lean"
    tool_name: str = "lean"

    # Policy
    treat_sorry_warning_as_failure: bool = True
    treat_any_warning_as_failure: bool = False
    auto_extract_code_block: bool = True
    cleanup_source_file: bool = False

    # Memory limit for the lake/lean subprocess (megabytes); 0 = no limit
    max_memory_megabytes: int = 4 * 1024  # 4 GiB

    # Optional prefix added to temp files
    filename_prefix: str = "lean_tool_"

    # Prompt-facing formatting
    max_formatted_messages: int = 8
    include_stdout_in_tool_response: bool = False
    include_stderr_in_tool_response: bool = True
    include_non_json_stdout_lines: bool = True


    # ------------------------------------------------------------------
    # CLI integration
    # ------------------------------------------------------------------

    @classmethod
    def add_cli_args(
        cls,
        parser: ArgumentParser,
        prefix: str = "lean",
        defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register CLI args for the user-facing LeanCompilerConfig fields.

        Only exposes the five fields that scripts typically surface as flags:
        project_dir, workspace_subdir, timeout_seconds, lean_jobs,
        max_memory_megabytes.  Policy/wiring fields are left at dataclass
        defaults and overridden in factory helpers.
        """
        d = defaults or {}
        pre = prefix  # e.g. "lean"
        dst = prefix.replace("-", "_")  # e.g. "lean"

        parser.add_argument(
            f"--{pre}-project-dir",
            dest=f"{dst}_project_dir",
            default=d.get("project_dir", "."),
            help="Path to a Lean project with Mathlib configured.",
        )
        parser.add_argument(
            f"--{pre}-workspace-subdir",
            dest=f"{dst}_workspace_subdir",
            default=d.get("workspace_subdir", cls.workspace_subdir),
            help="Subdirectory inside lean_project_dir where temporary .lean files are written.",
        )
        parser.add_argument(
            f"--{pre}-timeout-seconds",
            dest=f"{dst}_timeout_seconds",
            type=int,
            default=d.get("timeout_seconds", cls.timeout_seconds),
            help="Per-file Lean compilation timeout in seconds.",
        )
        parser.add_argument(
            f"--{pre}-jobs",
            dest=f"{dst}_jobs",
            type=int,
            default=d.get("lean_jobs", cls.lean_jobs),
            help="Number of parallel jobs passed to lake env lean -j.",
        )
        parser.add_argument(
            f"--{pre}-max-memory-megabytes",
            dest=f"{dst}_max_memory_megabytes",
            type=int,
            default=d.get("max_memory_megabytes", cls.max_memory_megabytes),
            help=(
                "Maximum virtual memory (megabytes) for each lake/lean subprocess. "
                "0 means no limit."
            ),
        )

    @classmethod
    def from_parsed_args(
        cls,
        args: Any,
        prefix: str = "lean",
        **overrides: Any,
    ) -> "LeanCompilerConfig":
        """Construct a LeanCompilerConfig from an argparse namespace.

        Fields not registered by ``add_cli_args`` stay at dataclass defaults
        unless provided via ``overrides``.
        """
        dst = prefix.replace("-", "_")
        kwargs: Dict[str, Any] = {
            "project_dir": getattr(args, f"{dst}_project_dir"),
            "workspace_subdir": getattr(args, f"{dst}_workspace_subdir"),
            "timeout_seconds": getattr(args, f"{dst}_timeout_seconds"),
            "lean_jobs": getattr(args, f"{dst}_jobs"),
            "max_memory_megabytes": getattr(args, f"{dst}_max_memory_megabytes"),
        }
        kwargs.update(overrides)
        return cls(**kwargs)


__all__ = [
    "LeanCompileResult",
    "LeanCompilerConfig"
]