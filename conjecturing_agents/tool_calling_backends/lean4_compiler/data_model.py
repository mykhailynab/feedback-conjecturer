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

    # Memory limit for the lake/lean subprocess (bytes); 0 = no limit
    max_memory_megabytes: int = 4 * 1024  # 4 GiB

    # Optional prefix added to temp files
    filename_prefix: str = "lean_tool_"

    # Prompt-facing formatting
    max_formatted_messages: int = 8
    include_stdout_in_tool_response: bool = False
    include_stderr_in_tool_response: bool = True
    include_non_json_stdout_lines: bool = True


__all__ = [
    "LeanCompileResult",
    "LeanCompilerConfig" 
]