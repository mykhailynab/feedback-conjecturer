from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from conjecturing_agents.agents.conjecture_formalizer import (
    extract_rhs_from_abbrev_declaration,
)
from conjecturing_agents.inference_backends.raw_backend import EventLoggerFn
from conjecturing_agents.tool_calling_backends.lean4_compiler import (
    Lean4CompilerBackend,
    LeanCompilerConfig,
)

from .result import CheckResult
from .string_match import check_string_match
from .lean_equiv import check_lean_equiv
from .goedel_equiv import check_goedel_equiv, check_goedel_inequiv


@dataclass
class AnswerCheckerConfig:
    # Lean project settings (required for heuristics 2 and 3)
    lean_project_dir: str = "."
    lean_workspace_subdir: str = ".conjecturing_agents/answer_checking"
    lean_timeout_seconds: int = 120
    lean_jobs: int = 4

    # Memory limit for the lake/lean subprocess (bytes); 0 = no limit
    lean_max_memory_megabytes: int = 4 * 1024  # 4 GiB

    # Which heuristics to run
    use_string_match: bool = True
    use_lean_equiv: bool = True
    use_goedel_prover: bool = False
    # Run a Goedel disproof attempt after a failed proof (requires use_goedel_prover=True)
    use_goedel_disprover: bool = False

    # Number of independent proof/disproof attempts before giving up (pass@N).
    # Each retry uses a different seed. 1 = try once (default, pass@1).
    goedel_proof_retries: int = 1
    goedel_disproof_retries: int = 1

    # ------------------------------------------------------------------ #
    # Goedel prover (heuristic 3)
    # ------------------------------------------------------------------ #
    goedel_chat_template_path: str = "templates/goedel_template.jinja"
    goedel_max_rounds: int = 2
    goedel_max_tokens: int = 16384
    goedel_temperature: float = 0.6
    goedel_top_p: float = 0.95
    goedel_repeat_penalty: float = 1.0
    goedel_context_tokens: int = 40960
    goedel_lean_workspace_subdir: str = ".conjecturing_agents/goedel_lean_runs"
    goedel_max_error_message_chars: int = 0

    # "ollama" or "vllm"
    goedel_backend_type: str = "ollama"

    # Ollama backend
    goedel_ollama_model: str = "goedel-v2"
    goedel_ollama_host: str = "http://localhost:11434"
    goedel_ollama_client_timeout: int = 240

    # vLLM backend (used when goedel_backend_type="vllm")
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

    # HF tokenizer path for exact token counting (required for either backend)
    goedel_tokenizer_path: str = ""

    # Print prompts and generated tokens to stdout in real time
    goedel_print_agent_conv: bool = False


class AnswerChecker:
    """
    Orchestrates answer-checking heuristics in order:

      1. String match (fast, no Lean needed)
      2. Lean equivalence proof via canned tactics
      3. Goedel-prover proof attempt (equivalent=True if proved)
      4. Goedel-prover disproof attempt (equivalent=False if disproved)

    Stops as soon as any heuristic returns a conclusive result (equivalent != None).
    Heuristic 4 only runs when use_goedel_disprover=True and heuristic 3 was
    inconclusive.

    Parameters
    ----------
    goedel_backend:
        Optional pre-built ``RawBackend`` to use for the Goedel heuristics.
        When provided, ``AnswerCheckerConfig.goedel_backend_type`` and related
        Ollama / vLLM config fields are ignored for backend construction.
        Pass a shared ``LoadBalancedRawBackend`` here to spread requests
        across multiple inference servers.
    """

    def __init__(
        self,
        cfg: Optional[AnswerCheckerConfig] = None,
        event_logger: Optional[EventLoggerFn] = None,
        goedel_backend=None,
    ):
        self.cfg = cfg or AnswerCheckerConfig()
        self._event_logger = event_logger
        self._compiler: Optional[Lean4CompilerBackend] = None
        self._goedel_agent = None
        # If an external backend is injected, use it; do not close it on close().
        self._goedel_backend = goedel_backend
        self._owns_goedel_backend = goedel_backend is None

    # ------------------------------------------------------------------
    # Lean compiler — lazy init (heuristic 2)
    # ------------------------------------------------------------------

    def _get_compiler(self) -> Lean4CompilerBackend:
        if self._compiler is None:
            lean_cfg = LeanCompilerConfig(
                project_dir=self.cfg.lean_project_dir,
                workspace_subdir=self.cfg.lean_workspace_subdir,
                timeout_seconds=self.cfg.lean_timeout_seconds,
                lean_jobs=self.cfg.lean_jobs,
                max_memory_megabytes=self.cfg.lean_max_memory_megabytes,
                treat_sorry_warning_as_failure=True,
                treat_any_warning_as_failure=False,
            )
            self._compiler = Lean4CompilerBackend(lean_cfg)
        return self._compiler

    # ------------------------------------------------------------------
    # Goedel agent + backend — lazy init (heuristic 3)
    # ------------------------------------------------------------------

    def _get_goedel(self):
        if self._goedel_agent is None:
            from conjecturing_agents.agents.goedel_prover import (
                GoedelProverAgent,
                GoedelProverConfig,
            )
            goedel_lean_cfg = LeanCompilerConfig(
                project_dir=self.cfg.lean_project_dir,
                workspace_subdir=self.cfg.goedel_lean_workspace_subdir,
                timeout_seconds=self.cfg.lean_timeout_seconds,
                lean_jobs=self.cfg.lean_jobs,
                max_memory_megabytes=self.cfg.lean_max_memory_megabytes,
                treat_sorry_warning_as_failure=True,
                treat_any_warning_as_failure=False,
            )
            goedel_cfg = GoedelProverConfig(
                chat_template_path=self.cfg.goedel_chat_template_path,
                max_rounds=self.cfg.goedel_max_rounds,
                max_tokens=self.cfg.goedel_max_tokens,
                temperature=self.cfg.goedel_temperature,
                top_p=self.cfg.goedel_top_p,
                repeat_penalty=self.cfg.goedel_repeat_penalty,
                context_tokens=self.cfg.goedel_context_tokens,
                lean=goedel_lean_cfg,
                max_error_message_chars=self.cfg.goedel_max_error_message_chars,
            )
            self._goedel_agent = GoedelProverAgent(goedel_cfg)

            # Only build a backend if one was not injected via __init__.
            if self._goedel_backend is None:
                if self.cfg.goedel_backend_type == "vllm":
                    from conjecturing_agents.inference_backends.vllm_raw import (
                        VLLMRawBackend,
                        VLLMRawConfig,
                    )
                    self._goedel_backend = VLLMRawBackend(VLLMRawConfig(
                        base_url=self.cfg.goedel_vllm_base_url,
                        served_model_name=self.cfg.goedel_vllm_model_name,
                        api_key=self.cfg.goedel_vllm_api_key,
                        client_timeout=self.cfg.goedel_vllm_client_timeout,
                        tokenizer_path=self.cfg.goedel_tokenizer_path,
                        manage_server=self.cfg.goedel_vllm_manage_server,
                        model_path=self.cfg.goedel_vllm_model_path,
                        port=self.cfg.goedel_vllm_port,
                        host=self.cfg.goedel_vllm_host,
                        server_timeout=self.cfg.goedel_vllm_server_timeout,
                        server_log_path=self.cfg.goedel_vllm_server_log_path,
                        dtype=self.cfg.goedel_vllm_dtype,
                        kv_cache_dtype=self.cfg.goedel_vllm_kv_cache_dtype,
                        context_tokens=self.cfg.goedel_context_tokens,
                        gpu_memory_utilization=self.cfg.goedel_vllm_gpu_memory_utilization,
                        max_num_seqs=self.cfg.goedel_vllm_max_num_seqs,
                        stream_interval=self.cfg.goedel_vllm_stream_interval,
                        enable_prefix_caching=self.cfg.goedel_vllm_enable_prefix_caching,
                        extra_server_args=self.cfg.goedel_vllm_extra_server_args,
                    ))
                else:
                    from conjecturing_agents.inference_backends.ollama_backend import (
                        OllamaBackend,
                        OllamaConfig,
                    )
                    self._goedel_backend = OllamaBackend(OllamaConfig(
                        model=self.cfg.goedel_ollama_model,
                        host=self.cfg.goedel_ollama_host,
                        client_timeout=self.cfg.goedel_ollama_client_timeout,
                        tokenizer_path=self.cfg.goedel_tokenizer_path,
                    ))
                self._owns_goedel_backend = True

                if self.cfg.goedel_print_agent_conv:
                    self._goedel_backend.set_verbose(True)
                if self._event_logger is not None:
                    self._goedel_backend.set_event_logger(self._event_logger)

        return self._goedel_agent, self._goedel_backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check a single formalization record from formalizations.jsonl.

        Returns a dict with keys:
          - ``check_result``: final CheckResult (as dict)
          - ``all_results``: list of CheckResult dicts, one per heuristic tried
          - ``equivalent``: bool or None
          - ``method``: method that gave the conclusive result (or "inconclusive")
        """
        status = record.get("status")
        if status != "success":
            result = CheckResult(
                equivalent=None,
                method="skipped",
                details={"reason": f"record_status={status}"},
            )
            return self._format_output(result, [result])

        proposed_decl = record.get("final_abbrev_declaration") or ""
        gt_rhs = record.get("ground_truth_extracted_answer") or ""
        lean_statement = record.get("lean_statement_without_comment") or ""
        abbrev_name = record.get("required_abbrev_name") or ""

        if not proposed_decl or not gt_rhs or not abbrev_name:
            result = CheckResult(
                equivalent=None,
                method="skipped",
                details={"reason": "missing_fields"},
            )
            return self._format_output(result, [result])

        proposed_rhs = extract_rhs_from_abbrev_declaration(proposed_decl)
        all_results: list[CheckResult] = []

        def _try(method_name: str, fn: Callable[[], CheckResult]) -> Optional[CheckResult]:
            """Run fn(), appending the result or an error record to all_results."""
            try:
                r = fn()
                all_results.append(r)
                return r
            except Exception as exc:
                all_results.append(CheckResult(
                    equivalent=None,
                    method=method_name,
                    details={
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                ))
                return None

        # --- Heuristic 1: string match ---
        if self.cfg.use_string_match:
            r1 = _try("string_match", lambda: check_string_match(proposed_rhs, gt_rhs))
            if r1 is not None and r1.equivalent is not None:
                return self._format_output(r1, all_results)

        # --- Heuristic 2: Lean equivalence proof via canned tactics ---
        if self.cfg.use_lean_equiv and lean_statement:
            compiler = self._get_compiler()
            r2 = _try("lean_equiv", lambda: check_lean_equiv(
                lean_statement, proposed_decl, gt_rhs, abbrev_name, compiler,
            ))
            if r2 is not None and r2.equivalent is not None:
                return self._format_output(r2, all_results)

        # --- Heuristics 3 & 4: Goedel-prover proof / disproof ---
        if self.cfg.use_goedel_prover and lean_statement:
            agent, backend = self._get_goedel()
            problem_id = record.get("problem_id") or 0
            attempt = record.get("attempt") or 0
            base_seed = hash((problem_id, attempt, abbrev_name)) & 0x7FFFFFFF

            for retry in range(self.cfg.goedel_proof_retries):
                seed = (base_seed + retry) & 0x7FFFFFFF
                r3 = _try("goedel_prover", lambda s=seed: check_goedel_equiv(
                    lean_statement, proposed_decl, gt_rhs, abbrev_name, agent, backend,
                    seed=s,
                    event_logger=self._event_logger,
                    metadata={
                        "problem_id": problem_id, "attempt": attempt,
                        "checking": "proof", "retry": retry
                    },
                ))
                if r3 is not None and r3.equivalent is not None:
                    return self._format_output(r3, all_results)

            if self.cfg.use_goedel_disprover:
                for retry in range(self.cfg.goedel_disproof_retries):
                    seed = (base_seed + retry) & 0x7FFFFFFF
                    r4 = _try("goedel_disprover", lambda s=seed: check_goedel_inequiv(
                        lean_statement, proposed_decl, gt_rhs, abbrev_name, agent, backend,
                        seed=s,
                        event_logger=self._event_logger,
                        metadata={
                            "problem_id": problem_id, "attempt": attempt,
                            "checking": "disproof", "retry": retry
                        },
                    ))
                    if r4 is not None and r4.equivalent is not None:
                        return self._format_output(r4, all_results)

        inconclusive = CheckResult(equivalent=None, method="inconclusive", details={})
        return self._format_output(inconclusive, all_results)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_output(
        final: CheckResult,
        all_results: list[CheckResult],
    ) -> Dict[str, Any]:
        return {
            "equivalent": final.equivalent,
            "method": final.method,
            "check_result": vars(final),
            "all_results": [vars(r) for r in all_results],
        }

    def close(self) -> None:
        if self._goedel_backend is not None and self._owns_goedel_backend:
            self._goedel_backend.close()

    def __enter__(self) -> "AnswerChecker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = ["AnswerCheckerConfig", "AnswerChecker"]
