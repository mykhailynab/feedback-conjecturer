from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from conjecturing_agents.agents.conjecture_formalizer import (
    extract_rhs_from_abbrev_declaration,
)
from conjecturing_agents.tool_calling_backends.lean4_compiler import (
    Lean4CompilerBackend,
    LeanCompilerConfig,
)

from .result import CheckResult
from .string_match import check_string_match
from .lean_equiv import check_lean_equiv


@dataclass
class AnswerCheckerConfig:
    # Lean project settings (required for heuristics 2 and 3)
    lean_project_dir: str = "."
    lean_workspace_subdir: str = ".conjecturing_agents/answer_checking"
    lean_timeout_seconds: int = 120
    lean_jobs: int = 4

    # Memory limit for the lake/lean subprocess (bytes); 0 = no limit
    lean_max_memory_megabytes: int = 2 * 1024  # 2 GiB

    # Which heuristics to run
    use_string_match: bool = True
    use_lean_equiv: bool = True
    use_goedel_prover: bool = False

    # ------------------------------------------------------------------ #
    # Goedel prover (heuristic 3)
    # ------------------------------------------------------------------ #
    goedel_chat_template_path: str = "goedel_template.jinja"
    goedel_max_rounds: int = 2
    goedel_max_tokens: int = 16384
    goedel_temperature: float = 0.6
    goedel_top_p: float = 0.95
    goedel_repeat_penalty: float = 1.0
    goedel_context_tokens: int = 40960
    goedel_lean_workspace_subdir: str = ".conjecturing_agents/goedel_lean_runs"

    # "ollama" or "vllm"
    goedel_backend_type: str = "ollama"

    # Ollama backend
    goedel_ollama_model: str = "goedel-v2"
    goedel_ollama_host: str = "http://localhost:11434"

    # vLLM backend (used when goedel_backend_type="vllm")
    goedel_vllm_base_url: str = "http://0.0.0.0:8001/v1"
    goedel_vllm_model_name: str = "goedel"

    # HF tokenizer path for exact token counting (required for either backend)
    goedel_tokenizer_path: str = ""


class AnswerChecker:
    """
    Orchestrates answer-checking heuristics in order:

      1. String match (fast, no Lean needed)
      2. Lean equivalence proof via canned tactics
      3. Goedel-prover proof attempt

    Stops as soon as any heuristic returns a conclusive result (equivalent != None).
    """

    def __init__(self, cfg: Optional[AnswerCheckerConfig] = None):
        self.cfg = cfg or AnswerCheckerConfig()
        self._compiler: Optional[Lean4CompilerBackend] = None
        self._goedel_agent = None
        self._goedel_backend = None

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
            )
            self._goedel_agent = GoedelProverAgent(goedel_cfg)

            if self.cfg.goedel_backend_type == "vllm":
                from conjecturing_agents.inference_backends.vllm_raw import (
                    VLLMRawBackend,
                    VLLMRawConfig,
                )
                self._goedel_backend = VLLMRawBackend(VLLMRawConfig(
                    base_url=self.cfg.goedel_vllm_base_url,
                    served_model_name=self.cfg.goedel_vllm_model_name,
                    tokenizer_path=self.cfg.goedel_tokenizer_path,
                ))
            else:
                from conjecturing_agents.inference_backends.ollama_backend import (
                    OllamaBackend,
                    OllamaConfig,
                )
                self._goedel_backend = OllamaBackend(OllamaConfig(
                    model=self.cfg.goedel_ollama_model,
                    host=self.cfg.goedel_ollama_host,
                    tokenizer_path=self.cfg.goedel_tokenizer_path,
                ))

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

        # --- Heuristic 1: string match ---
        if self.cfg.use_string_match:
            r1 = check_string_match(proposed_rhs, gt_rhs)
            all_results.append(r1)
            if r1.equivalent is not None:
                return self._format_output(r1, all_results)

        # --- Heuristic 2: Lean equivalence proof via canned tactics ---
        if self.cfg.use_lean_equiv and lean_statement:
            compiler = self._get_compiler()
            r2 = check_lean_equiv(
                lean_statement,
                proposed_decl,
                gt_rhs,
                abbrev_name,
                compiler,
            )
            all_results.append(r2)
            if r2.equivalent is not None:
                return self._format_output(r2, all_results)

        # --- Heuristic 3: Goedel-prover ---
        if self.cfg.use_goedel_prover and lean_statement:
            from .goedel_equiv import check_goedel_equiv
            agent, backend = self._get_goedel()
            problem_id = record.get("problem_id") or 0
            attempt = record.get("attempt") or 0
            seed = hash((problem_id, attempt, abbrev_name)) & 0x7FFFFFFF
            r3 = check_goedel_equiv(
                lean_statement,
                proposed_decl,
                gt_rhs,
                abbrev_name,
                agent,
                backend,
                seed=seed,
            )
            all_results.append(r3)
            if r3.equivalent is not None:
                return self._format_output(r3, all_results)

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
        if self._goedel_backend is not None:
            self._goedel_backend.close()

    def __enter__(self) -> "AnswerChecker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = ["AnswerCheckerConfig", "AnswerChecker"]
