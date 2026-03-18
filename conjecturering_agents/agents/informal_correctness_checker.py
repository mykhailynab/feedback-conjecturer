from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from openai_harmony import Message, ReasoningEffort, ToolNamespaceConfig

from conjecturering_agents.inference_backends.vllm_harmony import (
    HarmonyAgentSpec,
    HarmonyRunResult,
    HarmonySessionState,
    TerminationSignal,
    VLLMHarmonyBackend,
    extract_json_object_from_text,
)
from conjecturering_agents.tool_calling_backends.jupyter import (
    JupyterKernelConfig,
    JupyterToolBackend,
)


# ============================================================
# Default prompts
# ============================================================

DEFAULT_CHECKER_SYSTEM_PROMPT = (
    "You are a mathematical answer equivalence checker.\n"
    "Given:\n"
    "  (1) the problem statement,\n"
    "  (2) the official answer key text (ground truth), and\n"
    "  (3) a candidate answer (model output),\n"
    "determine whether the candidate answer is mathematically equivalent to the ground truth.\n\n"
    "Rules:\n"
    "- Treat the answer key as authoritative; it may be phrased as 'Show that ...' or 'Prove that ...', "
    "but the candidate answer need not include a full proof.\n"
    "- The candidate answer may be phrased differently, but it is equivalent if it asserts the same mathematical claim\n"
    "  (and any explicit final value/expression matches).\n"
    "- If needed, use the python tool (sympy/numpy) to check symbolic/numeric equivalence.\n"
    "- Output MUST be a single line of strict JSON with keys:\n"
    '    {"equivalent": true/false, "confidence": 0..1, "reason": "..."}\n'
    "  Keep reason concise.\n"
    "- Do not include any other text besides the JSON line."
)

DEFAULT_CHECKER_TOOL_PROMPT = (
    "Use this tool to execute Python code for mathematical answer checking.\n"
    "- Use sympy for symbolic equivalence checks\n"
    "- Use numerical substitution for sanity checks\n"
    "- Use print() to display results\n"
    "- Keep computations concise and directly relevant to equivalence checking\n\n"
    "The environment is a stateful Jupyter notebook. Code persists between executions."
)

DEFAULT_CHECKER_PREFERENCE_PROMPT = (
    "You have access to `math`, `numpy`, and `sympy` for:\n\n"
    "# Symbolic Computation (sympy):\n"
    "- Algebraic manipulation and simplification\n"
    "- Solving equations and systems of equations\n"
    "- Symbolic differentiation and integration\n"
    "- Number theory functions (primes, divisors, modular arithmetic)\n"
    "- Polynomial operations and factorization\n"
    "- Working with mathematical expressions symbolically\n\n"
    "# Numerical Computation (numpy):\n"
    "- Array operations and linear algebra\n"
    "- Efficient numerical calculations for large datasets\n"
    "- Matrix operations and eigenvalue problems\n"
    "- Statistical computations\n\n"
    "# Mathematical Functions (math):\n"
    "- Standard mathematical functions (trig, log, exp)\n"
    "- Constants like pi and e\n"
    "- Basic operations for single values\n\n"
    "Best Practices:\n"
    "- Use sympy for exact symbolic answers when possible\n"
    "- Use numpy for numerical verification and large-scale computation\n"
    "- Combine symbolic and numerical approaches: derive symbolically, verify numerically\n"
    "- Document your computational strategy clearly\n"
    "- Validate computational results against known cases or theoretical bounds"
)


# ============================================================
# JSON normalization
# ============================================================

def normalize_checker_result(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("checker output is not a dict")
    if "equivalent" not in obj:
        raise ValueError('"equivalent" missing from checker output')

    raw_equivalent = obj["equivalent"]
    if isinstance(raw_equivalent, bool):
        equivalent = raw_equivalent
    else:
        raise ValueError(f'invalid value for "equivalent": {raw_equivalent!r}')

    confidence = float(obj.get("confidence", 0.0))
    reason = str(obj.get("reason", ""))

    confidence = max(0.0, min(1.0, confidence))

    return {
        "equivalent": equivalent,
        "confidence": confidence,
        "reason": reason,
    }


def parse_partial_checker_result_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Checker-specific fallback for malformed / truncated JSON.

    Supports patterns like:
      {"equivalent": true, "confidence": 0.87
      {"equivalent": false
    and normalizes them into the expected result shape.
    """
    if not text:
        return None

    text = text.strip()

    # Case 1:
    #   {"equivalent": true/false, "confidence": 0.123
    pattern_with_conf = re.compile(
        r'\{\s*"equivalent"\s*:\s*(true|false)\s*,\s*"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        flags=re.DOTALL,
    )

    candidates_with_conf = list(pattern_with_conf.finditer(text))
    for candidate in reversed(candidates_with_conf):
        equivalent_str = candidate.group(1)
        confidence_str = candidate.group(2)

        obj = {
            "equivalent": equivalent_str == "true",
            "confidence": float(confidence_str),
            "reason": "",
        }

        try:
            return normalize_checker_result(obj)
        except Exception:
            continue

    # Case 2:
    #   {"equivalent": true/false
    pattern_only_equivalent = re.compile(
        r'\{\s*"equivalent"\s*:\s*(true|false)',
        flags=re.DOTALL,
    )

    candidates_only_equivalent = list(pattern_only_equivalent.finditer(text))
    for candidate in reversed(candidates_only_equivalent):
        equivalent_str = candidate.group(1)

        obj = {
            "equivalent": equivalent_str == "true",
            "confidence": 0.0,
            "reason": "",
        }

        try:
            return normalize_checker_result(obj)
        except Exception:
            continue

    return None


def parse_checker_result_from_text(text: str) -> Optional[Dict[str, Any]]:
    parsed = extract_json_object_from_text(
        text,
        required_key="equivalent",
        validator=normalize_checker_result,
    )
    if parsed is not None:
        return parsed

    parsed_partial = parse_partial_checker_result_from_text(text)
    if parsed_partial is not None:
        return parsed_partial

    return None


def fallback_checker_error_result(reason: str) -> Dict[str, Any]:
    return {
        "equivalent": False,
        "confidence": 0.0,
        "reason": reason,
    }


# ============================================================
# Config
# ============================================================

@dataclass
class InformalCorrectnessCheckerConfig:
    """
    Correctness-checker-specific defaults that used to live in the monolithic CFG.
    """
    name: str = "informal_correctness_checker"

    # Prompts
    system_prompt: str = DEFAULT_CHECKER_SYSTEM_PROMPT
    tool_prompt: str = DEFAULT_CHECKER_TOOL_PROMPT
    preference_prompt: str = DEFAULT_CHECKER_PREFERENCE_PROMPT

    # Harmony reasoning / decoding
    reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH
    temperature: float = 0.0
    min_p: float = 0.0
    top_logprobs: Optional[int] = None
    seed_offset: int = 99991

    # Session limits
    max_turns: int = 64
    timeout_seconds: int = 120
    buffer_tokens: int = 512
    stream_text_window: int = 32

    # Jupyter tool runtime
    jupyter: JupyterKernelConfig = field(default_factory=JupyterKernelConfig)

    # Short-circuit optimization
    treat_identical_strings_as_equivalent: bool = True

    # Whether to append the preference/tool-usage guidance after the problem
    append_preference_prompt: bool = False


# ============================================================
# Termination callbacks
# ============================================================

def checker_terminate_on_chunk(
    state: HarmonySessionState,
    recent_text: str,
    new_text: str,
) -> Optional[TerminationSignal]:
    parsed = parse_checker_result_from_text(recent_text)
    if parsed is None:
        return None

    return TerminationSignal(
        reason="checker_json_parsed_in_stream",
        parsed_output=parsed,
    )


def checker_terminate_on_message(
    state: HarmonySessionState,
    message: Message,
    completion_text: str,
) -> Optional[TerminationSignal]:
    # Parse from the message completion text first
    parsed = parse_checker_result_from_text(completion_text)
    if parsed is not None:
        return TerminationSignal(
            reason="checker_json_parsed",
            parsed_output=parsed,
        )

    # If the model explicitly finalized without valid JSON, mark that explicitly
    if message.channel == "final":
        return TerminationSignal(
            reason="checker_final_no_json",
            parsed_output=None,
        )

    return None


def checker_terminate_on_session_end(
    state: HarmonySessionState,
) -> Optional[TerminationSignal]:
    parsed = parse_checker_result_from_text(state.combined_completion_text())
    if parsed is None:
        return None

    return TerminationSignal(
        reason="checker_json_parsed_posthoc",
        parsed_output=parsed,
    )


# ============================================================
# Agent
# ============================================================

class InformalCorrectnessCheckerAgent:
    """
    Truth-vs-candidate correctness checker.

    This agent is intentionally narrower than the older combined equivalence agent:
    it focuses on checking whether a candidate answer is correct with respect to the
    official answer key text.

    It is implemented on top of the reusable VLLMHarmonyBackend and owns its own
    JupyterToolBackend by default, so each checker instance has its own persistent
    Python tool session.
    """

    def __init__(
        self,
        cfg: Optional[InformalCorrectnessCheckerConfig] = None,
        *,
        jupyter_backend: Optional[JupyterToolBackend] = None,
    ):
        self.cfg = cfg or InformalCorrectnessCheckerConfig()
        self.jupyter_backend = jupyter_backend or JupyterToolBackend(
            description=self.cfg.tool_prompt,
            cfg=self.cfg.jupyter,
        )

    # --------------------------------------------------------
    # Prompt construction
    # --------------------------------------------------------

    def build_user_prompt(
        self,
        *,
        problem_text: str,
        truth_answer_text: str,
        candidate_answer_text: str,
    ) -> str:
        initial_prompt = (
            "PROBLEM:\n"
            f"{problem_text}\n\n"
            "GROUND TRUTH ANSWER KEY:\n"
            f"{truth_answer_text}\n\n"
            "CANDIDATE ANSWER (from model):\n"
            f"{candidate_answer_text}\n"
        )

        if not self.cfg.append_preference_prompt:
            return initial_prompt

        return f"{initial_prompt}\n\n{self.cfg.preference_prompt}"

    # --------------------------------------------------------
    # Agent spec construction
    # --------------------------------------------------------

    def build_agent_spec(self) -> HarmonyAgentSpec:
        tool_config: ToolNamespaceConfig = self.jupyter_backend.tool_config

        return HarmonyAgentSpec(
            name=self.cfg.name,
            system_prompt=self.cfg.system_prompt,
            reasoning_effort=self.cfg.reasoning_effort,
            temperature=self.cfg.temperature,
            min_p=self.cfg.min_p,
            top_logprobs=self.cfg.top_logprobs,
            seed_offset=self.cfg.seed_offset,
            max_turns=self.cfg.max_turns,
            timeout_seconds=self.cfg.timeout_seconds,
            buffer_tokens=self.cfg.buffer_tokens,
            stream_text_window=self.cfg.stream_text_window,
            tool_configs=[tool_config],
            tool_handlers={
                self.cfg.jupyter.recipient_name: self.jupyter_backend.handle_invocation,
            },
            terminate_on_chunk=checker_terminate_on_chunk,
            terminate_on_message=checker_terminate_on_message,
            terminate_on_session_end=checker_terminate_on_session_end,
            stop_on_final_channel=True,
        )

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def check_vs_truth(
        self,
        *,
        backend: VLLMHarmonyBackend,
        problem_text: str,
        truth_answer_text: str,
        candidate_answer_text: str,
        base_seed: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if (
            self.cfg.treat_identical_strings_as_equivalent
            and truth_answer_text == candidate_answer_text
        ):
            return {
                "result": {
                    "equivalent": True,
                    "confidence": 1.0,
                    "reason": "strings_equal",
                },
                "raw_output": "",
                "termination_reason": "strings_equal",
                "tool_calls": [],
                "turns": [],
            }

        agent_spec = self.build_agent_spec()
        user_prompt = self.build_user_prompt(
            problem_text=problem_text,
            truth_answer_text=truth_answer_text,
            candidate_answer_text=candidate_answer_text,
        )

        result = backend.run_user_prompt(
            agent=agent_spec,
            user_prompt=user_prompt,
            seed=base_seed,
            metadata=metadata,
        )
        return self._to_checker_result(result)

    # --------------------------------------------------------
    # Result conversion
    # --------------------------------------------------------

    def _to_checker_result(self, result: HarmonyRunResult) -> Dict[str, Any]:
        parsed = result.parsed_output
        if parsed is None:
            if result.exception:
                parsed = fallback_checker_error_result(
                    f"checker_exception:{result.exception}"
                )
            else:
                parsed = fallback_checker_error_result("no_json_found")

        return {
            "result": normalize_checker_result(parsed),
            "raw_output": result.raw_output,
            "termination_reason": result.termination_reason,
            "tool_calls": result.tool_calls,
            "turns": result.turns,
        }

    # --------------------------------------------------------
    # Lifecycle helpers
    # --------------------------------------------------------

    def reset_tools(self) -> None:
        self.jupyter_backend.reset()

    def close(self) -> None:
        self.jupyter_backend.close()

    def __enter__(self) -> "InformalCorrectnessCheckerAgent":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "DEFAULT_CHECKER_SYSTEM_PROMPT",
    "DEFAULT_CHECKER_TOOL_PROMPT",
    "InformalCorrectnessCheckerConfig",
    "InformalCorrectnessCheckerAgent",
    "normalize_checker_result",
    "parse_checker_result_from_text",
    "fallback_checker_error_result",
    "checker_terminate_on_chunk",
    "checker_terminate_on_message",
    "checker_terminate_on_session_end",
]