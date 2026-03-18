from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from openai_harmony import Message, ReasoningEffort, ToolNamespaceConfig

from conjecturering_agents.inference_backends.vllm_harmony import (
    HarmonyAgentSpec,
    HarmonyRunResult,
    HarmonySessionState,
    TerminationSignal,
    VLLMHarmonyBackend,
    extract_last_boxed_content,
)
from conjecturering_agents.tool_calling_backends.jupyter import (
    JupyterKernelConfig,
    JupyterToolBackend,
)


# ============================================================
# Default prompts
# ============================================================

DEFAULT_SOLVER_SYSTEM_PROMPT = (
    "You are an elite mathematical problem solver with expertise at the International "
    "Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through "
    "rigorous mathematical reasoning.\n\n"
    "# Problem-Solving Approach:\n"
    "1. UNDERSTAND: Carefully read and rephrase the problem in your own words. "
    "Identify what is given, what needs to be found, and any constraints.\n"
    "2. EXPLORE: Consider multiple solution strategies. Think about relevant theorems, "
    "techniques, patterns, or analogous problems. Don't commit to one approach immediately.\n"
    "3. PLAN: Select the most promising approach and outline key steps before executing.\n"
    "4. EXECUTE: Work through your solution methodically. Show all reasoning steps clearly.\n"
    "5. VERIFY: Check your answer by substituting back, testing edge cases, or using "
    "alternative methods. Ensure logical consistency throughout.\n\n"
    "# Mathematical Reasoning Principles:\n"
    "- Break complex problems into smaller, manageable sub-problems\n"
    "- Look for patterns, symmetries, and special cases that provide insight\n"
    "- Use concrete examples to build intuition before generalizing\n"
    "- Consider extreme cases and boundary conditions\n"
    "- If stuck, try working backwards from the desired result\n"
    "- Be willing to restart with a different approach if needed\n\n"
    "# Verification Requirements:\n"
    "- Cross-check arithmetic and algebraic manipulations\n"
    "- Verify that your solution satisfies all problem constraints\n"
    "- Test your answer with simple cases or special values when possible\n"
    "- Ensure dimensional consistency and reasonableness of the result\n\n"
    "# Output Format:\n"
    "Place your final answer inside \\boxed{...}, e.g., \\boxed{There is no such function.} or "
    "\\boxed{\\frac{4\\pi}{\\log 2}}.\n\n"
    "Think step-by-step and show your complete reasoning process. Quality of reasoning "
    "is as important as the final answer."
)

DEFAULT_SOLVER_TOOL_PROMPT = (
    "Use this tool to execute Python code for:\n"
    "- Complex calculations that would be error-prone by hand\n"
    "- Numerical verification of analytical results\n"
    "- Generating examples or testing conjectures\n"
    "- Visualizing problem structure when helpful\n"
    "- Brute-force verification for small cases\n\n"
    "The environment is a stateful Jupyter notebook. Code persists between executions.\n"
    "Always use print() to display results. Write clear, well-commented code.\n\n"
    "Remember: Code should support your mathematical reasoning, not replace it. "
    "Explain what you're computing and why before running code."
)

DEFAULT_SOLVER_PREFERENCE_PROMPT = (
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
# Config
# ============================================================

@dataclass
class SolverAgentConfig:
    """
    Solver-specific config defaults.
    """
    name: str = "solver"

    # Prompts
    system_prompt: str = DEFAULT_SOLVER_SYSTEM_PROMPT
    tool_prompt: str = DEFAULT_SOLVER_TOOL_PROMPT
    preference_prompt: str = DEFAULT_SOLVER_PREFERENCE_PROMPT

    # Harmony reasoning / decoding
    reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH
    temperature: float = 0.5
    min_p: float = 0.02
    top_logprobs: Optional[int] = 5
    seed_offset: int = 0

    # Session limits
    max_turns: int = 128
    timeout_seconds: int = 600
    buffer_tokens: int = 512
    stream_text_window: int = 32

    # Jupyter tool runtime
    jupyter: JupyterKernelConfig = field(default_factory=JupyterKernelConfig)

    # Whether to append the preference/tool-usage guidance after the problem
    append_preference_prompt: bool = True


# ============================================================
# Termination callbacks
# ============================================================

def solver_terminate_on_chunk(
    state: HarmonySessionState,
    recent_text: str,
    new_text: str,
) -> Optional[TerminationSignal]:
    """
    Early-stop as soon as a boxed answer appears in the streamed assistant text.
    """
    if "\\boxed" not in recent_text and "}" not in recent_text:
        return None

    boxed = extract_last_boxed_content(recent_text)
    if boxed is None:
        return None

    return TerminationSignal(
        reason="boxed_detected_in_stream",
        parsed_output=boxed,
    )


def solver_terminate_on_message(
    state: HarmonySessionState,
    message: Message,
    completion_text: str,
) -> Optional[TerminationSignal]:
    """
    If the assistant emits a final-channel message, try to parse a boxed answer
    from that message specifically.
    """
    if message.channel != "final":
        return None

    text_parts = []
    for item in message.content or []:
        text = getattr(item, "text", None)
        if text is not None:
            text_parts.append(text)
    final_text = "\n".join(text_parts)

    boxed = extract_last_boxed_content(final_text)
    if boxed is not None:
        return TerminationSignal(
            reason="final_channel_answer",
            parsed_output=boxed,
        )

    return TerminationSignal(
        reason="final_channel_no_boxed",
        parsed_output=None,
    )


def solver_terminate_on_session_end(
    state: HarmonySessionState,
) -> Optional[TerminationSignal]:
    """
    Post-hoc fallback: scan all assistant completion text for a boxed answer if
    streaming/message-level termination did not already find one.
    """
    raw_output = state.combined_completion_text()
    boxed = extract_last_boxed_content(raw_output)
    if boxed is None:
        return None

    return TerminationSignal(
        reason="boxed_found_posthoc",
        parsed_output=boxed,
    )


# ============================================================
# Solver agent
# ============================================================

class SolverAgent:
    """
    Solver agent wrapper around:
      - prompts
      - default agent config
      - a per-agent Jupyter tool runtime
      - boxed-answer termination logic

    This class does not own the LLM server/backend. It is designed to run on top
    of VLLMHarmonyBackend.

    Each SolverAgent instance owns its own JupyterToolBackend by default, which
    gives it a separate persistent kernel.
    """

    def __init__(
        self,
        cfg: Optional[SolverAgentConfig] = None,
        *,
        jupyter_backend: Optional[JupyterToolBackend] = None,
    ):
        self.cfg = cfg or SolverAgentConfig()
        self.jupyter_backend = jupyter_backend or JupyterToolBackend(
            description=self.cfg.tool_prompt,
            cfg=self.cfg.jupyter,
        )

    # --------------------------------------------------------
    # Prompt construction
    # --------------------------------------------------------

    def build_user_prompt(self, problem_text: str) -> str:
        if not self.cfg.append_preference_prompt:
            return str(problem_text)

        return f"{problem_text}\n\n{self.cfg.preference_prompt}"

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
            terminate_on_chunk=solver_terminate_on_chunk,
            terminate_on_message=solver_terminate_on_message,
            terminate_on_session_end=solver_terminate_on_session_end,
            stop_on_final_channel=True,
        )

    # --------------------------------------------------------
    # Running
    # --------------------------------------------------------

    def run_attempt(
        self,
        *,
        backend: VLLMHarmonyBackend,
        problem_id: str,
        problem_text: str,
        attempt_index: int,
        base_seed: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run one solver attempt and return a legacy-compatible attempt record.

        This preserves the shape expected by the rest of your current pipeline
        while moving all session-driving logic into the reusable backend.
        """
        agent_spec = self.build_agent_spec()
        user_prompt = self.build_user_prompt(problem_text)

        merged_metadata = {
            "problem_id": problem_id,
            "attempt_index": attempt_index,
        }
        if metadata:
            merged_metadata.update(metadata)

        result = backend.run_user_prompt(
            agent=agent_spec,
            user_prompt=user_prompt,
            seed=base_seed + attempt_index,
            metadata=merged_metadata,
        )

        return self._to_attempt_record(
            problem_id=problem_id,
            attempt_index=attempt_index,
            result=result,
        )

    # --------------------------------------------------------
    # Result conversion
    # --------------------------------------------------------

    def _to_attempt_record(
        self,
        *,
        problem_id: str,
        attempt_index: int,
        result: HarmonyRunResult,
    ) -> Dict[str, Any]:
        python_calls = 0
        python_errors = 0

        for call in result.tool_calls:
            if call.get("recipient") != self.cfg.jupyter.recipient_name:
                continue
            python_calls += 1
            if bool(call.get("had_error", False)) or bool(call.get("timed_out", False)):
                python_errors += 1

        termination_reason = result.termination_reason
        if result.exception:
            termination_reason = f"{termination_reason} msg={result.exception}"

        return {
            "Problem ID": problem_id,
            "Attempt": attempt_index,
            "Response Length": result.total_tokens,
            "Python Calls": python_calls,
            "Python Errors": python_errors,
            "Entropy": result.mean_entropy,
            "Answer": result.parsed_output,
            "Trace": {
                "prompt_token_ids_initial": result.prompt_token_ids_initial,
                "prompt_text_initial": result.prompt_text_initial,
                "turns": result.turns,
                "full_completion_token_ids": result.full_completion_token_ids,
                "full_conversation_token_ids": result.full_conversation_token_ids,
                "raw_output": result.raw_output,
                "last_assistant_channel": result.last_assistant_channel,
                "last_assistant_recipient": result.last_assistant_recipient,
            },
            "Termination Reason": termination_reason,
            "Tool Calls": result.tool_calls,
            "Attempt Started TS": result.started_ts,
            "Attempt Finished TS": result.finished_ts,
            "Attempt Elapsed MS": result.elapsed_ms,
        }

    # --------------------------------------------------------
    # Lifecycle helpers
    # --------------------------------------------------------

    def reset_tools(self) -> None:
        self.jupyter_backend.reset()

    def close(self) -> None:
        self.jupyter_backend.close()

    def __enter__(self) -> "SolverAgent":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
