from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai_harmony import Message, ReasoningEffort, ToolNamespaceConfig

from conjecturing_agents.inference_backends.vllm_harmony import (
    HarmonyAgentSpec,
    HarmonyRunResult,
    HarmonySessionState,
    TerminationSignal,
    VLLMHarmonyBackend,
)
from conjecturing_agents.tool_calling_backends.jupyter import (
    JupyterKernelConfig,
    JupyterToolBackend,
)
from conjecturing_agents.tool_calling_backends.lean4_compiler import (
    LeanCompilerConfig,
    Lean4CompilerToolBackend,
    extract_lean_code_block_or_text,
)


# ============================================================
# Default prompts
# ============================================================

DEFAULT_CONJECTURE_FORMALIZER_SYSTEM_PROMPT = (
    "You are a Lean 4 formalization assistant.\n"
    "Your job is to formalize ONLY the conjectured answer of a problem, not the proof.\n\n"
    "You will be given:\n"
    "1. the informal problem statement,\n"
    "2. a tail of the model's informal solution,\n"
    "3. the extracted boxed answer,\n"
    "4. a Lean 4 file scaffold containing:\n"
    "   - an `abbrev ..._solution := sorry` placeholder,\n"
    "   - the main theorem statement ending in `sorry`.\n\n"
    "Your task is to produce a Lean 4 replacement for the abbrev only.\n\n"
    "Rules:\n"
    "- Do NOT modify the theorem statement.\n"
    "- Do NOT output a proof.\n"
    "- Do NOT output `sorry` in the abbrev.\n"
    "- Your final answer must be a single Lean 4 abbrev declaration defining the solution.\n"
    "- Prefer exact Lean expressions over prose.\n"
    "- If useful, use the python tool for algebraic sanity checks.\n"
    "- If useful, use the lean tool to compile a full Lean file with your abbrev inserted into the provided scaffold.\n"
    "- The theorem's existing trailing `sorry` is part of the scaffold and should be left untouched.\n\n"
    "Final output format:\n"
    "- Output a single ```lean4 ... ``` block containing only the abbrev declaration.\n"
)

DEFAULT_CONJECTURE_FORMALIZER_PYTHON_TOOL_PROMPT = (
    "Use this tool to execute Python code for formalization support.\n"
    "- Use it for algebraic simplification, symbolic checks, and sanity tests.\n"
    "- Use print() to display results.\n"
    "- Keep computations concise and directly relevant to constructing the Lean solution expression.\n\n"
    "The environment is a stateful Jupyter notebook. Code persists between executions.\n"
    "You have access to `math`, `numpy`, and `sympy`\n\n"
    "Write clear, well-commented code. Explain what you're computing and why before running code."
)

DEFAULT_CONJECTURE_FORMALIZER_LEAN_TOOL_PROMPT = (
    "Use this tool to compile Lean 4 code inside the configured Mathlib project.\n"
    "- Send the full Lean file you want compiled, preferably inside a ```lean4``` block.\n"
    "- You should typically take the provided scaffold, replace the abbrev placeholder with your candidate abbrev, and compile the whole file.\n"
    "- The theorem's existing trailing `sorry` belongs to the scaffold and is expected for this task.\n"
    "- Use the returned diagnostics to repair syntax, notation, typing, or namespace issues.\n"
)


# ============================================================
# Regex helpers / parsing utilities
# ============================================================

_ABBREV_NAME_RE = re.compile(
    r"^\s*(?:(?:noncomputable|unsafe|protected|private)\s+)*abbrev\s+([A-Za-z0-9_']+)\b",
    re.MULTILINE,
)
_SINGLE_LINE_ABBREV_RE = re.compile(
    r"^\s*(?:(?:noncomputable|unsafe|protected|private)\s+)*abbrev\s+([A-Za-z0-9_']+)\b[^\n]*:=.*$",
    re.MULTILINE,
)
_ABBREV_WITH_COMMENT_RE = re.compile(
    r"(?P<abbrev_line>^[ \t]*(?:(?:noncomputable|unsafe|protected|private)\s+)*abbrev[^\n]*:=.*\r?\n)"
    r"(?P<comment_line>^[ \t]*--[^\n]*\r?\n?)",
    re.MULTILINE,
)
_TOP_LEVEL_DECL_RE = re.compile(
    r"^\s*(?:(?:noncomputable|unsafe|protected|private)\s+)*"
    r"(?:abbrev|theorem|lemma|def|example|structure|class|inductive|instance|"
    r"namespace|end|section|open|import|#check|#eval|#print)\b"
)


def extract_abbrev_name_from_statement(lean_statement: str) -> Optional[str]:
    m = _ABBREV_NAME_RE.search(lean_statement or "")
    if not m:
        return None
    return m.group(1)


def extract_ground_truth_comment_and_strip_line(lean4_full_contents: str) -> Tuple[str, str]:
    """
    Extract the answer comment immediately following the abbrev placeholder and
    remove exactly that comment line, preserving imports/open statements and the
    rest of the scaffold.

    Expected shape:

        abbrev foo_solution : ... := sorry
        -- <ground truth answer>

        theorem ...
    """
    matches = list(_ABBREV_WITH_COMMENT_RE.finditer(lean4_full_contents or ""))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one abbrev+comment match, found {len(matches)}"
        )

    m = matches[0]
    comment_line = m.group("comment_line")
    ground_truth_answer = comment_line.split("--", 1)[1].strip()

    stripped = (
        lean4_full_contents[: m.start("comment_line")]
        + lean4_full_contents[m.end("comment_line") :]
    )

    # Make absolutely sure that the exact matched comment line was removed.
    if comment_line in stripped:
        raise ValueError("Failed to remove the matched abbrev answer comment line")

    return ground_truth_answer, stripped


def extract_last_abbrev_declaration(
    text: str,
    *,
    required_abbrev_name: Optional[str] = None,
) -> Optional[str]:
    """
    Extract the last plausible abbrev declaration from either:
      - a ```lean4``` code block, or
      - raw model text.

    We intentionally reject declarations containing `sorry`.
    """
    if not text:
        return None

    candidate_texts: List[str] = []

    code_block = extract_lean_code_block_or_text(text)
    if code_block:
        candidate_texts.append(code_block)

    if code_block != text.strip():
        candidate_texts.append(text)

    for candidate_text in candidate_texts:
        lines = candidate_text.splitlines()
        abbrev_indices = [
            i for i, line in enumerate(lines)
            if re.match(r"^\s*abbrev\b", line)
        ]

        for idx in reversed(abbrev_indices):
            head = lines[idx]
            m_name = re.match(r"^\s*abbrev\s+([A-Za-z0-9_']+)\b", head)
            if not m_name:
                continue

            abbrev_name = m_name.group(1)
            if required_abbrev_name is not None and abbrev_name != required_abbrev_name:
                continue

            block_lines = [head]
            for j in range(idx + 1, len(lines)):
                line = lines[j]
                if _TOP_LEVEL_DECL_RE.match(line):
                    break
                if line.strip().startswith("```"):
                    break
                block_lines.append(line)

            decl = "\n".join(block_lines).strip()
            if ":=" not in decl:
                continue
            if re.search(r"\bsorry\b", decl):
                continue

            return decl

    return None


def extract_rhs_from_abbrev_declaration(abbrev_declaration: str) -> Optional[str]:
    if ":=" not in abbrev_declaration:
        return None
    return abbrev_declaration.split(":=", 1)[1].strip()


def replace_abbrev_in_statement(
    lean_statement: str,
    new_abbrev_declaration: str,
    *,
    required_abbrev_name: Optional[str] = None,
) -> str:
    """
    Replace the unique abbrev declaration line in the scaffold with the generated
    abbrev declaration.

    This is useful both for external runners and for prompts/tool calls.
    """
    current_abbrev_name = extract_abbrev_name_from_statement(lean_statement)
    if current_abbrev_name is None:
        raise ValueError("Could not find abbrev name in Lean statement")

    if required_abbrev_name is not None and current_abbrev_name != required_abbrev_name:
        raise ValueError(
            f"Lean statement abbrev name {current_abbrev_name!r} does not match "
            f"required_abbrev_name={required_abbrev_name!r}"
        )

    new_abbrev_name = extract_abbrev_name_from_statement(new_abbrev_declaration)
    if new_abbrev_name is None:
        raise ValueError("Could not find abbrev name in new_abbrev_declaration")

    if new_abbrev_name != current_abbrev_name:
        raise ValueError(
            f"Generated abbrev name {new_abbrev_name!r} does not match scaffold "
            f"abbrev name {current_abbrev_name!r}"
        )

    matches = list(_SINGLE_LINE_ABBREV_RE.finditer(lean_statement))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one single-line abbrev placeholder in scaffold, found {len(matches)}"
        )

    m = matches[0]
    replacement = new_abbrev_declaration.strip()
    return lean_statement[: m.start()] + replacement + lean_statement[m.end() :]


# ============================================================
# Config
# ============================================================

@dataclass
class ConjectureFormalizerConfig:
    name: str = "conjecture_formalizer"

    # Prompts
    system_prompt: str = DEFAULT_CONJECTURE_FORMALIZER_SYSTEM_PROMPT
    python_tool_prompt: str = DEFAULT_CONJECTURE_FORMALIZER_PYTHON_TOOL_PROMPT
    lean_tool_prompt: str = DEFAULT_CONJECTURE_FORMALIZER_LEAN_TOOL_PROMPT

    # What the agent sees from the informal solution
    solution_tail_chars: int = 1000

    # Harmony reasoning / decoding
    reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH
    temperature: float = 0.2
    min_p: float = 0.0
    top_logprobs: Optional[int] = None
    seed_offset: int = 170001

    # Session limits
    max_turns: int = 48
    timeout_seconds: int = 300
    buffer_tokens: int = 512
    stream_text_window: int = 32

    # Tools
    use_python_tool: bool = True
    use_lean_tool: bool = True

    jupyter: JupyterKernelConfig = field(default_factory=JupyterKernelConfig)
    lean: Optional[LeanCompilerConfig] = None


# ============================================================
# Termination callbacks
# ============================================================

def conjecture_formalizer_terminate_on_message(
    state: HarmonySessionState,
    message: Message,
    completion_text: str,
) -> Optional[TerminationSignal]:
    if message.channel != "final":
        return None

    required_abbrev_name = state.metadata.get("required_abbrev_name")
    decl = extract_last_abbrev_declaration(
        completion_text,
        required_abbrev_name=required_abbrev_name,
    )
    if decl is None:
        return TerminationSignal(
            reason="formalizer_final_no_valid_abbrev",
            parsed_output=None,
        )

    return TerminationSignal(
        reason="formalizer_final_abbrev_extracted",
        parsed_output={
            "abbrev_name": extract_abbrev_name_from_statement(decl),
            "abbrev_declaration": decl,
            "rhs": extract_rhs_from_abbrev_declaration(decl),
        },
    )


def conjecture_formalizer_terminate_on_session_end(
    state: HarmonySessionState,
) -> Optional[TerminationSignal]:
    required_abbrev_name = state.metadata.get("required_abbrev_name")
    decl = extract_last_abbrev_declaration(
        state.combined_completion_text(),
        required_abbrev_name=required_abbrev_name,
    )
    if decl is None:
        return None

    return TerminationSignal(
        reason="formalizer_abbrev_extracted_posthoc",
        parsed_output={
            "abbrev_name": extract_abbrev_name_from_statement(decl),
            "abbrev_declaration": decl,
            "rhs": extract_rhs_from_abbrev_declaration(decl),
        },
    )


# ============================================================
# Agent
# ============================================================

class ConjectureFormalizerAgent:
    """
    Agent that formalizes only the conjectured answer, i.e. the abbrev
    placeholder in a Lean scaffold.

    It can use:
      - python tool (optional)
      - lean compiler tool (optional)

    Its final output is expected to be a single Lean abbrev declaration.
    """

    def __init__(
        self,
        cfg: Optional[ConjectureFormalizerConfig] = None,
        *,
        jupyter_backend: Optional[JupyterToolBackend] = None,
        lean_backend: Optional[Lean4CompilerToolBackend] = None,
    ):
        self.cfg = cfg or ConjectureFormalizerConfig()

        self.jupyter_backend: Optional[JupyterToolBackend] = None
        self.lean_backend: Optional[Lean4CompilerToolBackend] = None

        if self.cfg.use_python_tool:
            self.jupyter_backend = jupyter_backend or JupyterToolBackend(
                description=self.cfg.python_tool_prompt,
                cfg=self.cfg.jupyter,
            )

        if self.cfg.use_lean_tool:
            if lean_backend is not None:
                self.lean_backend = lean_backend
            else:
                if self.cfg.lean is None:
                    raise ValueError(
                        "ConjectureFormalizerConfig.use_lean_tool=True but no lean backend "
                        "and no cfg.lean were provided"
                    )
                self.lean_backend = Lean4CompilerToolBackend(
                    description=self.cfg.lean_tool_prompt,
                    cfg=self.cfg.lean,
                )

    # --------------------------------------------------------
    # Prompt construction
    # --------------------------------------------------------

    def build_user_prompt(
        self,
        *,
        informal_problem_text: str,
        proposed_solution_text: str,
        boxed_answer_text: str,
        lean_statement: str,
    ) -> str:
        solution_tail = (proposed_solution_text or "")[-self.cfg.solution_tail_chars :]
        target_abbrev_name = extract_abbrev_name_from_statement(lean_statement) or "<unknown_abbrev>"

        return (
            "Informal problem statement:\n"
            f"{informal_problem_text}\n\n"
            "Tail of the proposed informal solution:\n"
            f"{solution_tail}\n\n"
            "Extracted boxed answer:\n"
            f"{boxed_answer_text}\n\n"
            "Target abbrev name:\n"
            f"{target_abbrev_name}\n\n"
            "Lean 4 scaffold:\n"
            "```lean4\n"
            f"{lean_statement.strip()}\n"
            "```\n\n"
            "Task:\n"
            "- Produce only the abbrev declaration.\n"
            "- The abbrev name must exactly match the target abbrev name.\n"
            "- Do not output the theorem.\n"
            "- Do not output `sorry`.\n"
            "- If you use the lean tool, compile the full scaffold with your abbrev inserted.\n"
            "- Your final answer must be a single ```lean4``` block containing only the abbrev declaration.\n"
        )

    # --------------------------------------------------------
    # Agent spec
    # --------------------------------------------------------

    def build_agent_spec(self) -> HarmonyAgentSpec:
        tool_configs: List[ToolNamespaceConfig] = []
        tool_handlers: Dict[str, Any] = {}

        if self.jupyter_backend is not None:
            tool_configs.append(self.jupyter_backend.tool_config)
            tool_handlers[self.cfg.jupyter.recipient_name] = self.jupyter_backend.handle_invocation

        if self.lean_backend is not None:
            tool_configs.append(self.lean_backend.tool_config)
            tool_handlers[self.cfg.lean.recipient_name if self.cfg.lean is not None else "lean"] = (
                self.lean_backend.handle_invocation
            )

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
            tool_configs=tool_configs if tool_configs else None,
            tool_handlers=tool_handlers,
            terminate_on_chunk=None,
            terminate_on_message=conjecture_formalizer_terminate_on_message,
            terminate_on_session_end=conjecture_formalizer_terminate_on_session_end,
            stop_on_final_channel=True,
        )

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def formalize_conjecture(
        self,
        *,
        backend: VLLMHarmonyBackend,
        informal_problem_text: str,
        proposed_solution_text: str,
        boxed_answer_text: str,
        lean_statement: str,
        base_seed: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        required_abbrev_name = extract_abbrev_name_from_statement(lean_statement)
        if required_abbrev_name is None:
            raise ValueError("Could not extract target abbrev name from lean_statement")

        agent_spec = self.build_agent_spec()
        user_prompt = self.build_user_prompt(
            informal_problem_text=informal_problem_text,
            proposed_solution_text=proposed_solution_text,
            boxed_answer_text=boxed_answer_text,
            lean_statement=lean_statement,
        )

        merged_metadata = {
            "required_abbrev_name": required_abbrev_name,
        }
        if metadata:
            merged_metadata.update(metadata)

        result = backend.run_user_prompt(
            agent=agent_spec,
            user_prompt=user_prompt,
            seed=base_seed,
            metadata=merged_metadata,
        )
        return self._to_formalization_result(result)

    # --------------------------------------------------------
    # Result conversion
    # --------------------------------------------------------

    def _to_formalization_result(self, result: HarmonyRunResult) -> Dict[str, Any]:
        python_calls = 0
        python_errors = 0
        lean_calls = 0
        lean_errors = 0

        python_recipient = self.cfg.jupyter.recipient_name
        lean_recipient = self.cfg.lean.recipient_name if self.cfg.lean is not None else "lean"

        for call in result.tool_calls:
            recipient = call.get("recipient")
            if recipient == python_recipient:
                python_calls += 1
                if bool(call.get("had_error", False)) or bool(call.get("timed_out", False)):
                    python_errors += 1
            elif recipient == lean_recipient:
                lean_calls += 1
                if not bool(call.get("ok", False)):
                    lean_errors += 1

        parsed = result.parsed_output
        if parsed is None:
            parsed = {
                "abbrev_name": None,
                "abbrev_declaration": None,
                "rhs": None,
            }

        return {
            "result": parsed,
            "raw_output": result.raw_output,
            "termination_reason": result.termination_reason,
            "tool_calls": result.tool_calls,
            "turns": result.turns,
            "python_calls": python_calls,
            "python_errors": python_errors,
            "lean_calls": lean_calls,
            "lean_errors": lean_errors,
        }

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def reset_tools(self) -> None:
        if self.jupyter_backend is not None:
            self.jupyter_backend.reset()

    def close(self) -> None:
        if self.jupyter_backend is not None:
            self.jupyter_backend.close()
        if self.lean_backend is not None:
            self.lean_backend.close()

    def __enter__(self) -> "ConjectureFormalizerAgent":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "DEFAULT_CONJECTURE_FORMALIZER_SYSTEM_PROMPT",
    "DEFAULT_CONJECTURE_FORMALIZER_PYTHON_TOOL_PROMPT",
    "DEFAULT_CONJECTURE_FORMALIZER_LEAN_TOOL_PROMPT",
    "ConjectureFormalizerConfig",
    "ConjectureFormalizerAgent",
    "extract_abbrev_name_from_statement",
    "extract_ground_truth_comment_and_strip_line",
    "extract_last_abbrev_declaration",
    "extract_rhs_from_abbrev_declaration",
    "replace_abbrev_in_statement",
    "conjecture_formalizer_terminate_on_message",
    "conjecture_formalizer_terminate_on_session_end",
]