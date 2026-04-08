"""
Goedel prover agent.

Takes a Lean theorem statement (ending in ``:= by sorry``), renders a prompt
using the model's Jinja2 chat template, generates a proof via multi-round
self-correction, and compiles the result with Lean.

Prompts are kept verbatim from the original Goedel pipeline because the
fine-tuned model is sensitive to the exact prompt format it was trained on.
TODO: add tests for this. The original pipeline is at https://github.com/Goedel-LM/Goedel-Prover-V2

Supports both VLLMRawBackend and OllamaBackend via the RawBackend interface.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from jinja2 import Environment

from conjecturing_agents.inference_backends.raw_backend import (
    RawBackend,
    RawGenerationConfig,
)
from conjecturing_agents.tool_calling_backends.lean4_compiler import (
    Lean4CompilerBackend,
    LeanCompilerConfig,
)


# ============================================================
# Prompt constants
# (kept verbatim from test_goedel.py — do NOT paraphrase)
# ============================================================

# build_initial_messages content after .strip()
_INITIAL_USER_PROMPT = (
    "Complete the following Lean 4 code:\n\n"
    "```lean4\n"
    "{formal_statement}```\n\n"
    "Before producing the Lean 4 code to formally prove the given theorem, "
    "provide a detailed proof plan outlining the main proof steps and strategies.\n"
    "The plan should highlight key ideas, intermediate lemmas, and proof structures "
    "that will guide the construction of the final formal proof."
)

# build_correction_messages user content
_CORRECTION_USER_PROMPT = (
    "The proof (Round {round_num}) is not correct. "
    "Following is the compilation error message, where we use "
    "<error></error> to signal the position of the error.\n\n"
    "{error_feedback}\n\n"
    "Before producing the Lean 4 code to formally prove the given theorem, "
    "provide a detailed analysis of the error message."
)


# ============================================================
# Lean code utilities
# (copied verbatim from test_goedel.py — do NOT modify logic)
# ============================================================

_BY_CLAUSE_RE = re.compile(r":=\s*by\b", re.MULTILINE)


def _remove_comments(text: str) -> str:
    text = re.sub(r"/-.*?-/", "", text, flags=re.DOTALL)
    lines = text.split("\n")
    cleaned_lines = [line.split("--", 1)[0] for line in lines]
    return "\n".join(cleaned_lines).strip()


def _return_theorem_to_prove(text: str) -> Optional[Tuple[int, int]]:
    pattern = r"((?:theorem).*?:=\s*by\s*sorry)"
    match = re.search(pattern, text, re.DOTALL)
    return match.span() if match else None


def _return_theorem_to_replace(text: str) -> Optional[Tuple[int, int]]:
    pattern = r"((?:^|\s)theorem\s+.*?:=\s*by)"
    match = re.search(pattern, text, re.DOTALL)
    return match.span() if match else None


def normalize_for_prompt(statement: str) -> str:
    """Ensure the theorem statement ends with ``:= by sorry``."""
    m = _BY_CLAUSE_RE.search(statement)
    if not m:
        raise ValueError(
            "normalize_for_prompt: cannot find ':= by' in the input statement."
        )
    return statement[: m.start()] + ":= by sorry"


def replace_statement_in_proof(statement: str, proof: str) -> str:
    """
    Splice a generated proof block into the original theorem statement.
    Returns a string starting with ``**Error**`` on failure.
    """
    if ("apply?" in proof) or ("exact?" in proof):
        return "**Error**, 'apply?' or 'exact?' is used, which is not allowed."

    stats_re = _remove_comments(statement)
    stats_span_ = _return_theorem_to_prove(stats_re)
    if stats_span_ is None:
        error_app = "\n".join(["\n"] + ["-- " + x for x in statement.split("\n")])
        return f"**Error**, can not find 'theorem' and ':= sorry' in {error_app}"

    proof_str = _remove_comments(proof)
    span = _return_theorem_to_replace(proof_str)
    if span is None:
        error_app = "\n".join(["\n"] + ["-- " + x for x in proof.split("\n")])
        return f"**Error**, can not find 'theorem' and ':=' in {error_app}"

    return stats_re[: stats_span_[1]].replace("sorry", "") + proof_str[span[1] :]


def extract_lean4_code_block(model_text: str) -> Optional[str]:
    """Return the last ```lean4 / ```lean code block from model output."""
    patterns = [
        r"```lean4\n(.*?)\n```",
        r"```lean4\n(.*?)```",
        r"```lean\n(.*?)```",
    ]
    for pat in patterns:
        matches = re.findall(pat, model_text, re.DOTALL)
        if matches:
            return matches[-1]
    return None


def format_lean_errors(
    code: str,
    errors: List[Dict[str, Any]],
    *,
    truncate: bool = True,
) -> str:
    """
    Format Lean JSON errors with ``<error>...</error>`` markers.
    (Mirrors get_error_str from test_goedel.py.)
    """
    err_str = ""
    code_lines = code.split("\n")
    max_errors = 8 if truncate else len(errors)

    for i, error in enumerate(errors[:max_errors]):
        start_line = error["pos"]["line"] - 1
        start_col = error["pos"]["column"]

        if error.get("endPos") is None:
            end_line = start_line
            end_col = (
                len(code_lines[start_line])
                if 0 <= start_line < len(code_lines)
                else start_col
            )
        else:
            end_line = error["endPos"]["line"] - 1
            end_col = error["endPos"]["column"]

        err_str += f"\nError {i + 1}:\n"
        err_str += "\nCorresponding Code:\n```lean4\n"

        error_code = ""
        for ii in range(-4, 0):
            if 0 <= start_line + ii < len(code_lines):
                error_code += f"{code_lines[start_line + ii]}\n"

        start_line = max(0, min(start_line, len(code_lines) - 1))
        end_line = max(0, min(end_line, len(code_lines) - 1))
        start_col = max(0, min(start_col, len(code_lines[start_line])))
        end_col = max(0, min(end_col, len(code_lines[end_line])))

        if start_line != end_line:
            error_code += (
                code_lines[start_line][:start_col]
                + "<error>"
                + code_lines[start_line][start_col:]
                + "\n"
            )
            show_line = 6
            for j in range(start_line + 1, min(end_line, start_line + show_line)):
                error_code += f"{code_lines[j]}\n"
            if end_line > start_line + show_line:
                last_j = min(end_line - 1, start_line + show_line - 1)
                leading = len(code_lines[last_j]) - len(code_lines[last_j].lstrip(" "))
                error_code += " " * leading + "... --[Truncated]-- ...\n"
            error_code += (
                code_lines[end_line][:end_col]
                + "</error>"
                + code_lines[end_line][end_col:]
                + "\n"
            )
        else:
            error_code += (
                code_lines[start_line][:start_col]
                + "<error>"
                + code_lines[start_line][start_col:end_col]
                + "</error>"
                + code_lines[start_line][end_col:]
                + "\n"
            )

        if end_line + 1 < len(code_lines):
            error_code += f"{code_lines[end_line + 1]}\n"

        err_str += error_code
        err_str += "\n```\n"
        err_str += f"\nError Message: {error.get('data', '')}\n"

    if len(errors) > max_errors:
        err_str += f"\n... [Omitted {len(errors) - max_errors} more errors] ...\n"

    return err_str


# ============================================================
# Template rendering
# ============================================================

def render_with_template(
    chat_template: str,
    messages: List[Dict[str, Any]],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
) -> str:
    """
    Render an HF-style Jinja2 chat template into a raw prompt string.
    (Mirrors render_with_template from test_goedel.py.)
    """
    class _Obj:
        def __init__(self, d: Dict[str, Any]) -> None:
            for k, v in d.items():
                setattr(self, k, v)

    msg_objs = [_Obj(m) for m in messages]
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    tmpl = env.from_string(chat_template)
    return tmpl.render(
        tools=[],
        messages=msg_objs,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


# ============================================================
# Config / result types
# ============================================================

@dataclass
class GoedelProverConfig:
    name: str = "goedel_prover"

    # Jinja2 chat template — path to file (e.g. goedel_template.jinja)
    chat_template_path: str = "goedel_template.jinja"
    enable_thinking: bool = True

    # Generation
    max_tokens: int = 16384
    temperature: float = 0.6
    top_p: float = 0.95
    repeat_penalty: float = 1.0

    # Context budget — must match the model's max context length.
    # Prompts + max_tokens that exceed this are skipped rather than sent.
    context_tokens: int = 40960

    # Number of self-correction rounds after the initial attempt.
    # Total attempts = max_rounds + 1.
    max_rounds: int = 2

    # Lean compiler used inside the generation loop (required).
    lean: Optional[LeanCompilerConfig] = None

    # Whether to truncate error lists to 8 (matches default in test_goedel.py).
    truncate_errors: bool = True


@dataclass
class GoedelProverResult:
    proved: bool

    # "proved" | "max_rounds_exhausted" | "context_exceeded" |
    # "no_code_block" | "splice_error" | "exception"
    termination_reason: str

    raw_output: str          # last model generation
    proof_text: str          # last extracted lean4 code block (or "")
    full_code: str           # last assembled theorem+proof file (or "")

    rounds_used: int         # total rounds attempted (1 = initial only)
    elapsed_ms: int

    rounds: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================
# Agent
# ============================================================

class GoedelProverAgent:
    """
    Agent that attempts to prove a Lean theorem using the Goedel prover model.

    Usage::

        agent = GoedelProverAgent(cfg)
        result = agent.prove_theorem(theorem_statement, backend, seed=42)

    The theorem statement must contain a ``theorem ... := by sorry`` placeholder.
    The caller is responsible for providing a ``RawBackend`` instance (either
    ``VLLMRawBackend`` or ``OllamaBackend``).
    """

    def __init__(self, cfg: Optional[GoedelProverConfig] = None) -> None:
        self.cfg = cfg or GoedelProverConfig()
        if self.cfg.lean is None:
            raise ValueError("GoedelProverConfig.lean must be set")
        self.lean_backend = Lean4CompilerBackend(self.cfg.lean)
        self.chat_template = Path(self.cfg.chat_template_path).read_text(
            encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_initial_messages(
        self, formal_statement: str
    ) -> List[Dict[str, str]]:
        content = _INITIAL_USER_PROMPT.format(formal_statement=formal_statement)
        return [{"role": "user", "content": content}]

    def _build_correction_messages(
        self,
        prev_messages: List[Dict[str, str]],
        prev_assistant_output: str,
        error_feedback: str,
        failed_round_num: int,
    ) -> List[Dict[str, str]]:
        msgs = list(prev_messages)
        msgs.append({"role": "assistant", "content": prev_assistant_output})
        msgs.append({
            "role": "user",
            "content": _CORRECTION_USER_PROMPT.format(
                round_num=failed_round_num,
                error_feedback=error_feedback,
            ),
        })
        return msgs

    def _render_prompt(self, messages: List[Dict[str, str]]) -> str:
        return render_with_template(
            self.chat_template,
            messages,
            add_generation_prompt=True,
            enable_thinking=self.cfg.enable_thinking,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(
        self,
        prompt: str,
        backend: RawBackend,
        gen_cfg: RawGenerationConfig,
        stream_callback: Optional[Callable[[str], None]],
    ) -> str:
        if stream_callback is not None:
            chunks: List[str] = []
            for chunk in backend.generate_streaming(prompt, gen_cfg):
                chunks.append(chunk)
                stream_callback(chunk)
            return "".join(chunks)
        return backend.generate(prompt, gen_cfg).text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prove_theorem(
        self,
        theorem_statement: str,
        backend: RawBackend,
        *,
        seed: int = 0,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> GoedelProverResult:
        """
        Attempt to prove ``theorem_statement`` using up to
        ``cfg.max_rounds + 1`` generation + Lean compilation cycles.

        Args:
            theorem_statement: Full Lean file ending in ``theorem ... := by sorry``.
            backend: A ``RawBackend`` instance (vLLM or Ollama).
            seed: Base seed; each round adds the round index.
            stream_callback: If provided, called with each streamed text chunk.
        """
        t0 = time.time()
        rounds: List[Dict[str, Any]] = []

        try:
            formal_statement = normalize_for_prompt(theorem_statement)
        except ValueError as exc:
            return GoedelProverResult(
                proved=False,
                termination_reason="splice_error",
                raw_output="",
                proof_text="",
                full_code="",
                rounds_used=0,
                elapsed_ms=int((time.time() - t0) * 1000),
                rounds=[{"error": str(exc)}],
            )

        messages = self._build_initial_messages(formal_statement)

        termination_reason = "max_rounds_exhausted"
        raw_output = ""
        proof_text = ""
        full_code = ""

        for round_idx in range(self.cfg.max_rounds + 1):
            round_record: Dict[str, Any] = {"round": round_idx}
            round_t0 = time.time()

            # --- Render + context budget check ---
            prompt = self._render_prompt(messages)
            prompt_tokens = backend.count_tokens(prompt)
            round_record["prompt_tokens"] = prompt_tokens

            if prompt_tokens + self.cfg.max_tokens > self.cfg.context_tokens:
                round_record["termination_reason"] = "context_exceeded"
                rounds.append(round_record)
                termination_reason = "context_exceeded"
                break

            # --- Generate ---
            gen_cfg_round = RawGenerationConfig(
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                seed=seed + round_idx,
                repeat_penalty=self.cfg.repeat_penalty,
            )
            raw_output = self._generate(prompt, backend, gen_cfg_round, stream_callback)
            round_record["raw_output"] = raw_output

            # --- Extract code block ---
            code_block = extract_lean4_code_block(raw_output)
            if not code_block:
                round_record["termination_reason"] = "no_code_block"
                round_record["elapsed_ms"] = int((time.time() - round_t0) * 1000)
                rounds.append(round_record)
                if round_idx < self.cfg.max_rounds:
                    # No code to correct with; rebuild from scratch next round
                    messages = self._build_initial_messages(formal_statement)
                else:
                    termination_reason = "no_code_block"
                continue

            proof_text = code_block
            round_record["proof_text"] = proof_text

            # --- Splice proof into statement ---
            full_code = replace_statement_in_proof(formal_statement, code_block)
            round_record["full_code"] = full_code

            if full_code.startswith("**Error**"):
                round_record["termination_reason"] = "splice_error"
                round_record["elapsed_ms"] = int((time.time() - round_t0) * 1000)
                rounds.append(round_record)
                termination_reason = "splice_error"
                break

            # --- Lean compilation ---
            compile_result = self.lean_backend.compile_code(full_code)
            round_record["lean_ok"] = compile_result.ok
            round_record["lean_timed_out"] = compile_result.timed_out
            round_record["lean_error_count"] = len(compile_result.json_errors)
            round_record["lean_relative_path"] = compile_result.relative_path
            round_record["elapsed_ms"] = int((time.time() - round_t0) * 1000)
            rounds.append(round_record)

            if compile_result.ok:
                termination_reason = "proved"
                break

            # --- Build correction for next round ---
            if round_idx < self.cfg.max_rounds:
                all_errors = compile_result.json_errors + compile_result.sorry_warnings
                error_feedback = format_lean_errors(
                    full_code,
                    all_errors,
                    truncate=self.cfg.truncate_errors,
                )
                messages = self._build_correction_messages(
                    messages,
                    raw_output,
                    error_feedback,
                    failed_round_num=round_idx,
                )

        return GoedelProverResult(
            proved=(termination_reason == "proved"),
            termination_reason=termination_reason,
            raw_output=raw_output,
            proof_text=proof_text,
            full_code=full_code,
            rounds_used=len(rounds),
            elapsed_ms=int((time.time() - t0) * 1000),
            rounds=rounds,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.lean_backend.close()

    def __enter__(self) -> "GoedelProverAgent":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = [
    "GoedelProverConfig",
    "GoedelProverResult",
    "GoedelProverAgent",
    "normalize_for_prompt",
    "replace_statement_in_proof",
    "extract_lean4_code_block",
    "format_lean_errors",
    "render_with_template",
]
