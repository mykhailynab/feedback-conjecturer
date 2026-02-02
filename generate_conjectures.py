#!/usr/bin/env python3
"""
Generate Lean 4 theorem stubs (ending in `by sorry`) that, if proven, would certify a candidate integer answer.

This script:
  - Reads attempts.jsonl (your notebook output; one row per attempt)
  - Reads reference.csv (columns: id, problem, answer)
  - For each attempt with a candidate integer `attempt_answer` (or `pred_final_answer` fallback),
    asks gpt-oss-120b (served via a local vLLM OpenAI server) to produce a Lean 4 *single theorem*
    (plus any needed imports/defs) that ends in `by sorry`.
  - Writes results to an output jsonl file.
  - Logs progress to console.

Inference structure matches your notebook's “Harmony prompt_ids + OpenAI completions(stream=True)”
pattern, including stop_token_ids and return_token_ids.

Requirements:
  pip install openai openai-harmony pandas
  (and vLLM installed in the environment if you want the script to start the server)

Typical usage:
    TRANSFORMERS_NO_TF=1 \
    TRANSFORMERS_NO_FLAX=1 \
    CUDA_VISIBLE_DEVICES=0 \
    TOKENIZERS_PARALLELISM=false \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TIKTOKEN_ENCODINGS_BASE=/kaggle/tmp/setup/tiktoken_encodings \
    python generate_conjectures.py \
        --start_server \
        --attempts_path /kaggle/input/solution-logs/kaggle/working/aimo3_logs/attempts.jsonl \
        --reference_csv /kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv \
        --output_jsonl /kaggle/working/aimo3_logs/lean_theorems.jsonl \
        --model_path /kaggle/input/gpt-oss-120b/transformers/default/1 \
        --served_model_name gpt-oss
"""



from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from openai import OpenAI

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    SystemContent,
    ReasoningEffort,
    ToolNamespaceConfig,
    Author,
    Message,
    Role,
    TextContent,
    Conversation,
)

# ----------------------------- Utilities -----------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # skip malformed
                continue
    return rows

def append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, (int,)):
            return int(x)
        s = str(x).strip()
        if s == "" or s.lower() == "none":
            return None
        # allow commas
        s = s.replace(",", "")
        return int(s)
    except Exception:
        return None

def clip(s: str, max_len: int) -> str:
    if s is None:
        return ""
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"...[truncated:{len(s)-max_len}]"

def extract_lean_code(text: str) -> str:
    """
    Extract Lean code from markdown fences if present; else return raw text.
    Preference order: ```lean4``` then ```lean``` then whole text.
    """
    for pat in [
        r"```lean4\s*\n(.*?)```",
        r"```lean\s*\n(.*?)```",
    ]:
        m = re.findall(pat, text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m[-1].strip()
    return text.strip()

def ensure_single_theorem_ends_by_sorry(code: str) -> str:
    """
    Best-effort sanitizer:
      - Keep from first import/namespace/def/theorem onward (drops prose header).
      - Ensure there is at least one `theorem` or `lemma`.
      - Ensure the final theorem ends with `by sorry` (or `:= by sorry` style).
    We do NOT try to prove it; we only enforce the stub shape.
    """
    code = code.strip()

    # Drop leading prose before first Lean-ish token
    anchors = ["import ", "namespace ", "section ", "def ", "abbrev ", "theorem ", "lemma ", "set_option ", "open "]
    first_idx = None
    for a in anchors:
        i = code.find(a)
        if i != -1:
            first_idx = i if first_idx is None else min(first_idx, i)
    if first_idx is not None and first_idx > 0:
        code = code[first_idx:].lstrip()

    # If multiple theorems/lemmas, keep only up to the end of the first theorem stub region
    # (We allow defs before it.)
    # Find first occurrence of theorem/lemma:
    m = re.search(r"(^|\n)\s*(theorem|lemma)\s", code)
    if not m:
        # If the model didn't output a theorem, fabricate a minimal stub (still includes imports).
        # (Better than crashing; lets you detect and filter later.)
        fabricated = (
            "import Mathlib\n\n"
            "theorem generated_answer_certificate : True := by sorry\n"
        )
        return fabricated

    # Find second theorem/lemma occurrence after the first; if present, truncate before it.
    start_first = m.start()
    m2 = re.search(r"(^|\n)\s*(theorem|lemma)\s", code[m.end():])
    if m2:
        cut = m.end() + m2.start()
        code = code[:cut].rstrip()

    # Ensure it ends in `by sorry` for the theorem stub.
    # We only patch the tail if it looks like the theorem ends with `:= by` or `:= by\n`.
    # If it already contains `sorry` inside the theorem proof, we leave it.
    if re.search(r":=\s*by\s*sorry\s*$", code, flags=re.DOTALL) or re.search(r"\n\s*by\s*sorry\s*$", code, flags=re.DOTALL):
        return code

    # If the code ends with `:= by` (with optional whitespace/newlines), append ` sorry`
    if re.search(r":=\s*by\s*$", code, flags=re.DOTALL):
        return code + " sorry\n"

    # If the code ends with `by` on the last line, append ` sorry`
    if re.search(r"\n\s*by\s*$", code, flags=re.DOTALL):
        return code + " sorry\n"

    # If it ends with `by` followed by whitespace/newlines, append `sorry` with indentation
    if code.rstrip().endswith("by"):
        return code.rstrip() + " sorry\n"

    # Otherwise, as a last resort, if there is no `sorry` at all, append a new `by sorry` line.
    if "sorry" not in code:
        return code.rstrip() + "\nby sorry\n"

    return code


# ----------------------------- Harmony prompt building -----------------------------

class LeanTheoremTemplate:
    """
    Mimics your notebook's approach:
      system = SystemContent.with_model_identity(system_prompt).with_reasoning_effort(...).with_tools(...)
      user   = raw text prompt
    """
    def get_system_content(self, system_prompt: str, tool_config: ToolNamespaceConfig) -> SystemContent:
        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply(self, system_prompt: str, user_prompt: str, tool_config: ToolNamespaceConfig) -> List[Message]:
        system_content = self.get_system_content(system_prompt, tool_config)
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
        user_message = Message.from_role_and_content(Role.USER, user_prompt)
        return [system_message, user_message]


# ----------------------------- vLLM server management -----------------------------

@dataclass
class ServerConfig:
    port: int
    model_path: str
    served_model_name: str
    seed: int
    dtype: str
    kv_cache_dtype: str
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int
    stream_interval: int
    enable_prefix_caching: bool

class VLLMServer:
    def __init__(self, cfg: ServerConfig, log_path: str, start: bool = True):
        self.cfg = cfg
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None
        self.log_file = None
        self.base_url = f"http://0.0.0.0:{cfg.port}/v1"
        self.api_key = "sk-local"

        if start:
            self.start()

    def start(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--seed", str(self.cfg.seed),
            "--model", self.cfg.model_path,
            "--served-model-name", self.cfg.served_model_name,
            "--tensor-parallel-size", "1",
            "--max-num-seqs", str(self.cfg.max_num_seqs),
            "--gpu-memory-utilization", str(self.cfg.gpu_memory_utilization),
            "--host", "0.0.0.0",
            "--port", str(self.cfg.port),
            "--dtype", self.cfg.dtype,
            "--kv-cache-dtype", self.cfg.kv_cache_dtype,
            "--max-model-len", str(self.cfg.max_model_len),
            "--stream-interval", str(self.cfg.stream_interval),
            "--async-scheduling",
            "--disable-log-stats",
        ]
        if self.cfg.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")

        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def wait_ready(self, timeout_s: int = 180) -> OpenAI:
        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=timeout_s)
        start = time.time()

        print(f"[{now_iso()}] Waiting for vLLM server at {self.base_url} ...")
        while time.time() - start < timeout_s:
            if self.proc is not None:
                rc = self.proc.poll()
                if rc is not None:
                    self.log_file.flush()
                    with open(self.log_path, "r", encoding="utf-8") as f:
                        logs = f.read()
                    raise RuntimeError(f"vLLM server exited with code {rc}. Logs:\n{logs}")

            try:
                client.models.list()
                print(f"[{now_iso()}] vLLM server ready.")
                return client
            except Exception:
                time.sleep(1)

        raise RuntimeError("Timed out waiting for vLLM server.")

    def stop(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=30)
            except Exception:
                pass
        if self.log_file is not None:
            try:
                self.log_file.close()
            except Exception:
                pass

    def __del__(self):
        self.stop()


# ----------------------------- Main generator -----------------------------

DEFAULT_SYSTEM_PROMPT = r"""
You are a Lean 4 formalization assistant.

Goal
- Given (1) a math problem statement in natural language and (2) a candidate integer answer,
  produce Lean 4 code containing EXACTLY ONE theorem (or lemma) named:
    `problem_<ID>_answer_correct`
  (with <ID> replaced by the given problem id).
- The theorem must be a *correctness certificate*: if the theorem were proven, it would logically imply
  that the candidate integer answer is correct for the problem.

Hard constraints (must follow)
1) Your final response MUST contain exactly one markdown fenced code block:
      ```lean4
      ...Lean code...
      ```
   No other text outside the code block.
2) Inside the code block:
   - Include imports as needed (prefer `import Mathlib`).
   - You may define auxiliary `def`/`abbrev`/`structure`/`notation` to model the problem.
   - There must be EXACTLY ONE `theorem` or `lemma` in the entire file.
   - That theorem/lemma must end in `:= by sorry` (or `:= by` then `sorry` on the next line).
   - The word `sorry` must appear exactly once in the entire file (only at the end proof stub).
   - Do NOT include any additional theorems/lemmas (including helper lemmas).
3) The theorem statement should directly connect the problem to the candidate answer.
   Use quantifiers/constraints to reflect the problem, then conclude an equality or proposition that
   makes the candidate answer “the” correct answer.

Style guidance (strongly recommended)
- Prefer a statement of the form:
    theorem problem_<ID>_answer_correct : (ProblemSpec ...) → candidate = ... := by sorry
  or
    theorem problem_<ID>_answer_correct : candidate = <computed value> := by sorry
  when the problem admits a clean closed-form statement.
- If the problem is an "existence/uniqueness" question, express that with `∃!` and conclude that the
  extracted numeric answer equals `candidate_answer`.
- If the problem asks for “the value of …”, model the expression as a `Nat`/`Int`/`ℤ`/`ℚ`/`ℝ` expression,
  then state that it equals the candidate.
- If the problem is combinatorial (counts), define a finite set and use `Fintype.card` / `Finset.card`.
- If the problem is number theory, use `Nat` or `Int` with modular arithmetic; prefer Mathlib notations.

Few-shot examples (format + intent)

EXAMPLE 1 (simple algebraic value)
Input:
- id: demo1
- problem: "Compute (3+5)^2."
- candidate answer: 64

Output:
```lean4
import Mathlib

theorem problem_demo1_answer_correct : (3 + 5 : ℤ)^2 = (64 : ℤ) := by
  sorry
````

EXAMPLE 2 (counting / finite sets)
Input:

* id: demo2
* problem: "How many integers n in {1,2,3,4,5} satisfy n ≤ 3?"
* candidate answer: 3

Output:

```lean4
import Mathlib

def demo2Set : Finset ℕ := {1, 2, 3, 4, 5}

theorem problem_demo2_answer_correct :
    (demo2Set.filter (fun n => n ≤ 3)).card = 3 := by
  sorry
```

EXAMPLE 3 (number theory / modular condition)
Input:

* id: demo3
* problem: "Find the smallest nonnegative integer x such that x ≡ 2 (mod 5) and x ≡ 3 (mod 7)."
* candidate answer: 17

Output:

```lean4
import Mathlib

theorem problem_demo3_answer_correct :
    IsLeast {x : ℕ | x ≡ 2 [ZMOD 5] ∧ x ≡ 3 [ZMOD 7]} 17 := by
  sorry
```

Remember:

* Exactly one theorem/lemma total.
* Exactly one `sorry`.
* Final response is only the single `lean4` block.
""".strip()

def build_user_prompt(problem_id: str, problem_text: str, candidate_answer: int, raw_solution: str | None = None) -> str:
    """
    User prompt with lightweight “scaffolding” to encourage a good certificate theorem.

    We do NOT require the model to use raw_solution, but including a short excerpt can help.
    """
    raw_solution_section = ""
    if raw_solution:
        raw_solution_section = (
            "A previous attempt (may be wrong, for context only):\n"
            f"{raw_solution}\n\n"
        )

    return (
        f"ID: {problem_id}\n\n"
        f"Problem statement:\n{problem_text}\n\n"
        f"Candidate integer answer: {candidate_answer}\n\n"
        f"{raw_solution_section}"
        "Task:\n"
        "Produce Lean 4 code that contains exactly ONE theorem named:\n"
        f"  problem_{problem_id}_answer_correct\n\n"
        "The theorem must be a correctness certificate: if proven, it would imply that the candidate integer answer is correct.\n\n"
        "Mandatory output format:\n"
        "- Your entire response must be exactly one markdown fenced code block:\n"
        "  ```lean4\n"
        "  ...\n"
        "  ```\n"
        "- No text outside the code block.\n\n"
        "Mandatory constraints:\n"
        "- Exactly one theorem/lemma in the file.\n"
        "- Exactly one occurrence of `sorry`, at the end of that theorem proof.\n"
        "- Add imports (prefer `import Mathlib`) and any necessary defs, but no extra lemmas.\n"
        "- End the theorem with `:= by sorry` (or `:= by` then `sorry` on next line).\n\n"
        "Suggestion (not required):\n"
        "- If the problem asks for a number of objects, define the set/Finset and state its `card = <candidate>`.\n"
        "- If it asks for a value of an expression, state that expression equals `<candidate>` in ℤ/ℕ/ℚ/ℝ.\n"
        "- If it asks for smallest/largest such integer, use `IsLeast` / `IsGreatest`.\n"
    )

def generate_theorem_once(
    client: OpenAI,
    served_model_name: str,
    encoding,
    stop_token_ids: List[int],
    template: LeanTheoremTemplate,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float,
    min_p: float,
    seed: int,
    top_logprobs: int,
    max_new_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate using the same “prompt_ids -> OpenAI completions(stream=True)” structure.
    Returns (text, meta).
    """
    tool_config = ToolNamespaceConfig(name="python", description="(not used)", tools=[])

    messages = template.apply(system_prompt, user_prompt, tool_config)
    conversation = Conversation.from_messages(messages)

    prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

    # Notebook used: max_tokens = context_tokens - len(prompt_ids)
    # Here we cap with max_new_tokens for predictability.
    max_tokens = max_new_tokens

    text_chunks: List[str] = []
    total_token_ids = 0
    finish_reason = "unknown"

    stream = client.completions.create(
        model=served_model_name,
        temperature=temperature,
        logprobs=top_logprobs,
        max_tokens=max_tokens,
        prompt=prompt_ids,
        seed=seed,
        stream=True,
        extra_body={
            "min_p": min_p,
            "stop_token_ids": stop_token_ids,
            "return_token_ids": True,
        }
    )

    try:
        for chunk in stream:
            choice = chunk.choices[0]
            text = choice.text or ""
            if text:
                text_chunks.append(text)

            token_ids = getattr(choice, "token_ids", None)
            if token_ids:
                total_token_ids += len(token_ids)

            # vLLM often supplies finish_reason only at the end; we capture when available
            fr = getattr(choice, "finish_reason", None)
            if fr:
                finish_reason = fr
    finally:
        try:
            stream.close()
        except Exception:
            pass

    out_text = "".join(text_chunks)

    meta = {
        "finish_reason": finish_reason,
        "generated_token_ids": total_token_ids,
        "prompt_token_ids": len(prompt_ids),
    }
    return out_text, meta


def _preload_model_weights(model_path) -> None:
        print(f'[{now_iso()}] Loading model weights from {model_path} into OS Page Cache...')
        start_time = time.time()
        
        files_to_load = []
        total_size = 0
    
        for root, _, files in os.walk(model_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
    
                if os.path.isfile(file_path):
                    files_to_load.append(file_path)
                    total_size += os.path.getsize(file_path)
    
        def _read_file(path: str) -> None:
    
            with open(path, 'rb') as file_object:
                while file_object.read(1024 * 1024 * 1024):
                    pass
    
        with ThreadPoolExecutor(max_workers=16) as executor:
            list(executor.map(_read_file, files_to_load))
    
        elapsed = time.time() - start_time
        print(f'[{now_iso()}] Processed {len(files_to_load)} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n')


def main():
    ap = argparse.ArgumentParser(description="Generate Lean theorem stubs from attempts.jsonl using gpt-oss-120b via vLLM.")
    ap.add_argument("--attempts_path", required=True, help="Path to attempts.jsonl")
    ap.add_argument("--reference_csv", required=True, help="Path to reference.csv with columns id,problem,answer")
    ap.add_argument("--output_jsonl", required=True, help="Where to write generated theorem jsonl")

    # Server / model
    ap.add_argument("--start_server", action="store_true", help="Start a local vLLM OpenAI server (recommended on Kaggle).")
    ap.add_argument("--base_url", default=None, help="If not starting server, provide an existing OpenAI-compatible base_url.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model_path", default="", help="Path for vLLM --model (required if --start_server).")
    ap.add_argument("--served_model_name", default="gpt-oss")

    ap.add_argument("--server_log", default="vllm_server.log")
    ap.add_argument("--server_timeout", type=int, default=3600)

    # vLLM args (subset)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--kv_cache_dtype", default="fp8_e4m3")
    ap.add_argument("--max_model_len", type=int, default=65536)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.96)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--stream_interval", type=int, default=200)
    ap.add_argument("--enable_prefix_caching", action="store_true")

    # Generation params
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--min_p", type=float, default=0.02)
    ap.add_argument("--top_logprobs", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=2048)

    # Prompt config
    ap.add_argument("--system_prompt_file", default=None, help="Optional path to a system prompt text file.")
    ap.add_argument("--max_solution_chars", type=int, default=20000)

    # Which attempts to process
    ap.add_argument("--mode", choices=["per_attempt", "best_per_id"], default="per_attempt",
                    help="per_attempt: generate per attempt row; best_per_id: choose one attempt per id (lowest entropy among answered).")
    ap.add_argument("--use_pred_final_when_missing", action="store_true",
                    help="If attempt_answer is missing, fall back to pred_final_answer.")
    ap.add_argument("--only_ids", default=None,
                    help="Comma-separated list of ids to process (optional).")
    ap.add_argument("--max_records", type=int, default=0,
                    help="Optional cap on number of generations (0 = no cap).")

    args = ap.parse_args()

    # Load reference problems
    ref_df = pd.read_csv(args.reference_csv)
    if not {"id", "problem", "answer"}.issubset(set(ref_df.columns)):
        raise ValueError("reference.csv must contain columns: id, problem, answer")

    problem_by_id = dict(zip(ref_df["id"].astype(str), ref_df["problem"].astype(str)))
    true_answer_by_id = dict(zip(ref_df["id"].astype(str), ref_df["answer"]))

    # Load attempts
    attempts = read_jsonl(args.attempts_path)
    print(f"[{now_iso()}] Loaded attempts: {len(attempts)} rows from {args.attempts_path}")

    only_ids_set = None
    if args.only_ids:
        only_ids_set = set(x.strip() for x in args.only_ids.split(",") if x.strip())
        attempts = [r for r in attempts if str(r.get("id")) in only_ids_set]
        print(f"[{now_iso()}] Filtered to only_ids: {len(attempts)} rows remain")

    # Filter to attempts with an integer candidate answer
    filtered: List[Dict[str, Any]] = []
    for r in attempts:
        pid = str(r.get("id", ""))
        if pid not in problem_by_id:
            continue
        ans = safe_int(r.get("attempt_answer"))
        if ans is None and args.use_pred_final_when_missing:
            ans = safe_int(r.get("pred_final_answer"))
        if ans is None:
            continue
        filtered.append(r)

    print(f"[{now_iso()}] Attempts with candidate answers: {len(filtered)}")

    # If best_per_id, collapse by id (lowest entropy among rows with candidate answer)
    rows_to_process: List[Dict[str, Any]] = []
    if args.mode == "best_per_id":
        best: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        for r in filtered:
            pid = str(r.get("id"))
            ent = r.get("entropy")
            try:
                ent_f = float(ent) if ent is not None else float("inf")
            except Exception:
                ent_f = float("inf")
            if pid not in best or ent_f < best[pid][0]:
                best[pid] = (ent_f, r)
        rows_to_process = [v[1] for v in best.values()]
        rows_to_process.sort(key=lambda x: str(x.get("id")))
        print(f"[{now_iso()}] Mode best_per_id: {len(rows_to_process)} rows")
    else:
        # per_attempt
        rows_to_process = filtered
        rows_to_process.sort(key=lambda x: (str(x.get("id")), safe_int(x.get("attempt")) or 0))
        print(f"[{now_iso()}] Mode per_attempt: {len(rows_to_process)} rows")

    if args.max_records and args.max_records > 0:
        rows_to_process = rows_to_process[: args.max_records]
        print(f"[{now_iso()}] Capped to max_records={args.max_records}: {len(rows_to_process)} rows")

    # Load system prompt
    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")

    # Setup Harmony encoding and stop tokens
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()
    template = LeanTheoremTemplate()

    # Setup client (and optionally start server)
    server = None
    if args.start_server:
        if not args.model_path:
            raise ValueError("--model_path is required when --start_server is set")
        
        _preload_model_weights(model_path=args.model_path)

        scfg = ServerConfig(
            port=args.port,
            model_path=args.model_path,
            served_model_name=args.served_model_name,
            seed=args.seed,
            dtype=args.dtype,
            kv_cache_dtype=args.kv_cache_dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            stream_interval=args.stream_interval,
            enable_prefix_caching=args.enable_prefix_caching,
        )
        server = VLLMServer(scfg, log_path=args.server_log, start=True)
        client = server.wait_ready(timeout_s=args.server_timeout)
    else:
        if not args.base_url:
            raise ValueError("If not starting server, provide --base_url")
        client = OpenAI(base_url=args.base_url, api_key="sk-local", timeout=args.server_timeout)

    # Prepare output
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{now_iso()}] Writing outputs to: {out_path}")

    total = len(rows_to_process)
    start_all = time.time()
    n_ok = 0
    n_fail = 0

    try:
        for idx, r in enumerate(rows_to_process, start=1):
            pid = str(r.get("id"))
            attempt_no = safe_int(r.get("attempt"))
            ent = r.get("entropy")
            try:
                ent_f = float(ent) if ent is not None else None
            except Exception:
                ent_f = None

            problem_text = problem_by_id.get(pid, "")
            true_answer = safe_int(true_answer_by_id.get(pid))

            candidate = safe_int(r.get("attempt_answer"))
            if candidate is None and args.use_pred_final_when_missing:
                candidate = safe_int(r.get("pred_final_answer"))

            if candidate is None:
                continue

            raw_solution = clip(r.get("raw_output", ""), args.max_solution_chars)

            user_prompt = build_user_prompt(
                problem_id=pid,
                problem_text=problem_text,
                candidate_answer=candidate,
                # raw_solution=raw_solution,
            )

            print(
                f"[{now_iso()}] ({idx}/{total}) id={pid} attempt={attempt_no} "
                f"cand={candidate} true={true_answer} ent={ent_f}"
            )

            t0 = time.time()
            try:
                # Use a per-row seed for some diversity but reproducible.
                row_seed = int((args.seed + (attempt_no or 0) + hash(pid) % 100000) % 2_000_000_000)

                text, meta = generate_theorem_once(
                    client=client,
                    served_model_name=args.served_model_name,
                    encoding=encoding,
                    stop_token_ids=stop_token_ids,
                    template=template,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=args.temperature,
                    min_p=args.min_p,
                    seed=row_seed,
                    top_logprobs=args.top_logprobs,
                    max_new_tokens=args.max_new_tokens,
                )

                raw_out = text
                lean_code = extract_lean_code(raw_out)
                # lean_code = ensure_single_theorem_ends_by_sorry(lean_code)

                rec = {
                    "ts": now_iso(),
                    "id": pid,
                    "attempt": attempt_no,
                    "candidate_answer": candidate,
                    "true_answer": true_answer,
                    "entropy": ent_f,
                    "status": r.get("status"),
                    "reject_reason": r.get("reject_reason"),
                    "finish_reason": r.get("finish_reason"),
                    "generation_meta": meta,
                    "lean_code": lean_code,
                    # keep a clipped copy of model output for debugging
                    "model_output": raw_out,
                }
                append_jsonl(str(out_path), rec)

                dt = time.time() - t0
                print(f"[{now_iso()}]   generated in {dt:.2f}s | prompt_tok={meta.get('prompt_token_ids')} gen_tok={meta.get('generated_token_ids')}")
                n_ok += 1

            except Exception as exc:
                dt = time.time() - t0
                err = f"{type(exc).__name__}: {exc}"
                print(f"[{now_iso()}]   FAILED in {dt:.2f}s | {err}")
                n_fail += 1
                rec = {
                    "ts": now_iso(),
                    "id": pid,
                    "attempt": attempt_no,
                    "candidate_answer": candidate,
                    "true_answer": true_answer,
                    "entropy": ent_f,
                    "error": err,
                }
                append_jsonl(str(out_path), rec)

    finally:
        if server is not None:
            server.stop()

    elapsed = time.time() - start_all
    print(f"[{now_iso()}] Done. ok={n_ok} fail={n_fail} total={n_ok+n_fail} elapsed={elapsed:.2f}s")
    print(f"[{now_iso()}] Output jsonl: {out_path}")


if __name__ == "__main__":
    main()
