#!/usr/bin/env python3
"""
Single-thread Goedel pipeline + Lean 4 checking (Lake project w/ Mathlib) + self-correction.

Backends (configurable):
  - ollama: GGUF via Ollama (your current behavior)
  - vllm  : HF weights via vLLM OpenAI-compatible server (NEW)

Workflow (unchanged logically):
  1) Build prompt via your Jinja chat template (HF-style apply_chat_template emulation).
  2) Generate Lean proof (expects ```lean4 ... ``` in model output).
  3) Splice generated proof into the original theorem statement (replacing `:= by sorry`).
  4) Write a .lean file inside a Mathlib Lake project and compile/check with `lake env lean`.
  5) If it fails, format errors with <error>...</error> markers and self-correct for up to --max_rounds.

Notes for vLLM:
  - This script can start its own vLLM server (recommended) OR connect to an already running server.
  - Default port is 8001 to avoid clashing with your conjecturing server on 8000.
  - Default base_url targets that port: http://0.0.0.0:8001/v1
  - Default --gpu-memory-utilization is conservative for Goedel 8B/32B to coexist with other workloads.
    Adjust upward if Goedel is the only model on the GPU.
"""

from __future__ import annotations

import os
import re
import sys
import time
import json
import ollama
import argparse
import subprocess
from pathlib import Path
from openai import OpenAI
from jinja2 import Environment
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------- Template rendering -----------------------------

def render_with_template(
    chat_template: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
) -> str:
    """
    Render the HF-style chat template into a single prompt string.
    We adapt dicts to objects with attribute access by wrapping them in a tiny proxy.
    """
    class Obj:
        def __init__(self, d: Dict[str, Any]):
            for k, v in d.items():
                setattr(self, k, v)

    msg_objs = [Obj(m) for m in messages]
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    tmpl = env.from_string(chat_template)
    return tmpl.render(
        tools=tools or [],
        messages=msg_objs,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


# ----------------------------- Lean code utilities -----------------------------

def remove_comments(text: str) -> str:
    # Remove /- ... -/ blocks
    text = re.sub(r'/-.*?-/', '', text, flags=re.DOTALL)
    # Remove -- comments from each line
    lines = text.split('\n')
    cleaned_lines = [line.split('--', 1)[0] for line in lines]
    return '\n'.join(cleaned_lines).strip()


def return_theorem_to_prove(text: str) -> Optional[Tuple[int, int]]:
    pattern = r'((?:theorem).*?:=\s*by\s*sorry)'
    match = re.search(pattern, text, re.DOTALL)
    return match.span() if match else None


def return_theorem_to_replace(text: str) -> Optional[Tuple[int, int]]:
    pattern = r'((?:^|\s)theorem\s+.*?:=\s*by)'
    match = re.search(pattern, text, re.DOTALL)
    return match.span() if match else None


def replace_statement_in_proof(statement: str, proof: str) -> str:
    """
    Mirrors your utils.replace_statement_in_proof logic:
      - Rejects apply?/exact?
      - Removes comments
      - Replaces the `:= by sorry` in the statement with the generated proof tail
    """
    if ("apply?" in proof) or ("exact?" in proof):
        return "**Error**, 'apply?' or 'exact?' is used, which is not allowed."

    stats_re = remove_comments(statement)
    stats_span_ = return_theorem_to_prove(stats_re)
    if stats_span_ is None:
        error_app = '\n'.join(["\n"] + ['-- ' + x for x in statement.split('\n')])
        return f"**Error**, can not find 'theorem' and ':= sorry' in {error_app}"

    proof_str = remove_comments(proof)
    span = return_theorem_to_replace(proof_str)
    if span is None:
        error_app = '\n'.join(["\n"] + ['-- ' + x for x in proof.split('\n')])
        return f"**Error**, can not find 'theorem' and ':=' in {error_app}"

    # Replace sorry with empty, then append proof tail after `:= by`
    return stats_re[:stats_span_[1]].replace("sorry", "") + proof_str[span[1]:]


def extract_lean4_code_block(model_text: str) -> Optional[str]:
    """
    Extract the last ```lean4 ... ``` or ```lean ... ``` block.
    """
    patterns = [
        r'```lean4\n(.*?)\n```',
        r'```lean4\n(.*?)```',
        r'```lean\n(.*?)```',
    ]
    for pat in patterns:
        matches = re.findall(pat, model_text, re.DOTALL)
        if matches:
            return matches[-1]
    return None


_BY_CLAUSE_RE = re.compile(r":=\s*by\b", re.MULTILINE)

def normalize_for_prompt(statement: str) -> str:
    """
    Return a version of `statement` whose main theorem ends with `:= by sorry`.
      - finds the first occurrence of ':= by'
      - truncates everything after it
      - appends ':= by sorry'
    """
    m = _BY_CLAUSE_RE.search(statement)
    if not m:
        raise ValueError("normalize_for_prompt: cannot find ':= by' in the input statement.")
    prefix = statement[: m.start()]  # everything before ':= by'
    return prefix + ":= by sorry"


# ----------------------------- Lean error formatting ---------------------------

def get_error_str(code: str, errors: List[Dict[str, Any]], error_thres: bool = True) -> str:
    """
    Same spirit as your utils.get_error_str:
      - shows up to 8 errors if error_thres
      - injects <error>...</error> around the approximate span using line/column
      - includes 4 lines of context before start line, + 1 line after end line
    Expects Lean JSON-style errors with:
      error['pos']['line'], error['pos']['column'], error['endPos'] (or None), error['data'] message.
    """
    err_str = ""
    code_lines = code.split('\n')

    error_num_thres = 8 if error_thres else len(errors)

    for i, error in enumerate(errors[:error_num_thres]):
        start_line = error['pos']['line'] - 1
        start_col = error['pos']['column']

        if error.get('endPos') is None:
            end_line = start_line
            end_col = len(code_lines[start_line]) if 0 <= start_line < len(code_lines) else start_col
        else:
            end_line = error['endPos']['line'] - 1
            end_col = error['endPos']['column']

        # Build the highlighted snippet
        err_str += f"\nError {i + 1}:\n"
        err_str += "\nCorresponding Code:\n```lean4\n"

        error_code = ""
        for ii in range(-4, 0):
            if 0 <= start_line + ii < len(code_lines):
                error_code += f"{code_lines[start_line + ii]}\n"

        # safe clamp
        start_line = max(0, min(start_line, len(code_lines) - 1))
        end_line = max(0, min(end_line, len(code_lines) - 1))
        start_col = max(0, min(start_col, len(code_lines[start_line])))
        end_col = max(0, min(end_col, len(code_lines[end_line])))

        if start_line != end_line:
            error_code += code_lines[start_line][:start_col] + "<error>" + code_lines[start_line][start_col:] + "\n"

            if not error_thres:
                for j in range(start_line + 1, end_line):
                    error_code += f"{code_lines[j]}\n"
            else:
                show_line = 6
                for j in range(start_line + 1, min(end_line, start_line + show_line)):
                    error_code += f"{code_lines[j]}\n"
                if end_line > start_line + show_line:
                    leading_spaces = len(code_lines[j]) - len(code_lines[j].lstrip(' '))
                    error_code += "\n" + " " * leading_spaces + "... --[Truncated]-- ...\n"

            error_code += code_lines[end_line][:end_col] + "</error>" + code_lines[end_line][end_col:] + "\n"
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
        err_str += f"\nError Message: {error.get('data','')}\n"

    if len(errors) > error_num_thres:
        err_str += f"\n... [Omitted {len(errors) - error_num_thres} more errors] ...\n"

    return err_str


# ----------------------------- Lean compilation runner -------------------------

@dataclass
class LeanCheckResult:
    ok: bool
    stdout: str
    stderr: str
    json_errors: Optional[List[Dict[str, Any]]] = None


def run_lean_check(project_dir: Path, lean_file_rel: str, timeout_s: int = 120) -> LeanCheckResult:
    """
    Runs Lean in a Lake env, capturing stderr/stdout.
    Also tries to get structured errors using `lean --json`.
    """
    cmd = ["lake", "env", "lean", lean_file_rel]
    try:
        p = subprocess.run(
            cmd,
            cwd=str(project_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        return LeanCheckResult(ok=False, stdout=e.stdout or "", stderr=(e.stderr or "") + "\n[TIMEOUT]\n")

    ok = (p.returncode == 0)

    # Attempt a second pass with --json to extract structured errors (Lean supports `--json`).
    json_errors = None
    if not ok:
        try:
            cmd_json = ["lake", "env", "lean", "--json", lean_file_rel]
            pj = subprocess.run(
                cmd_json,
                cwd=str(project_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
            )
            # Lean writes JSON messages to stdout line-by-line.
            # We'll parse those and extract "error" severity items with pos/endPos.
            errs = []
            for line in pj.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                # Lean JSON format varies; common keys: "severity", "pos", "endPos", "data"
                print(f"Got msg: {msg}")
                if isinstance(msg, dict) and msg.get("severity") == "error":
                    if "pos" in msg and "data" in msg:
                        errs.append(msg)
            if errs:
                json_errors = errs
        except Exception:
            pass

    return LeanCheckResult(ok=ok, stdout=p.stdout, stderr=p.stderr, json_errors=json_errors)


# ----------------------------- vLLM server + client ----------------------------

@dataclass
class VLLMServerConfig:
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
    def __init__(self, cfg: VLLMServerConfig, log_path: str):
        self.cfg = cfg
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None
        self.log_file = None
        self.base_url = f"http://0.0.0.0:{cfg.port}/v1"
        self.api_key = "sk-local"

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

        print(f"[vLLM] waiting for server at {self.base_url} ...")
        while time.time() - start < timeout_s:
            if self.proc is not None:
                rc = self.proc.poll()
                if rc is not None:
                    self.log_file.flush()
                    logs = Path(self.log_path).read_text(encoding="utf-8", errors="ignore")
                    raise RuntimeError(f"vLLM server exited with code {rc}. Logs:\n{logs}")

            try:
                client.models.list()
                print("[vLLM] server ready.")
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


def vllm_generate_text(
    client: "OpenAI",
    served_model_name: str,
    prompt: str,
    *,
    seed: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stream: bool = True,
) -> str:
    """
    OpenAI-compat completions endpoint (like your gpt-oss script, but using a raw prompt string).
    """
    chunks: List[str] = []
    if not stream:
        resp = client.completions.create(
            model=served_model_name,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        return resp.choices[0].text or ""

    stream_iter = client.completions.create(
        model=served_model_name,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        stream=True,
    )

    try:
        for ev in stream_iter:
            txt = ev.choices[0].text or ""
            if txt:
                chunks.append(txt)
                print(txt, end="", flush=True)
    finally:
        try:
            stream_iter.close()
        except Exception:
            pass

    return "".join(chunks)


# ----------------------------- Ollama generation -------------------------------

def ollama_generate_text(model: str, prompt: str, options: Dict[str, Any], stream: bool = True) -> str:
    chunks: List[str] = []
    if stream:
        for part in ollama.generate(model=model, prompt=prompt, options=options, stream=True):
            chunk = part.get("response", "")
            if chunk:
                chunks.append(chunk)
                print(chunk, end="", flush=True)
    else:
        resp = ollama.generate(model=model, prompt=prompt, options=options, stream=False)
        chunks.append(resp.get("response", ""))
    return "".join(chunks)


# ----------------------------- Orchestration ------------------------------------

def build_initial_messages(formal_statement: str) -> List[Dict[str, str]]:
    prompt = """Complete the following Lean 4 code:

```lean4
{}```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()
    return [{"role": "user", "content": prompt.format(formal_statement)}]


def build_correction_messages(
    prev_messages: List[Dict[str, str]],
    prev_assistant_output: str,
    error_feedback: str,
    correction_round_num: int,
) -> List[Dict[str, str]]:
    """
    Mirrors your DeepSeekCoTHandler.generate_correction_prompt:
      - start from previous messages
      - append assistant failed attempt
      - append user feedback with error snippet and request for analysis
    """
    current_messages = list(prev_messages)
    current_messages.append({"role": "assistant", "content": prev_assistant_output})
    user_feedback_content = (
        f"The proof (Round {correction_round_num - 1}) is not correct. "
        "Following is the compilation error message, where we use <error></error> to signal the position of the error.\n\n"
        f"{error_feedback}\n\n"
        "Before producing the Lean 4 code to formally prove the given theorem, provide a detailed analysis of the error message."
    )
    current_messages.append({"role": "user", "content": user_feedback_content})
    return current_messages


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_lean_file(project_dir: Path, rel_path: str, content: str) -> Path:
    out_path = project_dir / rel_path
    ensure_dir(out_path.parent)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def get_next_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    max_idx = -1
    for path in base_dir.glob("run_*"):
        if path.is_dir():
            m = re.search(r"run_(\d+)$", path.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return base_dir / f"run_{max_idx + 1}"


def _preload_model_weights(model_path) -> None:
    print(f'Loading model weights from {model_path} into OS Page Cache...')
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
    print(f'Processed {len(files_to_load)} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n')


def main():
    ap = argparse.ArgumentParser(description="Goedel + Lean checking + self-correction (single-thread).")

    # Backend selection
    ap.add_argument("--backend", choices=["ollama", "vllm"], default="vllm",
                    help="Inference backend. Default: vllm")

    # ---- Ollama backend args ----
    ap.add_argument("--ollama_model", default="goedel-v2", help="Ollama model name, e.g. goedel-v2")

    # ---- vLLM backend args ----
    ap.add_argument("--start_server", action="store_true",
                    help="(vllm) Start a local vLLM OpenAI server.")
    ap.add_argument("--model_path", default="",
                    help="(vllm) Path for vLLM --model (required if --start_server).")
    ap.add_argument("--served_model_name", default="goedel",
                    help="(vllm) Name exposed by vLLM (used in OpenAI calls).")
    ap.add_argument("--port", type=int, default=8001,
                    help="(vllm) Server port. Default 8001 to avoid your 8000 conjecture server.")
    ap.add_argument("--base_url", default="http://0.0.0.0:8001/v1",
                    help="(vllm) If not starting server, connect here. Default matches --port 8001.")
    ap.add_argument("--server_log", default="vllm_goedel_server.log")
    ap.add_argument("--server_timeout", type=int, default=240)

    # vLLM server tuning (defaults are conservative to coexist with another server on the same H100)
    ap.add_argument("--dtype", default="bfloat16",
                    help="(vllm) --dtype. Default bfloat16 for H100.")
    ap.add_argument("--kv_cache_dtype", default="fp8_e4m3",
                    help="(vllm) --kv-cache-dtype. Default fp8_e4m3 for H100.")
    ap.add_argument("--max_model_len", type=int, default=40960,
                    help="(vllm) --max-model-len. Default 40960.")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.96,
                    help="(vllm) --gpu-memory-utilization. Default 0.35 (coexist-friendly on H100).")
    ap.add_argument("--max_num_seqs", type=int, default=32,
                    help="(vllm) --max-num-seqs. Default 32 for a single-thread pipeline.")
    ap.add_argument("--stream_interval", type=int, default=200)
    ap.add_argument("--enable_prefix_caching", action="store_true")

    # Generation params (used by both backends)
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repeat_penalty", type=float, default=None)
    ap.add_argument("--num_ctx", type=int, default=None, help="(ollama) Optional: num_ctx")
    ap.add_argument("--num_predict", type=int, default=32768,
                    help="Max tokens to generate (ollama num_predict; vllm max_tokens).")

    # Template
    ap.add_argument("--template_path", default="goedel_template.jinja", help="Path to Jinja chat template")
    ap.add_argument("--enable_thinking", action="store_true", help="If set, enables thinking blocks in template render (default: True)")
    ap.add_argument("--no_enable_thinking", dest="enable_thinking", action="store_false")
    ap.set_defaults(enable_thinking=True)

    # Lean project
    ap.add_argument("--project_dir", default="/Users/mila/lean/mathlib4", help="Lake project directory")
    ap.add_argument("--lean_relpath", default="GoedelRun.lean", help="Lean file path relative to project_dir")
    ap.add_argument("--lean_timeout", type=int, default=120)

    # Pipeline
    ap.add_argument("--max_rounds", type=int, default=2, help="Max correction rounds (0 = just initial)")
    ap.add_argument("--error_thres", action="store_true", help="Show at most 8 errors to the agent")
    ap.add_argument("--no_error_thres", dest="error_thres", action="store_false")
    ap.set_defaults(error_thres=True)

    # Inputs/outputs
    ap.add_argument("--statement_file", default=None, help="Path to a .lean file containing theorem statement with := by sorry")
    ap.add_argument("--out_dir", default="goedel_run_outputs", help="Directory to store logs/round outputs")

    args = ap.parse_args()

    out_dir = get_next_run_dir(Path(args.out_dir).expanduser().resolve())
    ensure_dir(out_dir)

    project_dir = Path(args.project_dir).expanduser().resolve()
    if not project_dir.exists():
        print(f"ERROR: project_dir does not exist: {project_dir}", file=sys.stderr)
        sys.exit(1)

    template_path = Path(args.template_path).expanduser().resolve()
    if not template_path.exists():
        print(f"ERROR: template_path does not exist: {template_path}", file=sys.stderr)
        sys.exit(1)

    chat_template = template_path.read_text(encoding="utf-8")

    # Load formal statement
    if args.statement_file:
        formal_statement = Path(args.statement_file).expanduser().read_text(encoding="utf-8").strip()
    else:
        # Default placeholder statement
        formal_statement = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
Prove that 2 + 2 = 4
-/
theorem sample_theorem :
    2 + 2 = 4 := by sorry
""".strip()

    formal_statement = normalize_for_prompt(formal_statement)

    # Backend setup
    vllm_server: Optional[VLLMServer] = None
    vllm_client = None

    if args.backend == "vllm":
        if args.start_server:
            if not args.model_path:
                raise ValueError("--model_path is required when --start_server is set (backend=vllm).")
            
            _preload_model_weights(model_path=args.model_path)

            scfg = VLLMServerConfig(
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
            vllm_server = VLLMServer(scfg, log_path=str(out_dir / args.server_log))
            vllm_server.start()
            vllm_client = vllm_server.wait_ready(timeout_s=args.server_timeout)
        else:
            # Connect to existing server
            vllm_client = OpenAI(base_url=args.base_url, api_key="sk-local", timeout=args.server_timeout)

    # Ollama options
    ollama_options: Dict[str, Any] = {
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.repeat_penalty is not None:
        ollama_options["repeat_penalty"] = args.repeat_penalty
    if args.num_ctx is not None:
        ollama_options["num_ctx"] = args.num_ctx
    if args.num_predict is not None and args.backend == "ollama":
        ollama_options["num_predict"] = args.num_predict

    # Run loop
    round_messages = build_initial_messages(formal_statement)
    prev_assistant_output = ""
    best_pass_path: Optional[Path] = None

    try:
        for r in range(0, args.max_rounds + 1):
            print(f"\n==============================")
            print(f"Round {r} / {args.max_rounds}")
            print(f"Backend: {args.backend}")
            print(f"==============================")

            rendered_prompt = render_with_template(
                chat_template=chat_template,
                messages=round_messages,
                tools=None,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
            (out_dir / f"round_{r}_prompt.txt").write_text(rendered_prompt, encoding="utf-8")

            # Generate
            start = time.time()
            if args.backend == "ollama":
                model_text = ollama_generate_text(args.ollama_model, rendered_prompt, options=ollama_options, stream=True)
            else:
                assert vllm_client is not None
                model_text = vllm_generate_text(
                    client=vllm_client,
                    served_model_name=args.served_model_name,
                    prompt=rendered_prompt,
                    seed=args.seed + r,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=int(args.num_predict),
                    stream=True,
                )
            dt = time.time() - start

            (out_dir / f"round_{r}_model_output.txt").write_text(model_text, encoding="utf-8")
            print(f"\n[Round {r}] Generation done in {dt:.2f}s, output saved.")

            prev_assistant_output = model_text

            # Extract Lean code block
            code_block = extract_lean4_code_block(model_text)
            if not code_block:
                print(f"[Round {r}] ERROR: no ```lean4``` block found. Stopping.")
                break

            # Splice proof into statement
            full_code = replace_statement_in_proof(formal_statement, code_block)
            (out_dir / f"round_{r}_full_code.lean").write_text(full_code, encoding="utf-8")

            if full_code.startswith("**Error**"):
                print(f"[Round {r}] Splicing error: {full_code}")
                break

            # Write into Lake project and check
            relpath = args.lean_relpath
            lean_file_path = write_lean_file(project_dir, relpath, full_code)
            print(f"[Round {r}] Wrote Lean file: {lean_file_path}")

            check = run_lean_check(project_dir, relpath, timeout_s=args.lean_timeout)
            (out_dir / f"round_{r}_lean_stdout.txt").write_text(check.stdout, encoding="utf-8")
            (out_dir / f"round_{r}_lean_stderr.txt").write_text(check.stderr, encoding="utf-8")

            if check.ok:
                print(f"[Round {r}] Lean check PASSED.")
                best_pass_path = lean_file_path
                break

            print(f"[Round {r}] Lean check FAILED. Preparing correction prompt...")

            # Build structured error feedback
            if check.json_errors:
                error_feedback = get_error_str(full_code, check.json_errors, error_thres=args.error_thres)
            else:
                stderr_lines = check.stderr.strip().splitlines()
                clipped = stderr_lines[:2000]
                error_feedback = "Lean stderr (clipped):\n```text\n" + "\n".join(clipped) + "\n```"

            (out_dir / f"round_{r}_error_feedback.txt").write_text(error_feedback, encoding="utf-8")

            if r >= args.max_rounds:
                print(f"[Round {r}] Reached max rounds; stopping.")
                break

            round_messages = build_correction_messages(
                prev_messages=round_messages,
                prev_assistant_output=prev_assistant_output,
                error_feedback=error_feedback,
                correction_round_num=r + 1,
            )

    finally:
        if vllm_server is not None:
            vllm_server.stop()

    print("\n==============================")
    if best_pass_path:
        print(f"SUCCESS: Proof checked. File: {best_pass_path}")
    else:
        print(f"FAILED: No passing proof within {args.max_rounds} correction rounds.")
        print(f"See logs in: {out_dir}")
    print("==============================\n")


if __name__ == "__main__":
    main()
