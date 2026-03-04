#!/usr/bin/env python3
"""
Goedel-Formalizer autoformalization pipeline + Lean 4 checking (Lake project w/ Mathlib) + self-correction.

Backends (configurable):
  - vllm        : HF weights via vLLM OpenAI-compatible server (recommended for shared GPU)

Workflow:
  1) Build a chat prompt for Goedel-Formalizer using your provided Jinja chat template.
  2) Generate Lean code (expects ```lean4 ... ``` in model output).
  3) Extract Lean code block.
  4) Optionally splice into a wrapper theorem (optional mode), else compile the generated code directly.
  5) Write a .lean file inside a Mathlib Lake project and compile/check with `lake env lean`.
  6) If it fails, format errors with <error>...</error> markers and self-correct for up to --max_rounds.

Inputs:
  - statement string: natural language problem statement to formalize
  - theorem name: the name to use in Lean output

Outputs:
  - round logs (prompt/model output/full code/errors) into --out_dir/run_*/...
  - final Lean file in your Lake project at --lean_relpath

Typical vLLM usage (start server):
  python formalize.py \
    --backend vllm \
    --start_server \
    --model_path /path/to/Goedel-Formalizer-V2-32B \
    --served_model_name goedel-formalizer \
    --template_path goedel_formalizer_template.jinja \
    --project_dir /Users/mila/lean/mathlib4 \
    --lean_relpath FormalizerRun.lean \
    --theorem_name test_problem \
    --statement_file statement.txt

Or connect to existing server:
  python formalize.py \
    --backend vllm \
    --base_url http://0.0.0.0:8002/v1 \
    --served_model_name goedel-formalizer \
    ...

Notes:
  - For vLLM: default port=8002, base_url=http://0.0.0.0:8002/v1
  - Defaults are tuned to coexist with another vLLM server on an H100.
"""

from __future__ import annotations

import os
import re
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment
from openai import OpenAI
import torch


# ----------------------------- Template rendering -----------------------------

def render_with_template(
    chat_template: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
) -> str:
    """
    Render HF-style chat template into a single prompt string.
    Same approach as your earlier script: wrap dict into attribute-access objects.
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

def extract_lean4_code_block(model_text: str) -> Optional[str]:
    """
    Extract the last ```lean4 ... ``` (preferred) or ```lean ... ``` code block.
    """
    patterns = [
        r"```lean4\n(.*?)\n```",
        r"```lean4\n(.*?)```",
        r"```lean\n(.*?)```",
    ]
    for pat in patterns:
        matches = re.findall(pat, model_text, re.DOTALL)
        if matches:
            return matches[-1].strip()
    return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def get_next_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    max_idx = -1
    for path in base_dir.glob("run_*"):
        if path.is_dir():
            m = re.search(r"run_(\d+)$", path.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return base_dir / f"run_{max_idx + 1}"


# ----------------------------- Lean error formatting ---------------------------

def get_error_str(code: str, errors: List[Dict[str, Any]], error_thres: bool = True) -> str:
    """
    Produce error feedback with <error>...</error> markers, similar to your old pipeline.
    Expects Lean --json messages with:
      msg['pos']['line'], msg['pos']['column'], optional msg['endPos'], msg['data']
    """
    err_str = ""
    code_lines = code.split("\n")
    max_errs = 8 if error_thres else len(errors)

    for i, e in enumerate(errors[:max_errs]):
        start_line = e["pos"]["line"] - 1
        start_col = e["pos"]["column"]
        if e.get("endPos") is None:
            end_line = start_line
            end_col = len(code_lines[start_line]) if 0 <= start_line < len(code_lines) else start_col
        else:
            end_line = e["endPos"]["line"] - 1
            end_col = e["endPos"]["column"]

        # clamp
        start_line = max(0, min(start_line, len(code_lines) - 1))
        end_line = max(0, min(end_line, len(code_lines) - 1))
        start_col = max(0, min(start_col, len(code_lines[start_line])))
        end_col = max(0, min(end_col, len(code_lines[end_line])))

        err_str += f"\nError {i+1}:\n"
        err_str += "\nCorresponding Code:\n```lean4\n"

        snippet = ""
        for ii in range(-4, 0):
            if 0 <= start_line + ii < len(code_lines):
                snippet += code_lines[start_line + ii] + "\n"

        if start_line != end_line:
            snippet += code_lines[start_line][:start_col] + "<error>" + code_lines[start_line][start_col:] + "\n"
            show_line = 6
            for j in range(start_line + 1, min(end_line, start_line + show_line)):
                snippet += code_lines[j] + "\n"
            if end_line > start_line + show_line:
                snippet += "  ... --[Truncated]-- ...\n"
            snippet += code_lines[end_line][:end_col] + "</error>" + code_lines[end_line][end_col:] + "\n"
        else:
            snippet += (
                code_lines[start_line][:start_col]
                + "<error>"
                + code_lines[start_line][start_col:end_col]
                + "</error>"
                + code_lines[start_line][end_col:]
                + "\n"
            )

        if end_line + 1 < len(code_lines):
            snippet += code_lines[end_line + 1] + "\n"

        err_str += snippet
        err_str += "```\n"
        err_str += f"\nError Message: {e.get('data','')}\n"

    if len(errors) > max_errs:
        err_str += f"\n... [Omitted {len(errors) - max_errs} more errors] ...\n"

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
    Compile/check via Lake env.
    If failing, attempt a second pass with `lean --json` to parse errors.
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
            errs: List[Dict[str, Any]] = []
            for line in pj.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                if isinstance(msg, dict) and msg.get("severity") == "error":
                    if "pos" in msg and "data" in msg:
                        errs.append(msg)
            if errs:
                json_errors = errs
        except Exception:
            pass

    return LeanCheckResult(ok=ok, stdout=p.stdout, stderr=p.stderr, json_errors=json_errors)


def write_lean_file(project_dir: Path, rel_path: str, content: str) -> Path:
    out_path = project_dir / rel_path
    ensure_dir(out_path.parent)
    out_path.write_text(content, encoding="utf-8")
    return out_path


# ----------------------------- vLLM server management --------------------------

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

        print(f"[vLLM] waiting for server at {self.base_url} ...")
        start = time.time()
        while time.time() - start < timeout_s:
            if self.proc is not None:
                rc = self.proc.poll()
                if rc is not None:
                    try:
                        self.log_file.flush()
                    except Exception:
                        pass
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
    client: OpenAI,
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
    OpenAI completions endpoint with a raw prompt string (rendered from your template).
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

    it = client.completions.create(
        model=served_model_name,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        stream=True,
    )
    try:
        for ev in it:
            txt = ev.choices[0].text or ""
            if txt:
                chunks.append(txt)
                # optional live stream to console, like your old script
                print(txt, end="", flush=True)
    finally:
        try:
            it.close()
        except Exception:
            pass

    return "".join(chunks)


# ----------------------------- Prompt building -----------------------------

def build_initial_messages(theorem_name: str, informal_statement: str) -> List[Dict[str, str]]:
    """
    Autoformalization request. This is the *only* place you should need to tweak task instruction.
    """
    user_prompt = (
        f"Please autoformalize the following natural language problem statement in Lean 4. "
        f"Use the following theorem name: {theorem_name}\n"
        f"The natural language statement is: \n"
        f"{informal_statement}"
        f"Think before you provide the lean statement."
    )
    return [{"role": "user", "content": user_prompt}]


def build_correction_messages(
    prev_messages: List[Dict[str, str]],
    prev_assistant_output: str,
    error_feedback: str,
    correction_round_num: int,
) -> List[Dict[str, str]]:
    current = list(prev_messages)
    current.append({"role": "assistant", "content": prev_assistant_output})
    user_feedback = (
        f"The Lean code from Round {correction_round_num - 1} did not compile.\n\n"
        "Here is the compiler error message (with <error></error> marking approximate locations):\n\n"
        f"{error_feedback}\n\n"
        "Please produce a corrected Lean 4 file that compiles.\n"
        "Return ONLY one ```lean4 ...``` code block.\n"
        "Do not include any prose outside the code block.\n"
    )
    current.append({"role": "user", "content": user_feedback})
    return current


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


# ----------------------------- Main -------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Goedel-Formalizer autoformalization + Lean checking + self-correction.")

    # Backend selection
    ap.add_argument("--backend", choices=["vllm"], default="vllm",
                    help="Inference backend. Default: vllm")

    # Inputs
    ap.add_argument("--theorem_name", required=True,
                    help="Lean theorem name to use in the generated code.")
    ap.add_argument("--statement_file", default=None,
                    help="Path to a text file with the natural language statement.")
    ap.add_argument("--statement", default=None,
                    help="Natural language statement directly on CLI (alternative to --statement_file).")

    # Template
    ap.add_argument("--template_path", required=True,
                    help="Path to the Goedel-Formalizer Jinja chat template.")
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--no_enable_thinking", dest="enable_thinking", action="store_false")
    ap.set_defaults(enable_thinking=True)

    # Lean project
    ap.add_argument("--project_dir", required=True, help="Lake project directory (Mathlib project).")
    ap.add_argument("--lean_relpath", default="FormalizerRun.lean", help="Lean file path relative to project_dir.")
    ap.add_argument("--lean_timeout", type=int, default=180)

    # Self-correction
    ap.add_argument("--max_rounds", type=int, default=2)
    ap.add_argument("--error_thres", action="store_true", help="If enabled (default), will output at most 8 errors to the agent.")
    ap.add_argument("--no_error_thres", dest="error_thres", action="store_false")
    ap.set_defaults(error_thres=True)

    # Output logs
    ap.add_argument("--out_dir", default="formalizer_run_outputs")

    # vLLM args
    ap.add_argument("--start_server", action="store_true",
                    help="(vllm) Start a local vLLM OpenAI server.")
    ap.add_argument("--model_path", default="",
                    help="(vllm) Path for vLLM --model (required if --start_server).")
    ap.add_argument("--served_model_name", default="goedel-formalizer",
                    help="(vllm) Model name exposed by vLLM, used in OpenAI calls.")
    ap.add_argument("--port", type=int, default=8002,
                    help="(vllm) Port for the formalizer server. Default 8002.")
    ap.add_argument("--base_url", default="http://0.0.0.0:8002/v1",
                    help="(vllm) Base URL if connecting to existing server. Default matches port 8002.")
    ap.add_argument("--server_log", default="vllm_formalizer_server.log")
    ap.add_argument("--server_timeout", type=int, default=180)

    # vLLM tuning
    ap.add_argument("--dtype", default="bfloat16",
                    help="(vllm) --dtype. Default bfloat16 for H100.")
    ap.add_argument("--kv_cache_dtype", default="fp8_e4m3",
                    help="(vllm) --kv-cache-dtype. Default fp8_e4m3 for H100.")
    ap.add_argument("--max_model_len", type=int, default=40960,
                    help="(vllm) --max-model-len. Default 40960.")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.96,
                    help="(vllm) --gpu-memory-utilization. Default 0.96.")
    ap.add_argument("--max_num_seqs", type=int, default=32,
                    help="(vllm) --max-num-seqs. Default 32.")
    ap.add_argument("--stream_interval", type=int, default=200)
    ap.add_argument("--enable_prefix_caching", action="store_true")

    # Generation params
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=32768,
                    help="Max tokens to generate (vllm max_tokens).")
    ap.add_argument("--no_stream", action="store_true",
                    help="Disable streaming printing for vllm (still saves full output).")

    args = ap.parse_args()

    # Input statement
    if args.statement_file:
        informal_statement = Path(args.statement_file).read_text(encoding="utf-8").strip()
    elif args.statement:
        informal_statement = args.statement.strip()
    else:
        raise ValueError("Provide --statement_file or --statement.")

    # Paths
    project_dir = Path(args.project_dir).expanduser().resolve()
    if not project_dir.exists():
        raise FileNotFoundError(f"project_dir does not exist: {project_dir}")

    template_path = Path(args.template_path).expanduser().resolve()
    if not template_path.exists():
        raise FileNotFoundError(f"template_path does not exist: {template_path}")

    out_dir = get_next_run_dir(Path(args.out_dir).expanduser().resolve())
    ensure_dir(out_dir)

    chat_template = template_path.read_text(encoding="utf-8")

    # Backend setup
    vllm_server: Optional[VLLMServer] = None
    vllm_client = None

    if args.backend == "vllm":
        if OpenAI is None:
            raise RuntimeError("openai package missing; pip install openai")

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
            vllm_client = OpenAI(base_url=args.base_url, api_key="sk-local", timeout=args.server_timeout)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # Main loop
    round_messages = build_initial_messages(args.theorem_name, informal_statement)
    prev_assistant_output = ""
    best_pass_path: Optional[Path] = None

    try:
        for r in range(0, args.max_rounds + 1):
            print("\n==============================")
            print(f"Round {r} / {args.max_rounds}")
            print(f"Backend: {args.backend}")
            print("==============================")

            # Render prompt string for vLLM
            rendered_prompt = render_with_template(
                chat_template=chat_template,
                messages=round_messages,
                tools=None,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
            (out_dir / f"round_{r}_prompt.txt").write_text(rendered_prompt, encoding="utf-8")

            # Generate
            t0 = time.time()
            if args.backend == "vllm":
                assert vllm_client is not None
                model_text = vllm_generate_text(
                    client=vllm_client,
                    served_model_name=args.served_model_name,
                    prompt=rendered_prompt,
                    seed=args.seed + r,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    stream=(not args.no_stream),
                )

            dt = time.time() - t0
            (out_dir / f"round_{r}_model_output.txt").write_text(model_text, encoding="utf-8")
            print(f"\n[Round {r}] Generation done in {dt:.2f}s, output saved.")

            prev_assistant_output = model_text

            # Extract Lean code block
            code_block = extract_lean4_code_block(model_text)
            if not code_block:
                print(f"[Round {r}] ERROR: no ```lean4``` block found. Stopping.")
                break

            # Compile the generated code directly
            full_code = code_block.strip()
            (out_dir / f"round_{r}_full_code.lean").write_text(full_code, encoding="utf-8")

            # Write into Lake project and check
            lean_file_path = write_lean_file(project_dir, args.lean_relpath, full_code)
            print(f"[Round {r}] Wrote Lean file: {lean_file_path}")

            check = run_lean_check(project_dir, args.lean_relpath, timeout_s=args.lean_timeout)
            (out_dir / f"round_{r}_lean_stdout.txt").write_text(check.stdout, encoding="utf-8")
            (out_dir / f"round_{r}_lean_stderr.txt").write_text(check.stderr, encoding="utf-8")

            if check.ok:
                print(f"[Round {r}] Lean check PASSED.")
                best_pass_path = lean_file_path
                break

            print(f"[Round {r}] Lean check FAILED. Preparing correction prompt...")

            # Error feedback
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

            # Next round messages
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
        print(f"SUCCESS: Lean compiled. File: {best_pass_path}")
    else:
        print(f"FAILED: No compiling output within {args.max_rounds} correction rounds.")
        print(f"See logs in: {out_dir}")
    print("==============================\n")


if __name__ == "__main__":
    main()
