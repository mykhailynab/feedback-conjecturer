#!/usr/bin/env python3
import os
import gc
import re
import sys
import csv
import json
import math
import time
import queue
import argparse
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Tuple, Iterable, Set

import polars as pl
from openai import OpenAI
from jupyter_client import KernelManager
from collections import defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor, Future

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

# ============================================================
# Config
# ============================================================
@dataclass
class CFG:
    # Paths
    reference_path: str = "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv"
    log_dir: str = "/kaggle/working/aimo3_logs"
    attempts_filename: str = "attempts.jsonl"
    solutions_filename: str = "solutions.csv"
    submission_filename: str = "submission.csv"

    # Model serving
    served_model_name: str = "gpt-oss"
    model_path: str = "/kaggle/input/models/danielhanchen/gpt-oss-20b/transformers/default/1"
    port: int = 8000
    api_key: str = "sk-local"
    kv_cache_dtype: str = "fp8_e4m3"
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.96

    # Timeouts
    server_timeout: int = 180
    session_timeout: int = 960
    jupyter_timeout: int = 6
    sandbox_timeout: int = 3

    # attempt + checker timeouts
    attempt_timeout_seconds: int = 600
    checker_timeout_seconds: int = 120

    # Decoding / sampling / batching
    stream_interval: int = 200
    context_tokens: int = 65536
    buffer_tokens: int = 512
    search_tokens: int = 32
    top_logprobs: int = 5
    batch_size: int = 256
    attempts_per_problem: int = 16
    turns: int = 128
    checker_max_turns: int = 64
    seed: int = 42
    temperature: float = 0.5
    min_p: float = 0.02

    # Parallelism knobs (separated)
    agent_parallelism: int = 8           # caps concurrent attempt + equivalence + correctness agent tasks (global)
    kernel_workers: int = 32             # number of persistent Jupyter kernels
    preload_workers: int = 32            # threads used for model weight page-caching

    # Optional cap on problems processed (0 means all)
    max_problems: int = 0

    # Logging
    verbose: bool = True

    # --- Prompts ---
    system_prompt: str = (
        "You are an elite mathematical problem solver with expertise at the International "
        "Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through "
        "rigorous mathematical reasoning.\n\n"
        "# Problem-Solving Approach:\n"
        "1. UNDERSTAND: Carefully read and rephrase the problem in your own words. "
        "Identify what is given, what needs to be found, and any constraints.\n"
        "2. EXPLORE: Consider multiple solution strategies. Think about relevant theorems, "
        "techniques, patterns, or analogous problems. Don\"t commit to one approach immediately.\n"
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
        "Place your final answer inside \\boxed{...}, e.g., \\boxed{There is no such function.} or \\boxed{\\frac{4\\pi}{\\log 2}}.\n\n"
        "Think step-by-step and show your complete reasoning process. Quality of reasoning "
        "is as important as the final answer."
    )

    tool_prompt: str = (
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

    preference_prompt: str = (
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

    checker_system_prompt_truth: str = (
        "You are a mathematical answer equivalence checker.\n"
        "Given:\n"
        "  (1) the problem statement,\n"
        "  (2) the official answer key text (ground truth), and\n"
        "  (3) a candidate answer (model output),\n"
        "determine whether the candidate answer is mathematically equivalent to the ground truth.\n\n"
        "Rules:\n"
        "- Treat the answer key as authoritative; it may be phrased as 'Show that ...' or 'Prove that ...', but the candidte answer need not include a full proof.\n"
        "- The candidate answer may be phrased differently, but it is equivalent if it asserts the same mathematical claim\n"
        "  (and any explicit final value/expression matches).\n"
        "- If needed, use the python tool (sympy/numpy) to check symbolic/numeric equivalence.\n"
        "- Output MUST be a single line of strict JSON with keys:\n"
        "    {\"equivalent\": true/false, \"confidence\": 0..1, \"reason\": \"...\"}\n"
        "  Keep reason concise.\n"
        "- Do not include any other text besides the JSON line."
    )

    checker_system_prompt_pair: str = (
        "You are a mathematical answer equivalence checker.\n"
        "Given:\n"
        "  (1) the problem statement,\n"
        "  (2) Answer A,\n"
        "  (3) Answer B,\n"
        "determine whether Answer A and Answer B are mathematically equivalent (same final claim/value),\n"
        "even if phrased differently.\n\n"
        "Rules:\n"
        "- If needed, use the python tool (sympy/numpy) to check symbolic/numeric equivalence.\n"
        "- Output MUST be a single line of strict JSON with keys:\n"
        "    {\"equivalent\": true/false, \"confidence\": 0..1, \"reason\": \"...\"}\n"
        "  Keep reason concise.\n"
        "- Do not include any other text besides the JSON line."
    )

# ============================================================
# Logging
# ============================================================

class RunLogger:
    """
    Writes:
      - attempts.jsonl  : one record per attempt
      - solutions.csv   : one row per problem
      - events.jsonl    : timestamped high-level lifecycle events

    Trace:
      - Non-redundant: store initial prompt token ids (turn 0) and per-turn completion token ids.
      - Also store per-attempt started/finished timestamps and elapsed_ms.
    """

    def __init__(
        self,
        attempts_path: str,
        solutions_path: str,
        log_dir: str | None = None,
        verbose: bool = True,
    ):
        self.attempts_path = attempts_path
        self.solutions_path = solutions_path
        self.verbose = verbose
        self._lock = threading.Lock()

        if log_dir is None:
            log_dir = str(Path(solutions_path).parent)
        self.events_path = str(Path(log_dir) / "events.jsonl")

        self._init_solutions_csv()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _append_jsonl(self, path: str, records: list[dict]):
        with open(path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def log_event(self, event_type: str, payload: dict):
        rec = {"ts": self._now_iso(), "event": event_type, **payload}
        with self._lock:
            self._append_jsonl(self.events_path, [rec])

        if self.verbose:
            msg = payload.get("msg") or ""
            print(f"[LOG:{event_type}] {msg}".rstrip())

    def log_attempt(self, attempt_record: dict):
        ts = self._now_iso()
        summary = (
            f"id={attempt_record.get('id')} attempt={attempt_record.get('attempt')} "
            f"ans={self._safe_one_line(attempt_record.get('attempt_answer'))} "
            f"ent={attempt_record.get('entropy')} "
            f"py={attempt_record.get('python_calls')}/{attempt_record.get('python_errors')} "
            f"termination={attempt_record.get('termination_reason')} "
            f"elapsed_ms={attempt_record.get('attempt_elapsed_ms')}"
        )
        attempt_record_copy = dict(attempt_record)
        attempt_record_copy["ts"] = ts
        attempt_record_copy["summary"] = summary

        with self._lock:
            self._append_jsonl(self.attempts_path, [attempt_record_copy])

        if self.verbose:
            print(f"[LOG:attempts] Attempt summary: {attempt_record_copy.get('summary')}")

    def log_agent_call(self, *, kind: str, payload: dict):
        """
        kind: "equivalence" | "truth_check"
        payload should include at least:
          - problem_id
          - attempt (optional)
          - raw_output
          - parsed_result
        """
        rec = {"ts": self._now_iso(), "event": f"agent_{kind}", **payload}
        with self._lock:
            self._append_jsonl(self.events_path, [rec])
        if self.verbose:
            rid = payload.get("problem_id")
            att = payload.get("attempt")
            eq = payload.get("agent_call", {}).get("result", {}).get("equivalent", None)
            print(f"[LOG:agent_{kind}] id={rid} attempt={att} equivalent={eq}")

    @staticmethod
    def _safe_one_line(x: Any, max_len: int = 200) -> str:
        if x is None:
            return "None"
        s = str(x).replace("\n", "\\n")
        if len(s) > max_len:
            return s[:max_len] + f"...[truncated:{len(s)-max_len}]"
        return s
    
    def _init_solutions_csv(self):
        if os.path.exists(self.solutions_path):
            return

        header = [
            "id",
            "true_answer_text",
            "solve_started_ts",
            "solve_finished_ts",
            "solve_elapsed_ms",
            "attempts_total",
            "attempts_with_answer",
            "checker_summary",     # truth-check details
        ]
        with open(self.solutions_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    def log_solution_row(
        self,
        *,
        id_value: str,
        true_answer_text: str,
        solve_started_ts: Optional[str],
        solve_finished_ts: Optional[str],
        solve_elapsed_ms: Optional[int],
        attempts_total: int,
        attempts_with_answer: int,
        checker_summary: dict,
    ):
        row = [
            id_value,
            true_answer_text,
            solve_started_ts,
            solve_finished_ts,
            solve_elapsed_ms,
            attempts_total,
            attempts_with_answer,
            json.dumps(checker_summary or {}, ensure_ascii=False),
        ]

        with self._lock:
            with open(self.solutions_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

        if self.verbose:
            print(
                f"[LOG:attempts_finished] id={id_value}"
            )


# ============================================================
# Template + sandbox + tool
# ============================================================
class AIMO3Template:
    def __init__(self):
        pass

    def get_system_content(self, system_prompt: str, tool_config: ToolNamespaceConfig) -> SystemContent:
        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_config: ToolNamespaceConfig,
    ) -> list[Message]:
        system_content = self.get_system_content(system_prompt, tool_config)
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
        user_message = Message.from_role_and_content(Role.USER, user_prompt)
        return [system_message, user_message]


class AIMO3Sandbox:
    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list[int]:
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, timeout: float):
        self._default_timeout = timeout
        self._owns_kernel = False
        self._client = None
        self._km = None

        ports = self._get_next_ports(5)

        env = os.environ.copy()
        env["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
        env["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "0"
        env["JUPYTER_PLATFORM_DIRS"] = "1"
        env["PYTHONWARNINGS"] = "ignore"
        env["MPLBACKEND"] = "Agg"

        self._km = KernelManager()
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]

        self._km.start_kernel(env=env, extra_arguments=["--Application.log_level=CRITICAL"])

        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True

        self.execute(
            "import math\n"
            "import numpy\n"
            "import sympy\n"
            "import itertools\n"
            "import collections\n"
            "import mpmath\n"
            "mpmath.mp.dps = 64\n"
        )

    def _format_error(self, traceback_list: list[str]) -> str:
        clean_lines = []
        for frame in traceback_list:
            clean_frame = re.sub(r"\x1b\[[0-9;]*m", "", frame)
            if "File \"" in clean_frame and "ipython-input" not in clean_frame:
                continue
            clean_lines.append(clean_frame)
        return "".join(clean_lines)

    def execute(self, code: str, timeout: float | None = None) -> str:
        client = self._client
        effective_timeout = timeout or self._default_timeout

        msg_id = client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)

        stdout_parts = []
        stderr_parts = []
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > effective_timeout:
                self._km.interrupt_kernel()
                return f"[ERROR] Execution timed out after {effective_timeout} seconds"

            try:
                msg = client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)

            elif msg_type == "error":
                stderr_parts.append(self._format_error(content.get("traceback", [])))

            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")

            elif msg_type == "status":
                if content.get("execution_state") == "idle":
                    break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)
        if stderr:
            return f"{stdout.rstrip()}\n{stderr}" if stdout else stderr
        return stdout if stdout.strip() else "[WARN] No output. Use print() to see results."

    def close(self):
        try:
            if self._client:
                self._client.stop_channels()
        except Exception:
            pass

        if self._owns_kernel and self._km is not None:
            try:
                self._km.shutdown_kernel(now=True)
            except Exception:
                pass
            try:
                self._km.cleanup_resources()
            except Exception:
                pass

    def reset(self):
        self.execute(
            "%reset -f\n"
            "import math\n"
            "import numpy\n"
            "import sympy\n"
            "import itertools\n"
            "import collections\n"
            "import mpmath\n"
            "mpmath.mp.dps = 64\n"
        )

    def __del__(self):
        self.close()


class AIMO3Tool:
    def __init__(self, local_jupyter_timeout: float, tool_prompt: str, sandbox=None):
        self._local_jupyter_timeout = local_jupyter_timeout
        self._tool_prompt = tool_prompt
        self._jupyter_session = sandbox
        self._owns_session = sandbox is None

        self._execution_lock = threading.Lock()
        self._init_lock = threading.Lock()

    def _ensure_session(self):
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = AIMO3Sandbox(timeout=self._local_jupyter_timeout)

    def _ensure_last_print(self, code: str) -> str:
        lines = code.strip().split("\n")
        if not lines:
            return code

        last_line = lines[-1].strip()
        if "print" in last_line or "import" in last_line:
            return code
        if not last_line:
            return code
        if last_line.startswith("#"):
            return code
        if last_line.startswith(" "):
            return code

        lines[-1] = "print(" + last_line + ")"
        return "\n".join(lines)

    @property
    def instruction(self) -> str:
        return self._tool_prompt

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(name="python", description=self.instruction, tools=[])

    def _make_response(self, output: str, channel: str | None = None) -> Message:
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name="python")
        message = Message(author=author, content=[content]).with_recipient("assistant")
        if channel:
            message = message.with_channel(channel)
        return message

    def process_sync_plus(self, message: Message) -> list[Message]:
        self._ensure_session()
        raw_script = message.content[0].text
        final_script = self._ensure_last_print(raw_script)

        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(final_script)
            except TimeoutError as exc:
                output = f"[ERROR] {exc}"

        return [self._make_response(output, channel=message.channel)]


# ============================================================
# Utilities: robust \\boxed{...} extraction for arbitrary text
# ============================================================
def extract_last_boxed_content(text: str) -> Optional[str]:
    """
    Extract the *content* of the last \\boxed{...} in `text`, using a simple brace balancer.
    Returns None if not found or malformed.
    """
    if not text:
        return None

    idx = text.rfind("\\boxed")
    while idx != -1:
        # find first '{' after \boxed
        brace_start = text.find("{", idx)
        if brace_start == -1:
            idx = text.rfind("\\boxed", 0, idx)
            continue

        depth = 0
        i = brace_start
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    # content inside outer braces
                    return text[brace_start + 1 : i].strip()
            i += 1

        # unbalanced; look for previous boxed
        idx = text.rfind("\\boxed", 0, idx)

    return None


def _normalize_checker_obj(obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        raise ValueError("obj is not a dict")
    if "equivalent" not in obj:
        raise ValueError("\"equivalent\" not in obj")
    try:
        eq = bool(obj["equivalent"])
        conf = float(obj.get("confidence", 0.0))
        reason = str(obj.get("reason", ""))
        conf = max(0.0, min(1.0, conf))
        return {"equivalent": eq, "confidence": conf, "reason": reason}
    except Exception as e:
        raise e


def extract_checker_json_from_text(text: str) -> Dict[str, Any]:
    """
    Checker MUST emit a single JSON line.
    Robustly locate a JSON object, preferring ones that contain the key "equivalent".
    """
    if not text:
        return {"equivalent": False, "confidence": 0.0, "reason": "Error: empty_checker_output"}

    text = text.strip()

    last_err: Optional[BaseException] = None

    # Fast path: try parsing the entire string as JSON (works if checker truly outputs only JSON)
    try:
        obj = json.loads(text)
    except Exception:
        obj = None  # ignore exception if not a json.
    try:
        if obj is not None:
            return _normalize_checker_obj(obj)
    except Exception as e:
        last_err = e

    candidates = list(re.finditer(r"\{.*?\}", text, flags=re.DOTALL))

    for m in reversed(candidates):
        chunk = m.group(0)
        # if do_debug:
        #     print(f"Candidate 1: {chunk}")
        if '"equivalent"' not in chunk:
            continue
        try:
            obj = json.loads(chunk)
        except Exception as e:
            continue  # ignore exception if not a json.
        try:
            return _normalize_checker_obj(obj)
        except Exception as e:
            last_err = e
            continue

    decoder = json.JSONDecoder()
    for i in reversed([i for i, ch in enumerate(text) if ch == "{"]):
        try:
            obj, end = decoder.raw_decode(text[i:])  # can have extra text at the end
        except Exception:
            continue  # ignore exception if not a json.
        try:
            return _normalize_checker_obj(obj)
        except Exception as e:
            last_err = e
            continue
    
    pattern = re.compile(r'\{\s*"equivalent"\s*:\s*(true|false)\s*,\s*"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)', flags=re.DOTALL)
    candidates_noreason = list(pattern.finditer(text))
    for c in candidates_noreason:
        m_text = c.group(0) + '}'
        try:
            obj = json.loads(m_text)
        except Exception as e:
            continue  # ignore exception if not a json.
        try:
            return _normalize_checker_obj(obj)
        except Exception as e:
            last_err = e
            continue

    pattern = re.compile(r'\{\s*"equivalent"\s*:\s*(true|false)', flags=re.DOTALL)
    candidates_onlyeq = list(pattern.finditer(text))
    for c in candidates_onlyeq:
        m_text = c.group(0) + '}'
        try:
            obj = json.loads(m_text)
        except Exception as e:
            continue  # ignore exception if not a json.
        try:
            return _normalize_checker_obj(obj)
        except Exception as e:
            last_err = e
            continue

    return {
        "equivalent": False,
        "confidence": 0.0,
        "reason": f"Error: {type(last_err).__name__}: {last_err}" if last_err is not None else "Error: no_json_found"
    }


# ============================================================
# Solver
# ============================================================
class AIMO3Solver:
    """
    Owns:
      - Harmony encoding/template
      - vLLM server process + OpenAI client
      - persistent sandbox_pool (Jupyter kernels)

    IMPORTANT:
      - No early stopping.
      - No per-problem deadline/budget logic.
      - run_attempt executes exactly one attempt (for one problem, one attempt_index),
        using a per-attempt timeout.
      - Attempts return Answer as TEXT: the extracted content inside the last \\boxed{...}.
    """

    def __init__(self, cfg: CFG):
        self.cfg = cfg

        self.port = cfg.port
        self.base_url = f"http://0.0.0.0:{self.port}/v1"
        self.api_key = cfg.api_key

        self.template = AIMO3Template()
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        self._preload_model_weights()
        self.server_process = self._start_server()
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.cfg.session_timeout)

        self._wait_for_server()
        self._initialize_kernels()

    # ---- server + kernels ----
    def _preload_model_weights(self) -> None:
        # same logic as before, but uses cfg.preload_workers
        print(f"Loading model weights from {self.cfg.model_path} into OS Page Cache...")
        start_time = time.time()

        files_to_load = []
        total_size = 0

        for root, _, files in os.walk(self.cfg.model_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    files_to_load.append(file_path)
                    total_size += os.path.getsize(file_path)

        def _read_file(path: str) -> None:
            with open(path, "rb") as file_object:
                while file_object.read(1024 * 1024 * 1024):
                    pass

        with ThreadPoolExecutor(max_workers=self.cfg.preload_workers) as executor:
            list(executor.map(_read_file, files_to_load))

        elapsed = time.time() - start_time
        print(f"Processed {len(files_to_load)} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n")

    def _start_server(self) -> subprocess.Popen:
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--seed",
            str(self.cfg.seed),
            "--model",
            self.cfg.model_path,
            "--served-model-name",
            self.cfg.served_model_name,
            "--tensor-parallel-size",
            "1",
            "--max-num-seqs",
            str(self.cfg.batch_size),
            "--gpu-memory-utilization",
            str(self.cfg.gpu_memory_utilization),
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--dtype",
            self.cfg.dtype,
            "--kv-cache-dtype",
            self.cfg.kv_cache_dtype,
            "--max-model-len",
            str(self.cfg.context_tokens),
            "--stream-interval",
            str(self.cfg.stream_interval),
            "--async-scheduling",
            "--disable-log-stats",
            "--enable-prefix-caching",
        ]

        self.log_file = open("vllm_server.log", "w")
        return subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _wait_for_server(self):
        print("Waiting for vLLM server...")
        start_time = time.time()

        for _ in range(self.cfg.server_timeout):
            return_code = self.server_process.poll()

            if return_code is not None:
                self.log_file.flush()
                with open("vllm_server.log", "r") as log_file:
                    logs = log_file.read()
                raise RuntimeError(f"Server died with code {return_code}. Full logs:\n{logs}\n")

            try:
                self.client.models.list()
                elapsed = time.time() - start_time
                print(f"Server is ready (took {elapsed:.2f} seconds).\n")
                return
            except Exception:
                time.sleep(1)

        raise RuntimeError("Server failed to start (timeout).\n")

    def _initialize_kernels(self) -> None:
        print(f"Initializing {self.cfg.kernel_workers} persistent Jupyter kernels...")
        start_time = time.time()

        self.sandbox_pool = queue.Queue()

        def _create_sandbox():
            return AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)

        with ThreadPoolExecutor(max_workers=self.cfg.kernel_workers) as executor:
            futures = [executor.submit(_create_sandbox) for _ in range(self.cfg.kernel_workers)]
            for future in as_completed(futures):
                self.sandbox_pool.put(future.result())

        elapsed = time.time() - start_time
        print(f"Kernels initialized in {elapsed:.2f} seconds.\n")

    # ---- utilities ----
    def _decode_ids(self, ids: list[int]) -> str:
        return self.encoding.decode_utf8(ids)

    def _compute_mean_entropy(self, logprobs_buffer: list) -> float:
        if not logprobs_buffer:
            return float("inf")

        total_entropy = 0.0
        token_count = 0

        for top_logprobs_dict in logprobs_buffer:
            if not isinstance(top_logprobs_dict, dict):
                continue
            if not top_logprobs_dict:
                continue

            token_entropy = 0.0
            for _token_str, log_prob in top_logprobs_dict.items():
                prob = math.exp(log_prob)
                if prob > 0:
                    token_entropy -= prob * math.log2(prob)

            total_entropy += token_entropy
            token_count += 1

        if token_count == 0:
            return float("inf")

        return total_entropy / token_count

    def run_attempt(
        self,
        *,
        problem_id: str,
        problem_text: str,
        attempt_index: int,
    ) -> dict:
        """
        Execute ONE attempt for ONE problem, with per-attempt timeout.

        Returns a dict suitable to be stored in attempts.jsonl
        This function always returns a record, including failures/timeouts.

        Answer:
          - We extract answer_text = content of last \\boxed{...} encountered.
          - If not found, Answer is None.
        """
        attempt_started_ts = datetime.now(timezone.utc).isoformat()
        t0 = time.time()

        turns_compact: list[dict] = []
        full_completion_ids: list[int] = []
        tool_calls: list[dict] = []
        termination_reason = "unknown"
        python_calls = 0
        python_errors = 0
        total_tokens = 0
        final_answer_text: Optional[str] = None
        logprobs_buffer = []
        sandbox = None
        prompt_token_ids_initial: list[int] = []
        conversation = None

        deadline = time.time() + float(self.cfg.attempt_timeout_seconds)
        attempt_seed = int((self.cfg.seed + attempt_index) ** 2)

        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)
            local_tool = AIMO3Tool(
                local_jupyter_timeout=self.cfg.jupyter_timeout,
                tool_prompt=self.cfg.tool_prompt,
                sandbox=sandbox,
            )

            user_input = f"{problem_text} {self.cfg.preference_prompt}"

            encoding = self.encoding
            messages = self.template.apply_chat_template(self.cfg.system_prompt, user_input, local_tool.tool_config)
            conversation = Conversation.from_messages(messages)

            prompt_token_ids_initial = list(encoding.render_conversation_for_completion(conversation, Role.ASSISTANT))

            for _turn in range(self.cfg.turns):
                if time.time() > deadline:
                    termination_reason = "attempt_deadline_exceeded"
                    break

                prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)

                if max_tokens < self.cfg.buffer_tokens:
                    termination_reason = "context_exhausted"
                    break

                stream = None
                try:
                    stream = self.client.completions.create(
                        model=self.cfg.served_model_name,
                        temperature=self.cfg.temperature,
                        logprobs=self.cfg.top_logprobs,
                        max_tokens=max_tokens,
                        prompt=prompt_ids,
                        seed=attempt_seed,
                        stream=True,
                        extra_body={
                            "min_p": self.cfg.min_p,
                            "stop_token_ids": self.stop_token_ids,
                            "return_token_ids": True,
                        },
                    )

                    token_buffer: list[int] = []
                    completion_text_parts: list[str] = []

                    for chunk in stream:
                        if time.time() > deadline:
                            termination_reason = "attempt_deadline_exceeded"
                            break

                        new_tokens = chunk.choices[0].token_ids or []
                        new_text = chunk.choices[0].text or ""

                        if new_tokens:
                            token_buffer.extend(new_tokens)
                            full_completion_ids.extend(new_tokens)
                            total_tokens += len(new_tokens)

                        if new_text:
                            completion_text_parts.append(new_text)

                        chunk_logprobs = chunk.choices[0].logprobs
                        if chunk_logprobs is not None and chunk_logprobs.top_logprobs:
                            logprobs_buffer.extend(chunk_logprobs.top_logprobs)

                        # Fast boxed scan on streamed text (look only in the recent window)
                        if "\\boxed" in new_text or "}" in new_text:
                            search_text = "".join(completion_text_parts[-self.cfg.search_tokens :])
                            boxed = extract_last_boxed_content(search_text)
                            if boxed is not None:
                                final_answer_text = boxed
                                termination_reason = "boxed_detected_in_stream"
                                break
                finally:
                    try:
                        if stream is not None:
                            stream.close()
                    except Exception:
                        pass

                turns_compact.append(
                    {
                        "turn": _turn,
                        "completion_token_ids": token_buffer,
                        "completion_text": self._decode_ids(token_buffer) if token_buffer else "",
                    }
                )

                if final_answer_text is not None:
                    break

                if not token_buffer:
                    if termination_reason == "unknown":
                        termination_reason = "no_tokens"
                    break

                new_messages = encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
                conversation.messages.extend(new_messages)
                last_message = new_messages[-1]

                # If model explicitly ended, try to extract boxed from final content too
                if last_message.channel == "final":
                    answer_text_full = last_message.content[0].text if last_message.content else ""
                    boxed = extract_last_boxed_content(answer_text_full)
                    if boxed is not None:
                        final_answer_text = boxed
                        termination_reason = "final_channel_answer"
                    else:
                        termination_reason = "final_channel_no_boxed"
                    break

                if last_message.recipient == "python":
                    python_calls += 1
                    raw_script = last_message.content[0].text
                    final_script = local_tool._ensure_last_print(raw_script)

                    tool_responses = local_tool.process_sync_plus(last_message)
                    response_text = tool_responses[0].content[0].text

                    tool_calls.append({"code": final_script, "output": response_text})

                    if response_text.startswith("[ERROR]") or "Traceback" in response_text or "Error:" in response_text:
                        python_errors += 1

                    conversation.messages.extend(tool_responses)

            if termination_reason == "unknown":
                termination_reason = "max_turns_or_no_answer"

        except queue.Empty:
            python_errors += 1
            termination_reason = "sandbox_pool_timeout"

        except Exception as exc:
            python_errors += 1
            cause = repr(getattr(exc, "__cause__", None))
            context = repr(getattr(exc, "__context__", None))
            termination_reason = f"exception:{type(exc).__name__} msg={exc} cause={cause} context={context}"

        finally:
            if sandbox is not None:
                try:
                    sandbox.reset()
                finally:
                    self.sandbox_pool.put(sandbox)

        mean_entropy = self._compute_mean_entropy(logprobs_buffer)

        attempt_finished_ts = datetime.now(timezone.utc).isoformat()
        attempt_elapsed_ms = int((time.time() - t0) * 1000)

        return {
            "Problem ID": problem_id,
            "Attempt": attempt_index,
            "Response Length": total_tokens,
            "Python Calls": python_calls,
            "Python Errors": python_errors,
            "Entropy": mean_entropy,
            "Answer": final_answer_text,  # TEXT (boxed content) or None
            "Trace": {
                "prompt_token_ids_initial": prompt_token_ids_initial,
                "prompt_text_initial": self._decode_ids(prompt_token_ids_initial) if prompt_token_ids_initial else "",
                "turns": turns_compact,
                "full_completion_token_ids": full_completion_ids,
                "full_conversation_token_ids": self.encoding.render_conversation(conversation) if conversation is not None else [],
            },
            "Termination Reason": termination_reason,
            "Tool Calls": tool_calls,
            "Attempt Started TS": attempt_started_ts,
            "Attempt Finished TS": attempt_finished_ts,
            "Attempt Elapsed MS": attempt_elapsed_ms,
        }

    def close(self):
        if hasattr(self, "server_process") and self.server_process is not None:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
            except Exception:
                pass

        if hasattr(self, "log_file") and self.log_file is not None:
            try:
                self.log_file.flush()
                self.log_file.close()
            except Exception:
                pass

        if hasattr(self, "sandbox_pool") and self.sandbox_pool is not None:
            try:
                while not self.sandbox_pool.empty():
                    sb = self.sandbox_pool.get_nowait()
                    try:
                        sb.close()
                    except Exception:
                        pass
            except Exception:
                pass

    def __del__(self):
        self.close()


# ============================================================
# Checker agents (new)
# ============================================================
class AnswerEquivalenceAgent:
    """
    Runs Harmony+tool-enabled equivalence checks using the SAME vLLM server and SAME sandbox pool,
    and is governed by the SAME global LLMGate.

    Two modes:
      - truth_check: (problem, truth_answer, candidate_answer) -> equivalent?
      - pair_check:  (problem, answer_a, answer_b) -> equivalent?
    """

    def __init__(self, *, cfg: CFG, solver: AIMO3Solver):
        self.cfg = cfg
        self.solver = solver
        self.encoding = solver.encoding
        self.template = solver.template

    def _run_check(self, *, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        deadline = time.time() + float(self.cfg.checker_timeout_seconds)

        turns_compact: list[dict] = []
        tool_calls: list[dict] = []
        termination_reason = "unknown"
        sandbox = None

        attempt_seed = int((self.cfg.seed + 99991) ** 2)

        try:
            sandbox = self.solver.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)
            local_tool = AIMO3Tool(
                local_jupyter_timeout=self.cfg.jupyter_timeout,
                tool_prompt=self.cfg.tool_prompt,
                sandbox=sandbox,
            )

            messages = self.template.apply_chat_template(system_prompt, user_prompt, local_tool.tool_config)
            conversation = Conversation.from_messages(messages)

            for _turn in range(self.cfg.checker_max_turns):
                if time.time() > deadline:
                    termination_reason = "checker_deadline_exceeded"
                    break

                prompt_ids = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)
                if max_tokens < self.cfg.buffer_tokens:
                    termination_reason = "checker_context_exhausted"
                    break

                stream = None
                token_buffer: list[int] = []
                try:
                    stream = self.solver.client.completions.create(
                        model=self.cfg.served_model_name,
                        temperature=0.0,
                        logprobs=None,
                        max_tokens=max_tokens,
                        prompt=prompt_ids,
                        seed=attempt_seed,
                        stream=True,
                        extra_body={
                            "min_p": 0.0,
                            "stop_token_ids": self.solver.stop_token_ids,
                            "return_token_ids": True,
                        },
                    )

                    for chunk in stream:
                        if time.time() > deadline:
                            termination_reason = "checker_deadline_exceeded"
                            break
                        new_tokens = chunk.choices[0].token_ids or []
                        if new_tokens:
                            token_buffer.extend(new_tokens)
                finally:
                    try:
                        if stream is not None:
                            stream.close()
                    except Exception:
                        pass

                turns_compact.append(
                    {
                        "turn": _turn,
                        "completion_token_ids": token_buffer,
                        "completion_text": self.solver._decode_ids(token_buffer) if token_buffer else "",
                    }
                )

                if not token_buffer:
                    if termination_reason == "unknown":
                        termination_reason = "checker_no_tokens"
                    break

                new_messages = self.encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
                conversation.messages.extend(new_messages)
                last_message = new_messages[-1]

                if last_message.recipient == "python":
                    raw_script = last_message.content[0].text
                    final_script = local_tool._ensure_last_print(raw_script)

                    tool_responses = local_tool.process_sync_plus(last_message)
                    response_text = tool_responses[0].content[0].text
                    tool_calls.append({"code": final_script, "output": response_text})

                    conversation.messages.extend(tool_responses)
                    continue

                # try parse from this turn
                completion_text = turns_compact[-1]["completion_text"]
                parsed = extract_checker_json_from_text(completion_text)
                if parsed.get("reason") != "no_json_found":
                    termination_reason = "checker_json_parsed"
                    raw_output = "\n".join(t.get("completion_text", "") for t in turns_compact)
                    return {
                        "result": parsed,
                        "raw_output": raw_output,
                        "termination_reason": termination_reason,
                        "tool_calls": tool_calls,
                        "turns": turns_compact,
                    }

            # fallback: parse concat
            raw_output = "\n".join(t.get("completion_text", "") for t in turns_compact)
            parsed = extract_checker_json_from_text(raw_output)
            return {
                "result": parsed,
                "raw_output": raw_output,
                "termination_reason": termination_reason if termination_reason != "unknown" else "checker_finished_no_json",
                "tool_calls": tool_calls,
                "turns": turns_compact,
            }

        except queue.Empty:
            return {
                "result": {"equivalent": False, "confidence": 0.0, "reason": "checker_sandbox_pool_timeout"},
                "raw_output": "",
                "termination_reason": "checker_sandbox_pool_timeout",
                "tool_calls": tool_calls,
                "turns": turns_compact,
            }
        except Exception as exc:
            return {
                "result": {"equivalent": False, "confidence": 0.0, "reason": f"checker_exception:{type(exc).__name__}"},
                "raw_output": "",
                "termination_reason": f"checker_exception:{type(exc).__name__} msg={exc}",
                "tool_calls": tool_calls,
                "turns": turns_compact,
            }
        finally:
            if sandbox is not None:
                try:
                    sandbox.reset()
                finally:
                    self.solver.sandbox_pool.put(sandbox)


    def check_vs_truth(self, *, problem_text: str, truth_answer_text: str, candidate_answer_text: str) -> Dict[str, Any]:
        if truth_answer_text == candidate_answer_text:
            return {
                "result": {"equivalent": True, "confidence": 1.0, "reason": "strings_equal"},
                "raw_output": "",
                "termination_reason": "strings_equal",
                "tool_calls": [],
                "turns": [],
            }
        user_prompt = (
            "PROBLEM:\n"
            f"{problem_text}\n\n"
            "GROUND TRUTH ANSWER KEY:\n"
            f"{truth_answer_text}\n\n"
            "CANDIDATE ANSWER (from model):\n"
            f"{candidate_answer_text}\n"
        )
        return self._run_check(system_prompt=self.cfg.checker_system_prompt_truth, user_prompt=user_prompt)

    def check_pair(self, *, problem_text: str, answer_a: str, answer_b: str) -> Dict[str, Any]:
        if answer_a == answer_b:
            return {
                "result": {"equivalent": True, "confidence": 1.0, "reason": "strings_equal"},
                "raw_output": "",
                "termination_reason": "strings_equal",
                "tool_calls": [],
                "turns": [],
            }
        user_prompt = (
            "PROBLEM:\n"
            f"{problem_text}\n\n"
            "ANSWER A:\n"
            f"{answer_a}\n\n"
            "ANSWER B:\n"
            f"{answer_b}\n"
        )
        return self._run_check(system_prompt=self.cfg.checker_system_prompt_pair, user_prompt=user_prompt)


# ============================================================
# Validation
# ============================================================
def validate_cfg(cfg: CFG):
    errs = []

    if cfg.attempts_per_problem <= 0:
        errs.append("attempts_per_problem must be >= 1")

    if cfg.agent_parallelism <= 0:
        errs.append("agent_parallelism must be >= 1")

    if cfg.kernel_workers <= 0:
        errs.append("kernel_workers must be >= 1")
    if cfg.kernel_workers < cfg.agent_parallelism:
        errs.append("kernel_workers must be >= agent_parallelism (to cover concurrent attempts/checker calls)")

    if cfg.preload_workers <= 0:
        errs.append("preload_workers must be >= 1")

    if cfg.context_tokens <= cfg.buffer_tokens:
        errs.append("context_tokens must be > buffer_tokens")

    if cfg.batch_size <= 0:
        errs.append("batch_size must be >= 1")
    if cfg.turns <= 0:
        errs.append("turns must be >= 1")

    if cfg.temperature < 0:
        errs.append("temperature must be >= 0")
    if not (0 <= cfg.min_p <= 1):
        errs.append("min_p must be in [0, 1]")

    if cfg.attempt_timeout_seconds <= 0:
        errs.append("attempt_timeout_seconds must be >= 1")
    if cfg.checker_timeout_seconds <= 0:
        errs.append("checker_timeout_seconds must be >= 1")
    if cfg.checker_max_turns <= 0:
        errs.append("checker_max_turns must be >= 1")

    if errs:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errs))


def iter_reference(reference_df: pl.DataFrame) -> Iterable[Tuple[str, str, Optional[str]]]:
    """
    Ground truth is now TEXT.
    reference.csv "answer" may be numeric or freeform LaTeX/text; we store as string if present.
    """
    for row in reference_df.iter_rows(named=True):
        pid = str(row["id"])
        ptxt = str(row["problem"])
        true_answer_text = str(row["answer"])
        yield pid, ptxt, true_answer_text

# ============================================================
# Scheduler class
# ============================================================

@dataclass
class ProblemState:
    id_value: str
    problem_text: str
    true_answer_text: str
    total_attempts: int

    # attempt lifecycle
    next_attempt_idx: int = 0
    finished_attempts: List[dict] = field(default_factory=list)

    # per-attempt truth-check bookkeeping
    truth_scheduled: Dict[int, bool] = field(default_factory=dict)         # attempt_idx -> bool
    truth_done: Set[int] = field(default_factory=set)                      # {attempt_idx}
    truth_results: Dict[int, Dict[str, Any]] = field(default_factory=dict) # attempt_idx -> {"is_correct": bool|None, "skipped": bool, ...}

    # finalization
    finalized: bool = False

    # wallclock
    solve_started_ts: str = ""
    solve_finished_ts: str = ""
    solve_elapsed_ms: int = 0

    # mark when we emitted problem_start log (for “late start”)
    started_logged: bool = False


@dataclass(frozen=True)
class SchedulerTaskInfo:
    kind: str  # "attempt" | "truth_check"
    problem_idx: int
    attempt_idx: Optional[int] = None


class ProblemScheduler:
    """
    Global task loop across problems to fully utilize cfg.agent_parallelism.

    Inflight tasks include:
      - attempt tasks (solver.run_attempt)
      - truth check tasks (agent.check_vs_truth) for EVERY attempt (including failed/no-answer attempts -> skipped)

    Policy:
      - Fill capacity with attempt futures in problem order, exhausting each problem's attempts before moving on.
      - When an attempt finishes, immediately schedule a truth-check future for that attempt.
      - Finalize a problem when all attempts are done AND all per-attempt truth-checks are done.
      - Submission answer is the *lowest-entropy* attempt with a non-empty Answer (else empty string).
        is_correct reflects the truth-check result for the selected attempt (or None if skipped/unavailable).
    """

    def __init__(self, *, cfg: CFG, solver: AIMO3Solver, logger: RunLogger, problems: List[ProblemState]):
        self.cfg = cfg
        self.solver = solver
        self.logger = logger
        self.equivalence_agent = AnswerEquivalenceAgent(cfg=cfg, solver=solver)
        self.problems = problems

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def run_all(self) -> List[Dict[str, Any]]:
        submission_rows: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.cfg.agent_parallelism) as pool:
            inflight: Dict[Future, SchedulerTaskInfo] = {}

            # initial fill (attempt tasks only)
            self._fill_with_attempts(pool=pool, inflight=inflight)

            # global event loop
            while inflight:
                print(f"Completed: {sum([p.finalized for p in self.problems]) / len(self.problems) * 100:.2f}%")
                done_fut = next(as_completed(list(inflight.keys())))
                info = inflight.pop(done_fut)
                ps = self.problems[info.problem_idx]

                if info.kind == "attempt":
                    self._handle_attempt_done(
                        done_fut=done_fut,
                        pool=pool,
                        inflight=inflight,
                        ps=ps,
                        problem_idx=info.problem_idx,
                        attempt_idx=info.attempt_idx,
                    )
                elif info.kind == "truth_check":
                    self._handle_truth_check_done(
                        done_fut=done_fut,
                        ps=ps,
                        attempt_idx=int(info.attempt_idx),
                    )
                else:
                    raise ValueError(f"Unknown task kind: {info.kind}")

                # finalize any problems that became eligible
                for _pidx, pstate in enumerate(self.problems):
                    if self._ready_to_finalize_problem(pstate):
                        submission_rows.append(self._finalize_problem_and_log_solution(pstate))

                # refill capacity with more attempt tasks
                self._fill_with_attempts(pool=pool, inflight=inflight)

        return submission_rows

    # ------------------------------------------------------------
    # Attempt filling / problem-start logging
    # ------------------------------------------------------------
    def _log_problem_start_if_needed(self, ps: ProblemState) -> None:
        if ps.started_logged:
            return
        ps.solve_started_ts = datetime.now(timezone.utc).isoformat()
        self.logger.log_event("problem_start", {"id": ps.id_value, "msg": f"Problem id: {ps.id_value}"})
        ps.started_logged = True

    def _has_more_attempts(self, ps: ProblemState) -> bool:
        return ps.next_attempt_idx < ps.total_attempts

    def _can_submit_more_tasks_globally(self, inflight: Dict[Future, SchedulerTaskInfo]) -> bool:
        return len(inflight) < self.cfg.agent_parallelism

    def _submit_attempt(
        self,
        *,
        pool: ThreadPoolExecutor,
        inflight: Dict[Future, SchedulerTaskInfo],
        problem_idx: int,
        ps: ProblemState,
    ) -> bool:
        if not self._has_more_attempts(ps):
            return False

        self._log_problem_start_if_needed(ps)

        attempt_idx = ps.next_attempt_idx
        ps.next_attempt_idx += 1

        fut = pool.submit(
            self.solver.run_attempt,
            problem_id=ps.id_value,
            problem_text=ps.problem_text,
            attempt_index=attempt_idx,
        )
        inflight[fut] = SchedulerTaskInfo(kind="attempt", problem_idx=problem_idx, attempt_idx=attempt_idx)
        return True

    def _fill_with_attempts(
        self,
        *,
        pool: ThreadPoolExecutor,
        inflight: Dict[Future, SchedulerTaskInfo],
    ) -> None:
        """
        Fill remaining global capacity with attempt tasks, in problem order, exhausting each
        problem's attempts before moving on.
        """
        if not self._can_submit_more_tasks_globally(inflight):
            return

        for pidx, ps in enumerate(self.problems):
            if not self._can_submit_more_tasks_globally(inflight):
                break
            while self._can_submit_more_tasks_globally(inflight) and self._has_more_attempts(ps):
                self._submit_attempt(pool=pool, inflight=inflight, problem_idx=pidx, ps=ps)

    # ------------------------------------------------------------
    # Attempt completion -> schedule truth-check per attempt
    # ------------------------------------------------------------
    def _process_attempt_future_result(self, attempt_idx: int, done_fut: Future) -> dict:
        try:
            return done_fut.result()
        except Exception as exc:
            exc_text = f"future_exception:{type(exc).__name__} msg={exc}"
            print(f"[warn] {exc_text}")
            return {
                "Problem ID": "",
                "Attempt": attempt_idx,
                "Response Length": 0,
                "Python Calls": 0,
                "Python Errors": 1,
                "Entropy": float("inf"),
                "Answer": None,
                "Trace": {
                    "prompt_token_ids_initial": [],
                    "prompt_text_initial": "",
                    "turns": [],
                    "full_completion_token_ids": [],
                    "full_conversation_token_ids": [],
                },
                "Termination Reason": exc_text,
                "Tool Calls": [],
                "Attempt Started TS": "",
                "Attempt Finished TS": datetime.now(timezone.utc).isoformat(),
                "Attempt Elapsed MS": 0,
            }

    @staticmethod
    def _find_finished_attempt_record_in_problem(ps: ProblemState, attempt_idx: int) -> Optional[dict]:
        for attempt_record in ps.finished_attempts:
            if attempt_record["Attempt"] == attempt_idx:
                return attempt_record
        return None

    def _handle_attempt_done(
        self,
        *,
        done_fut: Future,
        pool: ThreadPoolExecutor,
        inflight: Dict[Future, SchedulerTaskInfo],
        ps: ProblemState,
        problem_idx: int,
        attempt_idx: int,
    ) -> None:
        attempt_record = self._process_attempt_future_result(attempt_idx, done_fut)
        ps.finished_attempts.append(attempt_record)

        self.logger.log_attempt(
            {
                "id": ps.id_value,
                "attempt": attempt_record.get("Attempt", attempt_idx),
                "attempt_answer": attempt_record.get("Answer", None),
                "entropy": attempt_record.get("Entropy", None),
                "response_length": attempt_record.get("Response Length", None),
                "python_calls": attempt_record.get("Python Calls", None),
                "python_errors": attempt_record.get("Python Errors", None),
                "termination_reason": attempt_record.get("Termination Reason", "unknown"),
                "attempt_started_ts": attempt_record.get("Attempt Started TS"),
                "attempt_finished_ts": attempt_record.get("Attempt Finished TS"),
                "attempt_elapsed_ms": attempt_record.get("Attempt Elapsed MS"),
                "trace": attempt_record.get("Trace", {}),
                "tool_calls": attempt_record.get("Tool Calls", []),
            }
        )
        # schedule truth-check for every attempt. The capacity was just freed by the finished attempt
        tfut = pool.submit(self._truth_check_task, problem_idx, attempt_idx)
        inflight[tfut] = SchedulerTaskInfo(kind="truth_check", problem_idx=problem_idx, attempt_idx=attempt_idx)

    # ------------------------------------------------------------
    # Truth-check task + completion
    # ------------------------------------------------------------
    def _truth_check_task(self, problem_idx: int, attempt_idx: int) -> Dict[str, Any]:
        """
        Truth-check a single attempt. Always returns agent_call (possibly skipped).
        """
        ps = self.problems[problem_idx]
        finished_attempt = self._find_finished_attempt_record_in_problem(ps, attempt_idx) or {}
        pred = str(finished_attempt.get("Answer") or "").strip()

        if not pred:
            return {
                "problem_idx": problem_idx,
                "attempt_idx": attempt_idx,
                "skipped": True,
                "skipped_reason": "no_prediction",
                "agent_call": {
                    "raw_output": "",
                    "result": {"equivalent": False, "confidence": 0.0, "reason": "no_prediction"},
                    "termination_reason": "skipped",
                },
            }

        agent_call = self.equivalence_agent.check_vs_truth(
            problem_text=ps.problem_text,
            truth_answer_text=ps.true_answer_text,
            candidate_answer_text=pred,
        )
        return {"problem_idx": problem_idx, "attempt_idx": attempt_idx, "skipped": False, "agent_call": agent_call}

    def _handle_truth_check_done(self, *, done_fut: Future, ps: ProblemState, attempt_idx: int) -> None:
        try:
            payload = done_fut.result()
        except Exception as exc:
            payload = {
                "skipped": False,
                "agent_call": {
                    "raw_output": "",
                    "result": {"equivalent": False, "confidence": 0.0, "reason": f"truth_future_exception:{type(exc).__name__}"},
                    "termination_reason": f"truth_future_exception:{type(exc).__name__} msg={exc}",
                },
            }

        agent_call = payload.get("agent_call", {}) or {}

        # Log raw truth-check agent output for THIS attempt
        self.logger.log_agent_call(
            kind="truth_check",
            payload={
                "problem_id": ps.id_value,
                "attempt": attempt_idx,
                "agent_call": agent_call,
            },
        )

        skipped = bool(payload.get("skipped", False))
        if skipped:
            ps.truth_results[attempt_idx] = {
                "attempt_idx": attempt_idx,
                "skipped": True,
                "skipped_reason": payload.get("skipped_reason", "unkown_reason"),
                "is_correct": None,
                "parsed_result": agent_call.get("result", {}),
                "termination_reason": agent_call.get("termination_reason", ""),
            }
        else:
            is_correct = bool(agent_call.get("result", {}).get("equivalent", False))
            ps.truth_results[attempt_idx] = {
                "attempt_idx": attempt_idx,
                "skipped": False,
                "is_correct": is_correct,
                "parsed_result": agent_call.get("result", {}),
                "termination_reason": agent_call.get("termination_reason", ""),
            }

        ps.truth_done.add(attempt_idx)

    # ------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------
    def _ready_to_finalize_problem(self, ps: ProblemState) -> bool:
        if ps.finalized:
            return False
        if len(ps.finished_attempts) < ps.total_attempts:
            return False
        # all attempts must have truth_done recorded (either real or skipped)
        return all(i in ps.truth_done for i in range(ps.total_attempts))


    def _finalize_problem_and_log_solution(self, ps: ProblemState) -> Dict[str, Any]:
        ps.solve_finished_ts = datetime.now(timezone.utc).isoformat()
        ps.solve_elapsed_ms = int(
            (datetime.fromisoformat(ps.solve_finished_ts) - datetime.fromisoformat(ps.solve_started_ts)).total_seconds() * 1000
        )

        attempts_total = len(ps.finished_attempts)
        attempts_with_answer = sum(
            1 for r in ps.finished_attempts if (r.get("Answer") is not None and str(r.get("Answer")).strip() != "")
        )

        self.logger.log_solution_row(
            id_value=ps.id_value,
            true_answer_text=ps.true_answer_text,
            solve_started_ts=ps.solve_started_ts,
            solve_finished_ts=ps.solve_finished_ts,
            solve_elapsed_ms=ps.solve_elapsed_ms,
            attempts_total=attempts_total,
            attempts_with_answer=attempts_with_answer,
            checker_summary={
                "per_attempt_truth": {
                    str(attempt_idx): {
                        "is_correct": call_result.get("is_correct", None),
                        "skipped": call_result.get("skipped", False),
                        "skipped_reason": call_result.get("skipped_reason", ""),
                        "parsed_result": call_result.get("parsed_result", {}),
                        "termination_reason": call_result.get("termination_reason", ""),
                    }
                    for attempt_idx, call_result in ps.truth_results.items()
                },
            },
        )
        self.logger.log_event("problem_end", {"id": ps.id_value, "msg": f"Problem id: {ps.id_value}"})
        ps.finalized = True
        return {"id": ps.id_value, "solve_elapsed_ms": ps.solve_elapsed_ms}

# ============================================================
# CLI
# ============================================================
def parse_args() -> CFG:
    p = argparse.ArgumentParser(
        description="AIMO3 solver with sequential per-problem attempts + async equivalence/correctness tasks (text answers)."
    )

    # Paths
    p.add_argument("--reference-path", default=CFG.reference_path,
                   help="Path to reference.csv containing columns: id, problem, answer (answer may be text).")
    p.add_argument("--log-dir", default=CFG.log_dir,
                   help="Directory to write logs: attempts.jsonl, solutions.csv, events.jsonl, submission.csv.")
    p.add_argument("--attempts-log", dest="attempts_filename", default=CFG.attempts_filename,
                   help="Filename (within --log-dir) for per-attempt JSONL logs.")
    p.add_argument("--solutions-log", dest="solutions_filename", default=CFG.solutions_filename,
                   help="Filename (within --log-dir) for per-problem CSV summary logs.")
    p.add_argument("--submission-out", dest="submission_filename", default=CFG.submission_filename,
                   help="Filename (within --log-dir) for the final submission CSV (id, answer_text).")

    # Model server
    p.add_argument("--served-model-name", default=CFG.served_model_name,
                   help="OpenAI-compatible model name exposed by the vLLM server.")
    p.add_argument("--model-path", default=CFG.model_path,
                   help="Local filesystem path to the HF/vLLM model directory.")
    p.add_argument("--port", type=int, default=CFG.port,
                   help="Port for the local vLLM OpenAI server.")
    p.add_argument("--api-key", default=CFG.api_key,
                   help="API key string used by the OpenAI client (for local server can be any value).")
    p.add_argument("--kv-cache-dtype", default=CFG.kv_cache_dtype,
                   help="vLLM KV cache dtype (e.g., fp8_e4m3).")
    p.add_argument("--dtype", default=CFG.dtype,
                   help="vLLM model dtype (e.g., auto, float16, bfloat16).")
    p.add_argument("--gpu-memory-utilization", type=float, default=CFG.gpu_memory_utilization,
                   help="Fraction of GPU memory vLLM is allowed to use (0-1).")

    # Timeouts
    p.add_argument("--server-startup-timeout-seconds", dest="server_timeout", type=int, default=CFG.server_timeout,
                   help="How long to wait (seconds) for vLLM server to become ready.")
    p.add_argument("--openai-client-timeout-seconds", dest="session_timeout", type=int, default=CFG.session_timeout,
                   help="Timeout (seconds) for a single OpenAI client request (stream).")
    p.add_argument("--jupyter-exec-timeout-seconds", dest="jupyter_timeout", type=int, default=CFG.jupyter_timeout,
                   help="Timeout (seconds) for a single python tool execution in a sandbox kernel.")
    p.add_argument("--sandbox-acquire-timeout-seconds", dest="sandbox_timeout", type=int, default=CFG.sandbox_timeout,
                   help="Timeout (seconds) to acquire a sandbox kernel from the pool.")
    p.add_argument("--attempt-timeout-seconds", dest="attempt_timeout_seconds", type=int, default=CFG.attempt_timeout_seconds,
                   help="Per-attempt wall-clock timeout (seconds). Attempts terminate if exceeded.")
    p.add_argument("--checker-timeout-seconds", dest="checker_timeout_seconds", type=int, default=CFG.checker_timeout_seconds,
                   help="Per-checker-call wall-clock timeout (seconds).")
    p.add_argument("--checker-max-turns", dest="checker_max_turns", type=int, default=CFG.checker_max_turns,
                   help="Max Harmony turns for equivalence checker agents (tool-enabled).")

    # Decoding/sampling
    p.add_argument("--stream-interval", type=int, default=CFG.stream_interval,
                   help="vLLM stream interval (tokens) for partial outputs.")
    p.add_argument("--context-tokens", type=int, default=CFG.context_tokens,
                   help="Maximum model context length (tokens).")
    p.add_argument("--buffer-tokens", type=int, default=CFG.buffer_tokens,
                   help="Minimum safety buffer of tokens; stop if remaining context drops below this.")
    p.add_argument("--boxed-scan-window-tokens", dest="search_tokens", type=int, default=CFG.search_tokens,
                   help="How many recent streamed text chunks to scan for \\boxed{...}.")
    p.add_argument("--top-logprobs", type=int, default=CFG.top_logprobs,
                   help="Number of top logprobs per token to request for entropy estimation.")
    p.add_argument("--max-num-seqs", dest="batch_size", type=int, default=CFG.batch_size,
                   help="vLLM --max-num-seqs (max concurrent sequences).")
    p.add_argument("--attempts-per-problem", dest="attempts_per_problem", type=int, default=CFG.attempts_per_problem,
                   help="Number of independent attempts (ensembling runs) per problem.")
    p.add_argument("--max-turns", dest="turns", type=int, default=CFG.turns,
                   help="Maximum Harmony turns per attempt.")
    p.add_argument("--seed", type=int, default=CFG.seed,
                   help="Random seed for reproducibility.")
    p.add_argument("--temperature", type=float, default=CFG.temperature,
                   help="Sampling temperature for completions.")
    p.add_argument("--min-p", type=float, default=CFG.min_p,
                   help="min_p nucleus-like sampling parameter passed via extra_body.")

    # Parallelism knobs
    p.add_argument(
        "--agent-parallelism",
        dest="agent_parallelism",
        type=int,
        default=CFG.agent_parallelism,
        help="Global cap on concurrent tasks: attempts + equivalence checks + truth checks.",
    )
    p.add_argument(
        "--jupyter-kernels",
        dest="kernel_workers",
        type=int,
        default=CFG.kernel_workers,
        help="Number of persistent Jupyter kernels to pre-initialize. Must be >= agent-parallelism.",
    )
    p.add_argument(
        "--preload-workers",
        dest="preload_workers",
        type=int,
        default=CFG.preload_workers,
        help="Thread count used to page-cache model weights from disk before starting vLLM.",
    )

    # Optional cap
    p.add_argument("--max-problems", type=int, default=CFG.max_problems,
                   help="Optional cap on number of problems to process (<=0 means all rows in reference.csv).")

    # Logging verbosity
    p.add_argument("--verbose", action="store_true", default=CFG.verbose,
                   help="Enable verbose console logging.")

    args = p.parse_args()

    cfg = CFG(
        reference_path=args.reference_path,
        log_dir=args.log_dir,
        attempts_filename=args.attempts_filename,
        solutions_filename=args.solutions_filename,
        submission_filename=args.submission_filename,

        served_model_name=args.served_model_name,
        model_path=args.model_path,
        port=args.port,
        api_key=args.api_key,
        kv_cache_dtype=args.kv_cache_dtype,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,

        server_timeout=args.server_timeout,
        session_timeout=args.session_timeout,
        jupyter_timeout=args.jupyter_timeout,
        sandbox_timeout=args.sandbox_timeout,
        attempt_timeout_seconds=args.attempt_timeout_seconds,
        checker_timeout_seconds=args.checker_timeout_seconds,
        checker_max_turns=args.checker_max_turns,

        stream_interval=args.stream_interval,
        context_tokens=args.context_tokens,
        buffer_tokens=args.buffer_tokens,
        search_tokens=args.search_tokens,
        top_logprobs=args.top_logprobs,
        batch_size=args.batch_size,
        attempts_per_problem=args.attempts_per_problem,
        turns=args.turns,
        seed=args.seed,
        temperature=args.temperature,
        min_p=args.min_p,

        agent_parallelism=args.agent_parallelism,
        kernel_workers=args.kernel_workers,
        preload_workers=args.preload_workers,

        max_problems=args.max_problems,

        verbose=args.verbose,
    )
    return cfg


# ============================================================
# Main
# ============================================================
def main():
    cfg = parse_args()
    validate_cfg(cfg)

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    attempts_path = str(log_dir / cfg.attempts_filename)
    solutions_path = str(log_dir / cfg.solutions_filename)
    submission_path = str(log_dir / cfg.submission_filename)

    reference_df = pl.read_csv(cfg.reference_path)
    if "id" not in reference_df.columns or "problem" not in reference_df.columns:
        raise ValueError("reference.csv must contain columns: id, problem, answer")

    if cfg.max_problems and cfg.max_problems > 0:
        reference_df = reference_df.head(cfg.max_problems)

    logger = RunLogger(attempts_path, solutions_path, log_dir=str(log_dir), verbose=cfg.verbose)

    solver = AIMO3Solver(cfg)

    print("Base URL:", solver.base_url)
    poll_result = solver.server_process.poll()
    print("Server poll:", poll_result)
    assert poll_result is None
    print("Models list:", solver.client.models.list())

    # Build problem states (truth is text)
    problems: List[ProblemState] = []
    for pid, ptxt, true_ans_text in iter_reference(reference_df):
        problems.append(
            ProblemState(
                id_value=pid,
                problem_text=str(ptxt),
                true_answer_text=str(true_ans_text),
                total_attempts=cfg.attempts_per_problem,
            )
        )

    scheduler = ProblemScheduler(cfg=cfg, solver=solver, logger=logger, problems=problems)

    gc.disable()
    try:
        submission_rows = scheduler.run_all()
    finally:
        gc.enable()
        gc.collect()
        solver.close()

    # Deterministic submission order
    order = reference_df["id"].to_list()
    pred_by_id = {r["id"]: r["answer"] for r in submission_rows}
    submission_out = [{"id": pid, "answer": str(pred_by_id.get(pid, ""))} for pid in order]

    submission_df = pl.DataFrame(submission_out)
    submission_df.write_csv(submission_path)
    print(f"\nWrote submission to: {submission_path}")
    print(submission_df.head(5))

if __name__ == "__main__":
    main()
