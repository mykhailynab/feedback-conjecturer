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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List, Tuple, Iterable

import pandas as pd
import polars as pl
from openai import OpenAI
from transformers import set_seed
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


# -----------------------------
# Config (args w/ defaults)
# -----------------------------
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
    attempt_timeout_seconds: int = 300

    # Decoding / sampling / batching
    stream_interval: int = 200
    context_tokens: int = 65536
    buffer_tokens: int = 512
    search_tokens: int = 32
    top_logprobs: int = 5
    batch_size: int = 256
    attempts_per_problem: int = 16
    turns: int = 128
    seed: int = 42
    temperature: float = 0.5
    min_p: float = 0.02

    # Parallelism knobs (separated)
    solver_parallelism: int = 8          # max concurrent in-flight attempts / LLM streams globally
    kernel_workers: int = 32             # number of persistent Jupyter kernels
    attempt_threads: int = 32            # host threads to execute attempts (should be >= solver_parallelism)
    preload_workers: int = 32            # threads used for model weight page-caching

    # Optional cap on problems processed (0 means all)
    max_problems: int = 0

    # Logging
    verbose: bool = True

    # --- Prompts (DO NOT CHANGE) ---
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
        "The final answer must be a non-negative integer between 0 and 99999.\n"
        "Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n"
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
        "Explain what you\"re computing and why before running code."
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


# -----------------------------
# Logging
# -----------------------------
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

    def _init_solutions_csv(self):
        if os.path.exists(self.solutions_path):
            return

        header = [
            "id",
            "pred_answer",
            "true_answer",
            "is_correct",
            "selected_attempt",
            "selected_entropy",
            "solve_started_ts",
            "solve_finished_ts",
            "solve_elapsed_ms",
            "attempts_total",
            "attempts_with_answer",
            "vote_summary",
        ]
        with open(self.solutions_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    def log_event(self, event_type: str, payload: dict):
        rec = {"ts": self._now_iso(), "event": event_type, **payload}
        with self._lock:
            self._append_jsonl(self.events_path, [rec])

        if self.verbose:
            msg = payload.get("msg") or ""
            print(f"[LOG:{event_type}] {msg}".rstrip())

    def log_attempts(self, attempt_records: list[dict]):
        ts = self._now_iso()
        enriched = []
        for r in attempt_records:
            summary = (
                f"id={r.get('id')} attempt={r.get('attempt')} "
                f"ans={r.get('attempt_answer')} final={r.get('pred_final_answer')} "
                f"ent={r.get('entropy')} "
                f"py={r.get('python_calls')}/{r.get('python_errors')} "
                f"termination={r.get('termination_reason')} "
                f"elapsed_ms={r.get('attempt_elapsed_ms')}"
            )
            rr = dict(r)
            rr["ts"] = ts
            rr["summary"] = summary
            enriched.append(rr)

        with self._lock:
            self._append_jsonl(self.attempts_path, enriched)

        if self.verbose and enriched:
            print(f"[LOG:attempts] attempts={len(enriched)}")
            for e in enriched:
                summary = e.get("summary")
                print(f"[LOG:attempts] Attermpt summary: {summary}")

    def log_solution_row(
        self,
        id_value: str,
        pred_answer: int,
        true_answer: Optional[int],
        is_correct: Optional[bool],
        selected_attempt: Optional[int],
        selected_entropy: Optional[float],
        solve_started_ts: Optional[str] = None,
        solve_finished_ts: Optional[str] = None,
        solve_elapsed_ms: Optional[int] = None,
        attempts_total: Optional[int] = None,
        attempts_with_answer: Optional[int] = None,
        vote_summary: Optional[dict] = None,
    ):
        row = [
            id_value,
            pred_answer,
            true_answer,
            is_correct,
            selected_attempt,
            selected_entropy,
            solve_started_ts,
            solve_finished_ts,
            solve_elapsed_ms,
            attempts_total,
            attempts_with_answer,
            json.dumps(vote_summary or {}, ensure_ascii=False),
        ]

        with self._lock:
            with open(self.solutions_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

        if self.verbose:
            print(
                f"[LOG:solution] id={id_value} pred={pred_answer} "
                f"true={true_answer} correct={is_correct} "
                f"selected_attempt={selected_attempt} ent={selected_entropy}"
            )


# -----------------------------
# Template + sandbox + tool
# -----------------------------
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


# -----------------------------
# Solver
# -----------------------------
class AIMO3Solver:
    """
    Owns:
      - Harmony encoding/template
      - vLLM server process + OpenAI client
      - persistent sandbox_pool (Jupyter kernels)

    IMPORTANT:
      - No early stopping.
      - No per-problem deadline/budget logic.
      - _run_attempt executes exactly one attempt (for one problem, one attempt_index),
        using a per-attempt timeout.
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

    def _scan_for_answer(self, text: str) -> int | None:
        pattern = r"\\boxed\s*\{\s*([0-9,]+)\s*\}"
        matches = re.findall(pattern, text)
        if matches:
            try:
                clean_value = matches[-1].replace(",", "")
                value = int(clean_value)
                if 0 <= value <= 99999:
                    return value
            except ValueError:
                pass

        pattern = r"final\s+answer\s+is\s*([0-9,]+)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                clean_value = matches[-1].replace(",", "")
                value = int(clean_value)
                if 0 <= value <= 99999:
                    return value
            except ValueError:
                pass

        return None

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

    # ---- one attempt (no global gate here; scheduling handles parallelism) ----
    def run_attempt(
        self,
        *,
        problem_id: str,
        problem_text: str,
        attempt_index: int,
    ) -> dict:
        """
        Execute ONE attempt for ONE problem, with per-attempt timeout.

        Returns a dict suitable to be stored in attempts.jsonl, but does NOT add:
          - id/status/reject_reason/pred_final_answer (those are produced at ensemble time)
        This function always returns a record, including failures/timeouts.
        """
        attempt_started_ts = datetime.now(timezone.utc).isoformat()
        t0 = time.time()

        # ALWAYS define
        turns_compact: list[dict] = []
        full_completion_ids: list[int] = []
        tool_calls: list[dict] = []
        termination_reason = "unknown"
        python_calls = 0
        python_errors = 0
        total_tokens = 0
        final_answer = None
        logprobs_buffer = []
        sandbox = None
        prompt_token_ids_initial: list[int] = []
        conversation = None

        deadline = time.time() + float(self.cfg.attempt_timeout_seconds)
        local_tool = None
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

                        if "}" in new_text:
                            search_text = "".join(completion_text_parts[-self.cfg.search_tokens :])
                            answer = self._scan_for_answer(search_text)
                            if answer is not None:
                                final_answer = answer
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

                if final_answer is not None:
                    break

                if not token_buffer:
                    if termination_reason == "unknown":
                        termination_reason = "no_tokens"
                    break

                new_messages = encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
                conversation.messages.extend(new_messages)
                last_message = new_messages[-1]

                if last_message.channel == "final":
                    answer_text = last_message.content[0].text
                    final_answer = self._scan_for_answer(answer_text)
                    if final_answer is not None:
                        termination_reason = "final_channel_answer"
                    else:
                        termination_reason = "final_channel_no_answer"
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
            "Attempt": attempt_index + 1,
            "Response Length": total_tokens,
            "Python Calls": python_calls,
            "Python Errors": python_errors,
            "Entropy": mean_entropy,
            "Answer": final_answer,
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


# -----------------------------
# Ensembling (after all attempts)
# -----------------------------
def ensemble_attempts(attempts: List[dict]) -> Tuple[int, Optional[int], pd.DataFrame]:
    """
    Same scoring as before: weight = 1/entropy, pick max total weight.
    Representative selected attempt: lowest entropy among attempts with chosen answer.
    """
    answer_weights: Dict[int, float] = defaultdict(float)
    answer_votes: Dict[int, int] = defaultdict(int)

    for r in attempts:
        ans = r.get("Answer")
        ent = r.get("Entropy", float("inf"))
        if ans is None:
            continue
        weight = 1.0 / max(float(ent), 1e-9)
        answer_weights[int(ans)] += weight
        answer_votes[int(ans)] += 1

    scored = [{"answer": a, "votes": answer_votes[a], "score": w} for a, w in answer_weights.items()]
    scored.sort(key=lambda x: x["score"], reverse=True)
    vote_df = pd.DataFrame(scored) if scored else pd.DataFrame(columns=["answer", "votes", "score"])

    if not scored:
        return 0, None, vote_df

    final_answer = int(scored[0]["answer"])
    candidates = [
        (i, float(r.get("Entropy", float("inf"))))
        for i, r in enumerate(attempts)
        if r.get("Answer") == final_answer
    ]
    selected_idx = min(candidates, key=lambda x: x[1])[0] if candidates else None
    return final_answer, selected_idx, vote_df


# -----------------------------
# Problem state + global scheduler
# -----------------------------
@dataclass
class ProblemState:
    id_value: str
    problem_text: str
    true_answer: Optional[int]
    total_attempts: int

    next_attempt_idx: int = 0
    attempts: List[dict] = None

    solve_started_ts: str = ""
    solve_finished_ts: str = ""
    solve_elapsed_ms: int = 0

    def __post_init__(self):
        if self.attempts is None:
            self.attempts = []


def validate_cfg(cfg: CFG):
    errs = []

    if cfg.attempts_per_problem <= 0:
        errs.append("attempts_per_problem must be >= 1")

    if cfg.solver_parallelism <= 0:
        errs.append("solver_parallelism must be >= 1")

    if cfg.kernel_workers <= 0:
        errs.append("kernel_workers must be >= 1")
    if cfg.kernel_workers < cfg.solver_parallelism:
        errs.append("kernel_workers must be >= solver_parallelism (to cover concurrent attempts)")

    if cfg.attempt_threads <= 0:
        errs.append("attempt_threads must be >= 1")
    if cfg.attempt_threads < cfg.solver_parallelism:
        errs.append("attempt_threads must be >= solver_parallelism (host threads to run concurrent attempts)")

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

    if errs:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errs))


def iter_reference(reference_df: pl.DataFrame) -> Iterable[Tuple[str, str, Optional[int]]]:
    for row in reference_df.iter_rows(named=True):
        pid = str(row["id"])
        ptxt = row["problem"]
        true_answer = None
        if "answer" in row and row["answer"] is not None:
            try:
                true_answer = int(row["answer"])
            except Exception:
                true_answer = None
        yield pid, ptxt, true_answer


def run_global_scheduler(
    *,
    solver: AIMO3Solver,
    logger: RunLogger,
    problems: List[ProblemState],
) -> List[Dict[str, Any]]:
    """
    Global attempt scheduler:
      - Maintain up to cfg.solver_parallelism in-flight attempts at once.
      - Attempts are per-problem sequential (attempt_index increases), but interleaved across problems.
      - When all attempts for a problem complete, ensemble + log + record submission row.
    """
    cfg = solver.cfg
    submission_rows: List[Dict[str, Any]] = []
    lock = threading.Lock()

    # Track per-problem solve start
    for ps in problems:
        ps.solve_started_ts = datetime.now(timezone.utc).isoformat()
        logger.log_event("problem_start", {"id": ps.id_value})

    # In-flight futures -> (problem_index, attempt_index)
    inflight: Dict[Future, Tuple[int, int]] = {}

    def submit_next_attempt(pool: ThreadPoolExecutor, problem_idx: int) -> bool:
        ps = problems[problem_idx]
        if ps.next_attempt_idx >= ps.total_attempts:
            return False
        attempt_idx = ps.next_attempt_idx
        ps.next_attempt_idx += 1

        fut = pool.submit(
            solver.run_attempt,
            problem_id=ps.id_value,
            problem_text=str(ps.problem_text),
            attempt_index=attempt_idx,
        )
        inflight[fut] = (problem_idx, attempt_idx)
        return True

    def finalize_problem(problem_idx: int):
        ps = problems[problem_idx]
        ps.solve_finished_ts = datetime.now(timezone.utc).isoformat()
        ps.solve_elapsed_ms = int(
            (datetime.fromisoformat(ps.solve_finished_ts) - datetime.fromisoformat(ps.solve_started_ts)).total_seconds()
            * 1000
        )

        # Ensemble AFTER all attempts
        final_answer, selected_idx, vote_df = ensemble_attempts(ps.attempts)

        # Build attempt records for logging with final answer and selection info
        attempt_records = []
        for i, r in enumerate(ps.attempts):
            ans = r.get("Answer", None)
            termination_reason = r.get("Termination Reason", "unknown")

            if selected_idx is not None and i == selected_idx:
                status = "selected"
                reject_reason = ""
            else:
                status = "rejected"
                if ans is None:
                    reject_reason = f"no_answer:{termination_reason}"
                elif int(ans) != int(final_answer):
                    reject_reason = "different_answer"
                else:
                    reject_reason = "same_answer_not_selected"

            attempt_records.append(
                {
                    "id": ps.id_value,
                    "attempt": r.get("Attempt", i + 1),
                    "status": status,
                    "reject_reason": reject_reason,
                    "pred_final_answer": int(final_answer),
                    "attempt_answer": ans,
                    "entropy": r.get("Entropy", None),
                    "response_length": r.get("Response Length", None),
                    "python_calls": r.get("Python Calls", None),
                    "python_errors": r.get("Python Errors", None),
                    "termination_reason": termination_reason,
                    "attempt_started_ts": r.get("Attempt Started TS"),
                    "attempt_finished_ts": r.get("Attempt Finished TS"),
                    "attempt_elapsed_ms": r.get("Attempt Elapsed MS"),
                    "trace": r.get("Trace", {}),
                    "tool_calls": r.get("Tool Calls", []),
                }
            )

        attempts_total = len(attempt_records)
        attempts_with_answer = sum(1 for x in attempt_records if x["attempt_answer"] is not None)

        vote_summary = {}
        try:
            vote_summary = {"top": vote_df.head(5).to_dict(orient="records")}
        except Exception:
            vote_summary = {}

        # correctness
        is_correct = None
        if ps.true_answer is not None:
            is_correct = (int(final_answer) == int(ps.true_answer))

        selected_entropy = None
        selected_attempt_number = None
        if selected_idx is not None and 0 <= selected_idx < len(ps.attempts):
            selected_entropy = ps.attempts[selected_idx].get("Entropy", None)
            selected_attempt_number = ps.attempts[selected_idx].get("Attempt", selected_idx + 1)

        logger.log_attempts(attempt_records)
        logger.log_solution_row(
            id_value=ps.id_value,
            pred_answer=int(final_answer),
            true_answer=ps.true_answer,
            is_correct=is_correct,
            selected_attempt=selected_attempt_number,
            selected_entropy=selected_entropy,
            solve_started_ts=ps.solve_started_ts,
            solve_finished_ts=ps.solve_finished_ts,
            solve_elapsed_ms=ps.solve_elapsed_ms,
            attempts_total=attempts_total,
            attempts_with_answer=attempts_with_answer,
            vote_summary=vote_summary,
        )

        logger.log_event(
            "problem_end",
            {"id": ps.id_value, "pred": int(final_answer), "true": ps.true_answer, "correct": is_correct},
        )

        submission_rows.append({"id": ps.id_value, "answer": int(final_answer)})

    # Scheduler: keep queue of problems with remaining attempts
    pending_problem_idxs = [i for i, ps in enumerate(problems) if ps.total_attempts > 0]
    unfinished = set(pending_problem_idxs)

    # We use attempt_threads for host-side execution
    with ThreadPoolExecutor(max_workers=cfg.attempt_threads) as pool:
        # Prime inflight up to solver_parallelism
        pi = 0
        while len(inflight) < cfg.solver_parallelism and pi < len(pending_problem_idxs):
            submit_next_attempt(pool, pending_problem_idxs[pi])
            pi += 1

        # If attempts_per_problem is small, this may leave capacity unused. Fill by cycling.
        while len(inflight) < cfg.solver_parallelism and unfinished:
            progressed = False
            for idx in list(unfinished):
                if len(inflight) >= cfg.solver_parallelism:
                    break
                # will not progress if attempts exhausted
                progressed |= submit_next_attempt(pool, idx)
            if not progressed:
                break

        # Event loop: as attempts finish, schedule more
        while inflight:
            done_futs = []
            for fut in as_completed(list(inflight.keys()), timeout=None):
                done_futs.append(fut)
                # Only handle one completion at a time to simplify fairness
                break

            for fut in done_futs:
                problem_idx, attempt_idx = inflight.pop(fut)
                ps = problems[problem_idx]

                try:
                    attempt_record = fut.result()
                except Exception as exc:
                    exc_text = f"future_exception:{type(exc).__name__} msg={exc}"
                    print(f'[warn] {exc_text}')
                    # Last-resort: synthesize an attempt record so we still "save all attempts"
                    attempt_record = {
                        "Attempt": attempt_idx + 1,
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

                ps.attempts.append(attempt_record)

                # If this problem finished all attempts, finalize it
                if len(ps.attempts) >= ps.total_attempts:
                    if problem_idx in unfinished:
                        unfinished.remove(problem_idx)
                    finalize_problem(problem_idx)

                # Fill available slots up to solver_parallelism by pulling next attempts
                while len(inflight) < cfg.solver_parallelism and unfinished:
                    progressed = False
                    # fairness: try to schedule one attempt from each unfinished problem in round-robin-ish order
                    for idx in list(unfinished):
                        if len(inflight) >= cfg.solver_parallelism:
                            break
                        if submit_next_attempt(pool, idx):
                            progressed = True
                    if not progressed:
                        break

    return submission_rows


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> CFG:
    p = argparse.ArgumentParser(description="AIMO3 solver with global attempt scheduler (no early stop, per-attempt timeout).")

    # Paths
    p.add_argument("--reference-path", default=CFG.reference_path,
                   help="Path to reference.csv containing columns: id, problem, answer.")
    p.add_argument("--log-dir", default=CFG.log_dir,
                   help="Directory to write logs: attempts.jsonl, solutions.csv, events.jsonl, submission.csv.")
    p.add_argument("--attempts-log", dest="attempts_filename", default=CFG.attempts_filename,
                   help="Filename (within --log-dir) for per-attempt JSONL logs.")
    p.add_argument("--solutions-log", dest="solutions_filename", default=CFG.solutions_filename,
                   help="Filename (within --log-dir) for per-problem CSV summary logs.")
    p.add_argument("--submission-out", dest="submission_filename", default=CFG.submission_filename,
                   help="Filename (within --log-dir) for the final submission CSV (id, answer).")

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
    p.add_argument("--solver-parallelism", dest="solver_parallelism", type=int, default=CFG.solver_parallelism,
                   help="Max number of concurrent in-flight attempts / LLM streams globally.")
    p.add_argument("--jupyter-kernels", dest="kernel_workers", type=int, default=CFG.kernel_workers,
                   help="Number of persistent Jupyter kernels to pre-initialize (tool sandboxes). Must be >= solver-parallelism.")
    p.add_argument("--attempt-threads", dest="attempt_threads", type=int, default=CFG.attempt_threads,
                   help="Host thread pool size for executing attempts (should be >= solver-parallelism).")
    p.add_argument("--preload-workers", dest="preload_workers", type=int, default=CFG.preload_workers,
                   help="Thread count used to page-cache model weights from disk before starting vLLM.")

    # Optional cap
    p.add_argument("--max-problems", type=int, default=CFG.max_problems,
                   help="Optional cap on number of problems to process (<=0 means all rows in reference.csv).")

    # Logging verbosity
    p.add_argument("--verbose", action="store_true", default=CFG.verbose,
                   help="Enable verbose console logging.")
    p.add_argument("--quiet", action="store_true", default=False,
                   help="Disable verbose console logging.")

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

        solver_parallelism=args.solver_parallelism,
        kernel_workers=args.kernel_workers,
        attempt_threads=args.attempt_threads,
        preload_workers=args.preload_workers,

        max_problems=args.max_problems,

        verbose=(False if args.quiet else args.verbose),
    )
    return cfg


def main():
    cfg = parse_args()
    validate_cfg(cfg)

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    attempts_path = str(log_dir / cfg.attempts_filename)
    solutions_path = str(log_dir / cfg.solutions_filename)
    submission_path = str(log_dir / cfg.submission_filename)

    set_seed(cfg.seed)

    reference_df = pl.read_csv(cfg.reference_path)
    if "id" not in reference_df.columns or "problem" not in reference_df.columns:
        raise ValueError("reference.csv must contain columns: id, problem, answer")

    if cfg.max_problems and cfg.max_problems > 0:
        reference_df = reference_df.head(cfg.max_problems)

    # ground truth map (optional)
    ref_ans_by_id: Dict[str, Any] = {}
    if "answer" in reference_df.columns:
        ref_ans_by_id = dict(zip(reference_df["id"].to_list(), reference_df["answer"].to_list()))

    logger = RunLogger(attempts_path, solutions_path, log_dir=str(log_dir), verbose=cfg.verbose)

    solver = AIMO3Solver(cfg)

    print("Base URL:", solver.base_url)
    poll_result = solver.server_process.poll()
    print("Server poll:", poll_result)
    assert poll_result is None
    print("Models list:", solver.client.models.list())  # should succeed

    # Build problem states
    problems: List[ProblemState] = []
    for pid, ptxt, true_ans in iter_reference(reference_df):
        if true_ans is None and pid in ref_ans_by_id:
            try:
                true_ans = int(ref_ans_by_id[pid])
            except Exception:
                true_ans = None
        problems.append(
            ProblemState(
                id_value=pid,
                problem_text=str(ptxt),
                true_answer=true_ans,
                total_attempts=cfg.attempts_per_problem,
            )
        )

    # Run global scheduler
    gc.disable()
    try:
        submission_rows = run_global_scheduler(solver=solver, logger=logger, problems=problems)
    finally:
        gc.enable()
        gc.collect()
        solver.close()

    # Deterministic submission order
    order = reference_df["id"].to_list()
    pred_by_id = {r["id"]: r["answer"] for r in submission_rows}
    submission_out = [{"id": pid, "answer": int(pred_by_id.get(pid, 0))} for pid in order]

    submission_df = pl.DataFrame(submission_out)
    submission_df.write_csv(submission_path)
    print(f"\nWrote submission to: {submission_path}")
    print(submission_df.head(5))


if __name__ == "__main__":
    main()