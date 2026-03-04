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
from typing import Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import polars as pl
from openai import OpenAI
from transformers import set_seed
from jupyter_client import KernelManager
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor

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

    # Timeouts / budgets
    high_problem_timeout: int = 900
    base_problem_timeout: int = 300
    notebook_limit: int = 17400
    server_timeout: int = 180
    session_timeout: int = 960
    jupyter_timeout: int = 6
    sandbox_timeout: int = 3

    # Decoding / sampling / batching
    stream_interval: int = 200
    context_tokens: int = 65536
    buffer_tokens: int = 512
    search_tokens: int = 32
    top_logprobs: int = 5
    batch_size: int = 256
    early_stop: int = 8
    attempts: int = 16
    workers: int = 32
    turns: int = 128
    seed: int = 42
    temperature: float = 0.5
    min_p: float = 0.02

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

    Additions:
      - More columns in solutions.csv (timing/budget/vote summary + counts)
      - Per-record timestamps, elapsed_ms, and a concise summary field
      - Optional verbose console printing
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
            "budget_seconds",
            "deadline_ts",
            "solve_started_ts",
            "solve_finished_ts",
            "solve_elapsed_ms",
            "attempts_total",
            "attempts_with_answer",
            "attempts_selected",
            "attempts_rejected",
            "finish_reason_counts",
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
                f"status={r.get('status')} ans={r.get('attempt_answer')} "
                f"final={r.get('pred_final_answer')} "
                f"ent={r.get('entropy')} "
                f"py={r.get('python_calls')}/{r.get('python_errors')} "
                f"finish={r.get('finish_reason')}"
            )
            rr = dict(r)
            rr["ts"] = ts
            rr["summary"] = summary
            enriched.append(rr)

        with self._lock:
            self._append_jsonl(self.attempts_path, enriched)

        if self.verbose and enriched:
            idv = enriched[0].get("id")
            sel = [x for x in enriched if x.get("status") == "selected"]
            sel_ans = sel[0].get("attempt_answer") if sel else None
            print(f"[LOG:attempts] id={idv} attempts={len(enriched)} selected_ans={sel_ans}")

    def log_solution_row(
        self,
        id_value: str,
        pred_answer: int,
        true_answer: Optional[int],
        is_correct: Optional[bool],
        selected_attempt: Optional[int],
        selected_entropy: Optional[float],
        budget_seconds: Optional[float] = None,
        deadline_ts: Optional[float] = None,
        solve_started_ts: Optional[str] = None,
        solve_finished_ts: Optional[str] = None,
        solve_elapsed_ms: Optional[int] = None,
        attempts_total: Optional[int] = None,
        attempts_with_answer: Optional[int] = None,
        attempts_selected: Optional[int] = None,
        attempts_rejected: Optional[int] = None,
        finish_reason_counts: Optional[dict] = None,
        vote_summary: Optional[dict] = None,
    ):
        row = [
            id_value,
            pred_answer,
            true_answer,
            is_correct,
            selected_attempt,
            selected_entropy,
            budget_seconds,
            deadline_ts,
            solve_started_ts,
            solve_finished_ts,
            solve_elapsed_ms,
            attempts_total,
            attempts_with_answer,
            attempts_selected,
            attempts_rejected,
            json.dumps(finish_reason_counts or {}, ensure_ascii=False),
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

        self.notebook_start_time = time.time()
        self.problems_remaining = 50

    def _preload_model_weights(self) -> None:
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

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
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
        print(f"Initializing {self.cfg.workers} persistent Jupyter kernels...")
        start_time = time.time()

        self.sandbox_pool = queue.Queue()

        def _create_sandbox():
            return AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            futures = [executor.submit(_create_sandbox) for _ in range(self.cfg.workers)]
            for future in as_completed(futures):
                self.sandbox_pool.put(future.result())

        elapsed = time.time() - start_time
        print(f"Kernels initialized in {elapsed:.2f} seconds.\n")

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
            for token_str, log_prob in top_logprobs_dict.items():
                prob = math.exp(log_prob)
                if prob > 0:
                    token_entropy -= prob * math.log2(prob)

            total_entropy += token_entropy
            token_count += 1

        if token_count == 0:
            return float("inf")

        return total_entropy / token_count

    def _process_attempt(
        self,
        problem: str,
        system_prompt: str,
        attempt_index: int,
        stop_event: threading.Event,
        deadline: float,
    ) -> dict:
        # Always define these so we can safely return them
        turn_traces = []
        full_completion_ids: list[int] = []
        tool_calls: list[dict] = []
        finish_reason = "unknown"
        python_calls = 0
        python_errors = 0
        total_tokens = 0
        final_answer = None
        logprobs_buffer = []
        sandbox = None

        if stop_event.is_set():
            finish_reason = "skipped_stop_event"
            return {
                "Attempt": attempt_index + 1,
                "Answer": None,
                "Trace": {"turns": [], "full_completion_token_ids": []},
                "Finish Reason": finish_reason,
                "Python Calls": 0,
                "Python Errors": 0,
                "Response Length": 0,
                "Entropy": float("inf"),
                "Tool Calls": [],
            }

        if time.time() > deadline:
            finish_reason = "skipped_deadline"
            return {
                "Attempt": attempt_index + 1,
                "Answer": None,
                "Trace": {"turns": [], "full_completion_token_ids": []},
                "Finish Reason": finish_reason,
                "Python Calls": 0,
                "Python Errors": 0,
                "Response Length": 0,
                "Entropy": float("inf"),
                "Tool Calls": [],
            }

        local_tool = None
        attempt_seed = int((self.cfg.seed + attempt_index) ** 2)

        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)
            local_tool = AIMO3Tool(
                local_jupyter_timeout=self.cfg.jupyter_timeout,
                tool_prompt=self.cfg.tool_prompt,
                sandbox=sandbox,
            )

            encoding = self.encoding
            messages = self.template.apply_chat_template(system_prompt, problem, local_tool.tool_config)
            conversation = Conversation.from_messages(messages)

            for _turn in range(self.cfg.turns):
                if stop_event.is_set():
                    finish_reason = "stopped_by_early_stop"
                    break
                if time.time() > deadline:
                    finish_reason = "deadline_exceeded"
                    break

                prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                prompt_ids_this_turn = list(prompt_ids)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)

                if max_tokens < self.cfg.buffer_tokens:
                    finish_reason = "context_exhausted"
                    break

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
                try:
                    for chunk in stream:
                        if stop_event.is_set():
                            finish_reason = "stopped_by_early_stop"
                            break
                        if time.time() > deadline:
                            finish_reason = "deadline_exceeded"
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
                                finish_reason = "boxed_detected_in_stream"
                                break
                finally:
                    stream.close()

                prompt_text = self._decode_ids(prompt_ids_this_turn)
                completion_text = self._decode_ids(token_buffer)
                turn_traces.append(
                    {
                        "turn": _turn,
                        "prompt_token_ids": prompt_ids_this_turn,
                        "completion_token_ids": token_buffer,
                        "prompt_text": prompt_text,
                        "completion_text": completion_text,
                    }
                )

                if final_answer is not None:
                    break

                if not token_buffer:
                    if finish_reason == "unknown":
                        finish_reason = "no_tokens"
                    break

                new_messages = encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
                conversation.messages.extend(new_messages)
                last_message = new_messages[-1]

                if last_message.channel == "final":
                    answer_text = last_message.content[0].text
                    final_answer = self._scan_for_answer(answer_text)
                    if final_answer is not None:
                        finish_reason = "final_channel_answer"
                    else:
                        finish_reason = "final_channel_no_answer"
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

            if finish_reason == "unknown":
                finish_reason = "max_turns_or_no_answer"

        except Exception as exc:
            python_errors += 1
            cause = repr(getattr(exc, "__cause__", None))
            context = repr(getattr(exc, "__context__", None))
            finish_reason = f"exception:{type(exc).__name__} msg={exc} cause={cause} context={context}"

        finally:
            if sandbox is not None:
                try:
                    sandbox.reset()
                finally:
                    self.sandbox_pool.put(sandbox)

        mean_entropy = self._compute_mean_entropy(logprobs_buffer)

        return {
            "Attempt": attempt_index + 1,
            "Response Length": total_tokens,
            "Python Calls": python_calls,
            "Python Errors": python_errors,
            "Entropy": mean_entropy,
            "Answer": final_answer,
            "Trace": {"turns": turn_traces, "full_completion_token_ids": full_completion_ids},
            "Finish Reason": finish_reason,
            "Tool Calls": tool_calls,
        }

    def _select_answer(self, detailed_results: list[dict]) -> tuple[int, Optional[int], pd.DataFrame]:
        answer_weights = defaultdict(float)
        answer_votes = defaultdict(int)

        for r in detailed_results:
            ans = r.get("Answer")
            ent = r.get("Entropy", float("inf"))
            if ans is None:
                continue
            weight = 1.0 / max(ent, 1e-9)
            answer_weights[ans] += weight
            answer_votes[ans] += 1

        scored = [{"answer": a, "votes": answer_votes[a], "score": w} for a, w in answer_weights.items()]
        scored.sort(key=lambda x: x["score"], reverse=True)
        vote_df = pd.DataFrame(scored) if scored else pd.DataFrame(columns=["answer", "votes", "score"])

        if not scored:
            return 0, None, vote_df

        final_answer = scored[0]["answer"]

        candidates = [
            (i, r.get("Entropy", float("inf")))
            for i, r in enumerate(detailed_results)
            if r.get("Answer") == final_answer
        ]
        selected_idx = min(candidates, key=lambda x: x[1])[0] if candidates else None

        return final_answer, selected_idx, vote_df

    def solve_problem(self, problem: str) -> tuple[int, dict]:
        user_input = f"{problem} {self.cfg.preference_prompt}"

        elapsed_global = time.time() - self.notebook_start_time
        time_left = self.cfg.notebook_limit - elapsed_global

        problems_left_others = max(0, self.problems_remaining - 1)
        reserved_time = problems_left_others * self.cfg.base_problem_timeout

        budget = time_left - reserved_time
        budget = min(budget, self.cfg.high_problem_timeout)
        budget = max(budget, self.cfg.base_problem_timeout)

        deadline = time.time() + budget

        tasks = [(self.cfg.system_prompt, attempt_index) for attempt_index in range(self.cfg.attempts)]

        detailed_results = []
        valid_answers = []
        stop_event = threading.Event()
        executor = ThreadPoolExecutor(max_workers=self.cfg.workers)

        try:
            futures = [
                executor.submit(
                    self._process_attempt,
                    user_input,
                    system_prompt,
                    attempt_index,
                    stop_event,
                    deadline,
                )
                for (system_prompt, attempt_index) in tasks
            ]

            for future in as_completed(futures):
                try:
                    r = future.result()
                    detailed_results.append(r)

                    if r.get("Answer") is not None:
                        valid_answers.append(r["Answer"])

                    counts = Counter(valid_answers).most_common(1)
                    if counts and counts[0][1] >= self.cfg.early_stop:
                        stop_event.set()
                        for f in futures:
                            f.cancel()
                        break

                except Exception:
                    continue

        finally:
            stop_event.set()
            executor.shutdown(wait=True, cancel_futures=True)
            self.problems_remaining = max(0, self.problems_remaining - 1)

        final_answer, selected_idx, vote_df = self._select_answer(detailed_results)

        artifact = {
            "budget_seconds": float(budget),
            "deadline_ts": float(deadline),
            "final_answer": int(final_answer),
            "selected_attempt_index": selected_idx,
            "vote_df": vote_df,
            "attempts": detailed_results,
        }

        return final_answer, artifact

    def close(self):
        # explicit shutdown (more reliable than __del__)
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
# CLI + main loop (does TODO)
# -----------------------------
def parse_args() -> CFG:
    p = argparse.ArgumentParser()
    p.add_argument("--reference-path", default=CFG.reference_path)
    p.add_argument("--log-dir", default=CFG.log_dir)
    p.add_argument("--attempts-filename", default=CFG.attempts_filename)
    p.add_argument("--solutions-filename", default=CFG.solutions_filename)
    p.add_argument("--submission-filename", default=CFG.submission_filename)

    p.add_argument("--served-model-name", default=CFG.served_model_name)
    p.add_argument("--model-path", default=CFG.model_path)
    p.add_argument("--port", type=int, default=CFG.port)
    p.add_argument("--api-key", default=CFG.api_key)
    p.add_argument("--kv-cache-dtype", default=CFG.kv_cache_dtype)
    p.add_argument("--dtype", default=CFG.dtype)
    p.add_argument("--gpu-memory-utilization", type=float, default=CFG.gpu_memory_utilization)

    p.add_argument("--high-problem-timeout", type=int, default=CFG.high_problem_timeout)
    p.add_argument("--base-problem-timeout", type=int, default=CFG.base_problem_timeout)
    p.add_argument("--notebook-limit", type=int, default=CFG.notebook_limit)
    p.add_argument("--server-timeout", type=int, default=CFG.server_timeout)
    p.add_argument("--session-timeout", type=int, default=CFG.session_timeout)
    p.add_argument("--jupyter-timeout", type=int, default=CFG.jupyter_timeout)
    p.add_argument("--sandbox-timeout", type=int, default=CFG.sandbox_timeout)

    p.add_argument("--stream-interval", type=int, default=CFG.stream_interval)
    p.add_argument("--context-tokens", type=int, default=CFG.context_tokens)
    p.add_argument("--buffer-tokens", type=int, default=CFG.buffer_tokens)
    p.add_argument("--search-tokens", type=int, default=CFG.search_tokens)
    p.add_argument("--top-logprobs", type=int, default=CFG.top_logprobs)
    p.add_argument("--batch-size", type=int, default=CFG.batch_size)
    p.add_argument("--early-stop", type=int, default=CFG.early_stop)
    p.add_argument("--attempts", type=int, default=CFG.attempts)
    p.add_argument("--workers", type=int, default=CFG.workers)
    p.add_argument("--turns", type=int, default=CFG.turns)
    p.add_argument("--seed", type=int, default=CFG.seed)
    p.add_argument("--temperature", type=float, default=CFG.temperature)
    p.add_argument("--min-p", type=float, default=CFG.min_p)

    p.add_argument("--verbose", action="store_true", default=CFG.verbose)
    p.add_argument("--quiet", action="store_true", default=False)

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
        high_problem_timeout=args.high_problem_timeout,
        base_problem_timeout=args.base_problem_timeout,
        notebook_limit=args.notebook_limit,
        server_timeout=args.server_timeout,
        session_timeout=args.session_timeout,
        jupyter_timeout=args.jupyter_timeout,
        sandbox_timeout=args.sandbox_timeout,
        stream_interval=args.stream_interval,
        context_tokens=args.context_tokens,
        buffer_tokens=args.buffer_tokens,
        search_tokens=args.search_tokens,
        top_logprobs=args.top_logprobs,
        batch_size=args.batch_size,
        early_stop=args.early_stop,
        attempts=args.attempts,
        workers=args.workers,
        turns=args.turns,
        seed=args.seed,
        temperature=args.temperature,
        min_p=args.min_p,
        verbose=(False if args.quiet else args.verbose),
    )
    return cfg


def main():
    cfg = parse_args()

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    attempts_path = str(log_dir / cfg.attempts_filename)
    solutions_path = str(log_dir / cfg.solutions_filename)
    submission_path = str(log_dir / cfg.submission_filename)

    set_seed(cfg.seed)

    reference_df = pl.read_csv(cfg.reference_path)
    ref_ans_by_id = dict(zip(reference_df["id"].to_list(), reference_df["answer"].to_list()))

    logger = RunLogger(attempts_path, solutions_path, log_dir=str(log_dir), verbose=cfg.verbose)

    solver = AIMO3Solver(cfg)
    solver.problems_remaining = reference_df.height

    print("Base URL:", solver.base_url)
    poll_result = solver.server_process.poll()
    print("Server poll:", poll_result)
    assert poll_result is None
    print("Models list:", solver.client.models.list())  # should succeed

    def predict(id_: pl.DataFrame, question: pl.DataFrame, answer: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        id_value = id_.item(0)
        logger.log_event("problem_start", {"id": id_value})
        question_text = str(question.item(0))

        # Ground-truth answer (prefer what the gateway passes; else fallback to preloaded reference)
        true_answer = None
        if answer is not None:
            try:
                true_answer = int(answer.item(0))
            except Exception:
                true_answer = None
        if true_answer is None and id_value in ref_ans_by_id:
            try:
                true_answer = int(ref_ans_by_id[id_value])
            except Exception:
                true_answer = None

        solve_started_ts = datetime.now(timezone.utc).isoformat()
        t0 = time.time()

        gc.disable()
        pred_answer, artifact = solver.solve_problem(question_text)
        gc.enable()
        gc.collect()

        solve_finished_ts = datetime.now(timezone.utc).isoformat()
        solve_elapsed_ms = int((time.time() - t0) * 1000)

        attempts = artifact["attempts"]
        selected_idx = artifact.get("selected_attempt_index", None)

        selected_entropy = None
        if selected_idx is not None and 0 <= selected_idx < len(attempts):
            selected_entropy = attempts[selected_idx].get("Entropy", None)

        # Compute correctness
        is_correct = None
        if true_answer is not None:
            is_correct = (int(pred_answer) == int(true_answer))

        # Build attempt log records with status + reject reason
        attempt_records = []
        for i, r in enumerate(attempts):
            ans = r.get("Answer", None)
            finish_reason = r.get("Finish Reason", "unknown")

            if selected_idx is not None and i == selected_idx:
                status = "selected"
                reject_reason = ""
            else:
                status = "rejected"
                if ans is None:
                    reject_reason = f"no_answer:{finish_reason}"
                elif ans != pred_answer:
                    reject_reason = "different_answer"
                else:
                    reject_reason = "same_answer_not_selected"

            attempt_records.append(
                {
                    "id": id_value,
                    "attempt": r.get("Attempt", i + 1),
                    "status": status,
                    "reject_reason": reject_reason,
                    "pred_final_answer": int(pred_answer),
                    "attempt_answer": ans,
                    "entropy": r.get("Entropy", None),
                    "response_length": r.get("Response Length", None),
                    "python_calls": r.get("Python Calls", None),
                    "python_errors": r.get("Python Errors", None),
                    "finish_reason": finish_reason,
                    "trace": r.get("Trace", {}),
                    "tool_calls": r.get("Tool Calls", []),
                }
            )

        attempts_total = len(attempt_records)
        attempts_selected = sum(1 for x in attempt_records if x["status"] == "selected")
        attempts_rejected = attempts_total - attempts_selected
        attempts_with_answer = sum(1 for x in attempt_records if x["attempt_answer"] is not None)

        finish_reason_counts = {}
        for x in attempt_records:
            fr = x.get("finish_reason", "unknown")
            finish_reason_counts[fr] = finish_reason_counts.get(fr, 0) + 1

        vote_summary = {}
        try:
            df = artifact["vote_df"]
            vote_summary = {"top": df.head(5).to_dict(orient="records")}
        except Exception:
            vote_summary = {}

        logger.log_attempts(attempt_records)
        logger.log_solution_row(
            id_value=id_value,
            pred_answer=int(pred_answer),
            true_answer=true_answer,
            is_correct=is_correct,
            selected_attempt=(
                attempts[selected_idx].get("Attempt")
                if selected_idx is not None and 0 <= selected_idx < len(attempts)
                else None
            ),
            selected_entropy=selected_entropy,
            budget_seconds=artifact.get("budget_seconds"),
            deadline_ts=artifact.get("deadline_ts"),
            solve_started_ts=solve_started_ts,
            solve_finished_ts=solve_finished_ts,
            solve_elapsed_ms=solve_elapsed_ms,
            attempts_total=attempts_total,
            attempts_with_answer=attempts_with_answer,
            attempts_selected=attempts_selected,
            attempts_rejected=attempts_rejected,
            finish_reason_counts=finish_reason_counts,
            vote_summary=vote_summary,
        )

        logger.log_event(
            "problem_end",
            {"id": id_value, "pred": int(pred_answer), "true": true_answer, "correct": is_correct},
        )

        return pl.DataFrame({"id": [id_value], "answer": [int(pred_answer)]})

    # -----------------------------
    # run predict
    # -----------------------------
    out_rows = []
    try:
        # Iterate reference.csv rows deterministically
        for row in reference_df.iter_rows(named=True):
            id_value = row["id"]
            problem_text = row["problem"]
            ans_value = row["answer"]

            # Wrap into 1x1 Polars DataFrames to match predict() signature
            id_df = pl.DataFrame({"id": [id_value]})[["id"]]
            q_df = pl.DataFrame({"problem": [problem_text]})[["problem"]]
            a_df = pl.DataFrame({"answer": [ans_value]})[["answer"]]

            pred_df = predict(id_df, q_df, a_df)
            out_rows.append(pred_df)

        submission_df = pl.concat(out_rows) if out_rows else pl.DataFrame({"id": [], "answer": []})
        submission_df.write_csv(submission_path)
        print(f"\nWrote submission to: {submission_path}")
        print(submission_df.head(5))

    finally:
        solver.close()


if __name__ == "__main__":
    main()