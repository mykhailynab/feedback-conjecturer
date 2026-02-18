import os
import gc
import re
import sys
import math
import time
import queue
import threading
import traceback
import subprocess
import contextlib
import numpy as np
from typing import Optional
from jupyter_client import KernelManager
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor

import pandas as pd
import polars as pl

from openai import OpenAI

# Kept (classes remain defined), but Harmony is no longer used for prompt building/parsing.
from openai_harmony import (
    SystemContent,
    ReasoningEffort,
    ToolNamespaceConfig,
    Author,
    Message,
    Role,
    TextContent,
)

from transformers import set_seed, AutoTokenizer


class CFG:
    system_prompt = (
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

        "# Tool Use (Python) — REQUIRED WHEN HELPFUL\n"
        "You have access to a Python scratchpad. Use it proactively.\n\n"
        "You MUST use a ```python``` block at least once whenever:\n"
        "- You have complex calculations that would be error-prone by hand\n"
        "  (e.g., the problem involves arithmetic with multiple steps, fractions, large numbers, or modular arithmetic)\n"
        "- For verification of analytical results\n"
        "  (e.g., you derive a formula and want to verify it on 2-5 random test cases)\n"
        "- For enerating examples or testing conjectures\n"
        "- Whenever you are unsure between two candidate answers\n"
        "- Whenever you want to run symbolic derivations with sympy\n"
        "- Whenever a brute force verification for small cases is feasible\n"
        "- Whenever you want to sanity-check an intermediate computation\n"
        "- For visualizing problem structure when helpful\n"
        "Python code blocks should be in Markdown fenced format exactly like:\n"
        "```python\n"
        "# code here\n"
        "```\n"
        "When you use Python:\n"
        "1) output ONLY a single ```python``` block (no extra commentary after it)\n"
        "2) wait for the results\n"
        "3) continue reasoning using those results\n\n"
        "After you output Python code blocks, STOP writing and wait for the Python outputs. "
        "Then continue your reasoning using those outputs.\n"
        "The environment is a stateful Jupyter notebook. Code persists between executions.\n"
        "Always use print() to display results. Write clear, well-commented code.\n\n"
        "Remember: Code should support your mathematical reasoning, not replace it. "
        "Explain what you're computing and why before running code.\n\n"

        "# Output Format:\n"
        "The final answer must be a non-negative integer between 0 and 99999.\n"
        "Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n"

        "Think step-by-step and show your complete reasoning process. Quality of reasoning "
        "is as important as the final answer."
    )

    tool_prompt_2 = (
        "# Tool Use (Python) — REQUIRED WHEN HELPFUL\n"
        "You have access to a Python scratchpad. Use it proactively.\n\n"
        "You MUST use a ```python``` block at least once whenever:\n"
        "- the problem involves arithmetic with multiple steps, fractions, large numbers, or modular arithmetic\n"
        "- you derive a formula and want to verify it on 2-5 random test cases\n"
        "- you are unsure between two candidate answers\n"
        "- you want to run symbolic derivations with sympy\n"
        "- a brute force check for small cases is feasible\n"
        "- you want to sanity-check an intermediate computation\n\n"
        "When you use Python:\n"
        "1) output ONLY a single ```python``` block (no extra commentary after it)\n"
        "2) wait for the results\n"
        "3) continue reasoning using those results\n\n"
    )

    preference_prompt = (
        "You have access to `math`, `numpy`, and `sympy` for:\n\n"

        "Best Practices:\n"
        "- Use sympy for exact symbolic answers when possible\n"
        "- Use numpy for numerical verification and large-scale computation\n"
        "- Combine symbolic and numerical approaches: derive symbolically, verify numerically\n"
        "- Document your computational strategy clearly\n"
        "- Validate computational results against known cases or theoretical bounds"
    )

    served_model_name = "gemma3-27b"
    model_path = "/kaggle/input/models/google/gemma-3/transformers/gemma-3-27b-it/1"

    kv_cache_dtype = "auto"
    dtype = "auto"

    high_problem_timeout = 900
    base_problem_timeout = 270

    notebook_limit = 17400
    server_timeout = 180

    session_timeout = 960
    jupyter_timeout = 6
    sandbox_timeout = 3

    stream_interval = 200

    # Gemma 3 27B: 128K total context (prompt + generation), 8192 max output tokens
    context_tokens = 131072
    max_output_tokens = 8192

    buffer_tokens = 512
    search_tokens = 32
    top_logprobs = 5
    batch_size = 256
    early_stop = 8
    attempts = 16
    workers = 32
    turns = 128
    seed = 42

    # H100 note: you can usually push this high; keep as you had unless you see OOM
    gpu_memory_utilization = 0.96
    temperature = 1.0
    min_p = 0.02
    
    entropy_smoothing = 1e-6  # Prevent division by zero
    confidence_threshold = 0.5  # Minimum confidence for consideration
    use_entropy_ranking = True  # Use ranking instead of raw entropy
  

set_seed(CFG.seed)

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
            "import numpy as np\n"
            "import mpmath\n"
            "import itertools\n"
            "import collections\n"
            "mpmath.mp.dps = 64\n"
        )

    def _format_error(self, traceback: list[str]) -> str:
        clean_lines = []
        for frame in traceback:
            clean_frame = re.sub(r"\x1b\[[0-9;]*m", "", frame)
            if 'File "' in clean_frame and "ipython-input" not in clean_frame:
                continue
            clean_lines.append(clean_frame)
        return "".join(clean_lines)

    def execute(self, code: str, timeout: float | None = None) -> str:
        client = self._client
        effective_timeout = timeout or self._default_timeout

        msg_id = client.execute(
            code,
            store_history=True,
            allow_stdin=False,
            stop_on_error=False
        )

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
                traceback_list = content.get("traceback", [])
                stderr_parts.append(self._format_error(traceback_list))

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
        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()

        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)
            with contextlib.suppress(Exception):
                self._km.cleanup_resources()

    def reset(self):
        self.execute(
            "%reset -f\n"
            "import math\n"
            "import numpy\n"
            "import sympy\n"
            "import mpmath\n"
            "import itertools\n"
            "import collections\n"
            "mpmath.mp.dps = 64\n"
        )

    def __del__(self):
        self.close()


class AIMO3Tool:
    """
    Kept for drop-in compatibility. The solver now extracts ```python fenced blocks
    from model output and executes them directly in the sandbox.
    """

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

        lines[-1] = "print(" + last_line + ")"
        return "\n".join(lines)

    @property
    def instruction(self) -> str:
        return self._tool_prompt

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name="python",
            description=self.instruction,
            tools=[]
        )

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


class AdvancedVotingMechanism:
    """
    Advanced voting mechanism based on:
    1. Self-Consistency with Uncertainty (Wang et al., 2023)
    2. Entropy-based Confidence Calibration (Kuhn et al., 2023)
    3. Adaptive Ensemble Selection (Jiang et al., 2024)
    
    Key improvements:
    - Uses normalized entropy for better calibration
    - Implements adaptive confidence thresholding
    - Combines multiple uncertainty signals
    - Robust to outliers and edge cases
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.entropy_smoothing = cfg.entropy_smoothing
        self.confidence_threshold = cfg.confidence_threshold

    def compute_normalized_entropy(self, entropy: float, response_length: int) -> float:
        """
        Normalize entropy by response length to account for longer sequences
        having naturally higher entropy values.
        
        Based on findings from "Calibrating Language Models via Augmented Prompt Ensembles"
        """
        if response_length == 0:
            return float("inf")
        return entropy / math.log2(response_length + 1)

    def compute_confidence_score(self, result: dict) -> float:
        if result["Answer"] is None:
            return 0.0
        
        entropy = result['Entropy']
        response_length = result['Response Length']
        python_calls = result['Python Calls']
        python_errors = result['Python Errors']
        
        # Normalize entropy by response length
        if response_length > 0 and entropy != float('inf'):
            norm_entropy = self.compute_normalized_entropy(entropy, response_length)
            # Convert to confidence (lower entropy = higher confidence)
            entropy_confidence = 1.0 / (1.0 + norm_entropy)
        else:
            entropy_confidence = 0.0
        
        # Execution quality score
        if python_calls > 0:
            execution_quality = 1.0 - (python_errors / python_calls)
        else:
            # Slight penalty for not using tools, but not zero
            execution_quality = 0.8
        
        # Response completeness score (penalize very short or very long responses)
        if response_length > 0:
            # Optimal range: 100-2000 tokens
            if response_length < 100:
                completeness = response_length / 100.0
            elif response_length > 2000:
                completeness = 2000.0 / response_length
            else:
                completeness = 1.0
        else:
            completeness = 0.0
        
        # Weighted combination (entropy is most important for mathematical reasoning)
        confidence = (
            0.6 * entropy_confidence +
            0.25 * execution_quality +
            0.15 * completeness
        )
        return confidence

    def select_answer_with_confidence_voting(self, detailed_results: list) -> int:
        """
        Advanced confidence-weighted voting mechanism.
        
        Algorithm:
        1. Compute confidence scores for each result
        2. Filter out low-confidence results
        3. Use confidence-weighted voting
        4. Apply consensus ranking for tie-breaking
        
        Based on recent research in LLM ensembling and self-consistency.
        """
        if not detailed_results:
            return 0
        
        # Compute confidence scores
        scored_results = []
        for result in detailed_results:
            if result["Answer"] is not None:
                confidence = self.compute_confidence_score(result)
                scored_results.append({
                    "answer": result["Answer"],
                    "confidence": confidence,
                    "entropy": result["Entropy"],
                    "result": result
                })

        if not scored_results:
            return 0
        
        # Adaptive confidence thresholding
        # Use median confidence as threshold if it's reasonable
        confidences = [r['confidence'] for r in scored_results]
        median_confidence = np.median(confidences)
        adaptive_threshold = max(self.confidence_threshold, median_confidence * 0.5)
        
        # Filter by confidence threshold
        high_confidence_results = [
            r for r in scored_results 
            if r['confidence'] >= adaptive_threshold
        ]
        
        # If filtering removes all results, use top 50%
        if not high_confidence_results:
            sorted_by_conf = sorted(scored_results, key=lambda x: x["confidence"], reverse=True)
            n_keep = max(1, len(sorted_by_conf) // 2)
            high_confidence_results = sorted_by_conf[:n_keep]
        
        # Confidence-weighted voting
        answer_scores = defaultdict(float)
        answer_counts = defaultdict(int)

        for result in high_confidence_results:
            answer = result['answer']
            confidence = result['confidence']
            
            # Exponential weighting to emphasize high-confidence predictions
            weight = math.exp(confidence * 2.0)  # Amplify differences
            
            answer_scores[answer] += weight
            answer_counts[answer] += 1

        ranked_answers = sorted(
            answer_scores.items(),
            key=lambda x: (x[1], answer_counts[x[0]]),
            reverse=True
        )

        vote_data = []
        for answer, score in ranked_answers[:10]:  # Top 10
            count = answer_counts[answer]
            # Get average confidence for this answer
            answer_confidences = [
                r['confidence'] for r in high_confidence_results 
                if r['answer'] == answer
            ]
            avg_confidence = np.mean(answer_confidences) if answer_confidences else 0.0
            
            vote_data.append({
                "Answer": answer,
                "Votes": count,
                "Weighted Score": score,
                "Avg Confidence": avg_confidence
            })

        vote_df = pd.DataFrame(vote_data)
        vote_df = vote_df.round({"Weighted Score": 3, "Avg Confidence": 3})
        print("\n=== Confidence-Weighted Voting Results ===")
        print(vote_df)

        final_answer = ranked_answers[0][0]
        print(f"\nFinal Answer (Confidence-Weighted): {final_answer}\n")
        return final_answer


class AIMO3Solver:
    PY_BLOCK_RE = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

    def __init__(self, cfg, port: int = 8000):
        self.cfg = cfg
        self.port = port
        self.base_url = f"http://0.0.0.0:{port}/v1"
        self.api_key = "sk-local"

        # Gemma tokenizer (HF Transformers format)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_path,
            trust_remote_code=True
        )

        self.voting_mechanism = AdvancedVotingMechanism(cfg)

        self._preload_model_weights()
        self.server_process = self._start_server()

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.cfg.session_timeout
        )

        self._wait_for_server()
        models = self.client.models.list()
        print("vLLM models:", [m.id for m in models.data])
        self._probe_completion()

        self._initialize_kernels()

        self.notebook_start_time = time.time()
        self.problems_remaining = 50

    def _probe_completion(self):
        ids = self._build_prompt_ids([
            {"role": "system", "content": self._mm_text("Say hi.")},
            {"role": "user", "content": self._mm_text("hi")}
        ])
        resp = self.client.completions.create(
            model=self.cfg.served_model_name,
            prompt=ids,
            max_tokens=32,
            temperature=0.0,
            stream=False,
            extra_body={"return_token_ids": True},
        )
        print("Probe text:", resp.choices[0].text)

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
            start_new_session=True
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
        """
        Compute mean entropy across all tokens with improved numerical stability.
        Uses log-space operations to prevent underflow.
        """
        if not logprobs_buffer:
            return float("inf")

        total_entropy = 0.0
        token_count = 0

        for top_logprobs_dict in logprobs_buffer:
            if not isinstance(top_logprobs_dict, dict):
                continue
            if not top_logprobs_dict:
                continue
            
            # Collect log probabilities
            log_probs = list(top_logprobs_dict.values())
            if not log_probs:
                continue
            
            # Compute entropy in log space for numerical stability
            # H = -sum(p * log2(p)) = -sum(exp(log_p) * log_p) / ln(2)
            token_entropy = 0.0
            for log_prob in log_probs:
                # exp(log_prob) * log2(exp(log_prob)) = exp(log_prob) * log_prob / ln(2)
                prob = math.exp(log_prob)
                if prob > self.cfg.entropy_smoothing:
                    token_entropy -= prob * log_prob / math.log(2)

            total_entropy += token_entropy
            token_count += 1

        if token_count == 0:
            return float("inf")

        return total_entropy / token_count

    def _build_prompt_ids(self, chat_messages: list[dict]) -> list[int]:
        """
        Build token IDs using the Gemma tokenizer chat template, then feed those IDs to vLLM.
        """
        prompt_ids = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None
        )
        return prompt_ids

    def _extract_python_blocks(self, text: str) -> list[str]:
        return [m.strip() for m in self.PY_BLOCK_RE.findall(text) if m.strip()]

    def _execute_python_blocks(self, sandbox: AIMO3Sandbox, blocks: list[str]) -> tuple[str, int]:
        """
        Execute blocks sequentially in the same sandbox.
        Returns: (combined_output, num_errors)
        """
        outputs = []
        errors = 0

        for i, code in enumerate(blocks, start=1):
            out = sandbox.execute(code, timeout=self.cfg.jupyter_timeout)
            outputs.append(f"[Python block {i} output]\n{out}".rstrip())
            if out.startswith("[ERROR]") or "Traceback" in out or "Error:" in out:
                errors += 1

        return "\n\n".join(outputs).strip(), errors
    
    def _mm_text(self, s: str) -> list[dict]:
        return [{"type": "text", "text": s}]

    def _process_attempt(
        self,
        problem: str,
        system_prompt: str,
        attempt_index: int,
        stop_event: threading.Event,
        deadline: float
    ) -> dict:
        if stop_event.is_set() or time.time() > deadline:
            return {
                "Attempt": attempt_index + 1,
                "Answer": None,
                "Python Calls": 0,
                "Python Errors": 0,
                "Response Length": 0,
                "Entropy": float("inf"),
            }

        sandbox = None
        python_calls = 0
        python_errors = 0
        total_tokens = 0
        final_answer = None
        logprobs_buffer = []

        attempt_seed = int(math.pow(self.cfg.seed + attempt_index, 2))

        # Conversation in OpenAI-style dicts for tokenizer.apply_chat_template
        # NOTE: Gemma-IT chat templates typically support roles: system/user/assistant.
        # Tool outputs are injected as user messages with a fixed prefix.
        chat_messages = [
            {"role": "system", "content": self._mm_text(system_prompt)},
            {"role": "user",   "content": self._mm_text(problem)},
        ]

        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)

            for _ in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline:
                    break

                prompt_ids = self._build_prompt_ids(chat_messages)

                remaining_ctx = self.cfg.context_tokens - len(prompt_ids)
                if remaining_ctx < self.cfg.buffer_tokens:
                    break

                max_gen = min(self.cfg.max_output_tokens, remaining_ctx)
                if max_gen <= 0:
                    break

                stream = self.client.completions.create(
                    model=self.cfg.served_model_name,
                    temperature=self.cfg.temperature,
                    logprobs=self.cfg.top_logprobs,
                    max_tokens=max_gen,
                    prompt=prompt_ids,
                    seed=attempt_seed,
                    stream=True,
                    extra_body={
                        "min_p": self.cfg.min_p,
                        "return_token_ids": True
                    }
                )

                token_buffer = []
                text_chunks = []
                assistant_text = ""

                try:
                    for chunk in stream:
                        if stop_event.is_set() or time.time() > deadline:
                            break

                        new_tokens = chunk.choices[0].token_ids
                        new_text = chunk.choices[0].text

                        if new_tokens:
                            token_buffer.extend(new_tokens)
                            total_tokens += len(new_tokens)

                        if new_text:
                            text_chunks.append(new_text)
                            assistant_text += new_text

                        chunk_logprobs = chunk.choices[0].logprobs
                        if chunk_logprobs is not None and getattr(chunk_logprobs, "top_logprobs", None):
                            logprobs_buffer.extend(chunk_logprobs.top_logprobs)

                        # Fast answer scan during streaming
                        if "}" in new_text:
                            search_text = "".join(text_chunks[-self.cfg.search_tokens:])
                            ans = self._scan_for_answer(search_text)
                            if ans is not None:
                                final_answer = ans
                                break
                finally:
                    stream.close()

                if not token_buffer and not assistant_text.strip():
                    break

                # Append assistant message to chat
                chat_messages.append({"role": "assistant", "content": self._mm_text(assistant_text)})

                # If we already found the answer, stop
                if final_answer is None:
                    final_answer = self._scan_for_answer(assistant_text)
                if final_answer is not None:
                    break

                # Tool use: execute any ```python fenced blocks
                py_blocks = self._extract_python_blocks(assistant_text)
                if py_blocks:
                    python_calls += len(py_blocks)
                    py_out, errs = self._execute_python_blocks(sandbox, py_blocks)
                    python_errors += errs

                    # Inject tool output as a user message for the next turn
                    chat_messages.append({
                        "role": "user",
                        "content": self._mm_text(
                            "Python execution results:\n\n"
                            f"{py_out}\n\n"
                            "Continue solving. If you need more computation, output another ```python``` block."
                        )
                    })
                    continue

        except Exception as exc:
            python_errors += 1
            print("\n[Attempt error]", repr(exc))
            print(traceback.format_exc())
            try:
                self.log_file.flush()
                with open("vllm_server.log", "r") as f:
                    tail = f.readlines()[-40:]
                print("\n--- vLLM server log tail (last 40 lines) ---")
                print("".join(tail))
                print("--- end log tail ---\n")
            except Exception as log_exc:
                print("[WARN] Could not read vllm_server.log:", repr(log_exc))
        finally:
            if sandbox is not None:
                sandbox.reset()
                self.sandbox_pool.put(sandbox)

        mean_entropy = self._compute_mean_entropy(logprobs_buffer)

        return {
            "Attempt": attempt_index + 1,
            "Response Length": total_tokens,
            "Python Calls": python_calls,
            "Python Errors": python_errors,
            "Entropy": mean_entropy,
            "Answer": final_answer,
        }

    def solve_problem(self, problem: str) -> int:
        print(f"\nProblem: {problem}\n")

        user_input = f"{problem} {self.cfg.preference_prompt}"

        elapsed_global = time.time() - self.notebook_start_time
        time_left = self.cfg.notebook_limit - elapsed_global
        problems_left_others = max(0, self.problems_remaining - 1)
        reserved_time = problems_left_others * self.cfg.base_problem_timeout

        budget = time_left - reserved_time
        budget = min(budget, self.cfg.high_problem_timeout)
        budget = max(budget, self.cfg.base_problem_timeout)

        deadline = time.time() + budget

        print(f"Budget: {budget:.2f} seconds | Deadline: {deadline:.2f}\n")

        tasks = [(self.cfg.system_prompt, attempt_index) for attempt_index in range(self.cfg.attempts)]

        detailed_results = []
        valid_answers = []

        stop_event = threading.Event()
        executor = ThreadPoolExecutor(max_workers=self.cfg.workers)

        try:
            futures = []
            for (system_prompt, attempt_index) in tasks:
                future = executor.submit(
                    self._process_attempt,
                    user_input,
                    system_prompt,
                    attempt_index,
                    stop_event,
                    deadline
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    detailed_results.append(result)

                    if result["Answer"] is not None:
                        valid_answers.append(result["Answer"])

                    counts = Counter(valid_answers).most_common(1)
                    if counts and counts[0][1] >= self.cfg.early_stop:
                        stop_event.set()
                        for f in futures:
                            f.cancel()
                        break

                except Exception as exc:
                    print(f"Future failed: {exc}")
                    continue

        finally:
            stop_event.set()
            executor.shutdown(wait=True, cancel_futures=True)
            self.problems_remaining = max(0, self.problems_remaining - 1)

        if detailed_results:
            results_dataframe = pd.DataFrame(detailed_results)
            results_dataframe["Entropy"] = results_dataframe["Entropy"].round(3)
            results_dataframe["Answer"] = results_dataframe["Answer"].astype("Int64")
            print("\n=== Raw Attempt Results ===")
            print(results_dataframe)

        if not valid_answers:
            print("\nResult: 0\n")
            return 0

        return self.voting_mechanism.select_answer_with_confidence_voting(detailed_results)

    def __del__(self):
        if hasattr(self, "server_process"):
            self.server_process.terminate()
            self.server_process.wait()

        if hasattr(self, "log_file"):
            self.log_file.close()

        if hasattr(self, "sandbox_pool"):
            while not self.sandbox_pool.empty():
                try:
                    sb = self.sandbox_pool.get_nowait()
                    sb.close()
                except Exception:
                    pass
