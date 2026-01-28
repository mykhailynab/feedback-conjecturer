# %%
%pip uninstall --yes 'keras' 'matplotlib' 'scikit-learn' 'tensorflow'

# %%
import warnings
warnings.simplefilter('ignore')

# %%
import os
import sys
import subprocess

# %%
def set_env(input_archive, temp_dir):

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        
        subprocess.run(['tar', '-xzf', input_archive, '-C', temp_dir], check=True)
    
    subprocess.run([
        sys.executable, 
        '-m', 
        'pip', 
        'install', 
        '--no-index', 
        '--find-links', 
        f'{temp_dir}/wheels', 
        'unsloth', 
        'trl', 
        'vllm', 
        'openai_harmony'
    ], check=True)

# %%
set_env(
    input_archive='/kaggle/input/aimo-3-utils/wheels.tar.gz', 
    temp_dir='/kaggle/tmp/setup'
)

# %%
subprocess.run(['ls', '/kaggle/tmp/setup/tiktoken_encodings'])

# %%
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda/bin/ptxas'
os.environ['TIKTOKEN_ENCODINGS_BASE'] = '/kaggle/tmp/setup/tiktoken_encodings'

# %%
import gc
import re
import math
import time
import queue
import threading
import contextlib
from typing import Optional
from jupyter_client import KernelManager
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor

import pandas as pd
import polars as pl

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
    Conversation
)

from transformers import set_seed
import kaggle_evaluation.aimo_3_inference_server

# %%
from pathlib import Path
import json
import csv

LOG_DIR = Path("/kaggle/working/aimo3_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

ATTEMPTS_PATH = LOG_DIR / "attempts.jsonl"
SOLUTIONS_PATH = LOG_DIR / "solutions.csv"


# %%
class CFG:
    
    system_prompt = (
        'You are an elite mathematical problem solver with expertise at the International '
        'Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through '
        'rigorous mathematical reasoning.\n\n'
        
        '# Problem-Solving Approach:\n'
        '1. UNDERSTAND: Carefully read and rephrase the problem in your own words. '
        'Identify what is given, what needs to be found, and any constraints.\n'
        '2. EXPLORE: Consider multiple solution strategies. Think about relevant theorems, '
        'techniques, patterns, or analogous problems. Don\'t commit to one approach immediately.\n'
        '3. PLAN: Select the most promising approach and outline key steps before executing.\n'
        '4. EXECUTE: Work through your solution methodically. Show all reasoning steps clearly.\n'
        '5. VERIFY: Check your answer by substituting back, testing edge cases, or using '
        'alternative methods. Ensure logical consistency throughout.\n\n'
        
        '# Mathematical Reasoning Principles:\n'
        '- Break complex problems into smaller, manageable sub-problems\n'
        '- Look for patterns, symmetries, and special cases that provide insight\n'
        '- Use concrete examples to build intuition before generalizing\n'
        '- Consider extreme cases and boundary conditions\n'
        '- If stuck, try working backwards from the desired result\n'
        '- Be willing to restart with a different approach if needed\n\n'
        
        '# Verification Requirements:\n'
        '- Cross-check arithmetic and algebraic manipulations\n'
        '- Verify that your solution satisfies all problem constraints\n'
        '- Test your answer with simple cases or special values when possible\n'
        '- Ensure dimensional consistency and reasonableness of the result\n\n'
        
        '# Output Format:\n'
        'The final answer must be a non-negative integer between 0 and 99999.\n'
        'Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n'
        
        'Think step-by-step and show your complete reasoning process. Quality of reasoning '
        'is as important as the final answer.'
    )
    
    tool_prompt = (
        'Use this tool to execute Python code for:\n'
        '- Complex calculations that would be error-prone by hand\n'
        '- Numerical verification of analytical results\n'
        '- Generating examples or testing conjectures\n'
        '- Visualizing problem structure when helpful\n'
        '- Brute-force verification for small cases\n\n'
        
        'The environment is a stateful Jupyter notebook. Code persists between executions.\n'
        'Always use print() to display results. Write clear, well-commented code.\n\n'
        
        'Remember: Code should support your mathematical reasoning, not replace it. '
        'Explain what you\'re computing and why before running code.'
    )
    
    preference_prompt = (
        'You have access to `math`, `numpy`, and `sympy` for:\n\n'
        
        '# Symbolic Computation (sympy):\n'
        '- Algebraic manipulation and simplification\n'
        '- Solving equations and systems of equations\n'
        '- Symbolic differentiation and integration\n'
        '- Number theory functions (primes, divisors, modular arithmetic)\n'
        '- Polynomial operations and factorization\n'
        '- Working with mathematical expressions symbolically\n\n'
        
        '# Numerical Computation (numpy):\n'
        '- Array operations and linear algebra\n'
        '- Efficient numerical calculations for large datasets\n'
        '- Matrix operations and eigenvalue problems\n'
        '- Statistical computations\n\n'
        
        '# Mathematical Functions (math):\n'
        '- Standard mathematical functions (trig, log, exp)\n'
        '- Constants like pi and e\n'
        '- Basic operations for single values\n\n'
        
        'Best Practices:\n'
        '- Use sympy for exact symbolic answers when possible\n'
        '- Use numpy for numerical verification and large-scale computation\n'
        '- Combine symbolic and numerical approaches: derive symbolically, verify numerically\n'
        '- Document your computational strategy clearly\n'
        '- Validate computational results against known cases or theoretical bounds'
    )
    
    served_model_name = 'gpt-oss'
    model_path = '/kaggle/input/gpt-oss-120b/transformers/default/1'
    
    kv_cache_dtype = 'fp8_e4m3'
    dtype = 'auto'

    high_problem_timeout = 900
    base_problem_timeout = 300

    notebook_limit = 17400
    server_timeout = 180

    session_timeout = 960
    jupyter_timeout = 6
    sandbox_timeout = 3

    stream_interval = 200
    context_tokens = 65536
    buffer_tokens = 512
    search_tokens = 32
    top_logprobs = 5
    batch_size = 256
    early_stop = 4
    attempts = 8
    workers = 16
    turns = 128
    seed = 42

    gpu_memory_utilization = 0.96
    temperature = 0.5
    min_p = 0.02

    log_dir = str(LOG_DIR)
    attempts_path = str(ATTEMPTS_PATH)
    solutions_path = str(SOLUTIONS_PATH)

# %%
set_seed(CFG.seed)

# %%
class RunLogger:
    def __init__(self, attempts_path: str, solutions_path: str):
        self.attempts_path = attempts_path
        self.solutions_path = solutions_path
        self._lock = threading.Lock()
        self._init_solutions_csv()

    def _init_solutions_csv(self):
        if os.path.exists(self.solutions_path):
            return
        header = [
            "id", "pred_answer", "true_answer", "is_correct",
            "selected_attempt", "selected_entropy",
            "selected_raw_output"
        ]
        with open(self.solutions_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)

    def log_attempts(self, attempt_records: list[dict]):
        # JSONL: one record per attempt
        with self._lock:
            with open(self.attempts_path, "a", encoding="utf-8") as f:
                for rec in attempt_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def log_solution_row(
        self,
        id_value: int,
        pred_answer: int,
        true_answer: Optional[int],
        is_correct: Optional[bool],
        selected_attempt: Optional[int],
        selected_entropy: Optional[float],
        selected_raw_output: str
    ):
        with self._lock:
            with open(self.solutions_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    id_value, pred_answer, true_answer,
                    is_correct, selected_attempt, selected_entropy,
                    selected_raw_output
                ])

logger = RunLogger(CFG.attempts_path, CFG.solutions_path)

# %%
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
        tool_config: ToolNamespaceConfig
    ) -> list[Message]:

        system_content = self.get_system_content(system_prompt, tool_config)        
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)

        user_message = Message.from_role_and_content(Role.USER, user_prompt)

        return [system_message, user_message]

# %%
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
        env['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
        env['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '0'
        env['JUPYTER_PLATFORM_DIRS'] = '1'
        env['PYTHONWARNINGS'] = 'ignore'
        env['MPLBACKEND'] = 'Agg'

        self._km = KernelManager()
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]

        self._km.start_kernel(env=env, extra_arguments=['--Application.log_level=CRITICAL'])

        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True

        self.execute(
            'import math\n'
            'import numpy\n'
            'import sympy\n'
            'import itertools\n'
            'import collections\n'
            'import mpmath\n'
            'mpmath.mp.dps = 64\n'
        )

    def _format_error(self, traceback: list[str]) -> str:

        clean_lines = []

        for frame in traceback:
            clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)

            if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
                continue

            clean_lines.append(clean_frame)

        return ''.join(clean_lines)

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

                return f'[ERROR] Execution timed out after {effective_timeout} seconds'

            try:
                msg = client.get_iopub_msg(timeout=1.0)

            except queue.Empty:
                continue

            if msg.get('parent_header', {}).get('msg_id') != msg_id:
                continue

            msg_type = msg.get('msg_type')
            content = msg.get('content', {})

            if msg_type == 'stream':
                text = content.get('text', '')

                if content.get('name') == 'stdout':
                    stdout_parts.append(text)

                else:
                    stderr_parts.append(text)

            elif msg_type == 'error':
                traceback_list = content.get('traceback', [])

                stderr_parts.append(self._format_error(traceback_list))

            elif msg_type in {'execute_result', 'display_data'}:
                data = content.get('data', {})
                text = data.get('text/plain')

                if text:
                    stdout_parts.append(text if text.endswith('\n') else f'{text}\n')

            elif msg_type == 'status':
                if content.get('execution_state') == 'idle':
                    break

        stdout = ''.join(stdout_parts)
        stderr = ''.join(stderr_parts)

        if stderr:
            return f'{stdout.rstrip()}\n{stderr}' if stdout else stderr

        return stdout if stdout.strip() else '[WARN] No output. Use print() to see results.'

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
            '%reset -f\n'
            'import math\n'
            'import numpy\n'
            'import sympy\n'
            'import itertools\n'
            'import collections\n'
            'import mpmath\n'
            'mpmath.mp.dps = 64\n'
        )

    def __del__(self):

        self.close()

# %%
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

        lines = code.strip().split('\n')

        if not lines:
            return code

        last_line = lines[-1].strip()

        if 'print' in last_line or 'import' in last_line:
            return code

        if not last_line:
            return code

        if last_line.startswith('#'):
            return code

        lines[-1] = 'print(' + last_line + ')'

        return '\n'.join(lines)

    @property
    def instruction(self) -> str:

        return self._tool_prompt

    @property
    def tool_config(self) -> ToolNamespaceConfig:

        return ToolNamespaceConfig(
            name='python', 
            description=self.instruction, 
            tools=[]
        )

    def _make_response(self, output: str, channel: str | None = None) -> Message:

        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name='python')
        message = Message(author=author, content=[content]).with_recipient('assistant')

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
                output = f'[ERROR] {exc}'

        return [self._make_response(output, channel=message.channel)]

# %%
class AIMO3Solver:

    def __init__(self, cfg, port: int = 8000):
    
        self.cfg = cfg
        self.port = port
        self.base_url = f'http://0.0.0.0:{port}/v1'
        self.api_key = 'sk-local'
        self.template = AIMO3Template()
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
    
        self._preload_model_weights()
        
        self.server_process = self._start_server()
    
        self.client = OpenAI(
            base_url=self.base_url, 
            api_key=self.api_key, 
            timeout=self.cfg.session_timeout
        )
    
        self._wait_for_server()
        self._initialize_kernels()
    
        self.notebook_start_time = time.time()
        self.problems_remaining = 50
    
    def _preload_model_weights(self) -> None:
    
        print(f'Loading model weights from {self.cfg.model_path} into OS Page Cache...')
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
    
            with open(path, 'rb') as file_object:
                while file_object.read(1024 * 1024 * 1024):
                    pass
    
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            list(executor.map(_read_file, files_to_load))
    
        elapsed = time.time() - start_time
        print(f'Processed {len(files_to_load)} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n')
    
    def _start_server(self) -> subprocess.Popen:
    
        cmd = [
            sys.executable, 
            '-m', 
            'vllm.entrypoints.openai.api_server', 
            '--seed', 
            str(self.cfg.seed), 
            '--model', 
            self.cfg.model_path, 
            '--served-model-name', 
            self.cfg.served_model_name, 
            '--tensor-parallel-size', 
            '1', 
            '--max-num-seqs', 
            str(self.cfg.batch_size), 
            '--gpu-memory-utilization', 
            str(self.cfg.gpu_memory_utilization), 
            '--host', 
            '0.0.0.0', 
            '--port', 
            str(self.port), 
            '--dtype', 
            self.cfg.dtype, 
            '--kv-cache-dtype', 
            self.cfg.kv_cache_dtype, 
            '--max-model-len', 
            str(self.cfg.context_tokens), 
            '--stream-interval', 
            str(self.cfg.stream_interval), 
            '--async-scheduling', 
            '--disable-log-stats', 
            '--enable-prefix-caching'
        ]
    
        self.log_file = open('vllm_server.log', 'w')
    
        return subprocess.Popen(
            cmd, 
            stdout=self.log_file, 
            stderr=subprocess.STDOUT, 
            start_new_session=True
        )
    
    def _wait_for_server(self):
    
        print('Waiting for vLLM server...')
        start_time = time.time()
    
        for _ in range(self.cfg.server_timeout):
            return_code = self.server_process.poll()
    
            if return_code is not None:
                self.log_file.flush()
    
                with open('vllm_server.log', 'r') as log_file:
                    logs = log_file.read()
    
                raise RuntimeError(f'Server died with code {return_code}. Full logs:\n{logs}\n')
    
            try:
                self.client.models.list()
                elapsed = time.time() - start_time
                print(f'Server is ready (took {elapsed:.2f} seconds).\n')
    
                return
    
            except Exception:
                time.sleep(1)
    
        raise RuntimeError('Server failed to start (timeout).\n')
    
    def _initialize_kernels(self) -> None:
    
        print(f'Initializing {self.cfg.workers} persistent Jupyter kernels...')
        start_time = time.time()
    
        self.sandbox_pool = queue.Queue()
    
        def _create_sandbox():
            
            return AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)
    
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            futures = [executor.submit(_create_sandbox) for _ in range(self.cfg.workers)]
    
            for future in as_completed(futures):
                self.sandbox_pool.put(future.result())
    
        elapsed = time.time() - start_time
        print(f'Kernels initialized in {elapsed:.2f} seconds.\n')
    
    def _scan_for_answer(self, text: str) -> int | None:
        
        pattern = r'\\boxed\s*\{\s*([0-9,]+)\s*\}'
        matches = re.findall(pattern, text)
    
        if matches:
            try:
                clean_value = matches[-1].replace(',', '')
                value = int(clean_value)
    
                if 0 <= value <= 99999:
                    return value
    
            except ValueError:
                pass
                
        pattern = r'final\s+answer\s+is\s*([0-9,]+)'
        matches = re.findall(pattern, text, re.IGNORECASE)
    
        if matches:
            try:
                clean_value = matches[-1].replace(',', '')
                value = int(clean_value)
    
                if 0 <= value <= 99999:
                    return value
    
            except ValueError:
                pass
    
        return None
    
    def _compute_mean_entropy(self, logprobs_buffer: list) -> float:
    
        if not logprobs_buffer:
            return float('inf')
    
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
            return float('inf')
    
        return total_entropy / token_count
    
    def _process_attempt(
        self,
        problem: str,
        system_prompt: str,
        attempt_index: int,
        stop_event: threading.Event,
        deadline: float
    ) -> dict:
    
        # Always define these so we can safely return them
        text_chunks: list[str] = []
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
                "Raw Output": "",
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
                "Raw Output": "",
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
                sandbox=sandbox
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
                        "return_token_ids": True
                    }
                )
    
                token_buffer = []
                try:
                    for chunk in stream:
                        if stop_event.is_set():
                            finish_reason = "stopped_by_early_stop"
                            break
                        if time.time() > deadline:
                            finish_reason = "deadline_exceeded"
                            break
    
                        new_tokens = chunk.choices[0].token_ids
                        new_text = chunk.choices[0].text or ""
    
                        if new_tokens:
                            token_buffer.extend(new_tokens)
                            total_tokens += len(new_tokens)
                            text_chunks.append(new_text)
    
                            chunk_logprobs = chunk.choices[0].logprobs
                            if chunk_logprobs is not None and chunk_logprobs.top_logprobs:
                                logprobs_buffer.extend(chunk_logprobs.top_logprobs)
    
                        # Fast scan when '}' appears (heuristic)
                        if "}" in new_text:
                            search_text = "".join(text_chunks[-self.cfg.search_tokens:])
                            answer = self._scan_for_answer(search_text)
                            if answer is not None:
                                final_answer = answer
                                finish_reason = "boxed_detected_in_stream"
                                break
                finally:
                    stream.close()
    
                if final_answer is not None:
                    break
    
                if not token_buffer:
                    # No tokens produced this turn
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
                    # Capture the code the model requested
                    raw_script = last_message.content[0].text
                    final_script = local_tool._ensure_last_print(raw_script)
    
                    tool_responses = local_tool.process_sync_plus(last_message)
                    response_text = tool_responses[0].content[0].text
    
                    tool_calls.append({
                        "code": final_script,
                        "output": response_text
                    })
    
                    if response_text.startswith("[ERROR]") or "Traceback" in response_text or "Error:" in response_text:
                        python_errors += 1
    
                    conversation.messages.extend(tool_responses)
    
            if finish_reason == "unknown":
                finish_reason = "max_turns_or_no_answer"
    
        except Exception as exc:
            python_errors += 1
            finish_reason = f"exception:{type(exc).__name__}\n{exc}"
    
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
            "Raw Output": "".join(text_chunks),
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
    
        scored = [
            {"answer": a, "votes": answer_votes[a], "score": w}
            for a, w in answer_weights.items()
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
    
        vote_df = pd.DataFrame(scored) if scored else pd.DataFrame(columns=["answer", "votes", "score"])
    
        if not scored:
            return 0, None, vote_df
    
        final_answer = scored[0]["answer"]
    
        # Choose a representative "selected attempt":
        # among attempts that produced final_answer, pick minimal entropy (most confident).
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
    
        # Use actual remaining problem count if you can (we’ll set it after solver init)
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
                    deadline
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
    
        # Decide final answer and representative attempt
        final_answer, selected_idx, vote_df = self._select_answer(detailed_results)
    
        artifact = {
            "budget_seconds": float(budget),
            "deadline_ts": float(deadline),
            "final_answer": int(final_answer),
            "selected_attempt_index": selected_idx,  # 0-based index into detailed_results
            "vote_df": vote_df,
            "attempts": detailed_results,
        }
    
        return final_answer, artifact

    
    def __del__(self):
    
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
            self.server_process.wait()
    
        if hasattr(self, 'log_file'):
            self.log_file.close()
    
        if hasattr(self, 'sandbox_pool'):
            while not self.sandbox_pool.empty():
                try:
                    sb = self.sandbox_pool.get_nowait()
                    sb.close()
    
                except Exception:
                    pass

# %%
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    REFERENCE_PATH = "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv"
    _ref_df = pl.read_csv(REFERENCE_PATH)
    # Expect columns: id, question, answer (per your description)
    REF_ANSWER_BY_ID = dict(zip(_ref_df["id"].to_list(), _ref_df["answer"].to_list()))
    REF_QUESTION_BY_ID = dict(zip(_ref_df["id"].to_list(), _ref_df["problem"].to_list()))
    TOTAL_PROBLEMS = _ref_df.height
else:
    REF_ANSWER_BY_ID = {}
    REF_QUESTION_BY_ID = {}
    TOTAL_PROBLEMS = 50

# %%
def predict(id_: pl.DataFrame, question: pl.DataFrame, answer: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    id_value = int(id_.item(0))
    question_text = str(question.item(0))

    # Ground-truth answer (prefer what the gateway passes; else fallback to preloaded reference)
    true_answer = None
    if answer is not None:
        try:
            true_answer = int(answer.item(0))
        except Exception:
            true_answer = None
    if true_answer is None and id_value in REF_ANSWER_BY_ID:
        try:
            true_answer = int(REF_ANSWER_BY_ID[id_value])
        except Exception:
            true_answer = None

    gc.disable()
    pred_answer, artifact = solver.solve_problem(question_text)
    gc.enable()
    gc.collect()

    attempts = artifact["attempts"]
    selected_idx = artifact.get("selected_attempt_index", None)

    # Identify the selected attempt record (raw solution text) if possible
    selected_raw = ""
    selected_entropy = None
    if selected_idx is not None and 0 <= selected_idx < len(attempts):
        selected_raw = attempts[selected_idx].get("Raw Output", "")
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

        attempt_records.append({
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
            "raw_output": r.get("Raw Output", ""),
            "tool_calls": r.get("Tool Calls", []),
        })

    # Persist logs (thread-safe)
    logger.log_attempts(attempt_records)
    logger.log_solution_row(
        id_value=id_value,
        pred_answer=int(pred_answer),
        true_answer=true_answer,
        is_correct=is_correct,
        selected_attempt=(attempts[selected_idx].get("Attempt") if selected_idx is not None and 0 <= selected_idx < len(attempts) else None),
        selected_entropy=selected_entropy,
        selected_raw_output=selected_raw
    )

    return pl.DataFrame({"id": id_value, "answer": int(pred_answer)})


# %%
inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

# %%
solver = AIMO3Solver(CFG)
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    solver.problems_remaining = TOTAL_PROBLEMS

# %%
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        ('/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv',)
    )

# %%
!zip -r /kaggle/working/aimo3_logs.zip /kaggle/working/aimo3_logs


