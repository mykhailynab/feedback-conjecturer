#!/usr/bin/env python3
"""
Benchmark concurrent llama.cpp server throughput.
Displays per-request chars/s (rolling window) and total chars/s (since start)
on a single updating line in real time.

Usage:
    python benchmark_llamacpp.py [-p <parallelism>] [-u <base_url>] [-m <model>] [-n <prompt>] [-w <window_s>]
"""
import argparse
import collections
import sys
import threading
import time

import openai


def run_request(idx: int, client: openai.OpenAI, model: str, prompt: str,
                states: list, lock: threading.Lock) -> None:
    with lock:
        states[idx] = {"status": "running", "chars": 0, "tokens": 0,
                        "chunks": collections.deque()}

    try:
        stream = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=4096,
            stream=True,
            stream_options={"include_usage": True},
        )
        for part in stream:
            if part.usage is not None:
                with lock:
                    states[idx]["tokens"] = part.usage.completion_tokens
            if part.choices:
                chunk = part.choices[0].text or ""
                if chunk:
                    now = time.time()
                    with lock:
                        s = states[idx]
                        s["chars"] += len(chunk)
                        s["chunks"].append((now, len(chunk)))

        with lock:
            s = states[idx]
            s["status"] = "done"

    except Exception as exc:
        with lock:
            states[idx] = {"status": "error", "error": str(exc)}


def window_chars_s(chunks: collections.deque, now: float, window: float) -> float | None:
    cutoff = now - window
    while chunks and chunks[0][0] < cutoff:
        chunks.popleft()
    if not chunks:
        return None
    total = sum(n for _, n in chunks)
    span = now - chunks[0][0]
    return total / span if span > 0 else None


def fmt_slot(s: dict, now: float, window: float) -> str:
    status = s["status"]
    if status == "done":
        return f"{s.get('chars', 0)} ch"
    if status == "running":
        chunks = s.get("chunks")
        if chunks:
            cs = window_chars_s(chunks, now, window)
            if cs is not None:
                return f"{cs:.0f} ch/s"
        return "waiting..."
    if status == "error":
        return "ERR"
    return "pending"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark concurrent llama.cpp server throughput")
    parser.add_argument("-p", "--parallelism", type=int, default=1)
    parser.add_argument("-u", "--base-url", default="http://localhost:8001/v1")
    parser.add_argument("-m", "--model", default="unsloth/Qwen3.6-35B-A3B")
    parser.add_argument("-n", "--prompt", default="hi")
    parser.add_argument("-w", "--window", type=float, default=10.0,
                        help="Rolling window in seconds for per-request chars/s (default: 10)")
    args = parser.parse_args()

    client = openai.OpenAI(base_url=args.base_url, api_key="no-key")

    states = [{"status": "pending"} for _ in range(args.parallelism)]
    lock = threading.Lock()

    print(f"Base URL:    {args.base_url}")
    print(f"Model:       {args.model}")
    print(f"Parallelism: {args.parallelism}")
    print(f"Prompt:      {args.prompt}")
    print(f"Window:      {args.window}s")
    print()

    threads = [
        threading.Thread(
            target=run_request,
            args=(i, client, args.model, args.prompt, states, lock),
            daemon=True,
        )
        for i in range(args.parallelism)
    ]
    wall_start = time.time()
    for t in threads:
        t.start()

    def total_chars_s(now: float) -> str:
        total_chars = sum(s.get("chars", 0) for s in states)
        elapsed = now - wall_start
        if total_chars > 0 and elapsed > 0:
            return f"{total_chars / elapsed:.0f} ch/s total"
        return ""

    def render(now: float) -> str:
        slots = [fmt_slot(s, now, args.window) for s in states]
        summary = total_chars_s(now)
        elapsed = now - wall_start
        return f"\r[{elapsed:.1f}s]  [{'  |  '.join(slots)}]  {summary}    "

    while any(t.is_alive() for t in threads):
        now = time.time()
        with lock:
            line = render(now)
        sys.stdout.write(line)
        sys.stdout.flush()
        time.sleep(0.2)

    now = time.time()
    elapsed = now - wall_start
    with lock:
        line = render(now)
        total_tokens = sum(s.get("tokens", 0) for s in states if s["status"] == "done")
        total_chars = sum(s.get("chars", 0) for s in states if s["status"] == "done")
        errors = [(i, s) for i, s in enumerate(states) if s["status"] == "error"]

    sys.stdout.write(line + "\n\n")
    sys.stdout.flush()

    print(f"Wall-clock time:   {elapsed:.2f}s")
    print(f"Total tokens:      {total_tokens}")
    print(f"Total chars:       {total_chars}")
    if total_tokens > 0:
        print(f"Aggregate tok/s:   {total_tokens / elapsed:.1f}")
    if total_chars > 0:
        print(f"Aggregate ch/s:    {total_chars / elapsed:.0f}")
    for i, s in errors:
        print(f"Request {i + 1} error: {s.get('error')}")


if __name__ == "__main__":
    main()
