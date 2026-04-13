#!/usr/bin/env python3
"""
Benchmark concurrent Ollama throughput.
Displays per-request chars/s (rolling window) and total chars/s (since start)
on a single updating line in real time.

Usage:
    python benchmark_ollama.py [-p <parallelism>] [-m <model>] [-n <prompt>] [-w <window_s>]
"""
import argparse
import collections
import sys
import threading
import time

import ollama


def run_request(idx: int, model: str, prompt: str, states: list, lock: threading.Lock) -> None:
    with lock:
        states[idx] = {"status": "running", "chars": 0, "chunks": collections.deque()}

    try:
        client = ollama.Client()
        for part in client.generate(model=model, prompt=prompt, stream=True):
            chunk = part.response or ""
            if chunk:
                now = time.time()
                with lock:
                    s = states[idx]
                    s["chars"] += len(chunk)
                    s["chunks"].append((now, len(chunk)))

            if part.done:
                eval_count = part.eval_count
                eval_duration_ns = part.eval_duration
                tok_s = (
                    eval_count / (eval_duration_ns / 1e9)
                    if eval_count and eval_duration_ns
                    else None
                )
                with lock:
                    states[idx] = {
                        "status": "done",
                        "tok_s": tok_s,
                        "tokens": eval_count or 0,
                        "chars": states[idx]["chars"],
                    }
                return

        with lock:
            states[idx] = {"status": "done", "tok_s": None, "tokens": 0, "chars": states[idx]["chars"]}

    except Exception as exc:
        with lock:
            states[idx] = {"status": "error", "error": str(exc)}


def window_chars_s(chunks: collections.deque, now: float, window: float) -> float | None:
    cutoff = now - window
    # drop old entries
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
        tok_s = s.get("tok_s")
        return f"{tok_s:.1f} tok/s" if tok_s is not None else "? tok/s"
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
    parser = argparse.ArgumentParser(description="Benchmark concurrent Ollama throughput")
    parser.add_argument("-p", "--parallelism", type=int, default=1)
    parser.add_argument("-m", "--model", default="goedel-v2")
    parser.add_argument("-n", "--prompt", default="hi")
    parser.add_argument("-w", "--window", type=float, default=10.0,
                        help="Rolling window in seconds for per-request chars/s (default: 10)")
    args = parser.parse_args()

    states = [{"status": "pending"} for _ in range(args.parallelism)]
    lock = threading.Lock()

    print(f"Model:       {args.model}")
    print(f"Parallelism: {args.parallelism}")
    print(f"Prompt:      {args.prompt}")
    print(f"Window:      {args.window}s")
    print()

    threads = [
        threading.Thread(
            target=run_request,
            args=(i, args.model, args.prompt, states, lock),
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
        errors = [(i, s) for i, s in enumerate(states) if s["status"] == "error"]

    sys.stdout.write(line + "\n\n")
    sys.stdout.flush()

    print(f"Wall-clock time:   {elapsed:.2f}s")
    print(f"Total eval tokens: {total_tokens}")
    if total_tokens > 0:
        print(f"Aggregate tok/s:   {total_tokens / elapsed:.1f}")
    for i, s in errors:
        print(f"Request {i + 1} error: {s.get('error')}")


if __name__ == "__main__":
    main()
