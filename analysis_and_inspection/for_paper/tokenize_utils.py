"""
Shared tokenization and data-loading helpers for paper analysis scripts.

Two session types:
- Goedel: tokenize the last round's (prompt + raw_output) — peak context window.
- TIR: concatenate all conversation text (thinking, content, tool calls) and tokenize.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm


def turns_to_chat(
    turns: list[dict[str, Any]],
    *,
    exclude_last_tool_call_result: bool = False,
) -> list[dict[str, Any]]:
    """Convert TIR turn records into OpenAI-style chat messages.

    Each turn becomes an assistant message (with optional ``reasoning_content``
    and ``tool_calls``), followed by one ``tool`` message per tool call.

    Parameters
    ----------
    turns:
        List of turn dicts as stored in ``proof_result.turns``
        (schema: ``{turn, thinking, content, tool_calls}``).
    exclude_last_tool_call_result:
        If True, omit the ``tool`` result message for tool calls in the last
        turn (mirrors the old ``exclude_last_tool_call_result`` flag).
    """
    messages: list[dict[str, Any]] = []
    n_turns = len(turns)
    for ti, t in enumerate(turns):
        thinking = t.get("thinking", "") or ""
        content = t.get("content", "") or ""
        tool_calls_raw = t.get("tool_calls", [])

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": content or None,
        }
        if thinking:
            assistant_msg["reasoning_content"] = thinking

        if tool_calls_raw:
            assistant_msg["tool_calls"] = [
                {
                    "id": f"call_{ti}_{ci}",
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": (
                            json.dumps(tc["arguments"])
                            if isinstance(tc.get("arguments"), dict)
                            else str(tc.get("arguments", ""))
                        ),
                    },
                }
                for ci, tc in enumerate(tool_calls_raw)
            ]

        messages.append(assistant_msg)

        # Append tool result messages.
        is_last_turn = ti == n_turns - 1
        for ci, tc in enumerate(tool_calls_raw):
            if is_last_turn and exclude_last_tool_call_result:
                continue
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{ti}_{ci}",
                "name": tc.get("name", ""),
                "content": tc.get("result", "") or "",
            })

    return messages


def goedel_session_tokens(
    results_path: Path,
    tokenizer,
    keys: set[tuple[str, int]] | None = None,
) -> dict[tuple[str, int], int]:
    """
    For each (problem_id, attempt), tokenize the last round's prompt + output.

    If *keys* is provided, only tokenize those (problem_id, attempt) pairs.
    """
    result = {}
    with results_path.open() as f:
        f_lines = f.readlines()
        p = tqdm(total=len(keys) if keys is not None else len(f_lines))
        for line in f_lines:
            if not line.strip():
                continue
            d = json.loads(line)
            pid, att = d["problem_id"], d["attempt"]
            if keys is not None and (pid, att) not in keys:
                continue
            p.update(1)
            pr = d.get("proof_result")
            if not pr:
                continue
            rounds = pr.get("rounds") or []
            if not rounds:
                continue
            last = rounds[-1]
            prompt = last.get("prompt", "") or ""
            output = last.get("raw_output", "") or ""
            text = prompt + output
            tokens = len(tokenizer.encode(text, add_special_tokens=False))
            result[(pid, att)] = tokens
        p.close()
    return result


def _get_best_proof_result(d: dict) -> dict | None:
    """Return the best proof result from a v1 or v2 record."""
    all_pr = d.get("all_proof_results")
    if all_pr and len(all_pr) > 0:
        return all_pr[-1]
    return d.get("proof_result")


def tir_session_tokens(
    results_path: Path,
    tokenizer,
    keys: set[tuple[str, int]] | None = None,
    exclude_last_tool_call_result=False,
) -> dict[tuple[str, int], int]:
    """
    For each (problem_id, attempt), tokenize the conversation via
    ``apply_chat_template``.

    If *keys* is provided, only tokenize those (problem_id, attempt) pairs.
    """
    result = {}
    with results_path.open() as f:
        f_lines = f.readlines()
        p = tqdm(total=len(keys) if keys is not None else len(f_lines))
        for line in f_lines:
            if not line.strip():
                continue
            d = json.loads(line)
            pid, att = d["problem_id"], d["attempt"]
            if keys is not None and (pid, att) not in keys:
                continue
            p.update(1)
            pr = _get_best_proof_result(d)
            if not pr:
                result[(pid, att)] = 0
                continue

            # v2: conversation_history inside session_result
            session = pr.get("session_result") or {}
            history = session.get("conversation_history")
            if history:
                msgs = list(history)
                if exclude_last_tool_call_result:
                    # Remove the last tool result message(s)
                    while msgs and msgs[-1].get("role") == "tool":
                        msgs.pop()
                tokens = tokenizer.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=False,
                )
                result[(pid, att)] = len(tokens)
                continue

            # v1 fallback: use turns
            turns = pr.get("turns") or []
            if not turns:
                result[(pid, att)] = 0
                continue
            # NOTE: V1 schema hack — no system/user messages available
            parts = [d.get("proved_lean", "") or ""]
            for t in turns:
                parts.append(t.get("thinking", "") or "")
                parts.append(t.get("content", "") or "")
                for tc in t.get("tool_calls", []):
                    args = tc.get("arguments", {})
                    if isinstance(args, dict):
                        parts.append(json.dumps(args))
                    else:
                        parts.append(str(args))
                    parts.append(tc.get("result", "") or "")
            text = "\n".join(parts)
            tokens = len(tokenizer.encode(text, add_special_tokens=False))
            result[(pid, att)] = tokens
        p.close()
    return result


def _proof_from_theorem(text: str) -> str:
    """Return the substring starting at the first 'theorem' keyword."""
    idx = text.find("theorem")
    return text[idx:] if idx >= 0 else text


def goedel_proof_tokens(
    results_path: Path,
    tokenizer,
    keys: set[tuple[str, int]] | None = None,
) -> dict[tuple[str, int], int]:
    """
    For each (problem_id, attempt), tokenize the last round's proof_text
    from the 'theorem' keyword onward.

    If *keys* is provided, only process those (problem_id, attempt) pairs.
    """
    result = {}
    with results_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            pid, att = d["problem_id"], d["attempt"]
            if keys is not None and (pid, att) not in keys:
                continue
            pr = d.get("proof_result")
            if not pr:
                continue
            proof_text = pr.get("proof_text", "")
            text = _proof_from_theorem(proof_text)
            result[(pid, att)] = len(tokenizer.encode(text, add_special_tokens=False))
    return result


def tir_proof_tokens(
    results_path: Path,
    tokenizer,
    keys: set[tuple[str, int]] | None = None,
) -> dict[tuple[str, int], int]:
    """
    For each (problem_id, attempt), tokenize the last lean_final tool call's
    code from the 'theorem' keyword onward.

    Falls back to the last 'lean' tool call if no lean_final exists.
    If *keys* is provided, only process those (problem_id, attempt) pairs.
    """
    result = {}
    with results_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            pid, att = d["problem_id"], d["attempt"]
            if keys is not None and (pid, att) not in keys:
                continue
            pr = _get_best_proof_result(d)
            if not pr:
                continue

            last_lean_final = ""
            last_lean = ""

            # v2: conversation_history inside session_result
            session = pr.get("session_result") or {}
            history = session.get("conversation_history")
            if history:
                for msg in history:
                    for tc in msg.get("tool_calls", []):
                        name = tc.get("function", {}).get("name", "")
                        args_str = tc.get("function", {}).get("arguments", "")
                        try:
                            args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                        code = args.get("code", "") if isinstance(args, dict) else str(args)
                        if name == "lean_final":
                            last_lean_final = code
                        elif name == "lean":
                            last_lean = code
            else:
                # v1 fallback: use turns
                turns = pr.get("turns") or []
                for t in turns:
                    for tc in t.get("tool_calls", []):
                        name = tc.get("name", "")
                        args = tc.get("arguments", {})
                        code = args.get("code", "") if isinstance(args, dict) else str(args)
                        if name == "lean_final":
                            last_lean_final = code
                        elif name == "lean":
                            last_lean = code

            code = last_lean_final or last_lean
            if not code:
                continue
            text = _proof_from_theorem(code)
            result[(pid, att)] = len(tokenizer.encode(text, add_special_tokens=False))
    return result


def load_proved_status(path: Path) -> dict[tuple[str, int], bool]:
    results = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            results[(d["problem_id"], d["attempt"])] = bool(d.get("proved"))
    return results


def aggregate_per_problem(
    tokens_map: dict[tuple[str, int], int],
    proved_map: dict[tuple[str, int], bool],
) -> dict[str, tuple[bool, int]]:
    """
    Aggregate (problem_id, attempt) → per-problem (solved, representative_tokens).

    solved = any attempt proved.
    representative_tokens = min tokens among successful attempts if solved,
                            else min tokens among all attempts.
    """
    by_pid: dict[str, list[tuple[int, bool]]] = {}
    for (pid, att), toks in tokens_map.items():
        proved = proved_map.get((pid, att), False)
        by_pid.setdefault(pid, []).append((toks, proved))

    result = {}
    for pid, entries in by_pid.items():
        success = [(t, p) for t, p in entries if p]
        if success:
            result[pid] = (True, min(t for t, _ in success))
        else:
            result[pid] = (False, min(t for t, _ in entries))
    return result
