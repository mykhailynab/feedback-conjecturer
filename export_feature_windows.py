#!/usr/bin/env python3
import os
import time
import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def get_sequences_from_attempt_row(row, which_turn="all") -> List[Tuple[List[int], int, Dict[str, Any]]]:
    """
    Returns list of (full_token_ids, prompt_len, meta) from one attempt row.
    meta includes attempt id, turn index, etc.
    """
    trace = row.get("trace") or row.get("Trace")
    if not trace:
        return []
    turns = trace.get("turns") or []
    if not turns:
        return []

    if which_turn == "last":
        chosen = [(len(turns) - 1, turns[-1])]
    elif which_turn == "all":
        chosen = list(enumerate(turns))
    else:
        idx = int(which_turn)
        if idx < 0 or idx >= len(turns):
            return []
        chosen = [(idx, turns[idx])]

    out = []
    for turn_idx, t in chosen:
        prompt_ids = t.get("prompt_token_ids") or []
        comp_ids = t.get("completion_token_ids") or []
        if not prompt_ids or not comp_ids:
            continue
        full = list(prompt_ids) + list(comp_ids)
        meta = {
            "attempt_id": row.get("id"),
            "turn_index": turn_idx,
            "prompt_len": len(prompt_ids),
            "full_len": len(full),
        }
        out.append((full, len(prompt_ids), meta))
    return out

def pad_batch(seqs: List[List[int]], pad_id: int, device: torch.device):
    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
    seq_lengths = []
    for i, s in enumerate(seqs):
        seq_length = len(s)
        seq_lengths.append(seq_length)
        input_ids[i, :seq_length] = torch.tensor(s, dtype=torch.long, device=device)
        attn[i, :seq_length] = 1
    return input_ids, attn, seq_lengths

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


# ---------------- Feature-window selection ----------------

def find_active_segment_around_peak(
    a: np.ndarray,
    peak_i: int,
    *,
    thr: float,
    min_width: int,
    max_width: int,
    buffer: int,
) -> Tuple[int, int]:
    """
    Given 1D activations `a` for a feature over tokens (completion-only),
    pick an active window [L, R) around peak_i.

    Strategy:
      - start at peak_i, expand left/right while a > thr
      - enforce min_width
      - then add `buffer` tokens on each side (clamped)
      - clamp to max_width around peak if needed
    """
    n = a.shape[0]
    left_i = peak_i
    right_i = peak_i + 1

    # Expand to contiguous region above threshold
    while left_i - 1 >= 0 and a[left_i - 1] > thr:
        left_i -= 1
    while right_i < n and a[right_i] > thr:
        right_i += 1

    # Enforce min width (centered near peak)
    if (right_i - left_i) < min_width:
        half_w = min_width // 2
        left_i = max(0, peak_i - half_w)
        right_i = min(n, left_i + min_width)
        left_i = max(0, right_i - min_width)

    # Add buffer
    left_i_buf = max(0, left_i - buffer)
    right_i_buf = min(n, right_i + buffer)

    # Clamp max width (keep peak near middle)
    if (right_i_buf - left_i_buf) > max_width:
        half_w = max_width // 2
        left_i_buf = max(0, peak_i - half_w)
        right_i_buf = min(n, left_i_buf + max_width)
        left_i_buf = max(0, right_i_buf - max_width)

    return left_i_buf, right_i_buf

def decode_with_char_spans(tok: AutoTokenizer, token_ids: List[int]) -> Tuple[str, List[Tuple[int,int,str]]]:
    """
    Returns (text, token_char_spans) where token_char_spans is
    [(char_start, char_end, token_text)] for each token in order.
    Uses per-token decoding; robust for later highlighting.
    """
    pieces = []
    spans = []
    cur = 0
    for tid in token_ids:
        s = tok.decode([tid], clean_up_tokenization_spaces=False)
        pieces.append(s)
        start = cur
        cur += len(s)
        end = cur
        spans.append((start, end, s))
    return "".join(pieces), spans

def spans_for_active_tokens(
    token_spans: List[Tuple[int,int,str]],
    active_token_idx: List[int],
) -> Tuple[int,int]:
    """
    Convert a set of active token positions (indices in window) into a single
    char span [start,end] that covers from first to last active token.
    """
    if not active_token_idx:
        return (0, 0)
    i0 = min(active_token_idx)
    i1 = max(active_token_idx)
    start = token_spans[i0][0]
    end = token_spans[i1][1]
    return (start, end)


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--reference_path", type=str, required=True)
    ap.add_argument("--features", type=str, required=True,
                    help="Comma-separated feature ids, or path to txt with one id per line.")
    ap.add_argument("--out_json", type=str, default="feature_windows.json")

    ap.add_argument("--only_correct", action="store_true")
    ap.add_argument("--only_with_answers", action="store_true")
    ap.add_argument("--only_selected", action="store_true")

    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--turn", type=str, default="all")
    ap.add_argument("--max_attempts", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1)

    ap.add_argument("--max_tokens", type=int, default=(65536 - 500),
                    help="Max token cap per single turn (truncate each sequence).")
    ap.add_argument("--max_examples_per_feature", type=int, default=100)

    ap.add_argument("--activation_threshold", type=float, default=0.0,
                    help="Token considered 'active' if act > threshold.")
    ap.add_argument("--min_active_width", type=int, default=1,
                    help="Min width for the active region before adding buffer.")
    ap.add_argument("--max_window_width", type=int, default=256,
                    help="Max tokens returned in a snippet (including buffer).")
    ap.add_argument("--buffer_tokens", type=int, default=64,
                    help="Add this many tokens on each side of active region.")

    ap.add_argument("--per_feature_topk_candidates", type=int, default=5000,
                    help="Keep only the top-K peaks per feature globally (speed/memory).")

    ap.add_argument("--include_token_debug", action="store_true",
                    help="Include token ids/text/char spans for debugging/frontend refinement.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    # Parse features list
    feat_list: List[int] = []
    prompt_length = Path(args.features)
    if prompt_length.exists():
        for line in prompt_length.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                feat_list.append(int(line))
    else:
        feat_list = [int(x.strip()) for x in args.features.split(",") if x.strip()]

    feat_list = sorted(set(feat_list))
    if not feat_list:
        raise ValueError("No features specified.")

    feature_set = set(feat_list)

    _preload_model_weights(args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
        dtype="auto",
        attn_implementation="kernels-community/vllm-flash-attn3",
    )
    model.eval()

    # Input tensor device (for device_map="auto" on 1xH100 typically cuda:0)
    model_device = getattr(model, "device", None)
    model_device = torch.device(model_device) if model_device is not None else torch.device("cuda")

    sae = SAE.from_pretrained(
        release="gpt-oss-20b-andyrdt",
        sae_id="resid_post_layer_19_trainer_0",
        device="cuda",
    )

    # Hook layer
    try:
        layer_module = model.model.layers[args.layer]
    except Exception as e:
        raise RuntimeError(f"Could not access model.model.layers[{args.layer}]: {e}")

    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(mod, inp, out):
        hs = out[0] if isinstance(out, (tuple, list)) else out
        # detach so we don't hold onto graph / buffer refs
        captured["hs"] = hs.detach()

    handle = layer_module.register_forward_hook(hook_fn)

    # Load reference for filtering
    reference_csv = pd.read_csv(args.reference_path)

    # We will maintain a per-feature list of best examples:
    # entries are dicts with:
    #   score (peak activation), feature_id, attempt_id, turn_index, global metadata,
    #   window token ids, window activations (for that feature), etc.
    per_feat_examples: Dict[int, List[Dict[str, Any]]] = {f: [] for f in feat_list}

    def maybe_push_example(f: int, ex: Dict[str, Any]):
        lst = per_feat_examples[f]
        lst.append(ex)
        # Keep bounded by candidates, then prune to top by score
        if len(lst) > args.per_feature_topk_candidates:
            lst.sort(key=lambda d: d["score"], reverse=True)
            del lst[args.per_feature_topk_candidates:]

    attempts_path = Path(args.attempts)

    seqs: List[List[int]] = []
    prompt_lengths: List[int] = []
    metas: List[Dict[str, Any]] = []

    used_attempts = 0
    for attempt in iter_jsonl(attempts_path):
        if args.only_with_answers and attempt.get("attempt_answer") is None:
            continue

        # reference answer lookup (assumes unique id)
        ref_row = reference_csv[reference_csv["id"] == attempt["id"]]
        if len(ref_row) == 0:
            continue
        reference_answer = ref_row["answer"].iloc[0]
        if args.only_correct and attempt.get("attempt_answer") != reference_answer:
            continue
        if args.only_selected and attempt.get("status") != "selected":
            continue

        if used_attempts >= args.max_attempts:
            break

        extracted = get_sequences_from_attempt_row(attempt, which_turn=args.turn)
        if not extracted:
            continue

        for full_token_ids, prompt_length, meta in extracted:
            # per-turn truncation
            full_token_ids = full_token_ids[: min(len(full_token_ids), args.max_tokens)]
            prompt_length = min(prompt_length, len(full_token_ids))
            meta = dict(meta)
            meta.update({
                "status": attempt.get("status"),
                "attempt_answer": attempt.get("attempt_answer"),
                "reference_answer": reference_answer,
            })
            seqs.append(full_token_ids)
            prompt_lengths.append(prompt_length)
            metas.append(meta)

        used_attempts += 1

    if not seqs:
        raise RuntimeError("No usable sequences found after filtering.")

    # Main pass: run sequences, compute SAE on completion tokens, update per-feature best windows.
    with torch.no_grad():
        for start in tqdm(range(0, len(seqs), args.batch_size), desc="Scanning tokens"):
            batch_seqs = seqs[start:start+args.batch_size]
            batch_prompt_lengths = prompt_lengths[start:start+args.batch_size]
            batch_meta = metas[start:start+args.batch_size]

            input_ids, attention_mask, pad_seq_lengths = pad_batch(batch_seqs, tokenizer.pad_token_id, device=model_device)

            captured.clear()
            _ = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            hs = captured.pop("hs", None)
            if hs is None:
                raise RuntimeError("Hook failed to capture hs.")

            # Process each item separately to keep bookkeeping simple
            for bi in range(hs.shape[0]):
                seq_length = int(pad_seq_lengths[bi])
                prompt_length = int(batch_prompt_lengths[bi])
                meta = batch_meta[bi]

                if prompt_length >= seq_length:
                    continue

                # completion tokens only
                comp_hs = hs[bi, prompt_length:seq_length, :]  # [Tcomp, d_model]
                if comp_hs.numel() == 0:
                    continue

                completion_token_ids = batch_seqs[bi][prompt_length:seq_length]  # python list length Tcomp

                # SAE encode
                feat_acts = sae.encode(comp_hs.to("cuda"))  # [Tcomp, n_feat]
                # Pull only features of interest
                # (feat_acts can be float16/bf16; convert to float32 for robust peak)
                feat_acts_selected = feat_acts[:, feat_list].float().detach().cpu().numpy()  # [Tcomp, K]

                # For each feature, find local peaks and push best window(s)
                for k, f in enumerate(feat_list):
                    feat_acts_for_feature = feat_acts_selected[:, k]  # [Tcomp]
                    if args.activation_threshold > 0.0:
                        if not np.any(feat_acts_for_feature > args.activation_threshold):
                            continue
                    else:
                        if not np.any(feat_acts_for_feature != 0):
                            continue

                    peak_i = int(np.argmax(feat_acts_for_feature))
                    peak_v = float(feat_acts_for_feature[peak_i])

                    # Derive snippet window (token indices in completion-space)
                    left_win_i, right_win_i = find_active_segment_around_peak(
                        feat_acts_for_feature,
                        peak_i,
                        thr=args.activation_threshold,
                        min_width=args.min_active_width,
                        max_width=args.max_window_width,
                        buffer=args.buffer_tokens,
                    )

                    # Identify active tokens within the window (for highlight)
                    feat_acts_window = feat_acts_for_feature[left_win_i:right_win_i]
                    if args.activation_threshold > 0.0:
                        active_i_rel_window = np.where(feat_acts_window > args.activation_threshold)[0].tolist()
                    else:
                        active_i_rel_window = np.where(feat_acts_window != 0)[0].tolist()

                    # If no active tokens survived (possible with threshold and peak on boundary)
                    if not active_i_rel_window:
                        # fallback: highlight just the peak token
                        active_i_rel_window = [peak_i - left_win_i] if (left_win_i <= peak_i < right_win_i) else [0]

                    window_token_ids = completion_token_ids[left_win_i:right_win_i]

                    # Decode snippet + compute char spans
                    text, token_char_spans = decode_with_char_spans(tokenizer, window_token_ids)

                    # Compute char span to highlight (cover contiguous min..max active token)
                    char_start, char_end = spans_for_active_tokens(token_char_spans, active_i_rel_window)

                    # Summary strength for that highlight region
                    active_vals = feat_acts_window[np.array(active_i_rel_window, dtype=np.int64)]
                    ex = {
                        "feature_id": f,
                        "score": peak_v,
                        "peak_activation": peak_v,
                        "peak_token_index_in_window": int(peak_i - left_win_i),
                        "attempt_id": meta.get("attempt_id"),
                        "turn_index": meta.get("turn_index"),
                        "status": meta.get("status"),
                        "attempt_answer": meta.get("attempt_answer"),
                        "reference_answer": meta.get("reference_answer"),

                        "text": text,
                        "highlight": {
                            "char_start": int(char_start),
                            "char_end": int(char_end),
                            "max_act_in_highlight": float(active_vals.max()) if active_vals.size else 0.0,
                            "mean_act_in_highlight": float(active_vals.mean()) if active_vals.size else 0.0,
                        },
                    }

                    if args.include_token_debug:
                        ex["token_debug"] = {
                            "token_ids": window_token_ids,
                            "token_spans": [
                                {"i": i, "char_start": int(s), "char_end": int(e), "token_text": t}
                                for i, (s, e, t) in enumerate(token_char_spans)
                            ],
                            "acts": feat_acts_window.astype(np.float32).tolist(),  # feature acts over window tokens
                            "active_token_indices": active_i_rel_window,
                        }

                    maybe_push_example(f, ex)

    handle.remove()

    # Final prune to max_examples_per_feature and sort
    out: Dict[str, Any] = {
        "meta": {
            "model_path": args.model_path,
            "layer": args.layer,
            "sae_release": "gpt-oss-20b-andyrdt",
            "sae_id": "resid_post_layer_19_trainer_0",
            "features": feat_list,
            "max_examples_per_feature": args.max_examples_per_feature,
            "activation_threshold": args.activation_threshold,
            "min_active_width": args.min_active_width,
            "max_window_width": args.max_window_width,
            "buffer_tokens": args.buffer_tokens,
            "max_tokens_per_turn": args.max_tokens,
            "filters": {
                "only_correct": bool(args.only_correct),
                "only_with_answers": bool(args.only_with_answers),
                "only_selected": bool(args.only_selected),
                "turn": args.turn,
                "max_attempts": args.max_attempts,
            },
        },
        "features": {},
    }

    for f in feat_list:
        lst = per_feat_examples[f]
        lst.sort(key=lambda d: d["score"], reverse=True)
        lst = lst[:args.max_examples_per_feature]
        out["features"][str(f)] = {
            "feature_id": f,
            "num_examples": len(lst),
            "examples": lst,
        }

    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json}")
    # Print quick summary
    for f in feat_list:
        n = out["features"][str(f)]["num_examples"]
        print(f"feature {f}: {n} examples")

if __name__ == "__main__":
    main()
