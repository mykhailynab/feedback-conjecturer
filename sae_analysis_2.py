#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from typing import List, Tuple
import os
import time
from concurrent.futures import as_completed, ThreadPoolExecutor

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def get_sequences_from_attempt_row(row, which_turn="last") -> List[Tuple[List[int], int]]:
    """
    Returns list of (full_token_ids, prompt_len) sequences from one attempt row.
    `prompt_len` is used to slice out completion positions later.
    """
    trace = row.get("trace") or row.get("Trace")
    if not trace:
        return []
    turns = (trace.get("turns") or [])
    if not turns:
        return []

    if which_turn == "last":
        chosen = [turns[-1]]
    elif which_turn == "all":
        chosen = turns
    else:
        idx = int(which_turn)
        if idx < 0 or idx >= len(turns):
            return []
        chosen = [turns[idx]]

    out = []
    for t in chosen:
        prompt_ids = t.get("prompt_token_ids") or []
        comp_ids = t.get("completion_token_ids") or []
        if not prompt_ids or not comp_ids:
            continue
        full = list(prompt_ids) + list(comp_ids)
        out.append((full, len(prompt_ids)))
    return out


def pad_batch(
    seqs: List[List[int]],
    pad_id: int,
    device: torch.device,
):
    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
    lengths = []
    for i, s in enumerate(seqs):
        L = len(s)
        lengths.append(L)
        input_ids[i, :L] = torch.tensor(s, dtype=torch.long, device=device)
        attn[i, :L] = 1
    return input_ids, attn, lengths


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", type=str, required=True, help="Path to attempts.jsonl")
    ap.add_argument("--model_path", type=str, required=True, help="HF model path (local folder or hub id)")
    ap.add_argument("--layer", type=int, default=19, help="Layer index to hook")
    ap.add_argument("--turn", type=str, default="last", help="'last', 'all', or integer turn index")
    ap.add_argument("--max_attempts", type=int, default=64, help="How many attempt rows to use")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    attempts_path = Path(args.attempts)

    # You can still pass --device cpu, but with device_map="auto" this is mostly for SAE + tensors.
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    _preload_model_weights(args.model_path)

    # --- Load tokenizer/model with the SAME settings that work for you (test3.py) ---
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
        dtype="auto",
        attn_implementation="kernels-community/vllm-flash-attn3",
    )
    model.eval()

    # Where to place input tensors:
    # For device_map="auto" on 1xH100 this is usually cuda:0; use model.device if present.
    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = device
    else:
        model_device = torch.device(model_device)

    # --- Load SAE (keep on args.device; typically cuda) ---
    sae = SAE.from_pretrained(
        release="gpt-oss-20b-andyrdt",
        sae_id="resid_post_layer_19_trainer_0",
        device=device.type,
    )

    # --- Collect sequences from jsonl ---
    seqs: List[List[int]] = []
    prompt_lens: List[int] = []

    used_rows = 0
    for row in iter_jsonl(attempts_path):
        if used_rows >= args.max_attempts:
            break
        extracted = get_sequences_from_attempt_row(row, which_turn=args.turn)
        if not extracted:
            continue

        for full_ids, p_len in extracted:
            seqs.append(full_ids)
            prompt_lens.append(p_len)

        used_rows += 1

    if not seqs:
        raise RuntimeError("No usable sequences found in attempts.jsonl (missing trace/turns/token ids).")

    # --- Hook layer N output (post-block hidden states) ---
    try:
        layer_module = model.model.layers[args.layer]
    except Exception as e:
        raise RuntimeError(f"Could not access model.model.layers[{args.layer}] (arch mismatch?): {e}")

    captured = {}

    def hook_fn(mod, inp, out):
        # HF blocks may return tuples; first element is usually hidden states.
        hs = out[0] if isinstance(out, (tuple, list)) else out
        captured["hs"] = hs

    handle = layer_module.register_forward_hook(hook_fn)

    all_feat_acts = []

    with torch.no_grad():
        for start in range(0, len(seqs), args.batch_size):
            batch_seqs = seqs[start : start + args.batch_size]
            batch_prompt_lens = prompt_lens[start : start + args.batch_size]

            input_ids, attention_mask, lengths = pad_batch(
                batch_seqs,
                tok.pad_token_id,
                device=model_device,
            )

            captured.clear()
            _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            if "hs" not in captured:
                raise RuntimeError("Hook did not capture hidden states; layer forward output format unexpected.")

            hs = captured["hs"]  # [B, T, d_model] typically

            comp_chunks = []
            for i in range(hs.shape[0]):
                L = lengths[i]
                p = batch_prompt_lens[i]
                if p >= L:
                    continue
                comp = hs[i, p:L, :]
                comp_chunks.append(comp)

            if not comp_chunks:
                continue

            comp_tokens = torch.cat(comp_chunks, dim=0)  # [N_tokens, d_model]

            # SAE encode on SAE device (usually cuda)
            feat_acts = sae.encode(comp_tokens.to(device))  # [N_tokens, n_features]
            all_feat_acts.append(feat_acts.detach().float().cpu())

    handle.remove()

    if not all_feat_acts:
        raise RuntimeError("No completion activations collected (maybe all turns had empty completion_token_ids?).")

    feat = torch.cat(all_feat_acts, dim=0)  # [total_completion_tokens, n_features]

    mean_act = feat.mean(dim=0)
    max_act = feat.max(dim=0).values

    topk = 20
    top_mean = torch.topk(mean_act, k=topk)
    top_max = torch.topk(max_act, k=topk)

    print(f"\nCollected {feat.shape[0]:,} completion tokens.")
    print(f"SAE features: {feat.shape[1]:,}")

    print("\nTop features by MEAN activation:")
    for val, idx in zip(top_mean.values.tolist(), top_mean.indices.tolist()):
        print(f"  feature {idx:6d}  mean={val:.6f}")

    print("\nTop features by MAX activation:")
    for val, idx in zip(top_max.values.tolist(), top_max.indices.tolist()):
        print(f"  feature {idx:6d}  max={val:.6f}")


if __name__ == "__main__":
    main()
