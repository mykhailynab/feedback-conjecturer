#!/usr/bin/env python3
import os
import time
import torch
import numpy as np
import pandas as pd
import argparse, json
from tqdm import tqdm
from sae_lens import SAE
from scipy import sparse
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def get_sequences_from_attempt_row(row, which_turn="all") -> List[Tuple[List[int], int]]:
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
    input_ids_padded = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
    seq_lengths = []
    for i, s in enumerate(seqs):
        L = len(s)
        seq_lengths.append(L)
        input_ids_padded[i, :L] = torch.tensor(s, dtype=torch.long, device=device)
        attn_mask[i, :L] = 1
    return input_ids_padded, attn_mask, seq_lengths


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


def truncate_to_token_budget(
    seqs: List[List[int]],
    prompt_lens: List[int],
    max_tokens: int,
) -> Tuple[List[List[int]], List[int]]:
    """
    Keep the earliest tokens across sequences until total token count hits max_tokens.
    Sequences may be truncated. prompt_len is adjusted accordingly.
    """
    if max_tokens <= 0:
        return [], []

    new_seqs: List[List[int]] = []
    new_prompt_lens: List[int] = []

    for seq, input_prompt_len in zip(seqs, prompt_lens):
        if not seq:
            continue

        trunc_len = min(len(seq), max_tokens)

        seq_cut = seq[:trunc_len]
        input_prompt_len_cut = min(input_prompt_len, trunc_len)

        new_seqs.append(seq_cut)
        new_prompt_lens.append(input_prompt_len_cut)

    return new_seqs, new_prompt_lens


class CSRSink:
    """
    Incrementally builds a single CSR matrix row-by-row from dense SAE activations.
    Stores only (col_idx, value) pairs for values passing a threshold.
    """
    def __init__(self, n_features: int, threshold: float = 0.0):
        self.n_features = int(n_features)
        self.threshold = float(threshold)

        self._indices_chunks = []
        self._data_chunks = []
        self._indptr = [0]          # CSR row pointer
        self._n_rows = 0            # total rows appended

    @property
    def n_rows(self) -> int:
        return self._n_rows

    def append_dense(self, acts: torch.Tensor) -> None:
        """
        acts: [N_rows, n_features] on GPU (or CPU). Appends all rows.
        """
        if acts.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {acts.shape}")

        n_rows = int(acts.shape[0])
        if n_rows == 0:
            return

        # mask of kept entries
        if self.threshold <= 0.0:
            mask = acts != 0
        else:
            mask = acts > self.threshold

        nz = mask.nonzero(as_tuple=False)  # [nnz, 2] (row, col)

        if nz.numel() == 0:
            # No nnz: just extend indptr with same pointer for each new row
            self._indptr.extend([self._indptr[-1]] * n_rows)
            self._n_rows += n_rows
            return

        vals = acts[nz[:, 0], nz[:, 1]]

        # Move to CPU numpy
        r = nz[:, 0].detach().cpu().numpy().astype(np.int64)
        c = nz[:, 1].detach().cpu().numpy().astype(np.int64)
        v = vals.detach().float().cpu().numpy()

        # Ensure sorted by (row, col) for CSR correctness
        order = np.lexsort((c, r))
        r, c, v = r[order], c[order], v[order]

        # Build indptr increments row-by-row
        counts = np.bincount(r, minlength=n_rows)
        running = self._indptr[-1]
        for cnt in counts:
            running += int(cnt)
            self._indptr.append(running)

        self._indices_chunks.append(c)
        self._data_chunks.append(v)
        self._n_rows += n_rows

    def to_csr(self) -> sparse.csr_matrix:
        indices = np.concatenate(self._indices_chunks) if self._indices_chunks else np.zeros((0,), dtype=np.int64)
        data = np.concatenate(self._data_chunks) if self._data_chunks else np.zeros((0,), dtype=np.float32)
        indptr = np.asarray(self._indptr, dtype=np.int64)
        return sparse.csr_matrix((data, indices, indptr), shape=(self._n_rows, self.n_features))

    def save_npz(self, path: str) -> None:
        csr = self.to_csr()
        sparse.save_npz(path, csr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", type=str, required=True, help="Path to attempts.jsonl")
    ap.add_argument("--model_path", type=str, required=True, help="HF model path (local folder or hub id)")
    ap.add_argument("--reference_path", type=str, required=True, help="reference.csv path")
    ap.add_argument("--only_correct", action="store_true", help="only consider correct answers")
    ap.add_argument("--only_with_answers", action="store_true", help="only consider attempts with answers")
    ap.add_argument("--only_selected", action="store_true", help="only consider attempts that were selected")
    ap.add_argument("--layer", type=int, default=19, help="Layer index to hook")
    ap.add_argument("--max_attempts", type=int, default=64, help="How many attempt rows to use")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--out_sparse", type=str, default="sae_sparse.npz", help="Output .npz (scipy CSR)")
    ap.add_argument("--threshold", type=float, default=0.0, help="Keep activations > threshold (0 keeps nonzeros)")
    ap.add_argument("--max_tokens", type=int, default=(65536-500), help="Max token cap per single turn")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script, but torch.cuda.is_available() is False.")

    device = torch.device("cuda")
    attempts_path = Path(args.attempts)

    _preload_model_weights(args.model_path)

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

    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = device
    else:
        model_device = torch.device(model_device)

    sae = SAE.from_pretrained(
        release="gpt-oss-20b-andyrdt",
        sae_id="resid_post_layer_19_trainer_0",
        device="cuda",
    )

    # --- Collect sequences from jsonl ---
    seqs: List[List[int]] = []
    input_prompt_lens: List[int] = []

    reference_csv = pd.read_csv(args.reference_path)

    used_attempts = 0
    for attempt in iter_jsonl(attempts_path):
        if args.only_with_answers and attempt['attempt_answer'] is None:
            continue
        
        reference_answer = reference_csv[reference_csv['id'] == attempt['id']]['answer'].iloc[0]
        if args.only_correct and attempt['attempt_answer'] != reference_answer:
            continue

        if args.only_selected and attempt['status'] != 'selected':
            continue

        if used_attempts >= args.max_attempts:
            break

        for full_ids, ip_len in get_sequences_from_attempt_row(attempt, which_turn="all"):
            seqs.append(full_ids)
            input_prompt_lens.append(ip_len)
        
        print(f"Adding attempt id={attempt['id']} turns={len(attempt['trace']['turns'])} attempt_answer={attempt['attempt_answer']} reference_answer={reference_answer} selected={attempt['status'] == 'selected'}")

        used_attempts += 1

    if not seqs:
        raise RuntimeError("No usable sequences found in attempts.jsonl (missing trace/turns/token ids).")

    # --- Apply token budget (first N tokens across sequences) ---
    # before_total = sum(len(s) for s in seqs)
    seqs_cut, input_prompt_lens_cut = truncate_to_token_budget(seqs, input_prompt_lens, args.max_tokens)
    # after_total = sum(len(s) for s in seqs_cut)
    # print(f"Token budget: max_per_turn={args.max_tokens}, before={before_total}, after={after_total}, sequences={len(seqs_cut)}")

    if not seqs_cut:
        raise RuntimeError("After applying --max_tokens, no tokens remain to process.")

    # --- Hook layer N output (post-block hidden states) ---
    try:
        layer_module = model.model.layers[args.layer]
    except Exception as e:
        raise RuntimeError(f"Could not access model.model.layers[{args.layer}] (arch mismatch?): {e}")

    captured = {}

    def hook_fn(mod, inp, out):
        hs = out[0] if isinstance(out, (tuple, list)) else out
        captured["hs"] = hs.detach()

    handle = layer_module.register_forward_hook(hook_fn)

    sink = CSRSink(n_features=int(sae.cfg.d_sae), threshold=args.threshold)

    with torch.no_grad():
        tqdm_iter = tqdm(range(0, len(seqs_cut), args.batch_size))
        for start in tqdm_iter:
            batch_seqs = seqs_cut[start : start + args.batch_size]
            batch_prompt_lens = input_prompt_lens_cut[start : start + args.batch_size]

            input_ids_padded, attention_mask, seq_lengths = pad_batch(
                batch_seqs,
                tok.pad_token_id,
                device=model_device,
            )

            captured.clear()
            try:
                _ = model.model(input_ids=input_ids_padded, attention_mask=attention_mask, use_cache=False)
            except Exception as e:
                print(f"Model failed: {e}. {seq_lengths = }")
                return

            if "hs" not in captured:
                raise RuntimeError("Hook did not capture hidden states; layer forward output format unexpected.")

            hs = captured.pop("hs")

            comp_chunks = []
            for i in range(hs.shape[0]):
                seq_len = seq_lengths[i]
                prompt_len = batch_prompt_lens[i]
                # print(f"{i = } {prompt_len = } {(seq_len - prompt_len) = }")
                if prompt_len >= seq_len:
                    continue
                comp = hs[i, prompt_len:seq_len, :]
                comp_chunks.append(comp)

            if not comp_chunks:
                continue

            comp_tokens = torch.cat(comp_chunks, dim=0)  # [N_tokens, d_model]
            feat_acts = sae.encode(comp_tokens.to(device))
            sink.append_dense(feat_acts)

    handle.remove()

    if sink.n_rows == 0:
        raise RuntimeError("No completion activations collected.")

    sink.save_npz(args.out_sparse)
    csr = sink.to_csr()
    print(f"Saved sparse SAE activations to {args.out_sparse}")
    print(f"CSR shape={csr.shape}, nnz={csr.nnz}")


if __name__ == "__main__":
    main()
