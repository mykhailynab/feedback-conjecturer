#!/usr/bin/env python3
"""Download tokenizer files (no *.safetensors) for configured HF models."""
from huggingface_hub import snapshot_download
import os

# --- Configure models here ---
MODELS = [
    "Qwen/Qwen3.6-35B-A3B",
    "google/gemma-4-31B-it",
    "Qwen/Qwen3.5-27B",
]
TOKENIZERS_DIR = "tokenizers"
IGNORE_PATTERNS = ["*.safetensors", "*.gguf", "*.bin"]
# ----------------------------

for repo_id in MODELS:
    name = repo_id.split("/")[-1]
    local_dir = os.path.join(TOKENIZERS_DIR, name)
    print(f"Downloading {repo_id} -> {local_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        ignore_patterns=IGNORE_PATTERNS,
    )
    print(f"  Done: {local_dir}")
