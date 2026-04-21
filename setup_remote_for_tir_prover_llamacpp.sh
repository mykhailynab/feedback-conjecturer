#!/bin/bash
# Setup script for running prove_formalizations with llama.cpp + Qwen3.6 TIR prover
# on a remote Linux machine (tested on vast.ai).
#
# Run from the project root after cloning the repo:
#   bash setup_remote_for_tir_prover_llamacpp.sh [-p <parallelism>]
#
# Re-running is safe: each step checks whether its work is already done.

set -euo pipefail

# ============================================================
# Argument parsing
# ============================================================

PARALLELISM=1
START_PROVER=false

usage() {
    echo "Usage: $0 [-p <parallelism>] [--start-prover]"
    echo "  -p / --parallelism   Number of parallel slots (-np for llama-server"
    echo "                       and --parallelism for prove_formalizations). Default: 1."
    echo "  --start-prover       After setup, launch prove_formalizations in a screen session."
    echo "  -h / --help          Show this message."
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--parallelism)   PARALLELISM="$2"; shift 2 ;;
        --start-prover)     START_PROVER=true; shift ;;
        -h|--help)          usage ;;
        *)                  usage ;;
    esac
done

# ============================================================
# Output helpers
# ============================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

step() { echo -e "\n${GREEN}==> $*${NC}"; }
warn() { echo -e "${YELLOW}WARNING: $*${NC}"; }
die()  { echo -e "${RED}ERROR: $*${NC}" >&2; exit 1; }

# ============================================================
# Paths / constants
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
MATHLIB4_DIR="$PROJECT_DIR/mathlib4"
MODEL_DIR="$PROJECT_DIR/unsloth/Qwen3.6-35B-A3B-GGUF"
GGUF_FILENAME="Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf"
GGUF_PATH="$MODEL_DIR/$GGUF_FILENAME"
MMPROJ_PATH="$MODEL_DIR/mmproj-F16.gguf"
MODEL_ALIAS="unsloth/Qwen3.6-35B-A3B"
MATHLIB4_TARBALL="$PROJECT_DIR/mathlib4.tar.gz"
MATHLIB4_GDRIVE_ID="1_0QVdYxrsaibi-eCqBrdLWnscpUbpLhA"
TOKENIZER_PATH="$PROJECT_DIR/tokenizers/Qwen3.5-27B"
ELAN_ENV="/root/.elan/env"
LOGS_DIR_NAME="conjecture_formalization_logs_20mins"
LLAMACPP_PORT=8001

# Max context per slot = 262144.  Total --ctx-size = per-slot * parallelism.
CTX_PER_SLOT=262144
CTX_SIZE=$(( CTX_PER_SLOT * PARALLELISM ))

# Sampling defaults (set at server startup — not per-request for top_k/min_p/repeat_penalty)
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
MIN_P=0.0

# ============================================================
# 1. apt packages
# ============================================================

step "Installing apt packages"
apt-get update -q || die "apt-get update failed"
apt-get install -y screen curl git wget python3 python3-pip lshw 2>&1 \
    | grep -v "^Get:\|^Fetched\|^Hit:\|^Ign:" \
    || die "apt-get install failed"

# ============================================================
# 2. Python dependencies
# ============================================================

step "Installing Python dependencies"
[[ -f "$PROJECT_DIR/requirements.txt" ]] \
    || die "requirements.txt not found in $PROJECT_DIR"
pip install -r "$PROJECT_DIR/requirements.txt" \
    || die "pip install -r requirements.txt failed"

# ============================================================
# 3. Parallel downloads: model (via hf) + mathlib4 cache
# ============================================================

step "Starting parallel downloads"

GGUF_LOG="$PROJECT_DIR/download_gguf.log"
MATHLIB4_LOG="$PROJECT_DIR/download_mathlib4.log"

download_model() {
    if [[ -f "$GGUF_PATH" ]] && [[ -f "$MMPROJ_PATH" ]]; then
        echo "already downloaded" > "$GGUF_LOG"
        return 0
    fi
    huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF \
        --local-dir "$MODEL_DIR" \
        --include "*mmproj-F16*" \
        --include "*UD-Q5_K_XL*" \
        >> "$GGUF_LOG" 2>&1 \
        || { echo "FAILED" >> "$GGUF_LOG"; return 1; }
    echo "done" >> "$GGUF_LOG"
}

download_mathlib4() {
    if [[ -f "$MATHLIB4_TARBALL" ]] || [[ -d "$MATHLIB4_DIR" ]]; then
        echo "already present" > "$MATHLIB4_LOG"
        return 0
    fi
    gdown "$MATHLIB4_GDRIVE_ID" \
        -O "$MATHLIB4_TARBALL" -q \
        >> "$MATHLIB4_LOG" 2>&1 \
        || { rm -f "$MATHLIB4_TARBALL"; echo "FAILED" >> "$MATHLIB4_LOG"; return 1; }
    echo "done" >> "$MATHLIB4_LOG"
}

# Human-readable file size (bytes → B / KB / MB / GB)
_fmtsize() {
    local bytes=$1
    if   (( bytes >= 1073741824 )); then printf "%.1f GB" "$(echo "scale=1; $bytes/1073741824" | bc)"
    elif (( bytes >= 1048576    )); then printf "%.1f MB" "$(echo "scale=1; $bytes/1048576"    | bc)"
    elif (( bytes >= 1024       )); then printf "%.1f KB" "$(echo "scale=1; $bytes/1024"       | bc)"
    else printf "%d B" "$bytes"
    fi
}

_filesize() { [[ -f "$1" ]] && stat -c%s "$1" 2>/dev/null || echo 0; }

download_model &
GGUF_PID=$!
download_mathlib4 &
MATHLIB4_PID=$!

echo "Downloading in parallel (logs: download_gguf.log, download_mathlib4.log)"
while kill -0 "$GGUF_PID" 2>/dev/null || kill -0 "$MATHLIB4_PID" 2>/dev/null; do
    GGUF_STATUS="running"
    MATHLIB4_STATUS="running"
    kill -0 "$GGUF_PID"     2>/dev/null || GGUF_STATUS="done"
    kill -0 "$MATHLIB4_PID" 2>/dev/null || MATHLIB4_STATUS="done"

    GGUF_SIZE="$(_fmtsize "$(_filesize "$GGUF_PATH")")"
    MATHLIB4_SIZE="$(_fmtsize "$(_filesize "$MATHLIB4_TARBALL")")"

    printf "\r  model: %-8s %-10s   mathlib4: %-8s %-10s" \
        "$GGUF_STATUS" "$GGUF_SIZE" "$MATHLIB4_STATUS" "$MATHLIB4_SIZE"
    sleep 0.5
done
printf "\n"

wait $GGUF_PID     || die "Model download failed (see $GGUF_LOG)"
wait $MATHLIB4_PID || die "mathlib4 download failed (see $MATHLIB4_LOG)"

step "Downloads complete"

# ============================================================
# 4. Install Lean (elan) + build mathlib4
# ============================================================

step "Setting up Lean / mathlib4"

if [[ ! -f "$ELAN_ENV" ]]; then
    echo "Installing elan (Lean version manager)..."
    curl -sSf https://elan.lean-lang.org/elan-init.sh -o /tmp/elan.sh \
        || die "Failed to download elan installer"
    bash /tmp/elan.sh -y || die "elan installation failed"
    rm /tmp/elan.sh
else
    echo "elan already installed"
fi

# shellcheck source=/dev/null
source "$ELAN_ENV"

if [[ ! -d "$MATHLIB4_DIR" ]]; then
    if [[ -f "$MATHLIB4_TARBALL" ]]; then
        echo "Unpacking mathlib4.tar.gz..."
        tar -xzf "$MATHLIB4_TARBALL" -C "$PROJECT_DIR" \
            || die "Failed to unpack $MATHLIB4_TARBALL"
        echo "Unpacked to $MATHLIB4_DIR"
    else
        echo "Cloning mathlib4..."
        git clone https://github.com/xinhjBrant/mathlib4.git "$MATHLIB4_DIR" \
            || die "Failed to clone mathlib4"
    fi
fi

echo "Running lake build..."
pushd "$MATHLIB4_DIR" > /dev/null
lake build || die "lake build failed"
popd > /dev/null
echo "mathlib4 ready"

# ============================================================
# 5. Install llama.cpp
# ============================================================

step "Installing llama.cpp"
if command -v llama-server &>/dev/null; then
    echo "llama-server already installed: $(llama-server --version 2>&1 | head -1)"
else
    echo "Installing llama.cpp via pip (llama-cpp-python[server])..."
    pip install llama-cpp-python[server] \
        || die "Failed to install llama-cpp-python"
fi

# Verify llama-server is available
command -v llama-server &>/dev/null \
    || die "llama-server not found after installation. Install llama.cpp manually."

# ============================================================
# 6. Start llama-server
# ============================================================

step "Starting llama-server (port $LLAMACPP_PORT, -np $PARALLELISM, ctx_size $CTX_SIZE)"

if curl -sf "http://localhost:$LLAMACPP_PORT/health" &>/dev/null; then
    echo "llama-server already running on port $LLAMACPP_PORT"
else
    echo "Launching llama-server..."
    nohup llama-server \
        --model "$GGUF_PATH" \
        --mmproj "$MMPROJ_PATH" \
        --alias "$MODEL_ALIAS" \
        --temp "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --min-p "$MIN_P" \
        --top-k "$TOP_K" \
        --ctx-size "$CTX_SIZE" \
        --port "$LLAMACPP_PORT" \
        -np "$PARALLELISM" \
        > /var/log/llama-server.log 2>&1 &
    LLAMA_PID=$!
    echo "Waiting for llama-server to become ready (PID $LLAMA_PID)..."
    for i in $(seq 1 120); do
        if curl -sf "http://localhost:$LLAMACPP_PORT/health" &>/dev/null; then
            echo "llama-server is ready"
            break
        fi
        if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
            die "llama-server process exited unexpectedly (check /var/log/llama-server.log)"
        fi
        if [[ $i -eq 120 ]]; then
            die "llama-server did not become ready after 120 s (check /var/log/llama-server.log)"
        fi
        sleep 1
    done
fi

# ============================================================
# 7. Print (and optionally launch) the run command
# ============================================================

step "Setup complete"

# --------------- TIR prover command ---------------
PROVER_SCREEN_CMD="screen -dmS prove_formalizations bash -c \
    'source $ELAN_ENV && \
     cd $PROJECT_DIR && \
     export PYTHONPATH=. && \
     python conjecturing_agents/prove_formalizations.py \
       --prover-type tir \
       --tir-temperature $TEMPERATURE \
       --tir-top-p $TOP_P \
       --tir-backend llamacpp \
       --tir-llamacpp-model $MODEL_ALIAS \
       --tir-llamacpp-base-url http://localhost:$LLAMACPP_PORT \
       --tir-llamacpp-client-timeout 960 \
       --tir-llamacpp-presence-penalty 0.0 \
       --tir-llamacpp-tokenizer-path $TOKENIZER_PATH \
       --formalizations-path logs/$LOGS_DIR_NAME/formalizations.jsonl \
       --lean-project-dir $MATHLIB4_DIR \
       --parallelism $PARALLELISM \
       --limit-prover-tokens $CTX_PER_SLOT \
       --continue &> prove_formalizations_tir_llamacpp.log'"

echo ""
echo "TIR prover command (opens a detached screen session):"
echo "  $PROVER_SCREEN_CMD"
echo ""
echo "For log monitoring:"
echo "  screen -dmS monitor_prove tail -f prove_formalizations_tir_llamacpp.log"
echo ""
echo "For llama-server monitoring:"
echo "  screen -dmS llama_mon tail -f /var/log/llama-server.log"
echo ""
echo "For GPU monitoring:"
echo "  screen -S gpu -dm watch -n 1 nvidia-smi"
echo ""
echo "To attach to a session:  screen -r prove_formalizations"

if $START_PROVER; then
    step "Launching prove_formalizations screen session"
    eval "$PROVER_SCREEN_CMD" || die "Failed to start prove_formalizations screen session"
    echo "Started.  Log: $PROJECT_DIR/prove_formalizations_tir_llamacpp.log"
fi
