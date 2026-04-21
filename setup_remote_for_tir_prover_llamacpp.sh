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

PARALLELISM=0  # 0 = auto (NUM_GPUS * NP_PER_SERVER)
NP_PER_SERVER=1
START_PROVER=false

usage() {
    echo "Usage: $0 [-p <parallelism>] [--np <slots>] [--start-prover]"
    echo "  -p / --parallelism   Number of parallel workers for prove_formalizations."
    echo "                       Default: NUM_GPUS * NP_PER_SERVER."
    echo "  --np                 Number of parallel slots per llama-server (-np flag)."
    echo "                       Also sets --tir-llamacpp-max-concurrent. Default: 1."
    echo "  --start-prover       After setup, launch prove_formalizations in a screen session."
    echo "  -h / --help          Show this message."
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--parallelism)   PARALLELISM="$2"; shift 2 ;;
        --np)               NP_PER_SERVER="$2"; shift 2 ;;
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

# Max context per slot = 262144.  Each llama-server gets ctx_size = per-slot * np.
CTX_PER_SLOT=262144
CTX_SIZE_PER_SERVER=$(( CTX_PER_SLOT * NP_PER_SERVER ))

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
    hf download unsloth/Qwen3.6-35B-A3B-GGUF \
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

# Size of a single file, or total size of all files in a directory tree.
_filesize() {
    if [[ -f "$1" ]]; then
        stat -c%s "$1" 2>/dev/null || echo 0
    elif [[ -d "$1" ]]; then
        du -sb "$1" 2>/dev/null | cut -f1 || echo 0
    else
        echo 0
    fi
}

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

    GGUF_SIZE="$(_fmtsize "$(_filesize "$MODEL_DIR")")"
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

step "Installing llama.cpp (build from source with CUDA)"

LLAMACPP_DIR="$PROJECT_DIR/llama.cpp"
LLAMA_SERVER="$LLAMACPP_DIR/llama-server"

if [[ -x "$LLAMA_SERVER" ]]; then
    echo "llama-server already built: $LLAMA_SERVER"
else
    apt-get update -q || die "apt-get update failed"
    apt-get install -y pciutils build-essential cmake curl libcurl4-openssl-dev \
        || die "Failed to install build dependencies"

    if [[ ! -d "$LLAMACPP_DIR/.git" ]]; then
        git clone https://github.com/ggml-org/llama.cpp "$LLAMACPP_DIR" \
            || die "Failed to clone llama.cpp"
    fi

    cmake "$LLAMACPP_DIR" -B "$LLAMACPP_DIR/build" \
        -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
        || die "cmake configure failed"

    cmake --build "$LLAMACPP_DIR/build" --config Release -j --clean-first \
        --target llama-cli llama-mtmd-cli llama-server llama-gguf-split \
        || die "cmake build failed"

    cp "$LLAMACPP_DIR"/build/bin/llama-* "$LLAMACPP_DIR/" \
        || die "Failed to copy built binaries"
fi

# Verify llama-server is available
[[ -x "$LLAMA_SERVER" ]] \
    || die "llama-server not found after build. Check build logs."

# ============================================================
# 6. Detect GPUs and start one llama-server per GPU
# ============================================================

NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
[[ "$NUM_GPUS" -ge 1 ]] || die "No GPUs detected by nvidia-smi"

# Auto-compute parallelism if not explicitly set
if [[ "$PARALLELISM" -eq 0 ]]; then
    PARALLELISM=$(( NUM_GPUS * NP_PER_SERVER ))
fi

step "Detected $NUM_GPUS GPU(s), np=$NP_PER_SERVER per server — launching one llama-server per GPU (base port $LLAMACPP_PORT)"

for GPU_IDX in $(seq 0 $(( NUM_GPUS - 1 ))); do
    PORT=$(( LLAMACPP_PORT + GPU_IDX ))
    SESSION_NAME="llama-server-${GPU_IDX}"
    LOG_FILE="/var/log/llama-server-${GPU_IDX}.log"

    if curl -sf "http://localhost:$PORT/health" &>/dev/null; then
        echo "  GPU $GPU_IDX: llama-server already running on port $PORT"
        continue
    fi

    echo "  GPU $GPU_IDX: launching on port $PORT (screen: $SESSION_NAME)"
    screen -dmS "$SESSION_NAME" bash -c \
        "CUDA_VISIBLE_DEVICES=$GPU_IDX \"$LLAMA_SERVER\" \
        --model \"$GGUF_PATH\" \
        --mmproj \"$MMPROJ_PATH\" \
        --alias \"$MODEL_ALIAS\" \
        --temp $TEMPERATURE \
        --top-p $TOP_P \
        --min-p $MIN_P \
        --top-k $TOP_K \
        --ctx-size $CTX_SIZE_PER_SERVER \
        --port $PORT \
        -np $NP_PER_SERVER \
        2>&1 | tee $LOG_FILE"
done

# Wait for all servers to become ready
step "Waiting for all llama-servers to become ready..."
for GPU_IDX in $(seq 0 $(( NUM_GPUS - 1 ))); do
    PORT=$(( LLAMACPP_PORT + GPU_IDX ))
    SESSION_NAME="llama-server-${GPU_IDX}"
    LOG_FILE="/var/log/llama-server-${GPU_IDX}.log"

    for i in $(seq 1 120); do
        if curl -sf "http://localhost:$PORT/health" &>/dev/null; then
            echo "  GPU $GPU_IDX (port $PORT): ready"
            break
        fi
        if ! screen -list | grep -q "$SESSION_NAME"; then
            die "llama-server on GPU $GPU_IDX exited unexpectedly (check $LOG_FILE)"
        fi
        if [[ $i -eq 120 ]]; then
            die "llama-server on GPU $GPU_IDX did not become ready after 120 s (check $LOG_FILE)"
        fi
        sleep 1
    done
done

# ============================================================
# 7. Print (and optionally launch) the run command
# ============================================================

step "Setup complete"

# --------------- Build base-urls list ---------------
BASE_URLS=""
for GPU_IDX in $(seq 0 $(( NUM_GPUS - 1 ))); do
    BASE_URLS="$BASE_URLS http://localhost:$(( LLAMACPP_PORT + GPU_IDX ))"
done

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
       --tir-llamacpp-base-urls $BASE_URLS \
       --tir-llamacpp-max-concurrent $NP_PER_SERVER \
       --tir-llamacpp-client-timeout 960 \
       --tir-llamacpp-presence-penalty 0.0 \
       --tir-llamacpp-tokenizer-path $TOKENIZER_PATH \
       --formalizations-path logs/$LOGS_DIR_NAME/formalizations.jsonl \
       --lean-project-dir $MATHLIB4_DIR \
       --parallelism $PARALLELISM \
       --limit-prover-tokens $CTX_PER_SLOT \
       --continue &> prove_formalizations_tir_llamacpp.log'"

echo ""
echo "Servers: $NUM_GPUS llama-server instances (ports $LLAMACPP_PORT–$(( LLAMACPP_PORT + NUM_GPUS - 1 )))"
echo ""
echo "TIR prover command (opens a detached screen session):"
echo "  $PROVER_SCREEN_CMD"
echo ""
echo "For log monitoring:"
echo "  screen -dmS monitor_prove tail -f prove_formalizations_tir_llamacpp.log"
echo ""
echo "For llama-server monitoring:"
for GPU_IDX in $(seq 0 $(( NUM_GPUS - 1 ))); do
    echo "  screen -dmS llama_mon_$GPU_IDX tail -f /var/log/llama-server-${GPU_IDX}.log"
done
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
