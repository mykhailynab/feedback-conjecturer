#!/bin/bash
# Setup script for running prove_formalizations with Ollama + Qwen3.6 TIR prover
# on a remote Linux machine (tested on vast.ai, ~26.6 GB VRAM required).
#
# Run from the project root after cloning the repo:
#   bash setup_remote_for_tir_prover.sh [-p <parallelism>]
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
    echo "  -p / --parallelism   Number of parallel TIR workers (sets OLLAMA_NUM_PARALLEL"
    echo "                       and --parallelism). Default: 1."
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
MODELS_DIR="$PROJECT_DIR/models"
GGUF_FILENAME="Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf"
GGUF_PATH="$MODELS_DIR/$GGUF_FILENAME"
GGUF_URL="https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/$GGUF_FILENAME"
MATHLIB4_TARBALL="$PROJECT_DIR/mathlib4.tar.gz"
MATHLIB4_GDRIVE_ID="1_0QVdYxrsaibi-eCqBrdLWnscpUbpLhA"
OLLAMA_MODEL_NAME="qwen3-6"   # matches --tir-ollama-ollama-model default used in printed command
MODELFILE_PATH="$PROJECT_DIR/Modelfile.qwen3-6"
ELAN_ENV="/root/.elan/env"
LOGS_DIR_NAME="conjecture_formalization_logs_20mins"
CTX_LEN=262144   # context window — kept consistent between Modelfile, server env, and run command

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
# 3. Parallel downloads: GGUF model (~26.6 GB VRAM) + mathlib4 cache
# ============================================================

step "Starting parallel downloads"
mkdir -p "$MODELS_DIR"

GGUF_LOG="$PROJECT_DIR/download_gguf.log"
MATHLIB4_LOG="$PROJECT_DIR/download_mathlib4.log"

download_gguf() {
    if [[ -f "$GGUF_PATH" ]]; then
        echo "already downloaded" > "$GGUF_LOG"
        return 0
    fi
    wget --continue -q -O "$GGUF_PATH" "$GGUF_URL" \
        >> "$GGUF_LOG" 2>&1 \
        || { rm -f "$GGUF_PATH"; echo "FAILED" >> "$GGUF_LOG"; return 1; }
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

download_gguf &
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

    printf "\r  gguf: %-8s %-10s   mathlib4: %-8s %-10s" \
        "$GGUF_STATUS" "$GGUF_SIZE" "$MATHLIB4_STATUS" "$MATHLIB4_SIZE"
    sleep 0.5
done
printf "\n"

wait $GGUF_PID     || die "GGUF download failed (see $GGUF_LOG)"
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
# 5. Install Ollama
# ============================================================

step "Installing Ollama"
if command -v ollama &>/dev/null; then
    echo "Ollama already installed: $(ollama --version 2>&1 | head -1)"
else
    curl -fsSL https://ollama.com/install.sh | sh || die "Ollama installation failed"
fi

# ============================================================
# 6. Start Ollama server (with context window + parallelism)
# ============================================================

step "Starting Ollama server"
if pgrep -x ollama &>/dev/null; then
    echo "Ollama server already running"
else
    echo "Launching ollama serve (OLLAMA_CONTEXT_LENGTH=$CTX_LEN, OLLAMA_NUM_PARALLEL=$PARALLELISM)..."
    nohup env \
        OLLAMA_CONTEXT_LENGTH="$CTX_LEN" \
        OLLAMA_NUM_PARALLEL="$PARALLELISM" \
        ollama serve > /var/log/ollama.log 2>&1 &
    echo "Waiting for Ollama to become ready..."
    for i in $(seq 1 30); do
        if ollama list &>/dev/null 2>&1; then
            echo "Ollama is ready"
            break
        fi
        if [[ $i -eq 30 ]]; then
            die "Ollama server did not become ready after 30 s (check /var/log/ollama.log)"
        fi
        sleep 1
    done
fi

# ============================================================
# 7. Create Modelfile and register model with Ollama
# ============================================================

step "Registering model '$OLLAMA_MODEL_NAME' with Ollama"

cat > "$MODELFILE_PATH" <<EOF
FROM $GGUF_PATH
PARAMETER num_ctx $CTX_LEN
EOF

echo "Modelfile written to $MODELFILE_PATH"
ollama create "$OLLAMA_MODEL_NAME" -f "$MODELFILE_PATH" \
    || die "Failed to create Ollama model '$OLLAMA_MODEL_NAME'"
echo "Model '$OLLAMA_MODEL_NAME' registered"

# ============================================================
# 8. Print (and optionally launch) the run command
# ============================================================

step "Setup complete"

# --------------- TIR prover command ---------------
# Sampling params: temp=0.6 top_p=0.95 top_k=20 min_p=0.0 (per Qwen3.6 recommendations).
# Python and Lean tools are both enabled (default).
# Context window of $CTX_LEN is set at the server level via OLLAMA_CONTEXT_LENGTH
# and in the Modelfile — no extra CLI flag needed here.
PROVER_SCREEN_CMD="screen -dmS prove_formalizations bash -c \
    'source $ELAN_ENV && \
     cd $PROJECT_DIR && \
     export PYTHONPATH=. && \
     python conjecturing_agents/prove_formalizations.py \
       --prover-type tir \
       --tir-temperature 0.6 \
       --tir-top-p 0.95 \
       --tir-backend ollama \
       --tir-ollama-ollama-model $OLLAMA_MODEL_NAME \
       --tir-ollama-top-k 20 \
       --tir-ollama-min-p 0.0 \
       --formalizations-path logs/$LOGS_DIR_NAME/formalizations.jsonl \
       --lean-project-dir $MATHLIB4_DIR \
       --parallelism $PARALLELISM \
       --continue &> prove_formalizations_tir.log'"

echo ""
echo "TIR prover command (opens a detached screen session):"
echo "  $PROVER_SCREEN_CMD"
echo ""
echo "For log monitoring:"
echo "  screen -dmS monitor_prove_formalizations tail -f prove_formalizations_tir.log"
echo ""
echo "For ollama monitoring:"
echo "  screen -dmS ollama_mon tail -f /var/log/ollama.log"
echo ""
echo "For GPU monitoring:"
echo "  screen -S gpu -dm watch -n 1 nvidia-smi"
echo ""
echo "To attach to a session:  screen -r prove_formalizations"

if $START_PROVER; then
    step "Launching prove_formalizations screen session"
    eval "$PROVER_SCREEN_CMD" || die "Failed to start prove_formalizations screen session"
    echo "Started.  Log: $PROJECT_DIR/prove_formalizations_tir.log"
fi
