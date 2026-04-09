#!/bin/bash
# Setup script for running check_formalizations with Ollama + Goedel-Prover-V2-32B
# on a remote Linux machine (tested on vast.ai 1xA6000 48G).
#
# Run from the project root after cloning the repo:
#   bash setup_remote.sh
#
# Re-running is safe: each step checks whether its work is already done.

set -euo pipefail

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
# Paths
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
MATHLIB4_DIR="$PROJECT_DIR/mathlib4"
MODELS_DIR="$PROJECT_DIR/models"
GGUF_FILENAME="Goedel-Prover-V2-32B.Q8_0.gguf"
GGUF_PATH="$MODELS_DIR/$GGUF_FILENAME"
GGUF_URL="https://huggingface.co/mradermacher/Goedel-Prover-V2-32B-GGUF/resolve/main/$GGUF_FILENAME"
OLLAMA_MODEL_NAME="goedel-v2"   # matches AnswerCheckerConfig.goedel_ollama_model default
MODELFILE_PATH="$PROJECT_DIR/Modelfile"
ELAN_ENV="/root/.elan/env"

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
# 3. Download GGUF model (~34 GB)
# ============================================================

step "Downloading GGUF model"
mkdir -p "$MODELS_DIR"
if [[ -f "$GGUF_PATH" ]]; then
    echo "Already downloaded: $GGUF_PATH"
else
    echo "Downloading $GGUF_FILENAME from HuggingFace — this will take a while..."
    wget --continue --show-progress -O "$GGUF_PATH" "$GGUF_URL" \
        || { rm -f "$GGUF_PATH"; die "Download failed: $GGUF_URL"; }
    echo "Download complete: $GGUF_PATH"
fi

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
    echo "Cloning mathlib4..."
    git clone https://github.com/xinhjBrant/mathlib4.git "$MATHLIB4_DIR" \
        || die "Failed to clone mathlib4"
fi

echo "Running lake build (downloads precompiled cache; may take 10–20 min on first run)..."
# TODO: check if mathlib4.tar.gz exists in the current folder (we may have scp'd it onto the machine)
# If yes, we should just unpack tar.gz, if not, then run lake build (and create tar.gz afterwards)
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
# 6. Start Ollama server
# ============================================================

step "Starting Ollama server"
if pgrep -x ollama &>/dev/null; then
    echo "Ollama server already running"
else
    echo "Launching ollama serve..."
    nohup ollama serve > /var/log/ollama.log 2>&1 &
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
PARAMETER num_ctx 40960
EOF

echo "Modelfile written to $MODELFILE_PATH"
ollama create "$OLLAMA_MODEL_NAME" -f "$MODELFILE_PATH" \
    || die "Failed to create Ollama model '$OLLAMA_MODEL_NAME'"
echo "Model '$OLLAMA_MODEL_NAME' registered"

# ============================================================
# 8. Smoke-test Ollama
# ============================================================

# Skip because this is a specialized model and will try to prove something (long output)
# step "Testing Ollama with model '$OLLAMA_MODEL_NAME'"
# TEST_OUT=$(echo "Reply with the single word OK." \
#     | ollama run "$OLLAMA_MODEL_NAME" --nowordwrap 2>&1) \
#     || die "Ollama test run failed"
# echo "Model response: $TEST_OUT"
# echo "Ollama test passed"

# ============================================================
# 9. Print the run command
# ============================================================

step "Setup complete"
echo ""
echo "Run the following command (opens a detached screen session):"
echo ""
echo "  screen -dmS check_formalizations bash -c \\"
echo "    'source $ELAN_ENV && \\"
echo "     cd $PROJECT_DIR && \\"
echo "     export PYTHONPATH=. && \\"
echo "     python conjecturing_agents/check_formalizations.py \\"
echo "       --formalizations-path logs/conjecture_formalization_logs_20mins/formalizations.jsonl \\"
echo "       --lean-project-dir $MATHLIB4_DIR \\"
echo "       --parallelism 12 \\"
echo "       --goedel --goedel-disprover \\"
echo "       --continue &> check_formalizations.log'"
echo ""
echo "For script monitoring:"
echo "  screen -dmS monitor_check_formalizations tail -f check_formalizations.log"
echo ""
echo "For GPU monitoring:"
echo "  screen -S gpu -dm watch -n 1 nvidia-smi"
echo ""
echo "To attach to the session:  screen -r check_formalizations"
echo "Log file:                  $PROJECT_DIR/check_formalizations.log"
