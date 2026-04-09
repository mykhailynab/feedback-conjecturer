#!/bin/bash
# Creates a self-contained zip for remote deployment.
# Run from the project root:  bash create_remote_zip.sh
#
# The zip contains everything needed by setup_remote.sh and
# check_formalizations, but not the bulk folders (data/, goedel_run_outputs/,
# tests/, goedel_original_scripts/, other logs/, etc.).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT="goedel_check_package.zip"
rm -f "$OUTPUT"

echo "Collecting files..."

# Directories that need __pycache__ / .pyc filtering
find \
    conjecturing_agents \
    logs/conjecture_formalization_logs_20mins \
    goedel_prover_hf_tokenizer \
    \( -name "__pycache__" -prune \) \
    -o \( \
        -type f \
        -not -name "*.pyc" \
        -not -name ".DS_Store" \
        -print \
    \) \
| zip "$OUTPUT" -@

rm -f conjecturing_agents.zip

find \
    conjecturing_agents \
    \( -name "__pycache__" -prune \) \
    -o \( \
        -type f \
        -not -name "*.pyc" \
        -not -name ".DS_Store" \
        -print \
    \) \
| zip conjecturing_agents.zip -@

# Individual root-level files
zip "$OUTPUT" \
    requirements.txt \
    setup_remote.sh \
    goedel_template.jinja \
    goedel_formalizer_template.jinja

SIZE=$(du -sh "$OUTPUT" | cut -f1)
COUNT=$(unzip -l "$OUTPUT" | tail -1 | awk '{print $2}')
echo ""
echo "Created: $OUTPUT  ($SIZE, $COUNT files)"
