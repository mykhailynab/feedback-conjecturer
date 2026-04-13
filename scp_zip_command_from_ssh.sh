#!/bin/bash
# Usage: bash scp_zip.sh <ssh command>
# Example: bash scp_zip.sh ssh -i ~/.ssh/vastai -p 7267 root@82.141.118.38 -L 8080:localhost:8080
#
# Extracts host, port, identity file, and user from the ssh command and prints
# the equivalent scp command for goedel_check_package.zip.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIP="$SCRIPT_DIR/goedel_check_package.zip"

[[ $# -eq 0 ]] && { echo "Usage: $0 <ssh command>"; exit 1; }

# Drop leading 'ssh' token if present
args=("$@")
[[ "${args[0]}" == "ssh" ]] && args=("${args[@]:1}")

PORT=""
IDENTITY=""
USER_HOST=""

i=0
while [[ $i -lt ${#args[@]} ]]; do
    arg="${args[$i]}"
    case "$arg" in
        -p) PORT="${args[$((i+1))]}"; i=$((i+2)) ;;
        -i) IDENTITY="${args[$((i+1))]}"; i=$((i+2)) ;;
        -L|-R|-D|-o|-l|-F|-J|-W|-b|-c|-D|-e|-m|-Q) i=$((i+2)) ;;  # skip known flag+value pairs
        -*) i=$((i+1)) ;;                                            # skip standalone flags
        *)  [[ -z "$USER_HOST" ]] && USER_HOST="$arg"; i=$((i+1)) ;;
    esac
done

[[ -z "$USER_HOST" ]] && { echo "Could not extract user@host from: $*"; exit 1; }

SCP_CMD="scp"
[[ -n "$IDENTITY" ]] && SCP_CMD+=" -i $IDENTITY"
[[ -n "$PORT"     ]] && SCP_CMD+=" -P $PORT"
SCP_CMD+=" $ZIP $USER_HOST:/workspace/"

echo "$SCP_CMD"
