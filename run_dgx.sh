#!/usr/bin/env bash
# ==============================================================================
# YOLO11s Fine-Tuning Launcher for Linux / NVIDIA DGX
# ==============================================================================

set -eo pipefail

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo " Web PII YOLO11s Fine-Tuning (Linux DGX)"
echo " Project Directory : $SCRIPT_DIR"
echo " Timestamp         : $(date)"
echo "=================================================="

# Check for python3
if ! command -v python3 &>/dev/null; then
    echo "[Error] python3 could not be found. Please ensure Python 3 is installed." >&2
    exit 1
fi

# Print GPU information if nvidia-smi is available
if command -v nvidia-smi &>/dev/null; then
    echo "[System] Available NVIDIA GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    echo "--------------------------------------------------"
fi

# Execute training with any user-provided arguments (e.g., ./run_dgx.sh --device 0,1)
exec python3 train_yolo.py "$@"
