#!/usr/bin/env bash
#set -euo pipefail
#https://gist.github.com/mrprajesh/352cbe661ee27a6b4627ae72d89479e6


SCENE_ID="2B_TYPICAL"

PYTHON="F:\Program Files\Python312\python.exe"
SCRIPT="kernel_prof.py"
OUT_DIR="profiles_nvprof"
mkdir -p "$OUT_DIR"

export PYTHONIOENCODING=UTF-8

nvprof \
    --print-gpu-trace \
    --metrics sm_efficiency \
    --metrics achieved_occupancy \
    --metrics branch_efficiency \
    --metrics gld_throughput \
    --metrics gst_throughput \
    --metrics local_load_throughput \
    --metrics local_store_throughput \
    --metrics shared_load_throughput \
    --metrics shared_store_throughput \
    --metrics shared_load_transactions_per_request \
    --metrics shared_efficiency \
    --metrics warp_execution_efficiency \
    "$PYTHON" -X utf8 "$SCRIPT"


read -rp "Press enter to exit..."