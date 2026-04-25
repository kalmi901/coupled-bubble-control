#!/usr/bin/env bash
#set -euo pipefail

# -- Measurement Settings --
SCENE_ID="2B_TYPICAL"   # [2B_TYPICAL, 3B_TYPICAL, 4B_TYPICAL]
BACKEND="numba"     # [cupy numba]
VARIANT="shared"    # [warp shared]
COMPILER="nvrtc"    # [nvcc nvrtc]  
SYSTEMS_PER_BLOCK=64
MAX_REGISTERS=192   #64, 128, 192

WARMUP_STEPS=1
MEAS_NUM_STEPS=5
KERNEL_STEPS=512

NUM_ENVS_LIST=(256 512 1024 2048 4096 8192 16384 32768)
SEED_LIST=(11 42 55 77 89 119)

PYTHON_BIN="$(which python)"
SCRIPT="kernel_prof.py"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SAVE_FILE_NAME="${SCENE_ID}_${BACKEND}_${VARIANT}_${COMPILER}_ns${NUM_SYSTEMS}_spb${SYSTEMS_PER_BLOCK}_ks${KERNEL_STEPS}_r${MAX_REGISTERS}_run${TIMESTAMP}.csv"
#mkdir -p "$OUT_DIR"

export PYTHONIOENCODING=UTF-8

for NUM_ENVS in "${NUM_ENVS_LIST[@]}"; do
  for SEED in "${SEED_LIST[@]}"; do
    echo "  -> NUM_EVS: ${NUM_ENVS}"
    echo "  -> SEED: ${SEED}"

    "$PYTHON_BIN" -X utf8 "$SCRIPT" \
        --scene_id "$SCENE_ID" \
        --backend "$BACKEND" \
        --variant "$VARIANT" \
        --num-systems "$NUM_ENVS" \
        --systems-per-block "$SYSTEMS_PER_BLOCK" \
        --kernel-steps "$KERNEL_STEPS" \
        --max-kernel-steps 10000000 \
        --warm-up-steps "$WARMUP_STEPS" \
        --measured-steps "$MEAS_NUM_STEPS" \
        --seed "$SEED" \
        --cuda_mode "release" \
        --compiler "$COMPILER" \
        --max-registers "$MAX_REGISTERS" \
        --save_file_name "$SAVE_FILE_NAME"
    done
done

read -rp "Press enter to exit..."