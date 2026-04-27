#!/usr/bin/env bash
#set -euo pipefail
#https://gist.github.com/mrprajesh/352cbe661ee27a6b4627ae72d89479e6

# -- Profiler Settings --
SCENE_ID="4B_TYPICAL"   # [2B_TYPICAL, 3B_TYPICAL, 4B_TYPICAL]
BACKEND="numba"     # cupy
VARIANT="warp"      # "warp"
COMPILER="nvrtc"    # "nvcc"  
NUM_SYSTEMS=2048
SYSTEMS_PER_BLOCK=64
MAX_REGISTERS=192   #64, 128, 192
KERNEL_STEPS=512
GPU="GTX1070"       #P100

NVPROF_BIN="$(which nvprof)"
PYTHON_BIN="$(which python)"
#NVPROF_BIN=nvprof
#PYTHON_BIN="F:\Program Files\Python312\python.exe"

SCRIPT="kernel_prof.py"
OUT_DIR="profiles_nvprof/${GPU}/${SCENE_ID}"
mkdir -p "$OUT_DIR"

# ---- METRIC PASSES ----
# Forma: "tag metric1,metric2,..."
PASSES=(
  "occ sm_efficiency,branch_efficiency,achieved_occupancy,eligible_warps_per_cycle,ipc,issued_ipc,issue_slot_utilization"
  "mem gld_throughput,gst_throughput,dram_read_throughput,dram_write_throughput,shared_load_transactions,shared_store_transactions,shared_efficiency,shared_utilization,local_load_transactions,local_store_transactions,ldst_issued,ldst_executed,l2_read_transactions,l2_write_transactions"
  "stall stall_inst_fetch,stall_exec_dependency,stall_memory_dependency,stall_sync,stall_other,stall_memory_throttle"
  "flop flop_sp_efficiency,flop_dp_efficiency,inst_fp_32,inst_fp_64,flop_count_hp_add,flop_count_hp_mul,flop_count_hp_fma"
  "util special_fu_utilization,single_precision_fu_utilization,double_precision_fu_utilization"
)

export PYTHONIOENCODING=UTF-8
FNAME_TAG="${SCENE_ID}_${BACKEND}_${VARIANT}_${COMPILER}_ns${NUM_SYSTEMS}_spb${SYSTEMS_PER_BLOCK}_ks${KERNEL_STEPS}_r${MAX_REGISTERS}"

echo "Using NVPROF: $NVPROF_BIN"
echo "Using Python: $PYTHON_BIN"

for pass in "${PASSES[@]}"; do
  read -r tag METRICS <<< "$pass"
  echo "  -> PASS: ${tag}"
  #sudo -E
  "$NVPROF_BIN" \
  --metrics "$METRICS" \
  --log-file "${OUT_DIR}/nvprof_${tag}_${FNAME_TAG}.log" \
  -f \
  "$PYTHON_BIN" -X utf8 "$SCRIPT" \
    --scene_id "$SCENE_ID" \
    --backend "$BACKEND" \
    --variant "$VARIANT" \
    --num-systems "$NUM_SYSTEMS" \
    --systems-per-block "$SYSTEMS_PER_BLOCK" \
    --kernel-steps "$KERNEL_STEPS" \
    --max-kernel-steps "$KERNEL_STEPS" \
    --warm-up-steps 0 \
    --measured-steps 1 \
    --cuda_mode "profile" \
    --compiler "$COMPILER" \
    --max-registers "$MAX_REGISTERS"
done


echo "NVVP EXPORT — RAW EVENTS"
"$NVPROF_BIN" \
  --events active_cycles_pm \
  --events active_warps_pm \
  --events warps_launched \
  --events inst_issued0 \
  --events inst_issued1 \
  --events inst_issued2 \
  --events inst_executed \
  --events local_store \
  --events local_load \
  --events shared_store \
  --events shared_load \
  --events global_store \
  --events global_load \
  --log-file "${OUT_DIR}/nvprof_events_${FNAME_TAG}.log" \
  -f \
  "$PYTHON_BIN" -X utf8 "$SCRIPT" \
    --scene_id "$SCENE_ID" \
    --backend "$BACKEND" \
    --variant "$VARIANT" \
    --num-systems "$NUM_SYSTEMS" \
    --systems-per-block "$SYSTEMS_PER_BLOCK" \
    --kernel-steps "$KERNEL_STEPS" \
    --max-kernel-steps "$KERNEL_STEPS" \
    --warm-up-steps 0 \
    --measured-steps 1 \
    --cuda_mode "profile" \
    --compiler "$COMPILER" \
    --max-registers "$MAX_REGISTERS"


echo "NVVP EXPORT "
"$NVPROF_BIN" \
  --print-gpu-trace \
  --print-api-trace \
  --events all \
  --metrics sm_efficiency \
  --metrics achieved_occupancy \
  --metrics branch_efficiency \
  --metrics ipc \
  --metrics issued_ipc \
  --metrics gld_throughput \
  --metrics gst_throughput \
  --metrics dram_read_throughput \
  --metrics dram_write_throughput \
  --metrics stall_exec_dependency \
  --metrics stall_memory_dependency \
  --metrics stall_sync \
  --metrics flop_sp_efficiency \
  --metrics flop_dp_efficiency \
  --metrics single_precision_fu_utilization \
  --metrics double_precision_fu_utilization \
  --metrics special_fu_utilization \
  --metrics local_load_throughput \
  --metrics local_store_throughput \
  --metrics shared_load_throughput \
  --metrics shared_store_throughput \
  --metrics shared_load_transactions_per_request \
  --metrics shared_efficiency \
  --metrics warp_execution_efficiency \
  --export-profile "${OUT_DIR}/nvprof_metrics_${FNAME_TAG}.nvvp" \
  -f \
  "$PYTHON_BIN" -X utf8 "$SCRIPT" \
    --scene_id "$SCENE_ID" \
    --backend "$BACKEND" \
    --variant "$VARIANT" \
    --num-systems "$NUM_SYSTEMS" \
    --systems-per-block "$SYSTEMS_PER_BLOCK" \
    --kernel-steps "$KERNEL_STEPS" \
    --max-kernel-steps "$KERNEL_STEPS" \
    --warm-up-steps 0 \
    --measured-steps 1 \
    --cuda_mode "profile" \
    --compiler "$COMPILER" \
    --max-registers "$MAX_REGISTERS"



read -rp "Press enter to exit..."