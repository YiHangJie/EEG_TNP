#!/usr/bin/env bash
set -euo pipefail

# EXP-031 正式后台运行：
#   mkdir -p logs/exp031
#   nohup setsid env EXP031_RUN_ID=exp031_full_$(date +%Y%m%d_%H%M%S) \
#     GPU_IDS=0,1,2,3,4,5,6 bash rpcf/run_exp031.sh \
#     > logs/exp031/controller.nohup.log 2>&1 < /dev/null &
# 仅生成/检查完整计划：
#   DRY_RUN=1 bash rpcf/run_exp031.sh
# smoke（不进入正式汇总）：
#   SMOKE=1 bash rpcf/run_exp031.sh
# 只调度 THU/EEGNet 五 seed 的攻击、TNP 与 BPDA 闭环：
#   EXP031_RUN_ID=<run_id> TASK_SCOPE=thu_eegnet_closure bash rpcf/run_exp031.sh

RUN_ID="${EXP031_RUN_ID:-exp031_$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6}"
START_STAGE="${START_STAGE:-0}"
STOP_STAGE="${STOP_STAGE:-7}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
TASK_ID="${TASK_ID:-}"
TASK_SCOPE="${TASK_SCOPE:-all}"
RESERVED_GPU_PROCESSES="${RESERVED_GPU_PROCESSES:-}"
CONDA_ENV="${CONDA_ENV:-torch}"
RUN_DIR="logs/exp031/${RUN_ID}"

mkdir -p "${RUN_DIR}"
printf '%s\n' "$$" > "${RUN_DIR}/controller.pid"

args=(
  --run-id "${RUN_ID}"
  --gpu-ids "${GPU_IDS}"
  --start-stage "${START_STAGE}"
  --stop-stage "${STOP_STAGE}"
  --task-scope "${TASK_SCOPE}"
)
if [[ -n "${RESERVED_GPU_PROCESSES}" ]]; then
  args+=(--reserved-gpu-processes "${RESERVED_GPU_PROCESSES}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  args+=(--dry-run)
fi
if [[ "${SMOKE}" == "1" ]]; then
  args+=(--smoke)
fi
if [[ -n "${TASK_ID}" ]]; then
  args+=(--task-id "${TASK_ID}")
fi

{
  echo "EXPERIMENT_ID=EXP-031"
  echo "RUN_ID=${RUN_ID}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "START_STAGE=${START_STAGE}"
  echo "STOP_STAGE=${STOP_STAGE}"
  echo "DRY_RUN=${DRY_RUN}"
  echo "SMOKE=${SMOKE}"
  echo "TASK_ID=${TASK_ID}"
  echo "TASK_SCOPE=${TASK_SCOPE}"
  echo "RESERVED_GPU_PROCESSES=${RESERVED_GPU_PROCESSES}"
} > "${RUN_DIR}/launcher_config.txt"

conda run -n "${CONDA_ENV}" --no-capture-output \
  python -u -m rpcf.exp031 run "${args[@]}" 2>&1 | tee -a "${RUN_DIR}/controller.log"

if [[ "${DRY_RUN}" == "0" && "${SMOKE}" == "0" ]] \
    && (( STOP_STAGE >= 7 )); then
  conda run -n "${CONDA_ENV}" --no-capture-output \
    python -u -m rpcf.summarize_exp031 --run-id "${RUN_ID}" \
    2>&1 | tee -a "${RUN_DIR}/controller.log"
fi
