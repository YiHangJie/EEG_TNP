#!/usr/bin/env bash
set -euo pipefail

# 等待 EXP-026 预筛与参数选择完成，再自动启动六 backbone 正式实验。
# 后台运行：
#   EXP026_PRESCREEN_ROOT=logs/exp026/<prescreen_id> \
#     EXP026_RUN_ID=exp026_formal_seed42_YYYYMMDD_HHMMSS \
#     EXP026_GPU_IDS=2,3,4,5,6,7 nohup setsid bash rpcf/run_exp026_after_prescreen.sh \
#     > logs/exp026/exp026_formal_seed42_YYYYMMDD_HHMMSS/watcher.log 2>&1 < /dev/null &

PRESCREEN_ROOT="${EXP026_PRESCREEN_ROOT:?EXP026_PRESCREEN_ROOT is required}"
FORMAL_RUN_ID="${EXP026_RUN_ID:-exp026_formal_seed42_$(date +%Y%m%d_%H%M%S)}"
FORMAL_ROOT="${EXP026_ROOT:-logs/exp026/${FORMAL_RUN_ID}}"
SELECTED_ENV="${PRESCREEN_ROOT}/selection/selected_hparams.env"
POLL_SECONDS="${EXP026_POLL_SECONDS:-60}"
GPU_IDS="${EXP026_GPU_IDS:-2,3,4,5,6,7}"
mkdir -p "${FORMAL_ROOT}"
printf '%s\n' "$$" > "${FORMAL_ROOT}/watcher.pid"

prescreen_pid=""
if [[ -f "${PRESCREEN_ROOT}/controller.pid" ]]; then
  prescreen_pid="$(tr -d '[:space:]' < "${PRESCREEN_ROOT}/controller.pid")"
fi

while [[ ! -f "${SELECTED_ENV}" ]]; do
  if [[ -n "${prescreen_pid}" ]] && ! kill -0 "${prescreen_pid}" 2>/dev/null; then
    echo "Prescreen controller ${prescreen_pid} exited without selected_hparams.env." >&2
    exit 1
  fi
  echo "[$(date -Is)] Waiting for EXP-026 selection: ${SELECTED_ENV}"
  sleep "${POLL_SECONDS}"
done

echo "[$(date -Is)] Selection ready; starting formal EXP-026: ${FORMAL_RUN_ID}"
exec env EXP026_RUN_ID="${FORMAL_RUN_ID}" EXP026_ROOT="${FORMAL_ROOT}" \
  EXP026_SELECTED_ENV="${SELECTED_ENV}" EXP026_GPU_IDS="${GPU_IDS}" \
  EXP026_GPU_IDLE_MAX_USED_MB="${EXP026_GPU_IDLE_MAX_USED_MB:-100}" \
  RPCF_EVAL_BATCH_SIZE="${RPCF_EVAL_BATCH_SIZE:-2}" \
  ONLINE_AT_BATCH_SIZE="${ONLINE_AT_BATCH_SIZE:-16}" \
  ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-8}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash rpcf/run_exp026_all_backbones.sh
