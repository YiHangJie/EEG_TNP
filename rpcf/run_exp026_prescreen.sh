#!/usr/bin/env bash
set -euo pipefail

# EXP-026 三代表 backbone 并行预筛与统一参数选择。
# 正式后台运行：
#   EXP026_PRESCREEN_ID=exp026_prescreen_seed42_YYYYMMDD_HHMMSS \
#     EXP026_GPU_IDS=2,3,4 nohup setsid bash rpcf/run_exp026_prescreen.sh \
#     > logs/exp026/exp026_prescreen_seed42_YYYYMMDD_HHMMSS/controller.log 2>&1 < /dev/null &

RUN_ID="${EXP026_PRESCREEN_ID:-exp026_prescreen_seed42_$(date +%Y%m%d_%H%M%S)}"
ROOT="${EXP026_PRESCREEN_ROOT:-logs/exp026/${RUN_ID}}"
MODELS="${EXP026_PRESCREEN_MODELS:-eegnet tsception conformer}"
GPU_IDS_CSV="${EXP026_GPU_IDS:-0,1,2,3,4,5,6,7}"
IDLE_MAX_USED_MB="${EXP026_GPU_IDLE_MAX_USED_MB:-100}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "${ROOT}"
printf '%s\n' "$$" > "${ROOT}/controller.pid"

IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"
declare -A RESERVED=()
declare -a PIDS=()

gpu_used_mb() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | sed -n "$((gpu + 1))p" | tr -d ' '
}

pick_gpu() {
  local gpu used
  for gpu in "${GPU_IDS[@]}"; do
    [[ -n "${RESERVED[${gpu}]:-}" ]] && continue
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '%s\n' "${gpu}"
      return
    fi
    used="$(gpu_used_mb "${gpu}")"
    if [[ -n "${used}" && "${used}" -le "${IDLE_MAX_USED_MB}" ]]; then
      printf '%s\n' "${gpu}"
      return
    fi
  done
  return 1
}

for model in ${MODELS}; do
  gpu=""
  until gpu="$(pick_gpu)"; do
    echo "[$(date -Is)] Waiting for an idle EXP-026 GPU..."
    sleep 30
  done
  RESERVED["${gpu}"]=1
  model_log="${ROOT}/${model}/controller.log"
  mkdir -p "$(dirname "${model_log}")"
  echo "[$(date -Is)] prescreen ${model} -> physical GPU ${gpu}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    EXP026_MODEL="${model}" EXP026_PRESCREEN_ID="${RUN_ID}" \
      EXP026_PRESCREEN_ROOT="${ROOT}" CUDA_VISIBLE_DEVICES="${gpu}" GPU_ID=0 \
      DRY_RUN=1 bash rpcf/run_exp026_prescreen_model.sh
    continue
  fi
  env EXP026_MODEL="${model}" EXP026_PRESCREEN_ID="${RUN_ID}" \
    EXP026_PRESCREEN_ROOT="${ROOT}" CUDA_VISIBLE_DEVICES="${gpu}" GPU_ID=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash rpcf/run_exp026_prescreen_model.sh > "${model_log}" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" != "0" ]]; then
  echo "At least one EXP-026 prescreen backbone failed." >&2
  exit "${status}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY_RUN] python -m rpcf.select_exp026_hparams --root ${ROOT} --output_dir ${ROOT}/selection"
else
  conda run -n "${CONDA_ENV:-torch}" --no-capture-output python -u \
    -m rpcf.select_exp026_hparams --root "${ROOT}" \
    --models eegnet tsception conformer --output_dir "${ROOT}/selection" \
    2>&1 | tee -a "${ROOT}/selection.log"
fi
echo "[$(date -Is)] EXP-026 prescreen and selection finished: ${RUN_ID}"
