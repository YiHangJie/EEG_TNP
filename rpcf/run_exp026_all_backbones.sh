#!/usr/bin/env bash
set -euo pipefail

# EXP-026 六 backbone 正式全流程。
# 正式后台运行：
#   EXP026_RUN_ID=exp026_formal_seed42_YYYYMMDD_HHMMSS \
#     EXP026_SELECTED_ENV=logs/exp026/<prescreen>/selection/selected_hparams.env \
#     EXP026_GPU_IDS=2,3,4,5,6,7 nohup setsid bash rpcf/run_exp026_all_backbones.sh \
#     > logs/exp026/exp026_formal_seed42_YYYYMMDD_HHMMSS/controller.log 2>&1 < /dev/null &

RUN_ID="${EXP026_RUN_ID:-exp026_formal_seed42_$(date +%Y%m%d_%H%M%S)}"
ROOT="${EXP026_ROOT:-logs/exp026/${RUN_ID}}"
MODELS="${EXP026_MODELS:-eegnet tsception atcnet conformer deepconvnet tcnet}"
GPU_IDS_CSV="${EXP026_GPU_IDS:-0,1,2,3,4,5,6,7}"
IDLE_MAX_USED_MB="${EXP026_GPU_IDLE_MAX_USED_MB:-100}"
DRY_RUN="${DRY_RUN:-0}"
SELECTED_ENV="${EXP026_SELECTED_ENV:-}"
mkdir -p "${ROOT}"
printf '%s\n' "$$" > "${ROOT}/controller.pid"

if [[ "${DRY_RUN}" != "1" && "${SMOKE:-0}" != "1" && ! -f "${SELECTED_ENV}" ]]; then
  echo "EXP026_SELECTED_ENV is required after prescreen selection." >&2
  exit 1
fi

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
  echo "[$(date -Is)] formal ${model} -> physical GPU ${gpu}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    EXP026_MODEL="${model}" EXP026_RUN_ID="${RUN_ID}" EXP026_ROOT="${ROOT}" \
      EXP026_SELECTED_ENV="${SELECTED_ENV}" CUDA_VISIBLE_DEVICES="${gpu}" GPU_ID=0 \
      DRY_RUN=1 bash rpcf/run_exp026_backbone.sh
    continue
  fi
  env EXP026_MODEL="${model}" EXP026_RUN_ID="${RUN_ID}" EXP026_ROOT="${ROOT}" \
    EXP026_SELECTED_ENV="${SELECTED_ENV}" CUDA_VISIBLE_DEVICES="${gpu}" GPU_ID=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash rpcf/run_exp026_backbone.sh > "${model_log}" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" != "0" ]]; then
  echo "At least one EXP-026 backbone failed." >&2
  exit "${status}"
fi

if (( ${STOP_STAGE:-4} < 4 )); then
  echo "[$(date -Is)] Skip all-backbone summary because STOP_STAGE < 4."
elif [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY_RUN] python -m rpcf.compare_exp026_all --root ${ROOT} --models ${MODELS}"
else
  conda run -n "${CONDA_ENV:-torch}" --no-capture-output python -u \
    -m rpcf.compare_exp026_all --root "${ROOT}" --models ${MODELS} \
    --output_dir "${ROOT}/comparison_all" 2>&1 | tee -a "${ROOT}/summary_all.log"
fi
echo "[$(date -Is)] EXP-026 all-backbone pipeline finished: ${RUN_ID}"
