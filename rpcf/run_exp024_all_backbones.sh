#!/usr/bin/env bash
set -euo pipefail

# EXP-024 GPU-aware 调度：优先选择空闲 GPU，并用 CUDA_VISIBLE_DEVICES
#
# 常用后台运行示例：
#   nohup env EXP024_MODELS="deepconvnet tcnet" EXP024_GPU_IDS="4,5" \
#     bash rpcf/run_exp024_all_backbones.sh > logs/exp024/exp024_deepconvnet_tcnet.nohup.log 2>&1 &
# 隔离每个子流程，避免 TN 净化内部默认 cuda:0 与外部 gpu_id 混用。

RUN_TAG="${EXP024_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
CHAIN_ID="${EXP024_CHAIN_ID:-exp024_other_backbones_seed${EXP024_SEED:-42}_${RUN_TAG}}"
CHAIN_ROOT="${EXP024_CHAIN_ROOT:-logs/exp024/${CHAIN_ID}}"
MODELS="${EXP024_MODELS:-tsception atcnet conformer}"
SEED="${EXP024_SEED:-42}"
FOLD="${EXP024_FOLD:-0}"
EPS="${EXP024_EPS:-0.03}"
GPU_IDS_RAW="${EXP024_GPU_IDS:-${GPU_IDS:-0,1,2,3,4,5,6,7}}"
GPU_IDLE_MAX_USED_MB="${EXP024_GPU_IDLE_MAX_USED_MB:-100}"
GPU_POLL_SECONDS="${EXP024_GPU_POLL_SECONDS:-60}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-9}"
CONDA_SH="${CONDA_SH:-/home/yihangjie/miniconda3/etc/profile.d/conda.sh}"

if [[ -f "${CONDA_SH}" ]]; then
  # 让非交互 nohup shell 也能找到 conda。
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
fi

split_words() {
  local value="$1"
  value="${value//,/ }"
  # shellcheck disable=SC2206
  SPLIT_WORDS=(${value})
}

split_words "${GPU_IDS_RAW}"
GPU_ID_LIST=("${SPLIT_WORDS[@]}")
if [[ "${#GPU_ID_LIST[@]}" -eq 0 ]]; then
  echo "EXP024_GPU_IDS/GPU_IDS must contain at least one GPU id." >&2
  exit 1
fi

is_tn_only_stage() {
  [[ "${START_STAGE}" == "${STOP_STAGE}" && ("${START_STAGE}" == "2" || "${START_STAGE}" == "6") ]]
}

slots_per_idle_gpu() {
  if is_tn_only_stage; then
    printf '%s\n' "${EXP024_TN_SLOTS_PER_IDLE_GPU:-2}"
  else
    printf '%s\n' "${EXP024_SLOTS_PER_IDLE_GPU:-1}"
  fi
}

gpu_used_mb() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null \
    | awk 'NR == 1 {gsub(/[^0-9]/, "", $1); print $1}'
}

refresh_idle_gpus() {
  IDLE_GPUS=()
  local used gpu
  for gpu in "${GPU_ID_LIST[@]}"; do
    used="$(gpu_used_mb "${gpu}")"
    if [[ -z "${used}" ]]; then
      echo "[$(date -Is)] Skip GPU ${gpu}: unable to query memory usage." >&2
      continue
    fi
    if (( used <= GPU_IDLE_MAX_USED_MB )); then
      IDLE_GPUS+=("${gpu}")
    else
      echo "[$(date -Is)] GPU ${gpu} busy: used=${used}MiB > ${GPU_IDLE_MAX_USED_MB}MiB"
    fi
  done
}

active_count_for_gpu() {
  local query_gpu="$1"
  local count=0
  local gpu
  for gpu in "${ACTIVE_GPUS[@]}"; do
    if [[ "${gpu}" == "${query_gpu}" ]]; then
      count=$((count + 1))
    fi
  done
  printf '%s\n' "${count}"
}

cleanup_finished_jobs() {
  local next_pids=()
  local next_models=()
  local next_gpus=()
  local pid model gpu status index
  for index in "${!ACTIVE_PIDS[@]}"; do
    pid="${ACTIVE_PIDS[$index]}"
    model="${ACTIVE_MODELS[$index]}"
    gpu="${ACTIVE_GPUS[$index]}"
    if kill -0 "${pid}" 2>/dev/null; then
      next_pids+=("${pid}")
      next_models+=("${model}")
      next_gpus+=("${gpu}")
    else
      status=0
      wait "${pid}" || status=$?
      echo "[$(date -Is)] Finished EXP-024 ${model}: status=${status}"
      if [[ "${status}" -ne 0 ]]; then
        FAILED_MODELS+=("${model}:${status}")
      fi
    fi
  done
  ACTIVE_PIDS=("${next_pids[@]}")
  ACTIVE_MODELS=("${next_models[@]}")
  ACTIVE_GPUS=("${next_gpus[@]}")
}

find_available_gpu() {
  local capacity gpu active
  capacity="$(slots_per_idle_gpu)"

  refresh_idle_gpus
  for gpu in "${IDLE_GPUS[@]}"; do
    active="$(active_count_for_gpu "${gpu}")"
    if (( active < capacity )); then
      ASSIGNED_GPU="${gpu}"
      return 0
    fi
  done

  if is_tn_only_stage; then
    for gpu in "${GPU_ID_LIST[@]}"; do
      active="$(active_count_for_gpu "${gpu}")"
      if (( active > 0 && active < capacity )); then
        ASSIGNED_GPU="${gpu}"
        return 0
      fi
    done
  fi

  return 1
}

wait_for_available_slot() {
  while true; do
    cleanup_finished_jobs
    if find_available_gpu; then
      return 0
    fi
    echo "[$(date -Is)] No idle GPU slot available; active=${#ACTIVE_PIDS[@]}, waiting ${GPU_POLL_SECONDS}s..."
    sleep "${GPU_POLL_SECONDS}"
  done
}

run_id_for_model() {
  local model="$1"
  local env_name="EXP024_RUN_ID_${model^^}"
  local override="${!env_name:-}"
  if [[ -n "${override}" ]]; then
    printf '%s\n' "${override}"
  else
    printf '%s\n' "${CHAIN_ID}_${model}"
  fi
}

mkdir -p "${CHAIN_ROOT}"
printf '%s\n' "$$" > "${CHAIN_ROOT}/controller.pid"
cat > "${CHAIN_ROOT}/run_config.txt" <<EOF
EXPERIMENT_ID=EXP-024
CHAIN_ID=${CHAIN_ID}
MODELS=${MODELS}
SEED=${SEED}
FOLD=${FOLD}
EPS=${EPS}
GPU_IDS=${GPU_ID_LIST[*]}
GPU_IDLE_MAX_USED_MB=${GPU_IDLE_MAX_USED_MB}
START_STAGE=${START_STAGE}
STOP_STAGE=${STOP_STAGE}
TN_ONLY_STAGE=$(is_tn_only_stage && echo 1 || echo 0)
SLOTS_PER_IDLE_GPU=$(slots_per_idle_gpu)
SKIP_EXISTING=${SKIP_EXISTING}
EOF

ACTIVE_PIDS=()
ACTIVE_MODELS=()
ACTIVE_GPUS=()
FAILED_MODELS=()
IDLE_GPUS=()
ASSIGNED_GPU=""

for model in ${MODELS}; do
  wait_for_available_slot
  assigned_gpu="${ASSIGNED_GPU}"
  run_id="$(run_id_for_model "${model}")"
  log_root="logs/exp024/${run_id}"
  mkdir -p "${log_root}"
  echo "[$(date -Is)] Start EXP-024 ${model}: ${run_id} on physical GPU ${assigned_gpu}"
  (
    CUDA_VISIBLE_DEVICES="${assigned_gpu}" \
    EXP024_MODEL="${model}" \
    EXP024_SEED="${SEED}" \
    EXP024_FOLD="${FOLD}" \
    EXP024_EPS="${EPS}" \
    EXP024_RUN_ID="${run_id}" \
    EXP024_LOG_ROOT="${log_root}" \
    GPU_ID="0" \
    DRY_RUN="${DRY_RUN}" \
    SKIP_EXISTING="${SKIP_EXISTING}" \
    START_STAGE="${START_STAGE}" \
    STOP_STAGE="${STOP_STAGE}" \
      bash rpcf/run_exp024_backbone.sh
  ) > "${CHAIN_ROOT}/${model}.controller.log" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_MODELS+=("${model}")
  ACTIVE_GPUS+=("${assigned_gpu}")
  sleep 2
done

while (( ${#ACTIVE_PIDS[@]} > 0 )); do
  cleanup_finished_jobs
  if (( ${#ACTIVE_PIDS[@]} > 0 )); then
    sleep "${GPU_POLL_SECONDS}"
  fi
done

if (( ${#FAILED_MODELS[@]} > 0 )); then
  echo "[$(date -Is)] EXP-024 chain finished with failures: ${FAILED_MODELS[*]}" >&2
  exit 1
fi

echo "[$(date -Is)] EXP-024 all-backbone chain finished: ${CHAIN_ID}"
