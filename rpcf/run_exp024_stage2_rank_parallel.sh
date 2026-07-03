#!/usr/bin/env bash
set -euo pipefail

# EXP-024 stage2 rank 级并行：先生成共享 base shard，再把每个 rank shard
# 作为独立任务调度到空闲 GPU。TN 净化阶段默认每张空闲 GPU 同时跑 2 个 rank。

DATASET="${EXP024_DATASET:-thubenchmark}"
MODELS="${EXP024_MODELS:-tsception atcnet conformer}"
SEED="${EXP024_SEED:-42}"
FOLD="${EXP024_FOLD:-0}"
EPS="${EXP024_EPS:-0.03}"
ATTACK="${EXP024_ATTACK:-autoattack}"
CONDA_ENV="${CONDA_ENV:-torch}"
CONDA_SH="${CONDA_SH:-/home/yihangjie/miniconda3/etc/profile.d/conda.sh}"
CHAIN_ID="${EXP024_CHAIN_ID:-exp024_stage2_rank_parallel_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
CHAIN_ROOT="${EXP024_CHAIN_ROOT:-logs/exp024/${CHAIN_ID}}"
GPU_IDS_RAW="${EXP024_GPU_IDS:-${GPU_IDS:-0,1,2,3}}"
GPU_IDLE_MAX_USED_MB="${EXP024_GPU_IDLE_MAX_USED_MB:-100}"
GPU_POLL_SECONDS="${EXP024_GPU_POLL_SECONDS:-60}"
SLOTS_PER_IDLE_GPU="${EXP024_RANK_SLOTS_PER_IDLE_GPU:-2}"
MAX_ACTIVE_RANK_JOBS="${EXP024_MAX_ACTIVE_RANK_JOBS:-6}"
TRAIN_CACHE_SAMPLE_NUM="${TRAIN_CACHE_SAMPLE_NUM:-512}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-32}"
CHECKPOINT_EVERY="${RPCF_CHECKPOINT_EVERY:-8}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

TRAIN_RANKS="15,20,25,30,35,40"
TRAIN_CONFIGS="PTR3d_8_2048_rank15_3d_interpolate.yaml,PTR3d_8_2048_rank20_3d_interpolate.yaml,PTR3d_8_2048_rank25_3d_interpolate.yaml,PTR3d_8_2048_rank30_3d_interpolate.yaml,PTR3d_8_2048_rank35_3d_interpolate.yaml,PTR3d_8_2048_rank40_3d_interpolate.yaml"
PROTOCOL="train_only_subject_no_ea_subject_split"

if [[ -f "${CONDA_SH}" ]]; then
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
split_words "${TRAIN_RANKS}"
RANK_LIST=("${SPLIT_WORDS[@]}")
split_words "${TRAIN_CONFIGS}"
CONFIG_LIST=("${SPLIT_WORDS[@]}")

if [[ "${#GPU_ID_LIST[@]}" -eq 0 ]]; then
  echo "EXP024_GPU_IDS/GPU_IDS must contain at least one GPU id." >&2
  exit 1
fi
if [[ "${#RANK_LIST[@]}" -ne "${#CONFIG_LIST[@]}" ]]; then
  echo "TRAIN_RANKS and TRAIN_CONFIGS length mismatch." >&2
  exit 1
fi

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

checkpoint_for_model() {
  local model="$1"
  local run_id tag
  run_id="$(run_id_for_model "${model}")"
  tag="${run_id//[^A-Za-z0-9_.-]/_}"
  printf '%s\n' "checkpoints/${DATASET}_${model}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${tag}_madry_at_best.pth"
}

cache_for_model() {
  local model="$1"
  local run_id tag
  run_id="$(run_id_for_model "${model}")"
  tag="${run_id//[^A-Za-z0-9_.-]/_}"
  printf '%s\n' "purified_data/exp024/rpcf_train/${tag}_six_rank.pth"
}

rank_shard_for_model() {
  local model="$1"
  local rank="$2"
  printf '%s\n' "$(cache_for_model "${model}").work/rank${rank}.pth"
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
  local next_labels=()
  local next_gpus=()
  local pid label gpu status index
  for index in "${!ACTIVE_PIDS[@]}"; do
    pid="${ACTIVE_PIDS[$index]}"
    label="${ACTIVE_LABELS[$index]}"
    gpu="${ACTIVE_GPUS[$index]}"
    if kill -0 "${pid}" 2>/dev/null; then
      next_pids+=("${pid}")
      next_labels+=("${label}")
      next_gpus+=("${gpu}")
    else
      status=0
      wait "${pid}" || status=$?
      echo "[$(date -Is)] Finished ${label}: status=${status}"
      if [[ "${status}" -ne 0 ]]; then
        FAILED_JOBS+=("${label}:${status}")
      fi
    fi
  done
  ACTIVE_PIDS=("${next_pids[@]}")
  ACTIVE_LABELS=("${next_labels[@]}")
  ACTIVE_GPUS=("${next_gpus[@]}")
}

find_available_gpu() {
  local gpu active
  if (( ${#ACTIVE_PIDS[@]} >= MAX_ACTIVE_RANK_JOBS )); then
    return 1
  fi
  refresh_idle_gpus
  for gpu in "${IDLE_GPUS[@]}"; do
    active="$(active_count_for_gpu "${gpu}")"
    if (( active < SLOTS_PER_IDLE_GPU )); then
      ASSIGNED_GPU="${gpu}"
      return 0
    fi
  done
  for gpu in "${GPU_ID_LIST[@]}"; do
    active="$(active_count_for_gpu "${gpu}")"
    if (( active > 0 && active < SLOTS_PER_IDLE_GPU )); then
      ASSIGNED_GPU="${gpu}"
      return 0
    fi
  done
  return 1
}

wait_for_available_slot() {
  while true; do
    cleanup_finished_jobs
    if find_available_gpu; then
      return 0
    fi
    echo "[$(date -Is)] No idle rank slot available; active=${#ACTIVE_PIDS[@]}, waiting ${GPU_POLL_SECONDS}s..."
    sleep "${GPU_POLL_SECONDS}"
  done
}

run_generate_cache() {
  local gpu="$1"
  shift
  CUDA_VISIBLE_DEVICES="${gpu}" conda run -n "${CONDA_ENV}" --no-capture-output python -u -m rpcf.generate_cache "$@"
}

base_args_for_model() {
  local model="$1"
  local run_id tag checkpoint cache
  run_id="$(run_id_for_model "${model}")"
  tag="${run_id//[^A-Za-z0-9_.-]/_}"
  checkpoint="$(checkpoint_for_model "${model}")"
  cache="$(cache_for_model "${model}")"
  BASE_ARGS=(
    --dataset "${DATASET}" --model "${model}" --fold "${FOLD}"
    --seed "${SEED}" --attack "${ATTACK}" --eps "${EPS}"
    --checkpoint_path "${checkpoint}" --sample_num "${TRAIN_CACHE_SAMPLE_NUM}"
    --attack_batch_size "${ATTACK_BATCH_SIZE}" --ranks "${TRAIN_RANKS}"
    --configs "${TRAIN_CONFIGS}" --gpu_id 0 --tag "${tag}"
    --output_path "${cache}" --checkpoint_every "${CHECKPOINT_EVERY}"
  )
}

ensure_base_for_model() {
  local model="$1"
  local run_id log_root cache checkpoint gpu
  run_id="$(run_id_for_model "${model}")"
  log_root="logs/exp024/${run_id}"
  cache="$(cache_for_model "${model}")"
  checkpoint="$(checkpoint_for_model "${model}")"
  mkdir -p "${log_root}"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${cache}" ]]; then
    echo "[$(date -Is)] Reuse completed RPCF cache for ${model}: ${cache}"
    return
  fi
  if [[ "${DRY_RUN}" != "1" && ! -f "${checkpoint}" ]]; then
    echo "Required artifact not found: ${checkpoint}" >&2
    exit 1
  fi
  wait_for_available_slot
  gpu="${ASSIGNED_GPU}"
  base_args_for_model "${model}"
  echo "[$(date -Is)] Ensure base shard for ${model} on physical GPU ${gpu}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    {
      printf 'DRY_RUN base %s GPU %s:' "${model}" "${gpu}"
      printf ' %q' "${BASE_ARGS[@]}"
      printf '\n'
    } | tee -a "${log_root}/stage2_base.log"
  else
    run_generate_cache "${gpu}" "${BASE_ARGS[@]}" --base_only 2>&1 | tee -a "${log_root}/stage2_base.log"
  fi
}

launch_rank_job() {
  local model="$1"
  local rank="$2"
  local config="$3"
  local gpu="$4"
  local run_id log_root cache shard label
  run_id="$(run_id_for_model "${model}")"
  log_root="logs/exp024/${run_id}"
  cache="$(cache_for_model "${model}")"
  shard="$(rank_shard_for_model "${model}" "${rank}")"
  label="${model}:rank${rank}"
  mkdir -p "${log_root}"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${cache}" ]]; then
    echo "[$(date -Is)] Skip ${label}: final cache exists."
    return
  fi
  if [[ "${SKIP_EXISTING}" == "1" && -f "${shard}" ]]; then
    echo "[$(date -Is)] Skip ${label}: rank shard exists."
    return
  fi
  base_args_for_model "${model}"
  echo "[$(date -Is)] Start ${label} on physical GPU ${gpu}"
  (
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf 'DRY_RUN rank %s GPU %s rank=%s config=%s\n' "${model}" "${gpu}" "${rank}" "${config}"
    else
      run_generate_cache "${gpu}" "${BASE_ARGS[@]}" --ranks "${rank}" --configs "${config}" --rank_shard_only
    fi
  ) > "${log_root}/stage2_rank${rank}.log" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LABELS+=("${label}")
  ACTIVE_GPUS+=("${gpu}")
}

finalize_model() {
  local model="$1"
  local run_id log_root cache gpu
  run_id="$(run_id_for_model "${model}")"
  log_root="logs/exp024/${run_id}"
  cache="$(cache_for_model "${model}")"
  mkdir -p "${log_root}"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${cache}" ]]; then
    echo "[$(date -Is)] Reuse completed RPCF cache for ${model}: ${cache}"
    return
  fi
  for rank in "${RANK_LIST[@]}"; do
    if [[ "${DRY_RUN}" != "1" && ! -f "$(rank_shard_for_model "${model}" "${rank}")" ]]; then
      echo "Missing rank shard for ${model} rank ${rank}; cannot finalize." >&2
      exit 1
    fi
  done
  wait_for_available_slot
  gpu="${ASSIGNED_GPU}"
  base_args_for_model "${model}"
  echo "[$(date -Is)] Finalize ${model} RPCF cache on physical GPU ${gpu}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN finalize %s GPU %s\n' "${model}" "${gpu}" | tee -a "${log_root}/stage2_finalize.log"
  else
    run_generate_cache "${gpu}" "${BASE_ARGS[@]}" --finalize_only 2>&1 | tee -a "${log_root}/stage2_finalize.log"
  fi
}

mkdir -p "${CHAIN_ROOT}" purified_data/exp024/rpcf_train log_purify
printf '%s\n' "$$" > "${CHAIN_ROOT}/controller.pid"
cat > "${CHAIN_ROOT}/run_config.txt" <<EOF
EXPERIMENT_ID=EXP-024
STAGE=2_rank_parallel
CHAIN_ID=${CHAIN_ID}
MODELS=${MODELS}
SEED=${SEED}
FOLD=${FOLD}
EPS=${EPS}
ATTACK=${ATTACK}
GPU_IDS=${GPU_ID_LIST[*]}
GPU_IDLE_MAX_USED_MB=${GPU_IDLE_MAX_USED_MB}
SLOTS_PER_IDLE_GPU=${SLOTS_PER_IDLE_GPU}
MAX_ACTIVE_RANK_JOBS=${MAX_ACTIVE_RANK_JOBS}
TRAIN_CACHE_SAMPLE_NUM=${TRAIN_CACHE_SAMPLE_NUM}
ATTACK_BATCH_SIZE=${ATTACK_BATCH_SIZE}
SKIP_EXISTING=${SKIP_EXISTING}
EOF

ACTIVE_PIDS=()
ACTIVE_LABELS=()
ACTIVE_GPUS=()
FAILED_JOBS=()
IDLE_GPUS=()
ASSIGNED_GPU=""

for model in ${MODELS}; do
  ensure_base_for_model "${model}"
done

for model in ${MODELS}; do
  for index in "${!RANK_LIST[@]}"; do
    wait_for_available_slot
    launch_rank_job "${model}" "${RANK_LIST[$index]}" "${CONFIG_LIST[$index]}" "${ASSIGNED_GPU}"
    sleep 2
  done
done

while (( ${#ACTIVE_PIDS[@]} > 0 )); do
  cleanup_finished_jobs
  if (( ${#ACTIVE_PIDS[@]} > 0 )); then
    sleep "${GPU_POLL_SECONDS}"
  fi
done

if (( ${#FAILED_JOBS[@]} > 0 )); then
  echo "[$(date -Is)] EXP-024 rank parallel stage2 failed: ${FAILED_JOBS[*]}" >&2
  exit 1
fi

for model in ${MODELS}; do
  finalize_model "${model}"
done

while (( ${#ACTIVE_PIDS[@]} > 0 )); do
  cleanup_finished_jobs
  if (( ${#ACTIVE_PIDS[@]} > 0 )); then
    sleep "${GPU_POLL_SECONDS}"
  fi
done

echo "[$(date -Is)] EXP-024 rank parallel stage2 finished: ${CHAIN_ID}"
