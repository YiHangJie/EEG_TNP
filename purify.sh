#!/usr/bin/env bash
set -u

# Batch-run every purification config under configs/thubenchmark.
# Usage examples:
#   bash purify.sh
#   MAX_JOBS=1 SAMPLE_NUM=128 ATTACK=pgd bash purify.sh
#   GPU_IDS=0,1 MAX_JOBS=2 bash purify.sh
#   DRY_RUN=1 CONFIG_PATTERN='PTR3d_*.yaml' bash purify.sh
#   nohup bash purify.sh > log_purify/batch_thubenchmark.log 2>&1 &
#   GPU_IDS=0 MAX_JOBS=1 CONFIG_PATTERN='PTR3d_8_2048_rank*_3d_interpolate.yaml' nohup bash purify.sh > log_purify/batch_PTR3d.log 2>&1 &


DATASET="${DATASET:-thubenchmark}"
CLASSIFIER="${CLASSIFIER:-eegnet}"
ATTACK="${ATTACK:-autoattack}"
EPS="${EPS:-0.1}"
SEED="${SEED:-42}"
FOLD="${FOLD:-0}"
GPU_ID="${GPU_ID:-0}"
GPU_IDS="${GPU_IDS:-${GPU_ID}}"
SAMPLE_NUM="${SAMPLE_NUM:-512}"
MAX_JOBS="${MAX_JOBS:-2}"
POLL_SECONDS="${POLL_SECONDS:-60}"
CONDA_ENV="${CONDA_ENV:-torch}"
CONFIG_PATTERN="${CONFIG_PATTERN:-*.yaml}"
DRY_RUN="${DRY_RUN:-0}"
CONFIG_DIR="configs/${DATASET}"
FAILED_LIST="log_purify/failed_${DATASET}_${ATTACK}_eps${EPS}_${SEED}.txt"

IFS=',' read -r -a gpu_ids <<< "${GPU_IDS}"
normalized_gpu_ids=()
for gpu_id in "${gpu_ids[@]}"; do
  gpu_id="${gpu_id//[[:space:]]/}"
  if [[ -z "${gpu_id}" ]]; then
    continue
  fi
  if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU id in GPU_IDS: ${gpu_id}" >&2
    exit 1
  fi
  normalized_gpu_ids+=("${gpu_id}")
done
gpu_ids=("${normalized_gpu_ids[@]}")

if [[ "${#gpu_ids[@]}" -eq 0 ]]; then
  echo "No valid GPU ids found in GPU_IDS=${GPU_IDS}" >&2
  exit 1
fi
gpu_count="${#gpu_ids[@]}"

target_pid="${WAIT_FOR_PID:-}"
if [[ -n "${target_pid}" ]]; then
  while kill -0 "${target_pid}" 2> /dev/null; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID ${target_pid} still running..."
    sleep 30
  done
fi

if command -v conda > /dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

mkdir -p log_purify
: > "${FAILED_LIST}"

if [[ ! -d "${CONFIG_DIR}" ]]; then
  echo "Config directory not found: ${CONFIG_DIR}" >&2
  exit 1
fi

mapfile -t configs < <(find "${CONFIG_DIR}" -maxdepth 1 -type f -name "${CONFIG_PATTERN}" -printf '%f\n' | sort)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Dataset: ${DATASET}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Attack: ${ATTACK}, eps: ${EPS}, seed: ${SEED}, fold: ${FOLD}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sample num: ${SAMPLE_NUM}, max jobs: ${MAX_JOBS}, GPUs: ${gpu_ids[*]}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Config pattern: ${CONFIG_PATTERN}, dry run: ${DRY_RUN}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Config count: ${#configs[@]}"

if [[ "${#configs[@]}" -eq 0 ]]; then
  echo "No configs matched ${CONFIG_DIR}/${CONFIG_PATTERN}" >&2
  exit 1
fi

run_one() {
  local config="$1"
  local assigned_gpu="$2"
  local log_file="log_purify/purify_${DATASET}_${CLASSIFIER}_${ATTACK}_eps${EPS}_${SEED}_${config}.log"

  if [[ -f "${log_file}" ]] && grep -q "Mean mse of purified clean data" "${log_file}"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skip completed: ${config}"
    return 0
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Start: ${config} on GPU ${assigned_gpu}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "python purify.py --config ${config} --sample_num ${SAMPLE_NUM} --dataset ${DATASET} --model ${CLASSIFIER} --attack ${ATTACK} --eps ${EPS} --seed ${SEED} --fold ${FOLD} --gpu_id ${assigned_gpu}"
    return 0
  fi

  python purify.py \
    --config "${config}" \
    --sample_num "${SAMPLE_NUM}" \
    --dataset "${DATASET}" \
    --model "${CLASSIFIER}" \
    --attack "${ATTACK}" \
    --eps "${EPS}" \
    --seed "${SEED}" \
    --fold "${FOLD}" \
    --gpu_id "${assigned_gpu}"
  local status=$?

  if [[ "${status}" -eq 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done: ${config} on GPU ${assigned_gpu}"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed(${status}): ${config} on GPU ${assigned_gpu}"
    echo "${config}" >> "${FAILED_LIST}"
  fi

  return "${status}"
}

gpu_index=0
for config in "${configs[@]}"; do
  while [[ "$(jobs -rp | wc -l)" -ge "${MAX_JOBS}" ]]; do
    sleep "${POLL_SECONDS}"
  done
  assigned_gpu="${gpu_ids[$((gpu_index % gpu_count))]}"
  ( run_one "${config}" "${assigned_gpu}" ) &
  gpu_index=$((gpu_index + 1))
  sleep 2
done

wait

if [[ -s "${FAILED_LIST}" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch finished with failures. See ${FAILED_LIST}"
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch finished successfully."
