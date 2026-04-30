#!/usr/bin/env bash
set -euo pipefail

# End-to-end purified-data augmentation pipeline.
# Usage:
#   bash purify_aug_pipeline.sh
#   DATASET=thubenchmark GPU_ID=0 EPS=0.03 RUN_TAG=puraug_rank15-30_n512_eps0p03 bash purify_aug_pipeline.sh
#   TRAIN_ADV_AUG=1 TRAIN_ADV_CHECKPOINT_PATH=checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_clean_eps0_42_fold0_best.pth RUN_TAG=puraug_clean_adv_rank25-30_n512_eps0p03 bash purify_aug_pipeline.sh
#   TRAIN_CONFIGS_CSV='PTR3d_8_2048_rank20_3d_interpolate.yaml' SAMPLE_NUM=2 EPOCHS=1 PATIENCE=1 bash purify_aug_pipeline.sh
#   GPU_IDS=0,1 MAX_JOBS=2 nohup bash purify_aug_pipeline.sh > logs/purify_aug_pipeline_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   AT_STRATEGY=madry nohup bash purify_aug_pipeline.sh > logs/purify_aug_pipeline_$(date +%Y%m%d_%H%M%S).log 2>&1 &




DATASET="${DATASET:-thubenchmark}"
MODEL="${MODEL:-eegnet}"
GPU_ID="${GPU_ID:-0}"
EPS="${EPS:-0.03}"
ATTACK="${ATTACK:-autoattack}"
AT_STRATEGY="${AT_STRATEGY:-clean}"
SEED="${SEED:-42}"
FOLD="${FOLD:-0}"
SAMPLE_NUM="${SAMPLE_NUM:-512}"
RUN_TAG="${RUN_TAG:-puraug_rank25-30_n${SAMPLE_NUM}_eps${EPS//./p}}"
USE_EA="${USE_EA:-0}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
MAX_JOBS="${MAX_JOBS:-2}"
GPU_IDS="${GPU_IDS:-${GPU_ID}}"
POLL_SECONDS="${POLL_SECONDS:-10}"
TRAIN_ADV_AUG="${TRAIN_ADV_AUG:-0}"
TRAIN_ADV_ATTACK="${TRAIN_ADV_ATTACK:-${ATTACK}}"
TRAIN_ADV_EPS="${TRAIN_ADV_EPS:-${EPS}}"
TRAIN_ADV_SAMPLE_NUM="${TRAIN_ADV_SAMPLE_NUM:-${SAMPLE_NUM}}"
TRAIN_ADV_BATCH_SIZE="${TRAIN_ADV_BATCH_SIZE:-32}"
TRAIN_ADV_CHECKPOINT_PATH="${TRAIN_ADV_CHECKPOINT_PATH:-}"

EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
PATIENCE="${PATIENCE:-20}"

CONDA_ENV="${CONDA_ENV:-torch}"
ACTIVATE_CONDA="${ACTIVATE_CONDA:-1}"

# TRAIN_CONFIGS_CSV="${TRAIN_CONFIGS_CSV:-PTR3d_8_2048_rank15_3d_interpolate.yaml,PTR3d_8_2048_rank20_3d_interpolate.yaml,PTR3d_8_2048_rank25_3d_interpolate.yaml,PTR3d_8_2048_rank30_3d_interpolate.yaml}"
TRAIN_CONFIGS_CSV="${TRAIN_CONFIGS_CSV:-PTR3d_8_2048_rank25_3d_interpolate.yaml,PTR3d_8_2048_rank30_3d_interpolate.yaml}"
PURIFY_CONFIGS_CSV="${PURIFY_CONFIGS_CSV:-${TRAIN_CONFIGS_CSV}}"

safe_token() {
  local value="$1"
  if [[ -z "${value}" ]]; then
    echo "none"
    return
  fi
  printf '%s' "${value}" | sed 's/[^A-Za-z0-9_.-]/_/g'
}

RUN_TAG_SAFE="$(safe_token "${RUN_TAG}")"

if [[ "${USE_EA}" == "1" ]]; then
  EA_ARG="--use_ea"
  PROTOCOL_SHORT="ea"
  PROTOCOL_TAG="train_only_subject_ea_subject_split"
else
  EA_ARG="--no_ea"
  PROTOCOL_SHORT="no_ea"
  PROTOCOL_TAG="train_only_subject_no_ea_subject_split"
fi

if [[ "${AT_STRATEGY}" == "clean" ]]; then
  TRAIN_EPS="0"
else
  TRAIN_EPS="${EPS}"
fi

if [[ "${TRAIN_ADV_AUG}" == "1" && -z "${TRAIN_ADV_CHECKPOINT_PATH}" ]]; then
  TRAIN_ADV_CHECKPOINT_PATH="checkpoints/${DATASET}_${MODEL}_${PROTOCOL_TAG}_clean_eps0_${SEED}_fold${FOLD}_best.pth"
fi

IFS=',' read -r -a TRAIN_CONFIGS <<< "${TRAIN_CONFIGS_CSV}"
IFS=',' read -r -a PURIFY_CONFIGS <<< "${PURIFY_CONFIGS_CSV}"
TRAIN_ADV_CONFIGS_CSV="${TRAIN_ADV_CONFIGS_CSV:-${TRAIN_CONFIGS_CSV}}"
IFS=',' read -r -a TRAIN_ADV_CONFIGS <<< "${TRAIN_ADV_CONFIGS_CSV}"

if [[ ! "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer, got: ${MAX_JOBS}" >&2
  exit 1
fi
if [[ ! "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_SECONDS must be a positive integer, got: ${POLL_SECONDS}" >&2
  exit 1
fi

IFS=',' read -r -a raw_gpu_ids <<< "${GPU_IDS}"
gpu_ids=()
for gpu_id in "${raw_gpu_ids[@]}"; do
  gpu_id="${gpu_id//[[:space:]]/}"
  if [[ -z "${gpu_id}" ]]; then
    continue
  fi
  if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU id in GPU_IDS: ${gpu_id}" >&2
    exit 1
  fi
  gpu_ids+=("${gpu_id}")
done
if [[ "${#gpu_ids[@]}" -eq 0 ]]; then
  echo "No valid GPU ids found in GPU_IDS=${GPU_IDS}" >&2
  exit 1
fi
gpu_count="${#gpu_ids[@]}"

if [[ "${ACTIVATE_CONDA}" == "1" ]] && command -v conda > /dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
mkdir -p "${NUMBA_CACHE_DIR}" "${XDG_CACHE_HOME}"

run_cmd() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

wait_for_slots() {
  local max_jobs="$1"
  while [[ "$(jobs -rp | wc -l)" -ge "${max_jobs}" ]]; do
    sleep "${POLL_SECONDS}"
  done
}

wait_for_stage() {
  local stage_name="$1"
  shift
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${stage_name} job failed: pid ${pid}" >&2
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${stage_name} failed. Stop pipeline." >&2
    exit 1
  fi
}

aug_paths=()

echo "======================================================"
echo "RUN_TAG=${RUN_TAG_SAFE}"
echo "DATASET=${DATASET}, MODEL=${MODEL}, FOLD=${FOLD}, SEED=${SEED}, GPU=${GPU_ID}"
echo "AT_STRATEGY=${AT_STRATEGY}, EPS=${EPS}, ATTACK=${ATTACK}, USE_EA=${USE_EA}"
echo "TRAIN_ADV_AUG=${TRAIN_ADV_AUG}, TRAIN_ADV_ATTACK=${TRAIN_ADV_ATTACK}, TRAIN_ADV_EPS=${TRAIN_ADV_EPS}"
if [[ "${TRAIN_ADV_AUG}" == "1" ]]; then
  echo "TRAIN_ADV_CHECKPOINT_PATH=${TRAIN_ADV_CHECKPOINT_PATH}"
fi
echo "PURIFY MAX_JOBS=${MAX_JOBS}, GPU_IDS=${gpu_ids[*]}"
echo "======================================================"

if [[ "${TRAIN_ADV_AUG}" == "1" && "${DRY_RUN}" != "1" && ! -f "${TRAIN_ADV_CHECKPOINT_PATH}" ]]; then
  echo "TRAIN_ADV_CHECKPOINT_PATH not found: ${TRAIN_ADV_CHECKPOINT_PATH}" >&2
  echo "Set TRAIN_ADV_CHECKPOINT_PATH to the classifier checkpoint used for train adversarial generation." >&2
  exit 1
fi

echo "[Stage 1] Purify clean training samples"
stage1_pids=()
stage1_gpu_index=0
for config in "${TRAIN_CONFIGS[@]}"; do
  config="${config//[[:space:]]/}"
  if [[ -z "${config}" ]]; then
    continue
  fi
  config_stem="${config%.yaml}"
  output_path="purified_data/train_clean/${DATASET}_${MODEL}_${PROTOCOL_SHORT}_fold${FOLD}_seed${SEED}_train_clean_${config_stem}_n${SAMPLE_NUM}_tag${RUN_TAG_SAFE}.pth"
  aug_paths+=("${output_path}")
  if [[ -f "${output_path}" && "${OVERWRITE}" != "1" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skip existing: ${output_path}"
    continue
  fi
  overwrite_args=()
  if [[ "${OVERWRITE}" == "1" ]]; then
    overwrite_args=(--overwrite)
  fi
  wait_for_slots "${MAX_JOBS}"
  assigned_gpu="${gpu_ids[$((stage1_gpu_index % gpu_count))]}"
  stage1_gpu_index=$((stage1_gpu_index + 1))
  (
    run_cmd python -u purify_train_clean.py \
      --dataset "${DATASET}" \
      --model "${MODEL}" \
      --fold "${FOLD}" \
      --config "${config}" \
      --sample_num "${SAMPLE_NUM}" \
      --seed "${SEED}" \
      --gpu_id "${assigned_gpu}" \
      "${EA_ARG}" \
      --output_tag "${RUN_TAG_SAFE}" \
      "${overwrite_args[@]}"
  ) &
  stage1_pids+=("$!")
done
wait_for_stage "Stage 1" "${stage1_pids[@]}"

if [[ "${TRAIN_ADV_AUG}" == "1" ]]; then
  echo "[Stage 1b] Purify adversarial training samples"
  stage1b_pids=()
  stage1b_gpu_index=0
  train_adv_eps_safe="$(safe_token "${TRAIN_ADV_EPS}")"
  for config in "${TRAIN_ADV_CONFIGS[@]}"; do
    config="${config//[[:space:]]/}"
    if [[ -z "${config}" ]]; then
      continue
    fi
    config_stem="${config%.yaml}"
    output_path="purified_data/train_ad/${DATASET}_${MODEL}_${PROTOCOL_SHORT}_fold${FOLD}_seed${SEED}_train_ad_${TRAIN_ADV_ATTACK}_eps${train_adv_eps_safe}_${config_stem}_n${TRAIN_ADV_SAMPLE_NUM}_tag${RUN_TAG_SAFE}.pth"
    aug_paths+=("${output_path}")
    if [[ -f "${output_path}" && "${OVERWRITE}" != "1" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skip existing: ${output_path}"
      continue
    fi
    overwrite_args=()
    if [[ "${OVERWRITE}" == "1" ]]; then
      overwrite_args=(--overwrite)
    fi
    wait_for_slots "${MAX_JOBS}"
    assigned_gpu="${gpu_ids[$((stage1b_gpu_index % gpu_count))]}"
    stage1b_gpu_index=$((stage1b_gpu_index + 1))
    (
      run_cmd python -u purify_train_adv.py \
        --dataset "${DATASET}" \
        --model "${MODEL}" \
        --fold "${FOLD}" \
        --config "${config}" \
        --sample_num "${TRAIN_ADV_SAMPLE_NUM}" \
        --attack "${TRAIN_ADV_ATTACK}" \
        --eps "${TRAIN_ADV_EPS}" \
        --checkpoint_path "${TRAIN_ADV_CHECKPOINT_PATH}" \
        --batch_size "${TRAIN_ADV_BATCH_SIZE}" \
        --seed "${SEED}" \
        --gpu_id "${assigned_gpu}" \
        "${EA_ARG}" \
        --output_tag "${RUN_TAG_SAFE}" \
        "${overwrite_args[@]}"
    ) &
    stage1b_pids+=("$!")
  done
  wait_for_stage "Stage 1b" "${stage1b_pids[@]}"
fi

echo "[Stage 2] Train with purified augmentation"
run_cmd python -u train_AT.py \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --at_strategy "${AT_STRATEGY}" \
  --epsilon "${EPS}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --patience "${PATIENCE}" \
  --seed "${SEED}" \
  --gpu_id "${GPU_ID}" \
  "${EA_ARG}" \
  --use_purified_aug \
  --purified_aug_tag "${RUN_TAG_SAFE}" \
  --purified_aug_paths "${aug_paths[@]}"

checkpoint_path="checkpoints/${DATASET}_${MODEL}_${PROTOCOL_TAG}_${AT_STRATEGY}_eps${TRAIN_EPS}_${SEED}_fold${FOLD}_aug_${RUN_TAG_SAFE}_best.pth"
adv_tag="aug_${RUN_TAG_SAFE}"
adv_data_path="ad_data/${DATASET}_${MODEL}_${PROTOCOL_SHORT}_${adv_tag}_${AT_STRATEGY}_${ATTACK}_eps${EPS}_seed${SEED}_fold${FOLD}.pth"

echo "[Stage 3] Attack augmented model"
run_cmd python -u attack.py \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --at_strategy "${AT_STRATEGY}" \
  --fold "${FOLD}" \
  --attack "${ATTACK}" \
  --eps "${EPS}" \
  --seed "${SEED}" \
  --gpu_id "${GPU_ID}" \
  "${EA_ARG}" \
  --checkpoint_path "${checkpoint_path}" \
  --save_adv \
  --adv_output_tag "${adv_tag}"

echo "[Stage 4] Purify attacked samples with selected configs"
stage4_pids=()
stage4_gpu_index=0
for config in "${PURIFY_CONFIGS[@]}"; do
  config="${config//[[:space:]]/}"
  if [[ -z "${config}" ]]; then
    continue
  fi
  wait_for_slots "${MAX_JOBS}"
  assigned_gpu="${gpu_ids[$((stage4_gpu_index % gpu_count))]}"
  stage4_gpu_index=$((stage4_gpu_index + 1))
  (
    run_cmd python -u purify.py \
      --dataset "${DATASET}" \
      --model "${MODEL}" \
      --fold "${FOLD}" \
      --attack "${ATTACK}" \
      --eps "${EPS}" \
      --sample_num "${SAMPLE_NUM}" \
      --seed "${SEED}" \
      --gpu_id "${assigned_gpu}" \
      "${EA_ARG}" \
      --config "${config}" \
      --ad_data_path "${adv_data_path}" \
      --checkpoint_path "${checkpoint_path}" \
      --model_tag "${adv_tag}"
  ) &
  stage4_pids+=("$!")
done
wait_for_stage "Stage 4" "${stage4_pids[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline complete."
echo "Checkpoint: ${checkpoint_path}"
echo "Adversarial data: ${adv_data_path}"
