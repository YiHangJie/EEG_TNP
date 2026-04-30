#!/usr/bin/env bash
set -euo pipefail

# 复用 purify_aug_pipeline.sh 已完成的训练和攻击产物，只补跑 stage4 的 rank 扫描。
# 适合在不重复训练、不重复生成 adversarial data 的情况下，比较更多 PTR3d stage=4 rank。
#
# 用法示例：
#   bash purify_aug_stage4_rank_sweep.sh
#   DRY_RUN=1 RANKS_CSV=2,3,5,10,15,20,25,30,35,40 bash purify_aug_stage4_rank_sweep.sh
#   RUN_TAG=puraug_rank5-40_n512_eps0p03 GPU_IDS=0,1 MAX_JOBS=2 bash purify_aug_stage4_rank_sweep.sh
#   STAGE4_CONFIGS_CSV='PTR3d_8_2048_rank25_3d_interpolate.yaml,PTR3d_8_2048_rank30_3d_interpolate.yaml' bash purify_aug_stage4_rank_sweep.sh
#   nohup bash purify_aug_stage4_rank_sweep.sh > logs/purify_aug_stage4_rank_sweep_$(date +%Y%m%d_%H%M%S).log 2>&1 &

DATASET="${DATASET:-thubenchmark}"
MODEL="${MODEL:-eegnet}"
GPU_ID="${GPU_ID:-0}"
GPU_IDS="${GPU_IDS:-${GPU_ID}}"
MAX_JOBS="${MAX_JOBS:-2}"
POLL_SECONDS="${POLL_SECONDS:-10}"

EPS="${EPS:-0.03}"
ATTACK="${ATTACK:-autoattack}"
AT_STRATEGY="${AT_STRATEGY:-clean}"
SEED="${SEED:-42}"
FOLD="${FOLD:-0}"
SAMPLE_NUM="${SAMPLE_NUM:-512}"
PURIFY_BATCH_SIZE="${PURIFY_BATCH_SIZE:-32}"
USE_EA="${USE_EA:-0}"

RUN_TAG="${RUN_TAG:-puraug_rank25-30_n${SAMPLE_NUM}_eps${EPS//./p}}"
MODEL_TAG="${MODEL_TAG:-auto}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-auto}"
AD_DATA_PATH="${AD_DATA_PATH:-auto}"
OUTPUT_TAG="${OUTPUT_TAG:-stage4_rank_sweep}"

CONFIG_DIR="${CONFIG_DIR:-configs/${DATASET}}"
CONFIG_GLOB="${CONFIG_GLOB:-PTR3d_8_2048_rank*_3d_interpolate.yaml}"
if [[ -z "${CONFIG_TEMPLATE:-}" ]]; then
  CONFIG_TEMPLATE='PTR3d_8_2048_rank{rank}_3d_interpolate.yaml'
fi
RANKS_CSV="${RANKS_CSV:-auto}"
STAGE4_CONFIGS_CSV="${STAGE4_CONFIGS_CSV:-auto}"

OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
CONDA_ENV="${CONDA_ENV:-torch}"
ACTIVATE_CONDA="${ACTIVATE_CONDA:-1}"

safe_token() {
  local value="$1"
  if [[ -z "${value}" ]]; then
    echo "none"
    return
  fi
  printf '%s' "${value}" | sed 's/[^A-Za-z0-9_.-]/_/g'
}

log_cmd() {
  printf '[%s]' "$(date '+%Y-%m-%d %H:%M:%S')"
  printf ' %q' "$@"
  printf '\n'
}

run_cmd() {
  log_cmd "$@"
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
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage4 rank sweep job failed: pid ${pid}" >&2
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage4 rank sweep failed." >&2
    exit 1
  fi
}

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

RUN_TAG_SAFE="$(safe_token "${RUN_TAG}")"
if [[ "${OUTPUT_TAG}" == "none" ]]; then
  OUTPUT_TAG=""
fi
OUTPUT_TAG_SAFE="$(safe_token "${OUTPUT_TAG}")"

if [[ "${USE_EA}" == "1" ]]; then
  EA_ARG=(--use_ea)
  PROTOCOL_SHORT="ea"
  PROTOCOL_TAG="train_only_subject_ea_subject_split"
else
  EA_ARG=(--no_ea)
  PROTOCOL_SHORT="no_ea"
  PROTOCOL_TAG="train_only_subject_no_ea_subject_split"
fi

if [[ "${AT_STRATEGY}" == "clean" ]]; then
  TRAIN_EPS="0"
else
  TRAIN_EPS="${EPS}"
fi

# 默认路径和 purify_aug_pipeline.sh 的 stage2/stage3 命名保持一致。
DEFAULT_MODEL_TAG="aug_${RUN_TAG_SAFE}"
if [[ "${MODEL_TAG}" == "auto" ]]; then
  MODEL_TAG="${DEFAULT_MODEL_TAG}"
fi
MODEL_TAG="$(safe_token "${MODEL_TAG}")"
if [[ "${CHECKPOINT_PATH}" == "auto" ]]; then
  CHECKPOINT_PATH="checkpoints/${DATASET}_${MODEL}_${PROTOCOL_TAG}_${AT_STRATEGY}_eps${TRAIN_EPS}_${SEED}_fold${FOLD}_${DEFAULT_MODEL_TAG}_best.pth"
fi
if [[ "${AD_DATA_PATH}" == "auto" ]]; then
  AD_DATA_PATH="ad_data/${DATASET}_${MODEL}_${PROTOCOL_SHORT}_${DEFAULT_MODEL_TAG}_${AT_STRATEGY}_${ATTACK}_eps${EPS}_seed${SEED}_fold${FOLD}.pth"
fi

if [[ "${ACTIVATE_CONDA}" == "1" ]] && command -v conda > /dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
mkdir -p "${NUMBA_CACHE_DIR}" "${XDG_CACHE_HOME}" log_purify purified_data/attacked

if [[ ! -d "${CONFIG_DIR}" ]]; then
  echo "Config directory not found: ${CONFIG_DIR}" >&2
  exit 1
fi

if [[ "${RANKS_CSV}" == "auto" ]]; then
  case "${DATASET}" in
    thubenchmark)
      RANKS_CSV="2,3,5,10,15,20,25,30,35,40"
      ;;
    bciciv2a)
      RANKS_CSV="5,10,15,20,25,30,35,40"
      ;;
    seediv)
      RANKS_CSV="5,10,15,20,25,30,35,40,45,50,55,60"
      ;;
    *)
      RANKS_CSV=""
      ;;
  esac
fi

configs=()
if [[ "${STAGE4_CONFIGS_CSV}" != "auto" ]]; then
  IFS=',' read -r -a configs <<< "${STAGE4_CONFIGS_CSV}"
elif [[ -n "${RANKS_CSV}" ]]; then
  IFS=',' read -r -a ranks <<< "${RANKS_CSV}"
  for rank in "${ranks[@]}"; do
    rank="${rank//[[:space:]]/}"
    if [[ -z "${rank}" ]]; then
      continue
    fi
    if [[ ! "${rank}" =~ ^[0-9]+$ ]]; then
      echo "Invalid rank in RANKS_CSV: ${rank}" >&2
      exit 1
    fi
    configs+=("${CONFIG_TEMPLATE//\{rank\}/${rank}}")
  done
else
  mapfile -t configs < <(find "${CONFIG_DIR}" -maxdepth 1 -type f -name "${CONFIG_GLOB}" -printf '%f\n' | sort -V)
fi

normalized_configs=()
for config in "${configs[@]}"; do
  config="${config//[[:space:]]/}"
  if [[ -z "${config}" ]]; then
    continue
  fi
  if [[ ! -f "${CONFIG_DIR}/${config}" ]]; then
    echo "Config not found: ${CONFIG_DIR}/${config}" >&2
    exit 1
  fi
  normalized_configs+=("${config}")
done
configs=("${normalized_configs[@]}")

if [[ "${#configs[@]}" -eq 0 ]]; then
  echo "No stage4 configs selected. Check CONFIG_GLOB, RANKS_CSV, or STAGE4_CONFIGS_CSV." >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "CHECKPOINT_PATH not found: ${CHECKPOINT_PATH}" >&2
    echo "Run purify_aug_pipeline.sh first, or set CHECKPOINT_PATH explicitly." >&2
    exit 1
  fi
  if [[ ! -f "${AD_DATA_PATH}" ]]; then
    echo "AD_DATA_PATH not found: ${AD_DATA_PATH}" >&2
    echo "Run purify_aug_pipeline.sh through stage3 first, or set AD_DATA_PATH explicitly." >&2
    exit 1
  fi
fi

echo "======================================================"
echo "Stage4 rank sweep"
echo "DATASET=${DATASET}, MODEL=${MODEL}, FOLD=${FOLD}, SEED=${SEED}, USE_EA=${USE_EA}"
echo "AT_STRATEGY=${AT_STRATEGY}, ATTACK=${ATTACK}, EPS=${EPS}, SAMPLE_NUM=${SAMPLE_NUM}"
echo "RUN_TAG=${RUN_TAG_SAFE}, MODEL_TAG=${MODEL_TAG}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "AD_DATA_PATH=${AD_DATA_PATH}"
echo "CONFIG_DIR=${CONFIG_DIR}"
echo "CONFIGS=${configs[*]}"
echo "MAX_JOBS=${MAX_JOBS}, GPU_IDS=${gpu_ids[*]}, DRY_RUN=${DRY_RUN}, OVERWRITE=${OVERWRITE}"
echo "======================================================"

stage4_pids=()
stage4_gpu_index=0
skipped=0
for config in "${configs[@]}"; do
  config_stem="${config%.yaml}"
  output_suffix=""
  if [[ -n "${OUTPUT_TAG}" ]]; then
    output_suffix="_tag${OUTPUT_TAG_SAFE}"
  fi
  ad_output_path="purified_data/attacked/${DATASET}_${MODEL}_${PROTOCOL_SHORT}_${MODEL_TAG}_${ATTACK}_eps${EPS}_seed${SEED}_fold${FOLD}_${config_stem}_n${SAMPLE_NUM}${output_suffix}_ad.pth"
  clean_output_path="purified_data/attacked/${DATASET}_${MODEL}_${PROTOCOL_SHORT}_${MODEL_TAG}_${ATTACK}_eps${EPS}_seed${SEED}_fold${FOLD}_${config_stem}_n${SAMPLE_NUM}${output_suffix}_clean.pth"

  if [[ "${OVERWRITE}" != "1" && -f "${ad_output_path}" && -f "${clean_output_path}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skip existing: ${config}"
    skipped=$((skipped + 1))
    continue
  fi

  wait_for_slots "${MAX_JOBS}"
  assigned_gpu="${gpu_ids[$((stage4_gpu_index % gpu_count))]}"
  stage4_gpu_index=$((stage4_gpu_index + 1))
  (
    output_tag_args=()
    if [[ -n "${OUTPUT_TAG}" ]]; then
      output_tag_args=(--output_tag "${OUTPUT_TAG_SAFE}")
    fi
    run_cmd python -u purify.py \
      --dataset "${DATASET}" \
      --model "${MODEL}" \
      --at_strategy "${AT_STRATEGY}" \
      --fold "${FOLD}" \
      --attack "${ATTACK}" \
      --eps "${EPS}" \
      --sample_num "${SAMPLE_NUM}" \
      --batch_size "${PURIFY_BATCH_SIZE}" \
      --seed "${SEED}" \
      --gpu_id "${assigned_gpu}" \
      "${EA_ARG[@]}" \
      --config "${config}" \
      --ad_data_path "${AD_DATA_PATH}" \
      --checkpoint_path "${CHECKPOINT_PATH}" \
      --model_tag "${MODEL_TAG}" \
      "${output_tag_args[@]}"
  ) &
  stage4_pids+=("$!")
done

wait_for_stage "${stage4_pids[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage4 rank sweep complete."
echo "Started jobs: ${#stage4_pids[@]}, skipped existing: ${skipped}"
