#!/usr/bin/env bash
set -euo pipefail

BASE_RUN_ID="${EXP018_BASE_RUN_ID:-exp018_full_20260612_124131}"
SENSITIVITY_RUN_ID="${EXP018_SENSITIVITY_RUN_ID:-exp018_rpcf_no_early_stop_20260614_2357}"
RUN_ID="${EXP018_RPCF_ALL_RUN_ID:-exp018_rpcf_all_layers_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${EXP018_RPCF_ALL_LOG_ROOT:-logs/exp018/${RUN_ID}}"
TAG="${RUN_ID//[^A-Za-z0-9_.-]/_}"
GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-torch}"
START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
EPOCHS="${RPCF_ALL_EPOCHS:-100}"
ATTACK_SAMPLE_NUM="${ATTACK_SAMPLE_NUM:-}"
EVAL_SAMPLE_NUM="${EVAL_SAMPLE_NUM:-512}"

DATASET="thubenchmark"
MODEL="eegnet"
SEED="42"
FOLD="0"
EPS="0.03"
PROTOCOL="train_only_subject_no_ea_subject_split"

BASE_TAG="${BASE_RUN_ID//[^A-Za-z0-9_.-]/_}"
AT_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${BASE_TAG}_at_best.pth"
CONSISTANCY_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${BASE_TAG}_consistancy_best.pth"
RPCF_CACHE="purified_data/exp018/rpcf_train/${BASE_TAG}_six_rank.pth"
AT_PURIFY="purified_data/exp018/eval/${BASE_TAG}_at_rank25-30.pth"
CONSISTANCY_PURIFY="purified_data/exp018/eval/${BASE_TAG}_consistancy_rank25-30.pth"
SENSITIVITY_PATH="logs/exp018/${SENSITIVITY_RUN_ID}/sensitivity.json"

HISTORY_PREFIX="${LOG_ROOT}/finetune"
RPCF_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_rpcf_all_layers_${TAG}_best.pth"
RPCF_ATTACK="ad_data/exp018/${TAG}_rpcf_all_layers_autoattack.pth"
RPCF_PURIFY="purified_data/exp018/eval/${TAG}_rpcf_all_layers_rank25-30.pth"

mkdir -p "${LOG_ROOT}" checkpoints ad_data/exp018 purified_data/exp018/eval
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
mkdir -p "${NUMBA_CACHE_DIR}" "${XDG_CACHE_HOME}"

printf '%s\n' "$$" > "${LOG_ROOT}/controller.pid"
cat > "${LOG_ROOT}/run_config.txt" <<EOF
RUN_ID=${RUN_ID}
BASE_RUN_ID=${BASE_RUN_ID}
SENSITIVITY_RUN_ID=${SENSITIVITY_RUN_ID}
EPOCHS=${EPOCHS}
ALL_LAYERS=1
AT_CHECKPOINT=${AT_CHECKPOINT}
RPCF_CACHE=${RPCF_CACHE}
SENSITIVITY_PATH=${SENSITIVITY_PATH}
RPCF_CHECKPOINT=${RPCF_CHECKPOINT}
EOF

run_logged() {
  local stage="$1"
  shift
  echo "[$(date -Is)] ${stage}: $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  "$@" 2>&1 | tee -a "${LOG_ROOT}/${stage}.log"
}

should_run() {
  local stage="$1"
  (( stage >= START_STAGE && stage <= STOP_STAGE ))
}

require_artifact() {
  local path="$1"
  if [[ "${DRY_RUN}" != "1" && ! -f "${path}" ]]; then
    echo "Required artifact not found: ${path}" >&2
    exit 1
  fi
}

overwrite_args=()
if [[ "${SKIP_EXISTING}" != "1" ]]; then
  overwrite_args=(--overwrite)
fi
attack_subset_args=()
if [[ -n "${ATTACK_SAMPLE_NUM}" ]]; then
  attack_subset_args=(--sample_num "${ATTACK_SAMPLE_NUM}")
fi

require_artifact "${AT_CHECKPOINT}"
require_artifact "${CONSISTANCY_CHECKPOINT}"
require_artifact "${RPCF_CACHE}"
require_artifact "${SENSITIVITY_PATH}"
require_artifact "${AT_PURIFY}"
require_artifact "${CONSISTANCY_PURIFY}"

if should_run 1; then
  run_logged stage1_finetune_all_layers \
    conda run -n "${CONDA_ENV}" --no-capture-output python -u -m rpcf.finetune \
    --cache_path "${RPCF_CACHE}" --sensitivity_path "${SENSITIVITY_PATH}" \
    --checkpoint_path "${AT_CHECKPOINT}" --output_checkpoint "${RPCF_CHECKPOINT}" \
    --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" --seed "${SEED}" \
    --epsilon "${EPS}" --epochs "${EPOCHS}" --batch_size 64 --lr 0.0001 \
    --weight_decay 0.0001 --rank_temperature 0.5 \
    --consistancy_temperature 2.0 --pgd_steps 10 --gpu_id "${GPU_ID}" \
    --all_layers --history_prefix "${HISTORY_PREFIX}"
fi

if should_run 2; then
  require_artifact "${RPCF_CHECKPOINT}"
  run_logged stage2_attack \
    conda run -n "${CONDA_ENV}" --no-capture-output python -u \
    -m rpcf.evaluate_attack \
    --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" --seed "${SEED}" \
    --checkpoint_path "${RPCF_CHECKPOINT}" --method_tag rpcf_all_layers \
    --attack autoattack --eps "${EPS}" --batch_size 32 --gpu_id "${GPU_ID}" \
    --output_path "${RPCF_ATTACK}" \
    "${attack_subset_args[@]}" "${overwrite_args[@]}"
fi

if should_run 3; then
  require_artifact "${RPCF_ATTACK}"
  run_logged stage3_purification \
    conda run -n "${CONDA_ENV}" --no-capture-output python -u \
    -m rpcf.evaluate_purification \
    --attack_path "${RPCF_ATTACK}" --checkpoint_path "${RPCF_CHECKPOINT}" \
    --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" --seed "${SEED}" \
    --eps "${EPS}" --sample_num "${EVAL_SAMPLE_NUM}" --ranks "25,30" \
    --configs "PTR3d_8_2048_rank25_3d_interpolate.yaml,PTR3d_8_2048_rank30_3d_interpolate.yaml" \
    --gpu_id "${GPU_ID}" --output_path "${RPCF_PURIFY}" "${overwrite_args[@]}"
fi

if should_run 4; then
  require_artifact "${RPCF_PURIFY}"
  run_logged stage4_summary \
    conda run -n "${CONDA_ENV}" --no-capture-output python -u \
    -m rpcf.compare_exp018 \
    --dataset "${DATASET}" --model "${MODEL}" --seed "${SEED}" --fold "${FOLD}" \
    --eps "${EPS}" --ranks "25,30" --sample_num "${EVAL_SAMPLE_NUM}" \
    --gpu_id "${GPU_ID}" --at_checkpoint "${AT_CHECKPOINT}" \
    --consistancy_checkpoint "${CONSISTANCY_CHECKPOINT}" \
    --rpcf_checkpoint "${RPCF_CHECKPOINT}" \
    --at_purification_path "${AT_PURIFY}" \
    --consistancy_purification_path "${CONSISTANCY_PURIFY}" \
    --rpcf_purification_path "${RPCF_PURIFY}" \
    --sensitivity_path "${SENSITIVITY_PATH}" \
    --history_path "${HISTORY_PREFIX}.json" --output_dir "${LOG_ROOT}"
fi

echo "[$(date -Is)] EXP-018 RPCF all-layers finished: ${RUN_ID}"
