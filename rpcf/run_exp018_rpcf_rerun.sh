#!/usr/bin/env bash
set -euo pipefail

BASE_RUN_ID="${EXP018_BASE_RUN_ID:-exp018_full_20260612_124131}"
RUN_ID="${EXP018_RPCF_RUN_ID:-exp018_rpcf_interlayer_kl_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${EXP018_RPCF_LOG_ROOT:-logs/exp018/${RUN_ID}}"
TAG="${RUN_ID//[^A-Za-z0-9_.-]/_}"
GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-torch}"
START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-5}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

DATASET="thubenchmark"
MODEL="eegnet"
SEED="42"
FOLD="0"
EPS="0.03"
PROTOCOL="train_only_subject_no_ea_subject_split"

BASE_TAG="${BASE_RUN_ID//[^A-Za-z0-9_.-]/_}"
BASE_LOG_ROOT="logs/exp018/${BASE_RUN_ID}"
AT_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${BASE_TAG}_at_best.pth"
CONSISTANCY_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${BASE_TAG}_consistancy_best.pth"
RPCF_CACHE="purified_data/exp018/rpcf_train/${BASE_TAG}_six_rank.pth"
AT_PURIFY="purified_data/exp018/eval/${BASE_TAG}_at_rank25-30.pth"
CONSISTANCY_PURIFY="purified_data/exp018/eval/${BASE_TAG}_consistancy_rank25-30.pth"

SENSITIVITY_PREFIX="${LOG_ROOT}/sensitivity"
HISTORY_PREFIX="${LOG_ROOT}/finetune"
RPCF_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_rpcf_${TAG}_best.pth"
RPCF_ATTACK="ad_data/exp018/${TAG}_rpcf_autoattack.pth"
RPCF_PURIFY="purified_data/exp018/eval/${TAG}_rpcf_rank25-30.pth"

mkdir -p "${LOG_ROOT}" checkpoints ad_data/exp018 purified_data/exp018/eval
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
mkdir -p "${NUMBA_CACHE_DIR}" "${XDG_CACHE_HOME}"

printf '%s\n' "$$" > "${LOG_ROOT}/controller.pid"
cat > "${LOG_ROOT}/run_config.txt" <<EOF
RUN_ID=${RUN_ID}
BASE_RUN_ID=${BASE_RUN_ID}
AT_CHECKPOINT=${AT_CHECKPOINT}
RPCF_CACHE=${RPCF_CACHE}
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

require_artifact "${AT_CHECKPOINT}"
require_artifact "${CONSISTANCY_CHECKPOINT}"
require_artifact "${RPCF_CACHE}"
require_artifact "${AT_PURIFY}"
require_artifact "${CONSISTANCY_PURIFY}"

if should_run 1; then
  run_logged stage1_sensitivity \
    conda run -n "${CONDA_ENV}" --no-capture-output python -u \
    -m rpcf.analyze_sensitivity \
    --cache_path "${RPCF_CACHE}" --checkpoint_path "${AT_CHECKPOINT}" \
    --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" --seed "${SEED}" \
    --eps "${EPS}" --sensitive_ratio 0.4 --gpu_id "${GPU_ID}" \
    --output_prefix "${SENSITIVITY_PREFIX}"
fi

if should_run 2; then
  require_artifact "${SENSITIVITY_PREFIX}.json"
  run_logged stage2_finetune \
    conda run -n "${CONDA_ENV}" --no-capture-output python -u -m rpcf.finetune \
    --cache_path "${RPCF_CACHE}" --sensitivity_path "${SENSITIVITY_PREFIX}.json" \
    --checkpoint_path "${AT_CHECKPOINT}" --output_checkpoint "${RPCF_CHECKPOINT}" \
    --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" --seed "${SEED}" \
    --epsilon "${EPS}" --epochs 100 --batch_size 64 --lr 0.0001 \
    --weight_decay 0.0001 --rank_temperature 0.5 \
    --consistancy_temperature 2.0 --pgd_steps 10 --gpu_id "${GPU_ID}" \
    --history_prefix "${HISTORY_PREFIX}"
fi

if should_run 3; then
  require_artifact "${RPCF_CHECKPOINT}"
  run_logged stage3_attack \
    conda run -n "${CONDA_ENV}" --no-capture-output python -u \
    -m rpcf.evaluate_attack \
    --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" --seed "${SEED}" \
    --checkpoint_path "${RPCF_CHECKPOINT}" --method_tag rpcf_interlayer_kl \
    --attack autoattack --eps "${EPS}" --batch_size 32 --gpu_id "${GPU_ID}" \
    --output_path "${RPCF_ATTACK}" "${overwrite_args[@]}"
fi

if should_run 4; then
  require_artifact "${RPCF_ATTACK}"
  run_logged stage4_purification \
    conda run -n "${CONDA_ENV}" --no-capture-output python -u \
    -m rpcf.evaluate_purification \
    --attack_path "${RPCF_ATTACK}" --checkpoint_path "${RPCF_CHECKPOINT}" \
    --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" --seed "${SEED}" \
    --eps "${EPS}" --sample_num 512 --ranks "25,30" \
    --configs "PTR3d_8_2048_rank25_3d_interpolate.yaml,PTR3d_8_2048_rank30_3d_interpolate.yaml" \
    --gpu_id "${GPU_ID}" --output_path "${RPCF_PURIFY}" "${overwrite_args[@]}"
fi

if should_run 5; then
  require_artifact "${RPCF_PURIFY}"
  run_logged stage5_summary \
    conda run -n "${CONDA_ENV}" --no-capture-output python -u \
    -m rpcf.compare_exp018 \
    --dataset "${DATASET}" --model "${MODEL}" --seed "${SEED}" --fold "${FOLD}" \
    --eps "${EPS}" --ranks "25,30" --sample_num 512 --gpu_id "${GPU_ID}" \
    --at_checkpoint "${AT_CHECKPOINT}" \
    --consistancy_checkpoint "${CONSISTANCY_CHECKPOINT}" \
    --rpcf_checkpoint "${RPCF_CHECKPOINT}" \
    --at_purification_path "${AT_PURIFY}" \
    --consistancy_purification_path "${CONSISTANCY_PURIFY}" \
    --rpcf_purification_path "${RPCF_PURIFY}" \
    --sensitivity_path "${SENSITIVITY_PREFIX}.json" \
    --history_path "${HISTORY_PREFIX}.json" --output_dir "${LOG_ROOT}"
fi

echo "[$(date -Is)] EXP-018 RPCF rerun finished: ${RUN_ID}"
