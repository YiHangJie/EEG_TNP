#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-thubenchmark}"
MODEL="${MODEL:-eegnet}"
SEED="${SEED:-42}"
FOLD="${FOLD:-0}"
EPS="${EPS:-0.03}"
GPU_ID="${GPU_ID:-0}"
ATTACK="${ATTACK:-autoattack}"
RANKS="${RANKS:-15,20,25,30,35,40}"
CONFIGS="${CONFIGS:-PTR3d_8_2048_rank15_3d_interpolate.yaml,PTR3d_8_2048_rank20_3d_interpolate.yaml,PTR3d_8_2048_rank25_3d_interpolate.yaml,PTR3d_8_2048_rank30_3d_interpolate.yaml,PTR3d_8_2048_rank35_3d_interpolate.yaml,PTR3d_8_2048_rank40_3d_interpolate.yaml}"
SAMPLE_NUM="${SAMPLE_NUM:-512}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-32}"
PATIENCE="${PATIENCE:-15}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
SENSITIVE_RATIO="${SENSITIVE_RATIO:-0.4}"
RANK_TEMPERATURE="${RANK_TEMPERATURE:-0.5}"
PGD_STEPS="${PGD_STEPS:-10}"
START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-7}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDA_ENV="${CONDA_ENV:-torch}"
RUN_ID="${RPCF_RUN_ID:-rpcf_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${RPCF_LOG_ROOT:-logs/rpcf/${RUN_ID}}"

if [[ "${SMOKE}" == "1" ]]; then
  SAMPLE_NUM="${SMOKE_SAMPLE_NUM:-1}"
  EPOCHS="${SMOKE_EPOCHS:-1}"
  PATIENCE="${SMOKE_PATIENCE:-1}"
  ATTACK_SAMPLE_NUM="${SMOKE_ATTACK_SAMPLE_NUM:-2}"
  EVAL_PURIFY_SAMPLE_NUM="${SMOKE_EVAL_PURIFY_SAMPLE_NUM:-2}"
else
  ATTACK_SAMPLE_NUM="${ATTACK_SAMPLE_NUM:-}"
  EVAL_PURIFY_SAMPLE_NUM="${EVAL_PURIFY_SAMPLE_NUM:-512}"
fi

if ! [[ "${START_STAGE}" =~ ^[1-7]$ && "${STOP_STAGE}" =~ ^[1-7]$ ]]; then
  echo "START_STAGE and STOP_STAGE must be in [1, 7]." >&2
  exit 1
fi
if (( START_STAGE > STOP_STAGE )); then
  echo "START_STAGE cannot be greater than STOP_STAGE." >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}" checkpoints ad_data/rpcf purified_data/rpcf_train purified_data/rpcf_eval
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
mkdir -p "${NUMBA_CACHE_DIR}" "${XDG_CACHE_HOME}"

PROTOCOL="train_only_subject_no_ea_subject_split"
EPS_TAG="${EPS//./p}"
TAG="${RUN_ID//[^A-Za-z0-9_.-]/_}"
AT_CHECKPOINT="${AT_CHECKPOINT:-checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_best.pth}"
CACHE_PATH="${CACHE_PATH:-purified_data/rpcf_train/${DATASET}_${MODEL}_no_ea_fold${FOLD}_seed${SEED}_rpcf_train_${ATTACK}_eps${EPS}_${SAMPLE_NUM}_${TAG}.pth}"
SENSITIVITY_PREFIX="${SENSITIVITY_PREFIX:-${LOG_ROOT}/sensitivity}"
RPCF_CHECKPOINT="${RPCF_CHECKPOINT:-checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_rpcf_${TAG}_best.pth}"
HISTORY_PREFIX="${HISTORY_PREFIX:-${LOG_ROOT}/finetune}"
AT_ATTACK_PATH="${AT_ATTACK_PATH:-ad_data/rpcf/${TAG}_at_${ATTACK}_eps${EPS_TAG}.pth}"
RPCF_ATTACK_PATH="${RPCF_ATTACK_PATH:-ad_data/rpcf/${TAG}_rpcf_${ATTACK}_eps${EPS_TAG}.pth}"
AT_PURIFY_PATH="${AT_PURIFY_PATH:-purified_data/rpcf_eval/${TAG}_at_rank_eval.pth}"
RPCF_PURIFY_PATH="${RPCF_PURIFY_PATH:-purified_data/rpcf_eval/${TAG}_rpcf_rank_eval.pth}"

printf '%s\n' "$$" > "${LOG_ROOT}/controller.pid"
cat > "${LOG_ROOT}/run_config.txt" <<EOF
RUN_ID=${RUN_ID}
DATASET=${DATASET}
MODEL=${MODEL}
SEED=${SEED}
FOLD=${FOLD}
EPS=${EPS}
RANKS=${RANKS}
CONFIGS=${CONFIGS}
SAMPLE_NUM=${SAMPLE_NUM}
AT_CHECKPOINT=${AT_CHECKPOINT}
CACHE_PATH=${CACHE_PATH}
RPCF_CHECKPOINT=${RPCF_CHECKPOINT}
EOF

run_logged() {
  local stage="$1"
  shift
  local log_path="${LOG_ROOT}/${stage}.log"
  echo "[$(date -Is)] ${stage}: $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  "$@" 2>&1 | tee -a "${log_path}"
}

should_run() {
  local stage="$1"
  (( stage >= START_STAGE && stage <= STOP_STAGE ))
}

artifact_args=()
if [[ "${SKIP_EXISTING}" != "1" ]]; then
  artifact_args=(--overwrite)
fi
attack_sample_args=()
if [[ -n "${ATTACK_SAMPLE_NUM}" ]]; then
  attack_sample_args=(--sample_num "${ATTACK_SAMPLE_NUM}")
fi

if should_run 1; then
  if [[ -f "${AT_CHECKPOINT}" ]]; then
    echo "[Stage 1] Reuse AT checkpoint: ${AT_CHECKPOINT}"
  elif [[ "${SMOKE}" == "1" ]]; then
    echo "Smoke refuses to create/overwrite a baseline AT checkpoint: ${AT_CHECKPOINT}" >&2
    exit 1
  else
    run_logged stage1_at \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u train_AT.py \
      --dataset "${DATASET}" --model "${MODEL}" --at_strategy madry --fold "${FOLD}" \
      --epsilon "${EPS}" --epochs 400 --batch_size "${BATCH_SIZE}" --patience 20 \
      --seed "${SEED}" --gpu_id "${GPU_ID}" --no_ea
  fi
fi

if should_run 2; then
  if [[ "${SKIP_EXISTING}" == "1" && -f "${CACHE_PATH}" ]]; then
    echo "[Stage 2] Skip existing cache: ${CACHE_PATH}"
  else
    run_logged stage2_cache \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u -m rpcf.generate_cache \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" --seed "${SEED}" \
      --attack "${ATTACK}" --eps "${EPS}" --checkpoint_path "${AT_CHECKPOINT}" \
      --sample_num "${SAMPLE_NUM}" --attack_batch_size "${ATTACK_BATCH_SIZE}" \
      --ranks "${RANKS}" --configs "${CONFIGS}" --gpu_id "${GPU_ID}" \
      --tag "${TAG}" --output_path "${CACHE_PATH}" "${artifact_args[@]}"
  fi
fi

if should_run 3; then
  if [[ "${SKIP_EXISTING}" == "1" && -f "${SENSITIVITY_PREFIX}.json" ]]; then
    echo "[Stage 3] Skip existing sensitivity: ${SENSITIVITY_PREFIX}.json"
  else
    run_logged stage3_sensitivity \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u -m rpcf.analyze_sensitivity \
      --cache_path "${CACHE_PATH}" --checkpoint_path "${AT_CHECKPOINT}" \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" --seed "${SEED}" \
      --eps "${EPS}" --sensitive_ratio "${SENSITIVE_RATIO}" --gpu_id "${GPU_ID}" \
      --output_prefix "${SENSITIVITY_PREFIX}"
  fi
fi

if should_run 4; then
  if [[ "${SKIP_EXISTING}" == "1" && -f "${RPCF_CHECKPOINT}" ]]; then
    echo "[Stage 4] Skip existing RPCF checkpoint: ${RPCF_CHECKPOINT}"
  else
    run_logged stage4_finetune \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u -m rpcf.finetune \
      --cache_path "${CACHE_PATH}" --sensitivity_path "${SENSITIVITY_PREFIX}.json" \
      --checkpoint_path "${AT_CHECKPOINT}" --output_checkpoint "${RPCF_CHECKPOINT}" \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" --seed "${SEED}" \
      --epsilon "${EPS}" --epochs "${EPOCHS}" --batch_size "${BATCH_SIZE}" \
      --lr "${LR}" --weight_decay "${WEIGHT_DECAY}" --patience "${PATIENCE}" \
      --rank_temperature "${RANK_TEMPERATURE}" --pgd_steps "${PGD_STEPS}" \
      --gpu_id "${GPU_ID}" --history_prefix "${HISTORY_PREFIX}"
  fi
fi

if should_run 5; then
  for method in at rpcf; do
    if [[ "${method}" == "at" ]]; then
      checkpoint="${AT_CHECKPOINT}"
      output="${AT_ATTACK_PATH}"
    else
      checkpoint="${RPCF_CHECKPOINT}"
      output="${RPCF_ATTACK_PATH}"
    fi
    if [[ "${SKIP_EXISTING}" == "1" && -f "${output}" ]]; then
      echo "[Stage 5] Skip existing ${method} attack: ${output}"
      continue
    fi
    run_logged "stage5_attack_${method}" \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u -m rpcf.evaluate_attack \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" --seed "${SEED}" \
      --checkpoint_path "${checkpoint}" --method_tag "${method}" --attack "${ATTACK}" \
      --eps "${EPS}" --batch_size "${ATTACK_BATCH_SIZE}" --gpu_id "${GPU_ID}" \
      --output_path "${output}" "${attack_sample_args[@]}" "${artifact_args[@]}"
  done
fi

if should_run 6; then
  for method in at rpcf; do
    if [[ "${method}" == "at" ]]; then
      checkpoint="${AT_CHECKPOINT}"
      attack_path="${AT_ATTACK_PATH}"
      output="${AT_PURIFY_PATH}"
    else
      checkpoint="${RPCF_CHECKPOINT}"
      attack_path="${RPCF_ATTACK_PATH}"
      output="${RPCF_PURIFY_PATH}"
    fi
    if [[ "${SKIP_EXISTING}" == "1" && -f "${output}" ]]; then
      echo "[Stage 6] Skip existing ${method} purification: ${output}"
      continue
    fi
    run_logged "stage6_purify_${method}" \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u -m rpcf.evaluate_purification \
      --attack_path "${attack_path}" --checkpoint_path "${checkpoint}" \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" --seed "${SEED}" \
      --eps "${EPS}" --sample_num "${EVAL_PURIFY_SAMPLE_NUM}" --ranks "${RANKS}" \
      --configs "${CONFIGS}" --gpu_id "${GPU_ID}" --output_path "${output}" \
      "${artifact_args[@]}"
  done
fi

if should_run 7; then
  run_logged stage7_summary \
    conda run -n "${CONDA_ENV}" --no-capture-output python -u -m rpcf.summarize \
    --at_attack_path "${AT_ATTACK_PATH}" --rpcf_attack_path "${RPCF_ATTACK_PATH}" \
    --at_purification_path "${AT_PURIFY_PATH}" \
    --rpcf_purification_path "${RPCF_PURIFY_PATH}" \
    --sensitivity_path "${SENSITIVITY_PREFIX}.json" \
    --history_path "${HISTORY_PREFIX}.json" --output_dir "${LOG_ROOT}"
fi

echo "[$(date -Is)] RPCF pipeline finished: ${RUN_ID}"
