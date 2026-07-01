#!/usr/bin/env bash
set -euo pipefail

# EXP-024：其他 backbone 的 Madry AT、RPCF_AT、净化测试和 baseline。

DATASET="${EXP024_DATASET:-thubenchmark}"
MODEL="${EXP024_MODEL:-conformer}"
SEED="${EXP024_SEED:-42}"
FOLD="${EXP024_FOLD:-0}"
EPS="${EXP024_EPS:-0.03}"
GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-torch}"
RUN_ID="${EXP024_RUN_ID:-exp024_${MODEL}_seed${SEED}_fold${FOLD}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${EXP024_LOG_ROOT:-logs/exp024/${RUN_ID}}"
TAG="${RUN_ID//[^A-Za-z0-9_.-]/_}"

START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-9}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"

AT_EPOCHS="${AT_EPOCHS:-400}"
AT_PATIENCE="${AT_PATIENCE:-20}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-400}"
BASELINE_PATIENCE="${BASELINE_PATIENCE:-20}"
RPCF_EPOCHS="${RPCF_EPOCHS:-100}"
TRAIN_SAMPLE_NUM="${TRAIN_SAMPLE_NUM:-}"
TRAIN_CACHE_SAMPLE_NUM="${TRAIN_CACHE_SAMPLE_NUM:-512}"
ATTACK_SAMPLE_NUM="${ATTACK_SAMPLE_NUM:-}"
EVAL_SAMPLE_NUM="${EVAL_SAMPLE_NUM:-512}"

AT_BATCH_SIZE="${AT_BATCH_SIZE:-128}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-128}"
RPCF_BATCH_SIZE="${RPCF_BATCH_SIZE:-64}"
ONLINE_AT_BATCH_SIZE="${ONLINE_AT_BATCH_SIZE:-128}"
ONLINE_AT_TRAIN_SAMPLE_NUM="${ONLINE_AT_TRAIN_SAMPLE_NUM:-}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-32}"
AT_LR="${AT_LR:-0.001}"
AT_WEIGHT_DECAY="${AT_WEIGHT_DECAY:-0.0001}"
RPCF_LR="${RPCF_LR:-0.0001}"
RPCF_WEIGHT_DECAY="${RPCF_WEIGHT_DECAY:-0.0001}"
ATTACK="${EXP024_ATTACK:-autoattack}"
BASELINE_STRATEGIES="${EXP024_BASELINE_STRATEGIES:-clean trades fbf}"

TRAIN_RANKS="15,20,25,30,35,40"
TRAIN_CONFIGS="PTR3d_8_2048_rank15_3d_interpolate.yaml,PTR3d_8_2048_rank20_3d_interpolate.yaml,PTR3d_8_2048_rank25_3d_interpolate.yaml,PTR3d_8_2048_rank30_3d_interpolate.yaml,PTR3d_8_2048_rank35_3d_interpolate.yaml,PTR3d_8_2048_rank40_3d_interpolate.yaml"
EVAL_RANKS="25,30"
EVAL_CONFIGS="PTR3d_8_2048_rank25_3d_interpolate.yaml,PTR3d_8_2048_rank30_3d_interpolate.yaml"
PROTOCOL="train_only_subject_no_ea_subject_split"

if [[ "${SMOKE}" == "1" ]]; then
  AT_EPOCHS="${SMOKE_EPOCHS:-1}"
  AT_PATIENCE=1
  BASELINE_EPOCHS="${SMOKE_EPOCHS:-1}"
  BASELINE_PATIENCE=1
  RPCF_EPOCHS="${SMOKE_EPOCHS:-1}"
  TRAIN_SAMPLE_NUM="${SMOKE_TRAIN_SAMPLE_NUM:-2}"
  TRAIN_CACHE_SAMPLE_NUM="${SMOKE_CACHE_SAMPLE_NUM:-1}"
  ONLINE_AT_TRAIN_SAMPLE_NUM="${SMOKE_TRAIN_SAMPLE_NUM:-2}"
  ATTACK_SAMPLE_NUM="${SMOKE_ATTACK_SAMPLE_NUM:-2}"
  EVAL_SAMPLE_NUM="${SMOKE_EVAL_SAMPLE_NUM:-2}"
fi

if ! [[ "${START_STAGE}" =~ ^[1-9]$ && "${STOP_STAGE}" =~ ^[1-9]$ ]]; then
  echo "START_STAGE and STOP_STAGE must be in [1, 9]." >&2
  exit 1
fi
if (( START_STAGE > STOP_STAGE )); then
  echo "START_STAGE cannot be greater than STOP_STAGE." >&2
  exit 1
fi
case "${MODEL}" in
  tsception|atcnet|conformer) ;;
  *)
    echo "EXP-024 targets other backbones only: tsception, atcnet, conformer. Got ${MODEL}." >&2
    exit 1
    ;;
esac

AT_TAG="${TAG}_madry_at"
RPCF_AT_TAG="${TAG}_rpcf_at"
BASELINE_TAG="${TAG}_baseline"
AT_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${AT_TAG}_best.pth"
RPCF_CACHE="purified_data/exp024/rpcf_train/${TAG}_six_rank.pth"
SENSITIVITY_PREFIX="${LOG_ROOT}/sensitivity"
SENSITIVITY_PATH="${SENSITIVITY_PREFIX}.json"
RPCF_AT_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${RPCF_AT_TAG}_best.pth"
RPCF_AT_HISTORY="${LOG_ROOT}/finetune_rpcf_at"

AT_ATTACK="ad_data/exp024/${TAG}_madry_at_${ATTACK}.pth"
RPCF_AT_ATTACK="ad_data/exp024/${TAG}_rpcf_at_${ATTACK}.pth"
AT_PURIFY="purified_data/exp024/eval/${TAG}_madry_at_rank25-30.pth"
RPCF_AT_PURIFY="purified_data/exp024/eval/${TAG}_rpcf_at_rank25-30.pth"
SUMMARY_DIR="${LOG_ROOT}/comparison"

mkdir -p "${LOG_ROOT}" checkpoints ad_data/exp024 purified_data/exp024/rpcf_train purified_data/exp024/eval log_train_AT
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
mkdir -p "${NUMBA_CACHE_DIR}" "${XDG_CACHE_HOME}"

printf '%s\n' "$$" > "${LOG_ROOT}/controller.pid"
cat > "${LOG_ROOT}/run_config.txt" <<EOF
EXPERIMENT_ID=EXP-024
RUN_ID=${RUN_ID}
DATASET=${DATASET}
MODEL=${MODEL}
SEED=${SEED}
FOLD=${FOLD}
EPS=${EPS}
ATTACK=${ATTACK}
TRAIN_CACHE_SAMPLE_NUM=${TRAIN_CACHE_SAMPLE_NUM}
EVAL_SAMPLE_NUM=${EVAL_SAMPLE_NUM}
BASELINE_STRATEGIES=${BASELINE_STRATEGIES}
AT_CHECKPOINT=${AT_CHECKPOINT}
RPCF_CACHE=${RPCF_CACHE}
SENSITIVITY_PATH=${SENSITIVITY_PATH}
RPCF_AT_CHECKPOINT=${RPCF_AT_CHECKPOINT}
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

checkpoint_for() {
  local strategy="$1"
  if [[ "${strategy}" == "madry" ]]; then
    printf '%s\n' "${AT_CHECKPOINT}"
  elif [[ "${strategy}" == "clean" ]]; then
    printf '%s\n' "checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_clean_eps0_${SEED}_fold${FOLD}_${BASELINE_TAG}_best.pth"
  else
    printf '%s\n' "checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_${strategy}_eps${EPS}_${SEED}_fold${FOLD}_${BASELINE_TAG}_best.pth"
  fi
}

attack_for() {
  local method="$1"
  case "${method}" in
    madry_at) printf '%s\n' "${AT_ATTACK}" ;;
    rpcf_at) printf '%s\n' "${RPCF_AT_ATTACK}" ;;
    *) printf '%s\n' "ad_data/exp024/${TAG}_${method}_${ATTACK}.pth" ;;
  esac
}

overwrite_args=()
if [[ "${SKIP_EXISTING}" != "1" ]]; then
  overwrite_args=(--overwrite)
fi
train_subset_args=()
if [[ -n "${TRAIN_SAMPLE_NUM}" ]]; then
  train_subset_args=(--train_sample_num "${TRAIN_SAMPLE_NUM}")
fi
online_subset_args=()
if [[ -n "${ONLINE_AT_TRAIN_SAMPLE_NUM}" ]]; then
  online_subset_args=(--online_train_sample_num "${ONLINE_AT_TRAIN_SAMPLE_NUM}")
fi
attack_subset_args=()
if [[ -n "${ATTACK_SAMPLE_NUM}" ]]; then
  attack_subset_args=(--sample_num "${ATTACK_SAMPLE_NUM}")
fi

if should_run 1; then
  if [[ "${SKIP_EXISTING}" == "1" && -f "${AT_CHECKPOINT}" ]]; then
    echo "[Stage 1] Reuse Madry AT checkpoint: ${AT_CHECKPOINT}"
  else
    run_logged stage1_train_madry_at \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u train_AT.py \
      --dataset "${DATASET}" --model "${MODEL}" --at_strategy madry \
      --fold "${FOLD}" --epsilon "${EPS}" --epochs "${AT_EPOCHS}" \
      --batch_size "${AT_BATCH_SIZE}" "${train_subset_args[@]}" \
      --lr "${AT_LR}" --weight_decay "${AT_WEIGHT_DECAY}" \
      --patience "${AT_PATIENCE}" --seed "${SEED}" --gpu_id "${GPU_ID}" \
      --no_ea --checkpoint_tag "${AT_TAG}"
  fi
fi

if should_run 2; then
  require_artifact "${AT_CHECKPOINT}"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${RPCF_CACHE}" ]]; then
    echo "[Stage 2] Reuse RPCF cache: ${RPCF_CACHE}"
  else
    run_logged stage2_generate_rpcf_cache \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.generate_cache \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
      --seed "${SEED}" --attack "${ATTACK}" --eps "${EPS}" \
      --checkpoint_path "${AT_CHECKPOINT}" \
      --sample_num "${TRAIN_CACHE_SAMPLE_NUM}" \
      --attack_batch_size "${ATTACK_BATCH_SIZE}" --ranks "${TRAIN_RANKS}" \
      --configs "${TRAIN_CONFIGS}" --gpu_id "${GPU_ID}" --tag "${TAG}" \
      --output_path "${RPCF_CACHE}" "${overwrite_args[@]}"
  fi
fi

if should_run 3; then
  require_artifact "${RPCF_CACHE}"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${SENSITIVITY_PATH}" ]]; then
    echo "[Stage 3] Reuse sensitivity: ${SENSITIVITY_PATH}"
  else
    run_logged stage3_analyze_sensitivity \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.analyze_sensitivity \
      --cache_path "${RPCF_CACHE}" --checkpoint_path "${AT_CHECKPOINT}" \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
      --seed "${SEED}" --eps "${EPS}" --sensitive_ratio 0.4 \
      --gpu_id "${GPU_ID}" --output_prefix "${SENSITIVITY_PREFIX}"
  fi
fi

if should_run 4; then
  require_artifact "${RPCF_CACHE}"
  require_artifact "${SENSITIVITY_PATH}"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${RPCF_AT_CHECKPOINT}" ]]; then
    echo "[Stage 4] Reuse RPCF_AT checkpoint: ${RPCF_AT_CHECKPOINT}"
  else
    run_logged stage4_finetune_rpcf_at \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.finetune \
      --cache_path "${RPCF_CACHE}" --sensitivity_path "${SENSITIVITY_PATH}" \
      --checkpoint_path "${AT_CHECKPOINT}" --output_checkpoint "${RPCF_AT_CHECKPOINT}" \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
      --seed "${SEED}" --epsilon "${EPS}" --epochs "${RPCF_EPOCHS}" \
      --batch_size "${RPCF_BATCH_SIZE}" --lr "${RPCF_LR}" \
      --weight_decay "${RPCF_WEIGHT_DECAY}" --rank_temperature 0.5 \
      --consistancy_temperature 2.0 --pgd_steps 10 --online_madry_at \
      --online_at_batch_size "${ONLINE_AT_BATCH_SIZE}" \
      --online_at_pgd_steps 10 --online_at_step_size 0.006 \
      "${online_subset_args[@]}" --gpu_id "${GPU_ID}" \
      --history_prefix "${RPCF_AT_HISTORY}"
  fi
fi

if should_run 5; then
  for method in madry_at rpcf_at; do
    checkpoint="${AT_CHECKPOINT}"
    [[ "${method}" == "rpcf_at" ]] && checkpoint="${RPCF_AT_CHECKPOINT}"
    output="$(attack_for "${method}")"
    require_artifact "${checkpoint}"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${output}" ]]; then
      echo "[Stage 5] Reuse ${method} attack: ${output}"
      continue
    fi
    run_logged "stage5_attack_${method}" \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.evaluate_attack \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
      --seed "${SEED}" --checkpoint_path "${checkpoint}" \
      --method_tag "${method}" --attack "${ATTACK}" --eps "${EPS}" \
      --batch_size "${ATTACK_BATCH_SIZE}" --gpu_id "${GPU_ID}" \
      --output_path "${output}" "${attack_subset_args[@]}" "${overwrite_args[@]}"
  done
fi

if should_run 6; then
  for method in madry_at rpcf_at; do
    checkpoint="${AT_CHECKPOINT}"
    attack_path="${AT_ATTACK}"
    output="${AT_PURIFY}"
    if [[ "${method}" == "rpcf_at" ]]; then
      checkpoint="${RPCF_AT_CHECKPOINT}"
      attack_path="${RPCF_AT_ATTACK}"
      output="${RPCF_AT_PURIFY}"
    fi
    require_artifact "${attack_path}"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${output}" ]]; then
      echo "[Stage 6] Reuse ${method} purification: ${output}"
      continue
    fi
    run_logged "stage6_purify_${method}" \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.evaluate_purification \
      --attack_path "${attack_path}" --checkpoint_path "${checkpoint}" \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
      --seed "${SEED}" --eps "${EPS}" --sample_num "${EVAL_SAMPLE_NUM}" \
      --ranks "${EVAL_RANKS}" --configs "${EVAL_CONFIGS}" \
      --gpu_id "${GPU_ID}" --output_path "${output}" "${overwrite_args[@]}"
  done
fi

if should_run 7; then
  for strategy in ${BASELINE_STRATEGIES}; do
    checkpoint="$(checkpoint_for "${strategy}")"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${checkpoint}" ]]; then
      echo "[Stage 7] Reuse ${strategy} baseline checkpoint: ${checkpoint}"
      continue
    fi
    run_logged "stage7_train_baseline_${strategy}" \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u train_AT.py \
      --dataset "${DATASET}" --model "${MODEL}" --at_strategy "${strategy}" \
      --fold "${FOLD}" --epsilon "${EPS}" --epochs "${BASELINE_EPOCHS}" \
      --batch_size "${BASELINE_BATCH_SIZE}" "${train_subset_args[@]}" \
      --lr "${AT_LR}" --weight_decay "${AT_WEIGHT_DECAY}" \
      --patience "${BASELINE_PATIENCE}" --seed "${SEED}" --gpu_id "${GPU_ID}" \
      --no_ea --checkpoint_tag "${BASELINE_TAG}"
  done
fi

if should_run 8; then
  for strategy in madry ${BASELINE_STRATEGIES}; do
    method="${strategy}_baseline"
    [[ "${strategy}" == "madry" ]] && method="madry_at"
    checkpoint="$(checkpoint_for "${strategy}")"
    output="$(attack_for "${method}")"
    require_artifact "${checkpoint}"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${output}" ]]; then
      echo "[Stage 8] Reuse ${method} attack: ${output}"
      continue
    fi
    run_logged "stage8_attack_${method}" \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.evaluate_attack \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
      --seed "${SEED}" --checkpoint_path "${checkpoint}" \
      --method_tag "${method}" --attack "${ATTACK}" --eps "${EPS}" \
      --batch_size "${ATTACK_BATCH_SIZE}" --gpu_id "${GPU_ID}" \
      --output_path "${output}" "${attack_subset_args[@]}" "${overwrite_args[@]}"
  done
fi

if should_run 9; then
  require_artifact "${AT_ATTACK}"
  require_artifact "${RPCF_AT_ATTACK}"
  require_artifact "${AT_PURIFY}"
  require_artifact "${RPCF_AT_PURIFY}"
  require_artifact "${RPCF_AT_HISTORY}.json"
  attack_pairs=(
    "madry_at=${AT_ATTACK}"
    "rpcf_at=${RPCF_AT_ATTACK}"
  )
  for strategy in ${BASELINE_STRATEGIES}; do
    attack_pairs+=("${strategy}_baseline=$(attack_for "${strategy}_baseline")")
  done
  purification_pairs=(
    "madry_at=${AT_PURIFY}"
    "rpcf_at=${RPCF_AT_PURIFY}"
  )
  run_logged stage9_summary \
    conda run -n "${CONDA_ENV}" --no-capture-output python -u \
    -m rpcf.compare_exp024 \
    --dataset "${DATASET}" --model "${MODEL}" --seed "${SEED}" \
    --fold "${FOLD}" --eps "${EPS}" --sample_num "${EVAL_SAMPLE_NUM}" \
    --attack_paths "${attack_pairs[@]}" \
    --purification_paths "${purification_pairs[@]}" \
    --sensitivity_path "${SENSITIVITY_PATH}" \
    --history_path "${RPCF_AT_HISTORY}.json" \
    --output_dir "${SUMMARY_DIR}"
fi

echo "[$(date -Is)] EXP-024 backbone pipeline finished: ${RUN_ID}"
