#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-thubenchmark}"
MODEL="${MODEL:-eegnet}"
SEED="${SEED:-42}"
FOLD="${FOLD:-0}"
EPS="${EPS:-0.03}"
GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-torch}"
RUN_ID="${EXP018_RUN_ID:-exp018_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${EXP018_LOG_ROOT:-logs/exp018/${RUN_ID}}"
TAG="${RUN_ID//[^A-Za-z0-9_.-]/_}"

START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-9}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"

AT_EPOCHS="${AT_EPOCHS:-400}"
AT_PATIENCE="${AT_PATIENCE:-20}"
CONSISTANCY_EPOCHS="${CONSISTANCY_EPOCHS:-400}"
CONSISTANCY_PATIENCE="${CONSISTANCY_PATIENCE:-20}"
RPCF_EPOCHS="${RPCF_EPOCHS:-100}"
RPCF_PATIENCE="${RPCF_PATIENCE:-15}"
TRAIN_SAMPLE_NUM="${TRAIN_SAMPLE_NUM:-}"
TRAIN_CACHE_SAMPLE_NUM="${TRAIN_CACHE_SAMPLE_NUM:-512}"
ATTACK_SAMPLE_NUM="${ATTACK_SAMPLE_NUM:-}"
EVAL_SAMPLE_NUM="${EVAL_SAMPLE_NUM:-512}"

AT_BATCH_SIZE="${AT_BATCH_SIZE:-128}"
CONSISTANCY_BATCH_SIZE="${CONSISTANCY_BATCH_SIZE:-128}"
RPCF_BATCH_SIZE="${RPCF_BATCH_SIZE:-64}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-32}"
AT_LR="${AT_LR:-0.001}"
AT_WEIGHT_DECAY="${AT_WEIGHT_DECAY:-0.0001}"
RPCF_LR="${RPCF_LR:-0.0001}"
RPCF_WEIGHT_DECAY="${RPCF_WEIGHT_DECAY:-0.0001}"

TRAIN_RANKS="15,20,25,30,35,40"
TRAIN_CONFIGS="PTR3d_8_2048_rank15_3d_interpolate.yaml,PTR3d_8_2048_rank20_3d_interpolate.yaml,PTR3d_8_2048_rank25_3d_interpolate.yaml,PTR3d_8_2048_rank30_3d_interpolate.yaml,PTR3d_8_2048_rank35_3d_interpolate.yaml,PTR3d_8_2048_rank40_3d_interpolate.yaml"
EVAL_RANKS="25,30"
EVAL_CONFIGS="PTR3d_8_2048_rank25_3d_interpolate.yaml,PTR3d_8_2048_rank30_3d_interpolate.yaml"

if [[ "${SMOKE}" == "1" ]]; then
  AT_EPOCHS="${SMOKE_EPOCHS:-1}"
  AT_PATIENCE=1
  CONSISTANCY_EPOCHS="${SMOKE_EPOCHS:-1}"
  CONSISTANCY_PATIENCE=1
  RPCF_EPOCHS="${SMOKE_EPOCHS:-1}"
  RPCF_PATIENCE=1
  TRAIN_SAMPLE_NUM="${SMOKE_TRAIN_SAMPLE_NUM:-2}"
  TRAIN_CACHE_SAMPLE_NUM="${SMOKE_CACHE_SAMPLE_NUM:-1}"
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
if [[ "${DATASET}" != "thubenchmark" || "${MODEL}" != "eegnet" || \
      "${SEED}" != "42" || "${FOLD}" != "0" || "${EPS}" != "0.03" ]]; then
  echo "EXP-018 is fixed to thubenchmark/eegnet/seed42/fold0/eps0.03." >&2
  exit 1
fi

mkdir -p \
  "${LOG_ROOT}" checkpoints ad_data/exp018 \
  purified_data/exp018/train_pair_consistancy \
  purified_data/exp018/rpcf_train purified_data/exp018/eval
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
mkdir -p "${NUMBA_CACHE_DIR}" "${XDG_CACHE_HOME}"

PROTOCOL="train_only_subject_no_ea_subject_split"
AT_TAG="${TAG}_at"
CONSISTANCY_TAG="${TAG}_consistancy"
AT_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${AT_TAG}_best.pth"
CONSISTANCY_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${CONSISTANCY_TAG}_best.pth"
RPCF_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_rpcf_${TAG}_best.pth"

PAIR25="purified_data/exp018/train_pair_consistancy/${TAG}_rank25.pth"
PAIR30="purified_data/exp018/train_pair_consistancy/${TAG}_rank30.pth"
RPCF_CACHE="purified_data/exp018/rpcf_train/${TAG}_six_rank.pth"
SENSITIVITY_PREFIX="${LOG_ROOT}/sensitivity"
HISTORY_PREFIX="${LOG_ROOT}/finetune"

AT_ATTACK="ad_data/exp018/${TAG}_at_autoattack.pth"
CONSISTANCY_ATTACK="ad_data/exp018/${TAG}_consistancy_autoattack.pth"
RPCF_ATTACK="ad_data/exp018/${TAG}_rpcf_autoattack.pth"
AT_PURIFY="purified_data/exp018/eval/${TAG}_at_rank25-30.pth"
CONSISTANCY_PURIFY="purified_data/exp018/eval/${TAG}_consistancy_rank25-30.pth"
RPCF_PURIFY="purified_data/exp018/eval/${TAG}_rpcf_rank25-30.pth"

printf '%s\n' "$$" > "${LOG_ROOT}/controller.pid"
cat > "${LOG_ROOT}/run_config.txt" <<EOF
RUN_ID=${RUN_ID}
DATASET=${DATASET}
MODEL=${MODEL}
SEED=${SEED}
FOLD=${FOLD}
EPS=${EPS}
TRAIN_CACHE_SAMPLE_NUM=${TRAIN_CACHE_SAMPLE_NUM}
EVAL_SAMPLE_NUM=${EVAL_SAMPLE_NUM}
AT_CHECKPOINT=${AT_CHECKPOINT}
CONSISTANCY_CHECKPOINT=${CONSISTANCY_CHECKPOINT}
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
train_subset_args=()
if [[ -n "${TRAIN_SAMPLE_NUM}" ]]; then
  train_subset_args=(--train_sample_num "${TRAIN_SAMPLE_NUM}")
fi
attack_subset_args=()
if [[ -n "${ATTACK_SAMPLE_NUM}" ]]; then
  attack_subset_args=(--sample_num "${ATTACK_SAMPLE_NUM}")
fi

if should_run 1; then
  if [[ "${SKIP_EXISTING}" == "1" && -f "${AT_CHECKPOINT}" ]]; then
    echo "[Stage 1] Reuse current-run AT checkpoint: ${AT_CHECKPOINT}"
  else
    run_logged stage1_train_at \
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
  for rank in 25 30; do
    if [[ "${rank}" == "25" ]]; then
      output="${PAIR25}"
    else
      output="${PAIR30}"
    fi
    if [[ "${SKIP_EXISTING}" == "1" && -f "${output}" ]]; then
      echo "[Stage 2] Reuse pair rank${rank}: ${output}"
      continue
    fi
    run_logged "stage2_pair_rank${rank}" \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      purify_train_pair_consistancy.py \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
      --config "PTR3d_8_2048_rank${rank}_3d_interpolate.yaml" \
      --sample_num "${TRAIN_CACHE_SAMPLE_NUM}" --attack autoattack --eps "${EPS}" \
      --checkpoint_path "${AT_CHECKPOINT}" --batch_size "${ATTACK_BATCH_SIZE}" \
      --seed "${SEED}" --gpu_id "${GPU_ID}" --no_ea \
      --output_tag "${CONSISTANCY_TAG}" \
      --output_dir "purified_data/exp018/train_pair_consistancy" \
      "${overwrite_args[@]}"
    generated="purified_data/exp018/train_pair_consistancy/${DATASET}_${MODEL}_no_ea_fold${FOLD}_seed${SEED}_train_pair_consistancy_autoattack_eps0.03_PTR3d_8_2048_rank${rank}_3d_interpolate_n${TRAIN_CACHE_SAMPLE_NUM}_tag${CONSISTANCY_TAG}.pth"
    if [[ "${DRY_RUN}" != "1" ]]; then
      mv "${generated}" "${output}"
    fi
  done
fi

if should_run 3; then
  require_artifact "${PAIR25}"
  require_artifact "${PAIR30}"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${CONSISTANCY_CHECKPOINT}" ]]; then
    echo "[Stage 3] Reuse current-run consistancy checkpoint: ${CONSISTANCY_CHECKPOINT}"
  else
    run_logged stage3_train_consistancy \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u train_AT_consistancy.py \
      --dataset "${DATASET}" --model "${MODEL}" --at_strategy madry \
      --fold "${FOLD}" --epsilon "${EPS}" --epochs "${CONSISTANCY_EPOCHS}" \
      --batch_size "${CONSISTANCY_BATCH_SIZE}" "${train_subset_args[@]}" \
      --lr "${AT_LR}" --weight_decay "${AT_WEIGHT_DECAY}" \
      --patience "${CONSISTANCY_PATIENCE}" --seed "${SEED}" --gpu_id "${GPU_ID}" \
      --no_ea --use_consistancy_aug --consistancy_aug_tag "${CONSISTANCY_TAG}" \
      --consistancy_aug_paths "${PAIR25}" "${PAIR30}"
  fi
fi

if should_run 4; then
  require_artifact "${AT_CHECKPOINT}"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${RPCF_CACHE}" ]]; then
    echo "[Stage 4] Reuse current-run RPCF cache: ${RPCF_CACHE}"
  else
    run_logged stage4_rpcf_cache \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u -m rpcf.generate_cache \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" --seed "${SEED}" \
      --attack autoattack --eps "${EPS}" --checkpoint_path "${AT_CHECKPOINT}" \
      --sample_num "${TRAIN_CACHE_SAMPLE_NUM}" \
      --attack_batch_size "${ATTACK_BATCH_SIZE}" --ranks "${TRAIN_RANKS}" \
      --configs "${TRAIN_CONFIGS}" --gpu_id "${GPU_ID}" --tag "${TAG}" \
      --output_path "${RPCF_CACHE}" "${overwrite_args[@]}"
  fi
fi

if should_run 5; then
  require_artifact "${RPCF_CACHE}"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${SENSITIVITY_PREFIX}.json" ]]; then
    echo "[Stage 5] Reuse sensitivity: ${SENSITIVITY_PREFIX}.json"
  else
    run_logged stage5_sensitivity \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.analyze_sensitivity \
      --cache_path "${RPCF_CACHE}" --checkpoint_path "${AT_CHECKPOINT}" \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" --seed "${SEED}" \
      --eps "${EPS}" --sensitive_ratio 0.4 --gpu_id "${GPU_ID}" \
      --output_prefix "${SENSITIVITY_PREFIX}"
  fi
fi

if should_run 6; then
  require_artifact "${RPCF_CACHE}"
  require_artifact "${SENSITIVITY_PREFIX}.json"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${RPCF_CHECKPOINT}" ]]; then
    echo "[Stage 6] Reuse current-run RPCF checkpoint: ${RPCF_CHECKPOINT}"
  else
    run_logged stage6_rpcf_finetune \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u -m rpcf.finetune \
      --cache_path "${RPCF_CACHE}" \
      --sensitivity_path "${SENSITIVITY_PREFIX}.json" \
      --checkpoint_path "${AT_CHECKPOINT}" --output_checkpoint "${RPCF_CHECKPOINT}" \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" --seed "${SEED}" \
      --epsilon "${EPS}" --epochs "${RPCF_EPOCHS}" \
      --batch_size "${RPCF_BATCH_SIZE}" --lr "${RPCF_LR}" \
      --weight_decay "${RPCF_WEIGHT_DECAY}" --patience "${RPCF_PATIENCE}" \
      --rank_temperature 0.5 --pgd_steps 10 --gpu_id "${GPU_ID}" \
      --history_prefix "${HISTORY_PREFIX}"
  fi
fi

if should_run 7; then
  require_artifact "${AT_CHECKPOINT}"
  require_artifact "${CONSISTANCY_CHECKPOINT}"
  require_artifact "${RPCF_CHECKPOINT}"
  for method in at consistancy rpcf; do
    case "${method}" in
      at) checkpoint="${AT_CHECKPOINT}"; output="${AT_ATTACK}" ;;
      consistancy) checkpoint="${CONSISTANCY_CHECKPOINT}"; output="${CONSISTANCY_ATTACK}" ;;
      rpcf) checkpoint="${RPCF_CHECKPOINT}"; output="${RPCF_ATTACK}" ;;
    esac
    if [[ "${SKIP_EXISTING}" == "1" && -f "${output}" ]]; then
      echo "[Stage 7] Reuse ${method} attack: ${output}"
      continue
    fi
    run_logged "stage7_attack_${method}" \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.evaluate_attack \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" --seed "${SEED}" \
      --checkpoint_path "${checkpoint}" --method_tag "${method}" \
      --attack autoattack --eps "${EPS}" --batch_size "${ATTACK_BATCH_SIZE}" \
      --gpu_id "${GPU_ID}" --output_path "${output}" \
      "${attack_subset_args[@]}" "${overwrite_args[@]}"
  done
fi

if should_run 8; then
  for method in at consistancy rpcf; do
    case "${method}" in
      at) checkpoint="${AT_CHECKPOINT}"; attack="${AT_ATTACK}"; output="${AT_PURIFY}" ;;
      consistancy) checkpoint="${CONSISTANCY_CHECKPOINT}"; attack="${CONSISTANCY_ATTACK}"; output="${CONSISTANCY_PURIFY}" ;;
      rpcf) checkpoint="${RPCF_CHECKPOINT}"; attack="${RPCF_ATTACK}"; output="${RPCF_PURIFY}" ;;
    esac
    require_artifact "${attack}"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${output}" ]]; then
      echo "[Stage 8] Reuse ${method} purification: ${output}"
      continue
    fi
    run_logged "stage8_purify_${method}" \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.evaluate_purification \
      --attack_path "${attack}" --checkpoint_path "${checkpoint}" \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" --seed "${SEED}" \
      --eps "${EPS}" --sample_num "${EVAL_SAMPLE_NUM}" --ranks "${EVAL_RANKS}" \
      --configs "${EVAL_CONFIGS}" --gpu_id "${GPU_ID}" \
      --output_path "${output}" "${overwrite_args[@]}"
  done
fi

if should_run 9; then
  require_artifact "${AT_PURIFY}"
  require_artifact "${CONSISTANCY_PURIFY}"
  require_artifact "${RPCF_PURIFY}"
  require_artifact "${SENSITIVITY_PREFIX}.json"
  require_artifact "${HISTORY_PREFIX}.json"
  run_logged stage9_summary \
    conda run -n "${CONDA_ENV}" --no-capture-output python -u \
    -m rpcf.compare_exp018 \
    --dataset "${DATASET}" --model "${MODEL}" --seed "${SEED}" --fold "${FOLD}" \
    --eps "${EPS}" --ranks "${EVAL_RANKS}" \
    --sample_num "${EVAL_SAMPLE_NUM}" --gpu_id "${GPU_ID}" \
    --at_checkpoint "${AT_CHECKPOINT}" \
    --consistancy_checkpoint "${CONSISTANCY_CHECKPOINT}" \
    --rpcf_checkpoint "${RPCF_CHECKPOINT}" \
    --at_purification_path "${AT_PURIFY}" \
    --consistancy_purification_path "${CONSISTANCY_PURIFY}" \
    --rpcf_purification_path "${RPCF_PURIFY}" \
    --sensitivity_path "${SENSITIVITY_PREFIX}.json" \
    --history_path "${HISTORY_PREFIX}.json" \
    --output_dir "${LOG_ROOT}"
fi

echo "[$(date -Is)] EXP-018 pipeline finished: ${RUN_ID}"
