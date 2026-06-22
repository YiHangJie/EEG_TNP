#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-thubenchmark}"
MODEL="${MODEL:-eegnet}"
SEED="${SEED:-43}"
FOLD="${FOLD:-0}"
EPS="${EPS:-0.05}"
GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-torch}"
RUN_ID="${EXP020_RUN_ID:-exp020_seed${SEED}_fold${FOLD}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${EXP020_LOG_ROOT:-logs/exp020/${RUN_ID}}"
TAG="${RUN_ID//[^A-Za-z0-9_.-]/_}"

START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-11}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"

AT_EPOCHS="${AT_EPOCHS:-400}"
AT_PATIENCE="${AT_PATIENCE:-20}"
CONSISTANCY_EPOCHS="${CONSISTANCY_EPOCHS:-400}"
CONSISTANCY_PATIENCE="${CONSISTANCY_PATIENCE:-20}"
RPCF_EPOCHS="${RPCF_EPOCHS:-100}"
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
  TRAIN_SAMPLE_NUM="${SMOKE_TRAIN_SAMPLE_NUM:-2}"
  TRAIN_CACHE_SAMPLE_NUM="${SMOKE_CACHE_SAMPLE_NUM:-1}"
  ATTACK_SAMPLE_NUM="${SMOKE_ATTACK_SAMPLE_NUM:-2}"
  EVAL_SAMPLE_NUM="${SMOKE_EVAL_SAMPLE_NUM:-2}"
fi

if ! [[ "${START_STAGE}" =~ ^([1-9]|1[01])$ && "${STOP_STAGE}" =~ ^([1-9]|1[01])$ ]]; then
  echo "START_STAGE and STOP_STAGE must be in [1, 11]." >&2
  exit 1
fi
if (( START_STAGE > STOP_STAGE )); then
  echo "START_STAGE cannot be greater than STOP_STAGE." >&2
  exit 1
fi
if [[ "${DATASET}" != "thubenchmark" || "${MODEL}" != "eegnet" || "${EPS}" != "0.05" ]]; then
  echo "EXP-020 currently supports thubenchmark/eegnet/eps0.05 only." >&2
  exit 1
fi

PROTOCOL="train_only_subject_no_ea_subject_split"
AT_TAG="${TAG}_at"
CONSISTANCY_TAG="${TAG}_consistancy_six_rank"
AT_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${AT_TAG}_best.pth"
CONSISTANCY_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${CONSISTANCY_TAG}_best.pth"
RPCF_SELECTIVE_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_rpcf_selective_${TAG}_best.pth"
RPCF_ALL_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_rpcf_all_layers_${TAG}_best.pth"
RPCF_UNIFORM_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_rpcf_rank_weight_uniform_${TAG}_best.pth"

PAIR_DIR="purified_data/exp020/train_pair_consistancy_six_rank"
IFS=',' read -r -a CONSISTANCY_TRAIN_RANKS <<< "${TRAIN_RANKS}"
PAIR_PATHS=()
for rank in "${CONSISTANCY_TRAIN_RANKS[@]}"; do
  PAIR_PATHS+=(
    "${PAIR_DIR}/${DATASET}_${MODEL}_no_ea_fold${FOLD}_seed${SEED}_train_pair_consistancy_autoattack_eps${EPS}_PTR3d_8_2048_rank${rank}_3d_interpolate_n${TRAIN_CACHE_SAMPLE_NUM}_tag${CONSISTANCY_TAG}.pth"
  )
done
RPCF_CACHE="purified_data/exp020/rpcf_train/${TAG}_six_rank.pth"
SENSITIVITY_PREFIX="${LOG_ROOT}/sensitivity"
SENSITIVITY_PATH="${SENSITIVITY_PREFIX}.json"
SELECTIVE_HISTORY="${LOG_ROOT}/finetune_selective"
ALL_HISTORY="${LOG_ROOT}/finetune_all_layers"
UNIFORM_HISTORY="${LOG_ROOT}/finetune_rank_weight_uniform"

AT_ATTACK="ad_data/exp020/${TAG}_madry_at_autoattack.pth"
CONSISTANCY_ATTACK="ad_data/exp020/${TAG}_consistancy_six_rank_autoattack.pth"
RPCF_SELECTIVE_ATTACK="ad_data/exp020/${TAG}_rpcf_selective_autoattack.pth"
RPCF_ALL_ATTACK="ad_data/exp020/${TAG}_rpcf_all_layers_autoattack.pth"
RPCF_UNIFORM_ATTACK="ad_data/exp020/${TAG}_rpcf_rank_weight_uniform_autoattack.pth"

AT_PURIFY="purified_data/exp020/eval/${TAG}_madry_at_rank25-30.pth"
CONSISTANCY_PURIFY="purified_data/exp020/eval/${TAG}_consistancy_six_rank_rank25-30.pth"
RPCF_SELECTIVE_PURIFY="purified_data/exp020/eval/${TAG}_rpcf_selective_rank25-30.pth"
RPCF_ALL_PURIFY="purified_data/exp020/eval/${TAG}_rpcf_all_layers_rank25-30.pth"
RPCF_UNIFORM_PURIFY="purified_data/exp020/eval/${TAG}_rpcf_rank_weight_uniform_rank25-30.pth"

SELECTIVE_SUMMARY_DIR="${LOG_ROOT}/comparison_selective"
ALL_SUMMARY_DIR="${LOG_ROOT}/comparison_all_layers"
UNIFORM_SUMMARY_DIR="${LOG_ROOT}/comparison_rank_weight_uniform"
FIVE_SUMMARY_DIR="${LOG_ROOT}/five_methods"

mkdir -p \
  "${LOG_ROOT}" "${PAIR_DIR}" checkpoints ad_data/exp020 \
  purified_data/exp020/rpcf_train purified_data/exp020/eval
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
mkdir -p "${NUMBA_CACHE_DIR}" "${XDG_CACHE_HOME}"

printf '%s\n' "$$" > "${LOG_ROOT}/controller.pid"
cat > "${LOG_ROOT}/run_config.txt" <<EOF
EXPERIMENT_ID=EXP-020
RUN_ID=${RUN_ID}
DATASET=${DATASET}
MODEL=${MODEL}
SEED=${SEED}
FOLD=${FOLD}
EPS=${EPS}
TRAIN_CACHE_SAMPLE_NUM=${TRAIN_CACHE_SAMPLE_NUM}
EVAL_SAMPLE_NUM=${EVAL_SAMPLE_NUM}
CONSISTANCY_TRAIN_RANKS=${TRAIN_RANKS}
AT_CHECKPOINT=${AT_CHECKPOINT}
CONSISTANCY_CHECKPOINT=${CONSISTANCY_CHECKPOINT}
RPCF_SELECTIVE_CHECKPOINT=${RPCF_SELECTIVE_CHECKPOINT}
RPCF_ALL_CHECKPOINT=${RPCF_ALL_CHECKPOINT}
RPCF_UNIFORM_CHECKPOINT=${RPCF_UNIFORM_CHECKPOINT}
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
    echo "[Stage 1] Reuse AT checkpoint: ${AT_CHECKPOINT}"
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
  for index in "${!CONSISTANCY_TRAIN_RANKS[@]}"; do
    rank="${CONSISTANCY_TRAIN_RANKS[${index}]}"
    output="${PAIR_PATHS[${index}]}"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${output}" ]]; then
      echo "[Stage 2] Reuse consistancy six-rank pair rank${rank}: ${output}"
      continue
    fi
    run_logged "stage2_pair_rank${rank}" \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      purify_train_pair_consistancy.py \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
      --config "PTR3d_8_2048_rank${rank}_3d_interpolate.yaml" \
      --sample_num "${TRAIN_CACHE_SAMPLE_NUM}" --attack autoattack \
      --eps "${EPS}" --checkpoint_path "${AT_CHECKPOINT}" \
      --batch_size "${ATTACK_BATCH_SIZE}" --seed "${SEED}" \
      --gpu_id "${GPU_ID}" --no_ea --output_tag "${CONSISTANCY_TAG}" \
      --output_dir "${PAIR_DIR}" "${overwrite_args[@]}"
  done
fi

if should_run 3; then
  for path in "${PAIR_PATHS[@]}"; do
    require_artifact "${path}"
  done
  if [[ "${SKIP_EXISTING}" == "1" && -f "${CONSISTANCY_CHECKPOINT}" ]]; then
    echo "[Stage 3] Reuse consistancy six-rank checkpoint: ${CONSISTANCY_CHECKPOINT}"
  else
    run_logged stage3_train_consistancy_six_rank \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      train_AT_consistancy.py \
      --dataset "${DATASET}" --model "${MODEL}" --at_strategy madry \
      --fold "${FOLD}" --epsilon "${EPS}" --epochs "${CONSISTANCY_EPOCHS}" \
      --batch_size "${CONSISTANCY_BATCH_SIZE}" "${train_subset_args[@]}" \
      --lr "${AT_LR}" --weight_decay "${AT_WEIGHT_DECAY}" \
      --patience "${CONSISTANCY_PATIENCE}" --seed "${SEED}" \
      --gpu_id "${GPU_ID}" --no_ea --use_consistancy_aug \
      --consistancy_aug_tag "${CONSISTANCY_TAG}" \
      --consistancy_aug_paths "${PAIR_PATHS[@]}"
  fi
fi

if should_run 4; then
  require_artifact "${AT_CHECKPOINT}"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${RPCF_CACHE}" ]]; then
    echo "[Stage 4] Reuse RPCF cache: ${RPCF_CACHE}"
  else
    run_logged stage4_generate_rpcf_cache \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.generate_cache \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
      --seed "${SEED}" --attack autoattack --eps "${EPS}" \
      --checkpoint_path "${AT_CHECKPOINT}" \
      --sample_num "${TRAIN_CACHE_SAMPLE_NUM}" \
      --attack_batch_size "${ATTACK_BATCH_SIZE}" --ranks "${TRAIN_RANKS}" \
      --configs "${TRAIN_CONFIGS}" --gpu_id "${GPU_ID}" --tag "${TAG}" \
      --output_path "${RPCF_CACHE}" "${overwrite_args[@]}"
  fi
fi

if should_run 5; then
  require_artifact "${RPCF_CACHE}"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${SENSITIVITY_PATH}" ]]; then
    echo "[Stage 5] Reuse sensitivity: ${SENSITIVITY_PATH}"
  else
    run_logged stage5_analyze_sensitivity \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.analyze_sensitivity \
      --cache_path "${RPCF_CACHE}" --checkpoint_path "${AT_CHECKPOINT}" \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
      --seed "${SEED}" --eps "${EPS}" --sensitive_ratio 0.4 \
      --gpu_id "${GPU_ID}" --output_prefix "${SENSITIVITY_PREFIX}"
  fi
fi

run_rpcf_finetune() {
  local stage_name="$1"
  local checkpoint="$2"
  local history_prefix="$3"
  shift 3
  if [[ "${SKIP_EXISTING}" == "1" && -f "${checkpoint}" ]]; then
    echo "[${stage_name}] Reuse RPCF checkpoint: ${checkpoint}"
    return
  fi
  run_logged "${stage_name}" \
    conda run -n "${CONDA_ENV}" --no-capture-output python -u \
    -m rpcf.finetune \
    --cache_path "${RPCF_CACHE}" --sensitivity_path "${SENSITIVITY_PATH}" \
    --checkpoint_path "${AT_CHECKPOINT}" --output_checkpoint "${checkpoint}" \
    --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
    --seed "${SEED}" --epsilon "${EPS}" --epochs "${RPCF_EPOCHS}" \
    --batch_size "${RPCF_BATCH_SIZE}" --lr "${RPCF_LR}" \
    --weight_decay "${RPCF_WEIGHT_DECAY}" --rank_temperature 0.5 \
    --consistancy_temperature 2.0 --pgd_steps 10 --gpu_id "${GPU_ID}" \
    --history_prefix "${history_prefix}" "$@"
}

if should_run 6; then
  require_artifact "${RPCF_CACHE}"
  require_artifact "${SENSITIVITY_PATH}"
  run_rpcf_finetune \
    stage6_finetune_rpcf_selective \
    "${RPCF_SELECTIVE_CHECKPOINT}" "${SELECTIVE_HISTORY}"
fi

if should_run 7; then
  require_artifact "${RPCF_CACHE}"
  require_artifact "${SENSITIVITY_PATH}"
  run_rpcf_finetune \
    stage7_finetune_rpcf_all_layers \
    "${RPCF_ALL_CHECKPOINT}" "${ALL_HISTORY}" --all_layers
fi

if should_run 8; then
  require_artifact "${RPCF_CACHE}"
  require_artifact "${SENSITIVITY_PATH}"
  run_rpcf_finetune \
    stage8_finetune_rpcf_rank_weight_uniform \
    "${RPCF_UNIFORM_CHECKPOINT}" "${UNIFORM_HISTORY}" --static_rank_weights
fi

method_checkpoint() {
  case "$1" in
    madry_at) printf '%s\n' "${AT_CHECKPOINT}" ;;
    consistancy) printf '%s\n' "${CONSISTANCY_CHECKPOINT}" ;;
    rpcf_selective) printf '%s\n' "${RPCF_SELECTIVE_CHECKPOINT}" ;;
    rpcf_all_layers) printf '%s\n' "${RPCF_ALL_CHECKPOINT}" ;;
    rpcf_rank_weight_uniform) printf '%s\n' "${RPCF_UNIFORM_CHECKPOINT}" ;;
    *) return 1 ;;
  esac
}

method_attack() {
  case "$1" in
    madry_at) printf '%s\n' "${AT_ATTACK}" ;;
    consistancy) printf '%s\n' "${CONSISTANCY_ATTACK}" ;;
    rpcf_selective) printf '%s\n' "${RPCF_SELECTIVE_ATTACK}" ;;
    rpcf_all_layers) printf '%s\n' "${RPCF_ALL_ATTACK}" ;;
    rpcf_rank_weight_uniform) printf '%s\n' "${RPCF_UNIFORM_ATTACK}" ;;
    *) return 1 ;;
  esac
}

method_purification() {
  case "$1" in
    madry_at) printf '%s\n' "${AT_PURIFY}" ;;
    consistancy) printf '%s\n' "${CONSISTANCY_PURIFY}" ;;
    rpcf_selective) printf '%s\n' "${RPCF_SELECTIVE_PURIFY}" ;;
    rpcf_all_layers) printf '%s\n' "${RPCF_ALL_PURIFY}" ;;
    rpcf_rank_weight_uniform) printf '%s\n' "${RPCF_UNIFORM_PURIFY}" ;;
    *) return 1 ;;
  esac
}

METHODS=(
  madry_at
  consistancy
  rpcf_selective
  rpcf_all_layers
  rpcf_rank_weight_uniform
)

if should_run 9; then
  for method in "${METHODS[@]}"; do
    checkpoint="$(method_checkpoint "${method}")"
    output="$(method_attack "${method}")"
    require_artifact "${checkpoint}"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${output}" ]]; then
      echo "[Stage 9] Reuse ${method} attack: ${output}"
      continue
    fi
    run_logged "stage9_attack_${method}" \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.evaluate_attack \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
      --seed "${SEED}" --checkpoint_path "${checkpoint}" \
      --method_tag "${method}" --attack autoattack --eps "${EPS}" \
      --batch_size "${ATTACK_BATCH_SIZE}" --gpu_id "${GPU_ID}" \
      --output_path "${output}" \
      "${attack_subset_args[@]}" "${overwrite_args[@]}"
  done
fi

if should_run 10; then
  for method in "${METHODS[@]}"; do
    checkpoint="$(method_checkpoint "${method}")"
    attack_path="$(method_attack "${method}")"
    output="$(method_purification "${method}")"
    require_artifact "${attack_path}"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${output}" ]]; then
      echo "[Stage 10] Reuse ${method} purification: ${output}"
      continue
    fi
    run_logged "stage10_purify_${method}" \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.evaluate_purification \
      --attack_path "${attack_path}" --checkpoint_path "${checkpoint}" \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
      --seed "${SEED}" --eps "${EPS}" --sample_num "${EVAL_SAMPLE_NUM}" \
      --ranks "${EVAL_RANKS}" --configs "${EVAL_CONFIGS}" \
      --gpu_id "${GPU_ID}" --output_path "${output}" "${overwrite_args[@]}"
  done
fi

run_three_method_summary() {
  local stage_name="$1"
  local rpcf_checkpoint="$2"
  local rpcf_purification="$3"
  local history_path="$4"
  local output_dir="$5"
  run_logged "${stage_name}" \
    conda run -n "${CONDA_ENV}" --no-capture-output python -u \
    -m rpcf.compare_exp018 \
    --dataset "${DATASET}" --model "${MODEL}" --seed "${SEED}" \
    --fold "${FOLD}" --eps "${EPS}" --ranks "${EVAL_RANKS}" \
    --sample_num "${EVAL_SAMPLE_NUM}" --gpu_id "${GPU_ID}" \
    --at_checkpoint "${AT_CHECKPOINT}" \
    --consistancy_checkpoint "${CONSISTANCY_CHECKPOINT}" \
    --rpcf_checkpoint "${rpcf_checkpoint}" \
    --at_purification_path "${AT_PURIFY}" \
    --consistancy_purification_path "${CONSISTANCY_PURIFY}" \
    --rpcf_purification_path "${rpcf_purification}" \
    --sensitivity_path "${SENSITIVITY_PATH}" \
    --history_path "${history_path}.json" --output_dir "${output_dir}" \
    --experiment_id EXP-020
}

if should_run 11; then
  for path in \
    "${AT_PURIFY}" "${CONSISTANCY_PURIFY}" "${RPCF_SELECTIVE_PURIFY}" \
    "${RPCF_ALL_PURIFY}" "${RPCF_UNIFORM_PURIFY}" \
    "${SELECTIVE_HISTORY}.json" "${ALL_HISTORY}.json" "${UNIFORM_HISTORY}.json"; do
    require_artifact "${path}"
  done
  run_three_method_summary \
    stage11_summary_selective "${RPCF_SELECTIVE_CHECKPOINT}" \
    "${RPCF_SELECTIVE_PURIFY}" "${SELECTIVE_HISTORY}" "${SELECTIVE_SUMMARY_DIR}"
  run_three_method_summary \
    stage11_summary_all_layers "${RPCF_ALL_CHECKPOINT}" \
    "${RPCF_ALL_PURIFY}" "${ALL_HISTORY}" "${ALL_SUMMARY_DIR}"
  run_three_method_summary \
    stage11_summary_rank_weight_uniform "${RPCF_UNIFORM_CHECKPOINT}" \
    "${RPCF_UNIFORM_PURIFY}" "${UNIFORM_HISTORY}" "${UNIFORM_SUMMARY_DIR}"
  run_logged stage11_summary_five_methods \
    conda run -n "${CONDA_ENV}" --no-capture-output python -u \
    -m rpcf.compare_exp018_five_methods \
    --base_summary "${SELECTIVE_SUMMARY_DIR}/summary.json" \
    --selective_summary "${SELECTIVE_SUMMARY_DIR}/summary.json" \
    --all_layers_summary "${ALL_SUMMARY_DIR}/summary.json" \
    --uniform_summary "${UNIFORM_SUMMARY_DIR}/summary.json" \
    --output_dir "${FIVE_SUMMARY_DIR}" --experiment_id EXP-020
fi

echo "[$(date -Is)] EXP-020 pipeline finished: ${RUN_ID}"
