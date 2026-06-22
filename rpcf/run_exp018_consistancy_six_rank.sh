#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-thubenchmark}"
MODEL="${MODEL:-eegnet}"
SEED="${SEED:-42}"
FOLD="${FOLD:-0}"
EPS="${EPS:-0.03}"
GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-torch}"

BASE_RUN_ID="${BASE_RUN_ID:-exp018_full_20260612_124131}"
SELECTIVE_RUN_ID="${SELECTIVE_RUN_ID:-exp018_rpcf_no_early_stop_20260614_2357}"
ALL_LAYERS_RUN_ID="${ALL_LAYERS_RUN_ID:-exp018_rpcf_all_layers_20260615_0933}"
UNIFORM_RUN_ID="${UNIFORM_RUN_ID:-exp018_rpcf_static_ranks_20260615_1155}"

RUN_ID="${EXP018_CONSISTANCY_SIX_RANK_RUN_ID:-exp018_seed${SEED}_fold${FOLD}_consistancy_six_rank_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${EXP018_LOG_ROOT:-logs/exp018/${RUN_ID}}"
TAG="${RUN_ID//[^A-Za-z0-9_.-]/_}"

START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-5}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"

CONSISTANCY_EPOCHS="${CONSISTANCY_EPOCHS:-400}"
CONSISTANCY_PATIENCE="${CONSISTANCY_PATIENCE:-20}"
TRAIN_SAMPLE_NUM="${TRAIN_SAMPLE_NUM:-}"
TRAIN_CACHE_SAMPLE_NUM="${TRAIN_CACHE_SAMPLE_NUM:-512}"
ATTACK_SAMPLE_NUM="${ATTACK_SAMPLE_NUM:-}"
EVAL_SAMPLE_NUM="${EVAL_SAMPLE_NUM:-512}"

CONSISTANCY_BATCH_SIZE="${CONSISTANCY_BATCH_SIZE:-128}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-32}"
AT_LR="${AT_LR:-0.001}"
AT_WEIGHT_DECAY="${AT_WEIGHT_DECAY:-0.0001}"

TRAIN_RANKS=(15 20 25 30 35 40)
TRAIN_RANKS_CSV="15,20,25,30,35,40"
EVAL_RANKS="25,30"
EVAL_CONFIGS="PTR3d_8_2048_rank25_3d_interpolate.yaml,PTR3d_8_2048_rank30_3d_interpolate.yaml"

if [[ "${SMOKE}" == "1" ]]; then
  CONSISTANCY_EPOCHS="${SMOKE_EPOCHS:-1}"
  CONSISTANCY_PATIENCE=1
  TRAIN_SAMPLE_NUM="${SMOKE_TRAIN_SAMPLE_NUM:-2}"
  TRAIN_CACHE_SAMPLE_NUM="${SMOKE_CACHE_SAMPLE_NUM:-1}"
  ATTACK_SAMPLE_NUM="${SMOKE_ATTACK_SAMPLE_NUM:-2}"
  EVAL_SAMPLE_NUM="${SMOKE_EVAL_SAMPLE_NUM:-2}"
fi

if ! [[ "${START_STAGE}" =~ ^[1-5]$ && "${STOP_STAGE}" =~ ^[1-5]$ ]]; then
  echo "START_STAGE and STOP_STAGE must be in [1, 5]." >&2
  exit 1
fi
if (( START_STAGE > STOP_STAGE )); then
  echo "START_STAGE cannot be greater than STOP_STAGE." >&2
  exit 1
fi
if [[ "${DATASET}" != "thubenchmark" || "${MODEL}" != "eegnet" || "${EPS}" != "0.03" ]]; then
  echo "EXP-018 consistancy six-rank rerun currently supports thubenchmark/eegnet/eps0.03 only." >&2
  exit 1
fi

PROTOCOL="train_only_subject_no_ea_subject_split"
CONSISTANCY_TAG="${TAG}_consistancy_six_rank"

AT_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${BASE_RUN_ID}_at_best.pth"
CONSISTANCY_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${CONSISTANCY_TAG}_best.pth"
RPCF_SELECTIVE_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_rpcf_${SELECTIVE_RUN_ID}_best.pth"
RPCF_ALL_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_rpcf_all_layers_${ALL_LAYERS_RUN_ID}_best.pth"
RPCF_UNIFORM_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_rpcf_static_ranks_${UNIFORM_RUN_ID}_best.pth"

PAIR_DIR="purified_data/exp018/train_pair_consistancy_six_rank"
PAIR_PATHS=()
for rank in "${TRAIN_RANKS[@]}"; do
  PAIR_PATHS+=(
    "${PAIR_DIR}/${DATASET}_${MODEL}_no_ea_fold${FOLD}_seed${SEED}_train_pair_consistancy_autoattack_eps${EPS}_PTR3d_8_2048_rank${rank}_3d_interpolate_n${TRAIN_CACHE_SAMPLE_NUM}_tag${CONSISTANCY_TAG}.pth"
  )
done

CONSISTANCY_ATTACK="ad_data/exp018/${TAG}_consistancy_six_rank_autoattack.pth"
CONSISTANCY_PURIFY="purified_data/exp018/eval/${TAG}_consistancy_six_rank_rank25-30.pth"

AT_PURIFY="purified_data/exp018/eval/${BASE_RUN_ID}_at_rank25-30.pth"
RPCF_SELECTIVE_PURIFY="purified_data/exp018/eval/${SELECTIVE_RUN_ID}_rpcf_rank25-30.pth"
RPCF_ALL_PURIFY="purified_data/exp018/eval/${ALL_LAYERS_RUN_ID}_rpcf_all_layers_rank25-30.pth"
RPCF_UNIFORM_PURIFY="purified_data/exp018/eval/${UNIFORM_RUN_ID}_rpcf_static_ranks_rank25-30.pth"

SENSITIVITY_PATH="logs/exp018/${SELECTIVE_RUN_ID}/sensitivity.json"
SELECTIVE_HISTORY="logs/exp018/${SELECTIVE_RUN_ID}/finetune"
ALL_HISTORY="logs/exp018/${ALL_LAYERS_RUN_ID}/finetune"
UNIFORM_HISTORY="logs/exp018/${UNIFORM_RUN_ID}/finetune"

SELECTIVE_SUMMARY_DIR="${LOG_ROOT}/comparison_selective"
ALL_SUMMARY_DIR="${LOG_ROOT}/comparison_all_layers"
UNIFORM_SUMMARY_DIR="${LOG_ROOT}/comparison_rank_weight_uniform"
FIVE_SUMMARY_DIR="${LOG_ROOT}/five_methods"

mkdir -p \
  "${LOG_ROOT}" "${PAIR_DIR}" checkpoints ad_data/exp018 purified_data/exp018/eval
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
mkdir -p "${NUMBA_CACHE_DIR}" "${XDG_CACHE_HOME}"

printf '%s\n' "$$" > "${LOG_ROOT}/controller.pid"
cat > "${LOG_ROOT}/run_config.txt" <<EOF
EXPERIMENT_ID=EXP-018-consistancy-six-rank
RUN_ID=${RUN_ID}
BASE_RUN_ID=${BASE_RUN_ID}
SELECTIVE_RUN_ID=${SELECTIVE_RUN_ID}
ALL_LAYERS_RUN_ID=${ALL_LAYERS_RUN_ID}
UNIFORM_RUN_ID=${UNIFORM_RUN_ID}
DATASET=${DATASET}
MODEL=${MODEL}
SEED=${SEED}
FOLD=${FOLD}
EPS=${EPS}
TRAIN_RANKS=${TRAIN_RANKS_CSV}
TRAIN_CACHE_SAMPLE_NUM=${TRAIN_CACHE_SAMPLE_NUM}
EVAL_RANKS=${EVAL_RANKS}
EVAL_SAMPLE_NUM=${EVAL_SAMPLE_NUM}
AT_CHECKPOINT=${AT_CHECKPOINT}
CONSISTANCY_CHECKPOINT=${CONSISTANCY_CHECKPOINT}
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
  require_artifact "${AT_CHECKPOINT}"
  for index in "${!TRAIN_RANKS[@]}"; do
    rank="${TRAIN_RANKS[${index}]}"
    output="${PAIR_PATHS[${index}]}"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${output}" ]]; then
      echo "[Stage 1] Reuse consistancy six-rank pair rank${rank}: ${output}"
      continue
    fi
    run_logged "stage1_pair_rank${rank}" \
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

if should_run 2; then
  for path in "${PAIR_PATHS[@]}"; do
    require_artifact "${path}"
  done
  if [[ "${SKIP_EXISTING}" == "1" && -f "${CONSISTANCY_CHECKPOINT}" ]]; then
    echo "[Stage 2] Reuse consistancy six-rank checkpoint: ${CONSISTANCY_CHECKPOINT}"
  else
    run_logged stage2_train_consistancy_six_rank \
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

if should_run 3; then
  require_artifact "${CONSISTANCY_CHECKPOINT}"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${CONSISTANCY_ATTACK}" ]]; then
    echo "[Stage 3] Reuse consistancy six-rank attack: ${CONSISTANCY_ATTACK}"
  else
    run_logged stage3_attack_consistancy_six_rank \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.evaluate_attack \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
      --seed "${SEED}" --checkpoint_path "${CONSISTANCY_CHECKPOINT}" \
      --method_tag consistancy_six_rank --attack autoattack --eps "${EPS}" \
      --batch_size "${ATTACK_BATCH_SIZE}" --gpu_id "${GPU_ID}" \
      --output_path "${CONSISTANCY_ATTACK}" \
      "${attack_subset_args[@]}" "${overwrite_args[@]}"
  fi
fi

if should_run 4; then
  require_artifact "${CONSISTANCY_ATTACK}"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${CONSISTANCY_PURIFY}" ]]; then
    echo "[Stage 4] Reuse consistancy six-rank purification: ${CONSISTANCY_PURIFY}"
  else
    run_logged stage4_purify_consistancy_six_rank \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.evaluate_purification \
      --attack_path "${CONSISTANCY_ATTACK}" \
      --checkpoint_path "${CONSISTANCY_CHECKPOINT}" \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
      --seed "${SEED}" --eps "${EPS}" --sample_num "${EVAL_SAMPLE_NUM}" \
      --ranks "${EVAL_RANKS}" --configs "${EVAL_CONFIGS}" \
      --gpu_id "${GPU_ID}" --output_path "${CONSISTANCY_PURIFY}" \
      "${overwrite_args[@]}"
  fi
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
    --experiment_id EXP-018
}

if should_run 5; then
  for path in \
    "${AT_PURIFY}" "${CONSISTANCY_PURIFY}" "${RPCF_SELECTIVE_PURIFY}" \
    "${RPCF_ALL_PURIFY}" "${RPCF_UNIFORM_PURIFY}" \
    "${SENSITIVITY_PATH}" "${SELECTIVE_HISTORY}.json" \
    "${ALL_HISTORY}.json" "${UNIFORM_HISTORY}.json"; do
    require_artifact "${path}"
  done
  run_three_method_summary \
    stage5_summary_selective "${RPCF_SELECTIVE_CHECKPOINT}" \
    "${RPCF_SELECTIVE_PURIFY}" "${SELECTIVE_HISTORY}" "${SELECTIVE_SUMMARY_DIR}"
  run_three_method_summary \
    stage5_summary_all_layers "${RPCF_ALL_CHECKPOINT}" \
    "${RPCF_ALL_PURIFY}" "${ALL_HISTORY}" "${ALL_SUMMARY_DIR}"
  run_three_method_summary \
    stage5_summary_rank_weight_uniform "${RPCF_UNIFORM_CHECKPOINT}" \
    "${RPCF_UNIFORM_PURIFY}" "${UNIFORM_HISTORY}" "${UNIFORM_SUMMARY_DIR}"
  run_logged stage5_summary_five_methods \
    conda run -n "${CONDA_ENV}" --no-capture-output python -u \
    -m rpcf.compare_exp018_five_methods \
    --base_summary "${SELECTIVE_SUMMARY_DIR}/summary.json" \
    --selective_summary "${SELECTIVE_SUMMARY_DIR}/summary.json" \
    --all_layers_summary "${ALL_SUMMARY_DIR}/summary.json" \
    --uniform_summary "${UNIFORM_SUMMARY_DIR}/summary.json" \
    --output_dir "${FIVE_SUMMARY_DIR}" --experiment_id EXP-018
fi

echo "[$(date -Is)] EXP-018 consistancy six-rank rerun finished: ${RUN_ID}"
