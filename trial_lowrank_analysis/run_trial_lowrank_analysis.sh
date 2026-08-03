#!/usr/bin/env bash
set -euo pipefail

# EXP-028：trial 级 HOSVD 低秩性分析。
#
# 正式运行：
#   RUN_ID=exp028_trial_lowrank_seed42_YYYYMMDD_HHMMSS
#   mkdir -p "logs/exp028/${RUN_ID}"
#   nohup setsid env TRIAL_LOWRANK_RUN_ID="${RUN_ID}" \
#     TRIAL_LOWRANK_PHYSICAL_GPU=7 \
#     bash trial_lowrank_analysis/run_trial_lowrank_analysis.sh \
#     > "logs/exp028/${RUN_ID}/controller.log" 2>&1 < /dev/null &
#
# Dry-run / 续跑：
#   DRY_RUN=1 bash trial_lowrank_analysis/run_trial_lowrank_analysis.sh
#   TRIAL_LOWRANK_RUN_ID=<same_run_id> START_STAGE=2 STOP_STAGE=2 \
#     bash trial_lowrank_analysis/run_trial_lowrank_analysis.sh

DATASET="${TRIAL_LOWRANK_DATASET:-thubenchmark}"
MODEL="${TRIAL_LOWRANK_MODEL:-eegnet}"
SEED="${TRIAL_LOWRANK_SEED:-42}"
FOLD="${TRIAL_LOWRANK_FOLD:-0}"
EPS="${TRIAL_LOWRANK_EPS:-0.03}"
SAMPLE_NUM="${TRIAL_LOWRANK_SAMPLE_NUM:-512}"
PGD_STEPS="${TRIAL_LOWRANK_PGD_STEPS:-200}"
PGD_ALPHA="${TRIAL_LOWRANK_PGD_ALPHA:-0.00784313725490196}"
ATTACK_BATCH_SIZE="${TRIAL_LOWRANK_ATTACK_BATCH_SIZE:-32}"
MIN_TRIALS="${TRIAL_LOWRANK_MIN_TRIALS:-3}"
VIEWS="${TRIAL_LOWRANK_VIEWS:-trial time channel frequency}"
FREQUENCY_REPRESENTATION="${TRIAL_LOWRANK_FREQUENCY_REPRESENTATION:-complex}"
CHANNEL_VIEW_SPACE="${TRIAL_LOWRANK_CHANNEL_VIEW_SPACE:-interpolated_grid}"

TRAIN_EPOCHS="${TRIAL_LOWRANK_TRAIN_EPOCHS:-400}"
TRAIN_BATCH_SIZE="${TRIAL_LOWRANK_TRAIN_BATCH_SIZE:-128}"
TRAIN_LR="${TRIAL_LOWRANK_TRAIN_LR:-0.001}"
TRAIN_WEIGHT_DECAY="${TRIAL_LOWRANK_TRAIN_WEIGHT_DECAY:-0.0001}"
TRAIN_PATIENCE="${TRIAL_LOWRANK_TRAIN_PATIENCE:-20}"

PHYSICAL_GPU="${TRIAL_LOWRANK_PHYSICAL_GPU:-7}"
GPU_IDLE_MAX_USED_MB="${TRIAL_LOWRANK_GPU_IDLE_MAX_USED_MB:-100}"
GPU_POLL_SECONDS="${TRIAL_LOWRANK_GPU_POLL_SECONDS:-60}"
CONDA_ENV="${CONDA_ENV:-torch}"
RUN_ID="${TRIAL_LOWRANK_RUN_ID:-exp028_trial_lowrank_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${TRIAL_LOWRANK_LOG_ROOT:-logs/exp028/${RUN_ID}}"
OUTPUT_DIR="${TRIAL_LOWRANK_OUTPUT_DIR:-trial_lowrank_analysis/outputs/${RUN_ID}}"
CHECKPOINT="checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_clean_eps0_42_fold0_best.pth"

START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

if ! [[ "${START_STAGE}" =~ ^[1-2]$ && "${STOP_STAGE}" =~ ^[1-2]$ ]]; then
  echo "START_STAGE and STOP_STAGE must be in [1, 2]." >&2
  exit 1
fi
if (( START_STAGE > STOP_STAGE )); then
  echo "START_STAGE cannot exceed STOP_STAGE." >&2
  exit 1
fi
if ! [[ "${PHYSICAL_GPU}" =~ ^[0-7]$ ]]; then
  echo "TRIAL_LOWRANK_PHYSICAL_GPU must be in [0, 7]." >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}" "${OUTPUT_DIR}" checkpoints log_train_AT
printf '%s\n' "$$" > "${LOG_ROOT}/controller.pid"
cat > "${LOG_ROOT}/run_config.txt" <<EOF
EXPERIMENT_ID=EXP-028
RUN_ID=${RUN_ID}
DATASET=${DATASET}
MODEL=${MODEL}
SEED=${SEED}
FOLD=${FOLD}
EPS=${EPS}
SAMPLE_NUM=${SAMPLE_NUM}
PGD_STEPS=${PGD_STEPS}
PGD_ALPHA=${PGD_ALPHA}
ATTACK_BATCH_SIZE=${ATTACK_BATCH_SIZE}
MIN_TRIALS=${MIN_TRIALS}
VIEWS=${VIEWS}
FREQUENCY_REPRESENTATION=${FREQUENCY_REPRESENTATION}
CHANNEL_VIEW_SPACE=${CHANNEL_VIEW_SPACE}
PHYSICAL_GPU=${PHYSICAL_GPU}
CHECKPOINT=${CHECKPOINT}
OUTPUT_DIR=${OUTPUT_DIR}
START_STAGE=${START_STAGE}
STOP_STAGE=${STOP_STAGE}
SKIP_EXISTING=${SKIP_EXISTING}
DRY_RUN=${DRY_RUN}
EOF

should_run() {
  local stage="$1"
  (( stage >= START_STAGE && stage <= STOP_STAGE ))
}

gpu_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    -i "${PHYSICAL_GPU}" 2>/dev/null \
    | awk 'NR == 1 {gsub(/[^0-9]/, "", $1); print $1}'
}

wait_for_gpu() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  while true; do
    local used
    used="$(gpu_used_mb)"
    if [[ -n "${used}" ]] && (( used <= GPU_IDLE_MAX_USED_MB )); then
      echo "[$(date -Is)] Physical GPU ${PHYSICAL_GPU} is idle (${used} MiB)."
      return
    fi
    echo "[$(date -Is)] Waiting for physical GPU ${PHYSICAL_GPU}: used=${used:-unknown} MiB, threshold=${GPU_IDLE_MAX_USED_MB} MiB."
    sleep "${GPU_POLL_SECONDS}"
  done
}

run_clean_training() {
  if [[ "${SKIP_EXISTING}" == "1" && -f "${CHECKPOINT}" ]]; then
    echo "[$(date -Is)] Reuse clean checkpoint: ${CHECKPOINT}"
    return
  fi
  wait_for_gpu
  echo "[$(date -Is)] Start clean EEGNet training on physical GPU ${PHYSICAL_GPU}."
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN clean training checkpoint=%s gpu=%s\n' "${CHECKPOINT}" "${PHYSICAL_GPU}"
    return
  fi
  CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" conda run -n "${CONDA_ENV}" --no-capture-output \
    python -u train_AT.py \
    --dataset "${DATASET}" --model "${MODEL}" --at_strategy clean \
    --fold "${FOLD}" --epsilon 0 --epochs "${TRAIN_EPOCHS}" \
    --batch_size "${TRAIN_BATCH_SIZE}" --lr "${TRAIN_LR}" \
    --weight_decay "${TRAIN_WEIGHT_DECAY}" --patience "${TRAIN_PATIENCE}" \
    --seed "${SEED}" --gpu_id 0 --no_ea \
    >> "${LOG_ROOT}/stage1_train_clean.log" 2>&1
  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Expected clean checkpoint was not produced: ${CHECKPOINT}" >&2
    exit 1
  fi
  echo "[$(date -Is)] Clean checkpoint ready: ${CHECKPOINT}"
}

run_lowrank_analysis() {
  if [[ "${SKIP_EXISTING}" == "1" && -f "${OUTPUT_DIR}/hosvd_summary.csv" ]]; then
    echo "[$(date -Is)] Reuse low-rank summary: ${OUTPUT_DIR}/hosvd_summary.csv"
    return
  fi
  if [[ "${DRY_RUN}" != "1" && ! -f "${CHECKPOINT}" ]]; then
    echo "Required clean checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
  fi
  wait_for_gpu
  # shellcheck disable=SC2206
  local views=( ${VIEWS} )
  echo "[$(date -Is)] Start trial low-rank analysis on physical GPU ${PHYSICAL_GPU}."
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN lowrank output=%s gpu=%s views=%s\n' \
      "${OUTPUT_DIR}" "${PHYSICAL_GPU}" "${views[*]}"
    return
  fi
  CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" conda run -n "${CONDA_ENV}" --no-capture-output \
    python -u trial_lowrank_analysis/analyze_trial_hosvd_lowrank.py \
    --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" --seed "${SEED}" \
    --eps "${EPS}" --pgd_steps "${PGD_STEPS}" --pgd_alpha "${PGD_ALPHA}" \
    --batch_size "${ATTACK_BATCH_SIZE}" --gpu_id 0 --min_trials "${MIN_TRIALS}" \
    --sample_num "${SAMPLE_NUM}" --views "${views[@]}" \
    --frequency_representation "${FREQUENCY_REPRESENTATION}" \
    --channel_view_space "${CHANNEL_VIEW_SPACE}" \
    --output_dir "${OUTPUT_DIR}" --no_ea \
    >> "${LOG_ROOT}/stage2_lowrank_analysis.log" 2>&1
}

if should_run 1; then
  run_clean_training
fi
if should_run 2; then
  run_lowrank_analysis
fi

echo "[$(date -Is)] EXP-028 pipeline finished: ${RUN_ID}"
