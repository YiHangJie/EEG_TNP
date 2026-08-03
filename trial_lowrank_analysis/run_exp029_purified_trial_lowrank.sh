#!/usr/bin/env bash
set -euo pipefail

# EXP-029：对 EXP-028 同批 clean/PGD-200 trial 的六-rank净化结果执行 HOSVD。
#
# 正式运行（每张 GPU 同时两个 EEG_TNP 进程）：
#   RUN_ID=exp029_purified_trial_lowrank_seed42_YYYYMMDD_HHMMSS
#   mkdir -p "logs/exp029/${RUN_ID}"
#   nohup setsid env EXP029_RUN_ID="${RUN_ID}" \
#     bash trial_lowrank_analysis/run_exp029_purified_trial_lowrank.sh \
#     > "logs/exp029/${RUN_ID}/controller.log" 2>&1 < /dev/null &
#
# Smoke / dry-run：
#   SMOKE=1 bash trial_lowrank_analysis/run_exp029_purified_trial_lowrank.sh
#   DRY_RUN=1 bash trial_lowrank_analysis/run_exp029_purified_trial_lowrank.sh
#
# 续跑：
#   EXP029_RUN_ID=<same_run_id> START_STAGE=2 STOP_STAGE=3 \
#     bash trial_lowrank_analysis/run_exp029_purified_trial_lowrank.sh

DATASET="${EXP029_DATASET:-thubenchmark}"
MODEL="${EXP029_MODEL:-eegnet}"
SEED="${EXP029_SEED:-42}"
FOLD="${EXP029_FOLD:-0}"
EPS="${EXP029_EPS:-0.03}"
CONDA_ENV="${CONDA_ENV:-torch}"
RUN_ID="${EXP029_RUN_ID:-exp029_purified_trial_lowrank_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${EXP029_LOG_ROOT:-logs/exp029/${RUN_ID}}"
ATTACK_ROOT="${EXP029_ATTACK_ROOT:-ad_data/exp029}"
PURIFICATION_ROOT="${EXP029_PURIFICATION_ROOT:-purified_data/exp029/eval}"
OUTPUT_DIR="${EXP029_OUTPUT_DIR:-trial_lowrank_analysis/outputs/${RUN_ID}}"

SOURCE_RUN_ID="${EXP029_SOURCE_RUN_ID:-exp028_trial_lowrank_seed42_20260716_1344}"
SOURCE_BUNDLE="${EXP029_SOURCE_BUNDLE:-trial_lowrank_analysis/outputs/${SOURCE_RUN_ID}/bundle.pt}"
SOURCE_SUMMARY="${EXP029_SOURCE_SUMMARY:-trial_lowrank_analysis/outputs/${SOURCE_RUN_ID}/hosvd_summary.csv}"
CHECKPOINT="${EXP029_CHECKPOINT:-checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_clean_eps0_42_fold0_best.pth}"
ATTACK_PAYLOAD="${ATTACK_ROOT}/${RUN_ID}_exp028_pgd200.pth"

GPU_IDS_RAW="${EXP029_GPU_IDS:-0,1,2,3,4,5,6,7}"
GPU_IDLE_MAX_USED_MB="${EXP029_GPU_IDLE_MAX_USED_MB:-100}"
GPU_POLL_SECONDS="${EXP029_GPU_POLL_SECONDS:-60}"
PROCESSES_PER_GPU=2
CHECKPOINT_EVERY="${EXP029_CHECKPOINT_EVERY:-8}"
EVAL_BATCH_SIZE="${EXP029_EVAL_BATCH_SIZE:-64}"
MIN_TRIALS="${EXP029_MIN_TRIALS:-3}"
VIEWS="${EXP029_VIEWS:-trial time channel frequency}"
FREQUENCY_REPRESENTATION="${EXP029_FREQUENCY_REPRESENTATION:-complex}"
CHANNEL_VIEW_SPACE="${EXP029_CHANNEL_VIEW_SPACE:-interpolated_grid}"

START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-3}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"

if [[ "${SMOKE}" == "1" ]]; then
  SAMPLE_NUM="${SMOKE_SAMPLE_NUM:-2}"
  RANKS=(15 20)
  MIN_TRIALS="${SMOKE_MIN_TRIALS:-1}"
else
  SAMPLE_NUM="${EXP029_SAMPLE_NUM:-512}"
  RANKS=(15 20 25 30 35 40)
fi

if ! [[ "${START_STAGE}" =~ ^[1-3]$ && "${STOP_STAGE}" =~ ^[1-3]$ ]]; then
  echo "START_STAGE and STOP_STAGE must be in [1, 3]." >&2
  exit 1
fi
if (( START_STAGE > STOP_STAGE )); then
  echo "START_STAGE cannot exceed STOP_STAGE." >&2
  exit 1
fi
if ! [[ "${GPU_IDLE_MAX_USED_MB}" =~ ^[0-9]+$ ]]; then
  echo "EXP029_GPU_IDLE_MAX_USED_MB must be a non-negative integer." >&2
  exit 1
fi

split_words() {
  local value="$1"
  value="${value//,/ }"
  # shellcheck disable=SC2206
  SPLIT_WORDS=(${value})
}

split_words "${GPU_IDS_RAW}"
GPU_IDS=("${SPLIT_WORDS[@]}")
for gpu in "${GPU_IDS[@]}"; do
  if ! [[ "${gpu}" =~ ^[0-7]$ ]]; then
    echo "Invalid physical GPU id: ${gpu}" >&2
    exit 1
  fi
done

mkdir -p "${LOG_ROOT}/purification" "${ATTACK_ROOT}" \
  "${PURIFICATION_ROOT}" "${OUTPUT_DIR}"
printf '%s\n' "$$" > "${LOG_ROOT}/controller.pid"

cat > "${LOG_ROOT}/run_config.txt" <<EOF
EXPERIMENT_ID=EXP-029
RUN_ID=${RUN_ID}
DATASET=${DATASET}
MODEL=${MODEL}
SEED=${SEED}
FOLD=${FOLD}
EPS=${EPS}
SAMPLE_NUM=${SAMPLE_NUM}
RANKS=${RANKS[*]}
SOURCE_RUN_ID=${SOURCE_RUN_ID}
SOURCE_BUNDLE=${SOURCE_BUNDLE}
SOURCE_SUMMARY=${SOURCE_SUMMARY}
CHECKPOINT=${CHECKPOINT}
ATTACK_PAYLOAD=${ATTACK_PAYLOAD}
OUTPUT_DIR=${OUTPUT_DIR}
GPU_IDS=${GPU_IDS[*]}
GPU_IDLE_MAX_USED_MB=${GPU_IDLE_MAX_USED_MB}
PROCESSES_PER_GPU=${PROCESSES_PER_GPU}
MIN_TRIALS=${MIN_TRIALS}
VIEWS=${VIEWS}
FREQUENCY_REPRESENTATION=${FREQUENCY_REPRESENTATION}
CHANNEL_VIEW_SPACE=${CHANNEL_VIEW_SPACE}
SMOKE=${SMOKE}
DRY_RUN=${DRY_RUN}
SKIP_EXISTING=${SKIP_EXISTING}
EOF

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

gpu_used_mb() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    -i "${gpu}" 2>/dev/null \
    | awk 'NR == 1 {gsub(/[^0-9]/, "", $1); print $1}'
}

select_idle_gpus() {
  local required="$1"
  SELECTED_GPUS=()
  if [[ "${DRY_RUN}" == "1" ]]; then
    SELECTED_GPUS=("${GPU_IDS[@]:0:required}")
    return
  fi
  while true; do
    SELECTED_GPUS=()
    local gpu used
    for gpu in "${GPU_IDS[@]}"; do
      used="$(gpu_used_mb "${gpu}")"
      if [[ -n "${used}" ]] && (( used <= GPU_IDLE_MAX_USED_MB )); then
        SELECTED_GPUS+=("${gpu}")
      fi
      if (( ${#SELECTED_GPUS[@]} == required )); then
        echo "[$(date -Is)] Selected idle physical GPUs: ${SELECTED_GPUS[*]}"
        return
      fi
    done
    echo "[$(date -Is)] Need ${required} idle GPUs for paired launch; found ${#SELECTED_GPUS[@]}. Waiting ${GPU_POLL_SECONDS}s..."
    sleep "${GPU_POLL_SECONDS}"
  done
}

rank_config() {
  local rank="$1"
  printf '%s\n' "PTR3d_8_2048_rank${rank}_3d_interpolate.yaml"
}

purification_output() {
  local rank="$1"
  printf '%s\n' "${PURIFICATION_ROOT}/${RUN_ID}_rank${rank}.pth"
}

prepare_attack_payload() {
  if [[ "${SKIP_EXISTING}" == "1" && -f "${ATTACK_PAYLOAD}" ]]; then
    echo "[$(date -Is)] Reuse EXP-028 purification input: ${ATTACK_PAYLOAD}"
    return
  fi
  require_artifact "${SOURCE_BUNDLE}"
  echo "[$(date -Is)] Prepare purification input from EXP-028 bundle."
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN prepare source=%s output=%s\n' \
      "${SOURCE_BUNDLE}" "${ATTACK_PAYLOAD}"
    return
  fi
  conda run -n "${CONDA_ENV}" --no-capture-output \
    python -u -m trial_lowrank_analysis.prepare_purification_input \
    --bundle_path "${SOURCE_BUNDLE}" --output_path "${ATTACK_PAYLOAD}" \
    --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
    --seed "${SEED}" --eps "${EPS}" \
    >> "${LOG_ROOT}/stage1_prepare_input.log" 2>&1
}

launch_purification_task() {
  local rank="$1"
  local gpu="$2"
  local output_path
  output_path="$(purification_output "${rank}")"
  echo "[$(date -Is)] Start rank${rank} on physical GPU ${gpu}"
  (
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf 'DRY_RUN purification rank=%s gpu=%s output=%s\n' \
        "${rank}" "${gpu}" "${output_path}"
    else
      CUDA_VISIBLE_DEVICES="${gpu}" conda run -n "${CONDA_ENV}" --no-capture-output \
        python -u -m rpcf.evaluate_purification \
        --attack_path "${ATTACK_PAYLOAD}" --checkpoint_path "${CHECKPOINT}" \
        --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
        --seed "${SEED}" --eps "${EPS}" --sample_num "${SAMPLE_NUM}" \
        --ranks "${rank}" --configs "$(rank_config "${rank}")" \
        --batch_size "${EVAL_BATCH_SIZE}" --checkpoint_every "${CHECKPOINT_EVERY}" \
        --gpu_id 0 --output_path "${output_path}"
    fi
  ) > "${LOG_ROOT}/purification/rank${rank}.log" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LABELS+=("rank${rank}")
}

run_purification_wave() {
  require_artifact "${ATTACK_PAYLOAD}"
  require_artifact "${CHECKPOINT}"
  TASK_RANKS=()
  local rank output_path
  for rank in "${RANKS[@]}"; do
    output_path="$(purification_output "${rank}")"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${output_path}" ]]; then
      echo "[$(date -Is)] Reuse rank${rank}: ${output_path}"
    else
      TASK_RANKS+=("${rank}")
    fi
  done
  if (( ${#TASK_RANKS[@]} == 0 )); then
    echo "[$(date -Is)] All purification ranks already completed."
    return
  fi

  local required_gpus=$(( (${#TASK_RANKS[@]} + PROCESSES_PER_GPU - 1) / PROCESSES_PER_GPU ))
  if (( required_gpus > ${#GPU_IDS[@]} )); then
    echo "Need ${required_gpus} GPUs; only ${#GPU_IDS[@]} candidates." >&2
    exit 1
  fi
  select_idle_gpus "${required_gpus}"
  echo "[$(date -Is)] Launch ${#TASK_RANKS[@]} ranks as ${required_gpus} GPU pairs."

  ACTIVE_PIDS=()
  ACTIVE_LABELS=()
  local index gpu_index gpu
  for index in "${!TASK_RANKS[@]}"; do
    gpu_index=$((index / PROCESSES_PER_GPU))
    gpu="${SELECTED_GPUS[$gpu_index]}"
    launch_purification_task "${TASK_RANKS[$index]}" "${gpu}"
  done

  local failed=()
  local status
  for index in "${!ACTIVE_PIDS[@]}"; do
    status=0
    wait "${ACTIVE_PIDS[$index]}" || status=$?
    echo "[$(date -Is)] Finished ${ACTIVE_LABELS[$index]}: status=${status}"
    if [[ "${status}" -ne 0 ]]; then
      failed+=("${ACTIVE_LABELS[$index]}:${status}")
    fi
  done
  if (( ${#failed[@]} > 0 )); then
    echo "EXP-029 purification failures: ${failed[*]}" >&2
    exit 1
  fi
}

run_purified_analysis() {
  if [[ "${SKIP_EXISTING}" == "1" && -f "${OUTPUT_DIR}/hosvd_summary.csv" ]]; then
    echo "[$(date -Is)] Reuse purified HOSVD summary: ${OUTPUT_DIR}/hosvd_summary.csv"
    return
  fi
  local paths=()
  local rank path
  for rank in "${RANKS[@]}"; do
    path="$(purification_output "${rank}")"
    require_artifact "${path}"
    paths+=("${path}")
  done
  # shellcheck disable=SC2206
  local views=( ${VIEWS} )
  local expected_ranks
  expected_ranks="$(IFS=,; echo "${RANKS[*]}")"
  echo "[$(date -Is)] Start CPU HOSVD analysis for purified ranks: ${RANKS[*]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN analyze ranks=%s output=%s paths=%s\n' \
      "${expected_ranks}" "${OUTPUT_DIR}" "${paths[*]}"
    return
  fi

  local source_summary_args=()
  if [[ "${SMOKE}" != "1" ]]; then
    require_artifact "${SOURCE_SUMMARY}"
    source_summary_args=(--source_summary "${SOURCE_SUMMARY}")
  fi
  CUDA_VISIBLE_DEVICES="" conda run -n "${CONDA_ENV}" --no-capture-output \
    python -u -m trial_lowrank_analysis.analyze_purified_trial_hosvd_lowrank \
    --source_bundle "${SOURCE_BUNDLE}" "${source_summary_args[@]}" \
    --purification_paths "${paths[@]}" --expected_ranks "${expected_ranks}" \
    --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
    --seed "${SEED}" --eps "${EPS}" --min_trials "${MIN_TRIALS}" \
    --views "${views[@]}" \
    --frequency_representation "${FREQUENCY_REPRESENTATION}" \
    --channel_view_space "${CHANNEL_VIEW_SPACE}" \
    --output_dir "${OUTPUT_DIR}" \
    >> "${LOG_ROOT}/stage3_purified_lowrank_analysis.log" 2>&1
}

if should_run 1; then
  prepare_attack_payload
fi
if should_run 2; then
  run_purification_wave
fi
if should_run 3; then
  run_purified_analysis
fi

echo "[$(date -Is)] EXP-029 pipeline finished: ${RUN_ID}"
