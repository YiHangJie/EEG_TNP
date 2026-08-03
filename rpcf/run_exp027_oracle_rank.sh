#!/usr/bin/env bash
set -euo pipefail

# EXP-027：EEGNet 样本级动态 rank oracle headroom。
#
# 正式运行（净化阶段等待五张空闲卡，并在每张卡同时启动两个进程）：
#   EXP027_RUN_ID=exp027_oracle_rank_seed42_$(date +%Y%m%d_%H%M%S) \
#   nohup setsid bash rpcf/run_exp027_oracle_rank.sh \
#     > logs/exp027/exp027_oracle_rank_seed42_YYYYMMDD_HHMMSS/controller.log \
#     2>&1 < /dev/null &
#
# Smoke / dry-run：
#   SMOKE=1 bash rpcf/run_exp027_oracle_rank.sh
#   DRY_RUN=1 bash rpcf/run_exp027_oracle_rank.sh
#
# 续跑：
#   EXP027_RUN_ID=<same_run_id> START_STAGE=2 STOP_STAGE=3 \
#     bash rpcf/run_exp027_oracle_rank.sh

DATASET="${EXP027_DATASET:-thubenchmark}"
MODEL="eegnet"
SEED="${EXP027_SEED:-42}"
FOLD="${EXP027_FOLD:-0}"
EPS="${EXP027_EPS:-0.03}"
ATTACK="autoattack"
CONDA_ENV="${CONDA_ENV:-torch}"
RUN_ID="${EXP027_RUN_ID:-exp027_oracle_rank_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${EXP027_LOG_ROOT:-logs/exp027/${RUN_ID}}"
ATTACK_ROOT="${EXP027_ATTACK_ROOT:-ad_data/exp027}"
PURIFICATION_ROOT="${EXP027_PURIFICATION_ROOT:-purified_data/exp027/eval}"
ORACLE_ROOT="${LOG_ROOT}/oracle"

GPU_IDS_RAW="${EXP027_GPU_IDS:-0,1,2,3,4,5,6,7}"
GPU_IDLE_MAX_USED_MB="${EXP027_GPU_IDLE_MAX_USED_MB:-100}"
GPU_POLL_SECONDS="${EXP027_GPU_POLL_SECONDS:-60}"
FORMAL_REQUIRED_GPUS=5
PROCESSES_PER_GPU=2

START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-3}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CHECKPOINT_EVERY="${EXP027_CHECKPOINT_EVERY:-8}"
ATTACK_BATCH_SIZE="${EXP027_ATTACK_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EXP027_EVAL_BATCH_SIZE:-64}"
BOOTSTRAP_SAMPLES="${EXP027_BOOTSTRAP_SAMPLES:-10000}"

MADRY_CHECKPOINT="${EXP027_MADRY_CHECKPOINT:-checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_exp025_eegnet_prep_seed42_20260706_1225_at_best.pth}"
RPCF_CHECKPOINT="${EXP027_RPCF_CHECKPOINT:-checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_exp025_layer_prefix_seed42_20260706_1225_eegnet_eegnet_budget_2_rpcf_at_best.pth}"
RPCF_REUSE_ATTACK="${EXP027_RPCF_REUSE_ATTACK:-logs/exp025/exp025_layer_prefix_seed42_20260706_1225_eegnet/eegnet/budget_2/rpcf_at_autoattack.pth}"
RPCF_REUSE_RANK25="${EXP027_RPCF_REUSE_RANK25:-logs/exp025/exp025_layer_prefix_seed42_20260706_1225_eegnet/eegnet/budget_2/rpcf_at_rank25.pth}"
RPCF_REUSE_RANK30="${EXP027_RPCF_REUSE_RANK30:-logs/exp025/exp025_layer_prefix_seed42_20260706_1225_eegnet/eegnet/budget_2/rpcf_at_rank30.pth}"

if [[ "${SMOKE}" == "1" ]]; then
  SAMPLE_NUM="${SMOKE_SAMPLE_NUM:-2}"
  RANKS=(15 20)
  BOOTSTRAP_SAMPLES="${SMOKE_BOOTSTRAP_SAMPLES:-100}"
else
  SAMPLE_NUM="${EXP027_SAMPLE_NUM:-512}"
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
  echo "EXP027_GPU_IDLE_MAX_USED_MB must be a non-negative integer." >&2
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
if (( ${#GPU_IDS[@]} < FORMAL_REQUIRED_GPUS )) && [[ "${SMOKE}" != "1" ]]; then
  echo "Formal EXP-027 requires at least ${FORMAL_REQUIRED_GPUS} candidate GPUs." >&2
  exit 1
fi
for gpu in "${GPU_IDS[@]}"; do
  if ! [[ "${gpu}" =~ ^[0-7]$ ]]; then
    echo "Invalid physical GPU id: ${gpu}" >&2
    exit 1
  fi
done

mkdir -p "${LOG_ROOT}" "${LOG_ROOT}/purification" "${ATTACK_ROOT}" \
  "${PURIFICATION_ROOT}" "${ORACLE_ROOT}"
printf '%s\n' "$$" > "${LOG_ROOT}/controller.pid"

MADRY_ATTACK_PATH="${ATTACK_ROOT}/${RUN_ID}_madry_at_autoattack.pth"
if [[ "${SMOKE}" == "1" ]]; then
  RPCF_ATTACK_PATH="${ATTACK_ROOT}/${RUN_ID}_rpcf_at_autoattack.pth"
else
  RPCF_ATTACK_PATH="${RPCF_REUSE_ATTACK}"
fi

cat > "${LOG_ROOT}/run_config.txt" <<EOF
EXPERIMENT_ID=EXP-027
RUN_ID=${RUN_ID}
DATASET=${DATASET}
MODEL=${MODEL}
SEED=${SEED}
FOLD=${FOLD}
EPS=${EPS}
SAMPLE_NUM=${SAMPLE_NUM}
RANKS=${RANKS[*]}
GPU_IDS=${GPU_IDS[*]}
GPU_IDLE_MAX_USED_MB=${GPU_IDLE_MAX_USED_MB}
FORMAL_REQUIRED_GPUS=${FORMAL_REQUIRED_GPUS}
PROCESSES_PER_GPU=${PROCESSES_PER_GPU}
MADRY_CHECKPOINT=${MADRY_CHECKPOINT}
RPCF_CHECKPOINT=${RPCF_CHECKPOINT}
MADRY_ATTACK_PATH=${MADRY_ATTACK_PATH}
RPCF_ATTACK_PATH=${RPCF_ATTACK_PATH}
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
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null \
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

run_attack() {
  local method="$1"
  local checkpoint="$2"
  local output_path="$3"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${output_path}" ]]; then
    echo "[$(date -Is)] Reuse ${method} attack: ${output_path}"
    return
  fi
  require_artifact "${checkpoint}"
  select_idle_gpus 1
  local gpu="${SELECTED_GPUS[0]}"
  local subset_args=()
  if [[ "${SMOKE}" == "1" ]]; then
    subset_args=(--sample_num "${SAMPLE_NUM}")
  fi
  echo "[$(date -Is)] Start ${method} AutoAttack on physical GPU ${gpu}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN attack method=%s gpu=%s output=%s\n' "${method}" "${gpu}" "${output_path}"
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" conda run -n "${CONDA_ENV}" --no-capture-output \
    python -u -m rpcf.evaluate_attack \
    --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" --seed "${SEED}" \
    --checkpoint_path "${checkpoint}" --method_tag "exp027_${method}" \
    --attack "${ATTACK}" --eps "${EPS}" --batch_size "${ATTACK_BATCH_SIZE}" \
    --gpu_id 0 --output_path "${output_path}" "${subset_args[@]}" \
    >> "${LOG_ROOT}/stage1_attack_${method}.log" 2>&1
}

rank_config() {
  local rank="$1"
  printf '%s\n' "PTR3d_8_2048_rank${rank}_3d_interpolate.yaml"
}

purification_output() {
  local method="$1"
  local rank="$2"
  printf '%s\n' "${PURIFICATION_ROOT}/${RUN_ID}_${method}_rank${rank}.pth"
}

add_purification_task() {
  local method="$1"
  local rank="$2"
  local checkpoint="$3"
  local attack_path="$4"
  local output_path="$5"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${output_path}" ]]; then
    echo "[$(date -Is)] Reuse ${method} rank${rank}: ${output_path}"
    return
  fi
  TASK_METHODS+=("${method}")
  TASK_RANKS+=("${rank}")
  TASK_CHECKPOINTS+=("${checkpoint}")
  TASK_ATTACKS+=("${attack_path}")
  TASK_OUTPUTS+=("${output_path}")
}

launch_purification_task() {
  local task_index="$1"
  local gpu="$2"
  local method="${TASK_METHODS[$task_index]}"
  local rank="${TASK_RANKS[$task_index]}"
  local checkpoint="${TASK_CHECKPOINTS[$task_index]}"
  local attack_path="${TASK_ATTACKS[$task_index]}"
  local output_path="${TASK_OUTPUTS[$task_index]}"
  local label="${method}_rank${rank}"
  require_artifact "${checkpoint}"
  require_artifact "${attack_path}"
  echo "[$(date -Is)] Start ${label} on physical GPU ${gpu}"
  (
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf 'DRY_RUN purification label=%s gpu=%s output=%s\n' "${label}" "${gpu}" "${output_path}"
    else
      CUDA_VISIBLE_DEVICES="${gpu}" conda run -n "${CONDA_ENV}" --no-capture-output \
        python -u -m rpcf.evaluate_purification \
        --attack_path "${attack_path}" --checkpoint_path "${checkpoint}" \
        --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" --seed "${SEED}" \
        --eps "${EPS}" --sample_num "${SAMPLE_NUM}" --ranks "${rank}" \
        --configs "$(rank_config "${rank}")" --batch_size "${EVAL_BATCH_SIZE}" \
        --checkpoint_every "${CHECKPOINT_EVERY}" --gpu_id 0 --output_path "${output_path}"
    fi
  ) > "${LOG_ROOT}/purification/${label}.log" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LABELS+=("${label}")
}

run_purification_wave() {
  TASK_METHODS=()
  TASK_RANKS=()
  TASK_CHECKPOINTS=()
  TASK_ATTACKS=()
  TASK_OUTPUTS=()

  local rank
  for rank in "${RANKS[@]}"; do
    add_purification_task \
      madry_at "${rank}" "${MADRY_CHECKPOINT}" "${MADRY_ATTACK_PATH}" \
      "$(purification_output madry_at "${rank}")"
  done
  if [[ "${SMOKE}" == "1" ]]; then
    for rank in "${RANKS[@]}"; do
      add_purification_task \
        rpcf_at "${rank}" "${RPCF_CHECKPOINT}" "${RPCF_ATTACK_PATH}" \
        "$(purification_output rpcf_at "${rank}")"
    done
  else
    for rank in 15 20 35 40; do
      add_purification_task \
        rpcf_at "${rank}" "${RPCF_CHECKPOINT}" "${RPCF_ATTACK_PATH}" \
        "$(purification_output rpcf_at "${rank}")"
    done
  fi

  local task_count="${#TASK_METHODS[@]}"
  if (( task_count == 0 )); then
    echo "[$(date -Is)] All purification tasks already completed."
    return
  fi
  local required_gpus=$(( (task_count + PROCESSES_PER_GPU - 1) / PROCESSES_PER_GPU ))
  if [[ "${SMOKE}" != "1" && "${SKIP_EXISTING}" != "1" && "${task_count}" -ne 10 ]]; then
    echo "Fresh formal run must contain exactly 10 purification tasks, got ${task_count}." >&2
    exit 1
  fi
  if (( required_gpus > ${#GPU_IDS[@]} )); then
    echo "Need ${required_gpus} GPUs for ${task_count} paired tasks; only ${#GPU_IDS[@]} candidates." >&2
    exit 1
  fi
  select_idle_gpus "${required_gpus}"
  echo "[$(date -Is)] Launch ${task_count} purification tasks as ${required_gpus} GPU pairs."

  ACTIVE_PIDS=()
  ACTIVE_LABELS=()
  local task_index gpu_index gpu
  for task_index in "${!TASK_METHODS[@]}"; do
    gpu_index=$((task_index / PROCESSES_PER_GPU))
    gpu="${SELECTED_GPUS[$gpu_index]}"
    launch_purification_task "${task_index}" "${gpu}"
  done

  local failed=()
  local status index
  for index in "${!ACTIVE_PIDS[@]}"; do
    status=0
    wait "${ACTIVE_PIDS[$index]}" || status=$?
    echo "[$(date -Is)] Finished ${ACTIVE_LABELS[$index]}: status=${status}"
    if [[ "${status}" -ne 0 ]]; then
      failed+=("${ACTIVE_LABELS[$index]}:${status}")
    fi
  done
  if (( ${#failed[@]} > 0 )); then
    echo "EXP-027 purification failures: ${failed[*]}" >&2
    exit 1
  fi
}

collect_method_paths() {
  local method="$1"
  METHOD_PATHS=()
  local rank
  for rank in "${RANKS[@]}"; do
    if [[ "${method}" == "rpcf_at" && "${SMOKE}" != "1" && "${rank}" == "25" ]]; then
      METHOD_PATHS+=("${RPCF_REUSE_RANK25}")
    elif [[ "${method}" == "rpcf_at" && "${SMOKE}" != "1" && "${rank}" == "30" ]]; then
      METHOD_PATHS+=("${RPCF_REUSE_RANK30}")
    else
      METHOD_PATHS+=("$(purification_output "${method}" "${rank}")")
    fi
  done
}

run_oracle_analysis() {
  if [[ "${SKIP_EXISTING}" == "1" && -f "${ORACLE_ROOT}/summary.json" ]]; then
    echo "[$(date -Is)] Reuse oracle summary: ${ORACLE_ROOT}/summary.json"
    return
  fi
  collect_method_paths madry_at
  local madry_paths=("${METHOD_PATHS[@]}")
  collect_method_paths rpcf_at
  local rpcf_paths=("${METHOD_PATHS[@]}")
  local path
  for path in "${madry_paths[@]}" "${rpcf_paths[@]}"; do
    require_artifact "${path}"
  done
  select_idle_gpus 1
  local gpu="${SELECTED_GPUS[0]}"
  local expected_ranks
  expected_ranks="$(IFS=,; echo "${RANKS[*]}")"
  echo "[$(date -Is)] Start oracle analysis on physical GPU ${gpu}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN oracle gpu=%s ranks=%s madry=%s rpcf=%s\n' \
      "${gpu}" "${expected_ranks}" "${madry_paths[*]}" "${rpcf_paths[*]}"
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" conda run -n "${CONDA_ENV}" --no-capture-output \
    python -u -m rpcf.analyze_exp027_oracle_rank \
    --method_payload madry_at "${madry_paths[@]}" \
    --method_payload rpcf_at "${rpcf_paths[@]}" \
    --expected_ranks "${expected_ranks}" --batch_size "${EVAL_BATCH_SIZE}" \
    --gpu_id 0 --bootstrap_samples "${BOOTSTRAP_SAMPLES}" --seed "${SEED}" \
    --output_dir "${ORACLE_ROOT}" \
    >> "${LOG_ROOT}/stage3_oracle.log" 2>&1
}

if should_run 1; then
  run_attack madry_at "${MADRY_CHECKPOINT}" "${MADRY_ATTACK_PATH}"
  if [[ "${SMOKE}" == "1" ]]; then
    run_attack rpcf_at "${RPCF_CHECKPOINT}" "${RPCF_ATTACK_PATH}"
  else
    require_artifact "${RPCF_ATTACK_PATH}"
    echo "[$(date -Is)] Reuse RPCF_AT attack: ${RPCF_ATTACK_PATH}"
  fi
fi

if should_run 2; then
  require_artifact "${MADRY_ATTACK_PATH}"
  require_artifact "${RPCF_ATTACK_PATH}"
  run_purification_wave
fi

if should_run 3; then
  run_oracle_analysis
fi

echo "[$(date -Is)] EXP-027 pipeline finished: ${RUN_ID}"
