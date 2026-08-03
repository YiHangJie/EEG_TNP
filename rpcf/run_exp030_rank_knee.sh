#!/usr/bin/env bash
set -euo pipefail

# EXP-030：复用 EXP-027 六秩 EEG_TNP payload，执行无标签 log-MSE knee 自动选秩。
#
# 正式运行：
#   RUN_ID=exp030_auto_rank_log_mse_knee_seed42_$(date +%Y%m%d_%H%M%S); \
#   mkdir -p "logs/exp030/${RUN_ID}"; \
#   nohup setsid env EXP030_KNEE_RUN_ID="${RUN_ID}" \
#     bash rpcf/run_exp030_rank_knee.sh \
#     > "logs/exp030/${RUN_ID}/controller.log" 2>&1 < /dev/null &
#
# 检查命令与阶段续跑：
#   DRY_RUN=1 bash rpcf/run_exp030_rank_knee.sh
#   START_STAGE=2 STOP_STAGE=2 EXP030_KNEE_RUN_ID=<run_id> \
#     bash rpcf/run_exp030_rank_knee.sh

CONDA_ENV="${CONDA_ENV:-torch}"
SEED="${EXP030_SEED:-42}"
RUN_ID="${EXP030_KNEE_RUN_ID:-exp030_auto_rank_log_mse_knee_seed42_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${EXP030_KNEE_LOG_ROOT:-logs/exp030/${RUN_ID}}"
OUTPUT_ROOT="${EXP030_KNEE_OUTPUT_ROOT:-purified_data/exp030}"
ANALYSIS_ROOT="${LOG_ROOT}/analysis"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-2}"
BOOTSTRAP_SAMPLES="${EXP030_BOOTSTRAP_SAMPLES:-10000}"
GPU_IDS_RAW="${EXP030_GPU_IDS:-0,1,2,3,4,5,6,7}"
GPU_IDLE_MAX_USED_MB="${EXP030_GPU_IDLE_MAX_USED_MB:-100}"
GPU_POLL_SECONDS="${EXP030_GPU_POLL_SECONDS:-60}"

MADRY_OUTPUT="${OUTPUT_ROOT}/${RUN_ID}_madry_at_auto_rank_log_mse_knee_n512.pth"
RPCF_OUTPUT="${OUTPUT_ROOT}/${RUN_ID}_rpcf_at_auto_rank_log_mse_knee_n512.pth"
MADRY_FIXED25="purified_data/exp027/eval/exp027_oracle_rank_seed42_20260716_1305_madry_at_rank25.pth"
RPCF_FIXED25="logs/exp025/exp025_layer_prefix_seed42_20260706_1225_eegnet/eegnet/budget_2/rpcf_at_rank25.pth"
ORACLE_ROOT="logs/exp027/exp027_oracle_rank_seed42_20260716_1305/oracle"

MADRY_PAYLOADS=()
for rank in 15 20 25 30 35 40; do
  MADRY_PAYLOADS+=("purified_data/exp027/eval/exp027_oracle_rank_seed42_20260716_1305_madry_at_rank${rank}.pth")
done
RPCF_PAYLOADS=(
  "purified_data/exp027/eval/exp027_oracle_rank_seed42_20260716_1305_rpcf_at_rank15.pth"
  "purified_data/exp027/eval/exp027_oracle_rank_seed42_20260716_1305_rpcf_at_rank20.pth"
  "${RPCF_FIXED25}"
  "logs/exp025/exp025_layer_prefix_seed42_20260706_1225_eegnet/eegnet/budget_2/rpcf_at_rank30.pth"
  "purified_data/exp027/eval/exp027_oracle_rank_seed42_20260716_1305_rpcf_at_rank35.pth"
  "purified_data/exp027/eval/exp027_oracle_rank_seed42_20260716_1305_rpcf_at_rank40.pth"
)

if ! [[ "${START_STAGE}" =~ ^[1-2]$ && "${STOP_STAGE}" =~ ^[1-2]$ ]]; then
  echo "START_STAGE and STOP_STAGE must be in [1, 2]." >&2
  exit 1
fi
if (( START_STAGE > STOP_STAGE )); then
  echo "START_STAGE cannot exceed STOP_STAGE." >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}" "${ANALYSIS_ROOT}"
printf '%s\n' "$$" > "${LOG_ROOT}/controller.pid"
cat > "${LOG_ROOT}/run_config.txt" <<EOF
EXPERIMENT_ID=EXP-030
RUN_ID=${RUN_ID}
VARIANT=auto_rank_log_mse_knee
SEED=${SEED}
SAMPLE_NUM=512
CANDIDATE_RANKS=15,20,25,30,35,40
SELECTION=maximum_chord_distance_on_log_mse_curve
RANK_INFERENCE_USES_LABELS=0
RANK_INFERENCE_USES_CLASSIFIER_LOGITS=0
MADRY_OUTPUT=${MADRY_OUTPUT}
RPCF_OUTPUT=${RPCF_OUTPUT}
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

build_method() {
  local method="$1"
  local output="$2"
  shift 2
  local paths=("$@")
  if [[ "${SKIP_EXISTING}" == "1" && -f "${output}" ]]; then
    echo "[$(date -Is)] Reuse ${method} knee payload: ${output}"
    return
  fi
  local path
  for path in "${paths[@]}"; do
    require_artifact "${path}"
  done
  echo "[$(date -Is)] Build ${method} log-MSE knee payload."
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN build method=%s output=%s ranks=15,20,25,30,35,40\n' \
      "${method}" "${output}"
    return
  fi
  local overwrite_args=()
  if [[ "${SKIP_EXISTING}" != "1" ]]; then
    overwrite_args=(--overwrite)
  fi
  conda run -n "${CONDA_ENV}" --no-capture-output python -u \
    -m rpcf.build_exp030_rank_knee_payload \
    --method "${method}" --payload_paths "${paths[@]}" \
    --expected_ranks 15,20,25,30,35,40 \
    --output_path "${output}" "${overwrite_args[@]}" \
    > "${LOG_ROOT}/stage1_${method}.log" 2>&1
  echo "[$(date -Is)] Finished ${method} knee payload."
}

split_words() {
  local value="$1"
  value="${value//,/ }"
  # shellcheck disable=SC2206
  SPLIT_WORDS=(${value})
}
split_words "${GPU_IDS_RAW}"
GPU_IDS=("${SPLIT_WORDS[@]}")

gpu_used_mb() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null \
    | awk 'NR == 1 {gsub(/[^0-9]/, "", $1); print $1}'
}

select_idle_gpu() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    SELECTED_GPU="${GPU_IDS[0]}"
    return
  fi
  while true; do
    local gpu used
    for gpu in "${GPU_IDS[@]}"; do
      used="$(gpu_used_mb "${gpu}")"
      if [[ -n "${used}" ]] && (( used <= GPU_IDLE_MAX_USED_MB )); then
        SELECTED_GPU="${gpu}"
        echo "[$(date -Is)] Selected idle physical GPU ${SELECTED_GPU}."
        return
      fi
    done
    echo "[$(date -Is)] No idle GPU found; waiting ${GPU_POLL_SECONDS}s."
    sleep "${GPU_POLL_SECONDS}"
  done
}

run_analysis() {
  if [[ "${SKIP_EXISTING}" == "1" && -f "${ANALYSIS_ROOT}/summary.json" ]]; then
    echo "[$(date -Is)] Reuse knee analysis: ${ANALYSIS_ROOT}/summary.json"
    return
  fi
  for path in \
    "${MADRY_OUTPUT}" "${RPCF_OUTPUT}" "${MADRY_FIXED25}" "${RPCF_FIXED25}" \
    "${ORACLE_ROOT}/summary.json" "${ORACLE_ROOT}/oracle_selected_rows.csv"; do
    require_artifact "${path}"
  done
  select_idle_gpu
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN analyze gpu=${SELECTED_GPU} output=${ANALYSIS_ROOT}"
    return
  fi
  local overwrite_args=()
  if [[ "${SKIP_EXISTING}" != "1" ]]; then
    overwrite_args=(--overwrite)
  fi
  CUDA_VISIBLE_DEVICES="${SELECTED_GPU}" conda run -n "${CONDA_ENV}" \
    --no-capture-output python -u -m rpcf.analyze_exp030_auto_rank \
    --method madry_at "${MADRY_OUTPUT}" "${MADRY_FIXED25}" \
    --method rpcf_at "${RPCF_OUTPUT}" "${RPCF_FIXED25}" \
    --oracle_summary "${ORACLE_ROOT}/summary.json" \
    --oracle_rows "${ORACLE_ROOT}/oracle_selected_rows.csv" \
    --batch_size 64 --gpu_id 0 --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
    --seed "${SEED}" --output_dir "${ANALYSIS_ROOT}" "${overwrite_args[@]}" \
    > "${LOG_ROOT}/stage2_analysis.log" 2>&1
}

if should_run 1; then
  build_method madry_at "${MADRY_OUTPUT}" "${MADRY_PAYLOADS[@]}"
  build_method rpcf_at "${RPCF_OUTPUT}" "${RPCF_PAYLOADS[@]}"
fi
if should_run 2; then
  run_analysis
fi

echo "[$(date -Is)] EXP-030 log-MSE knee pipeline finished: ${RUN_ID}"
