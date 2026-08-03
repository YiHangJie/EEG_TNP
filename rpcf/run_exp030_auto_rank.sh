#!/usr/bin/env bash
set -euo pipefail

# EXP-030：EEG_TNP 内生自动定秩防御。
#
# n64 pilot（推荐首次启动；一张物理 GPU 同时运行 Madry/RPCF_AT 两个进程）：
#   EXP030_RUN_ID=exp030_auto_rank_pilot_seed42_$(date +%Y%m%d_%H%M%S) \
#     nohup setsid env PILOT=1 bash rpcf/run_exp030_auto_rank.sh \
#     > logs/exp030/exp030_auto_rank_pilot_seed42_YYYYMMDD_HHMMSS/controller.log \
#     2>&1 < /dev/null &
#
# smoke / dry-run：
#   SMOKE=1 bash rpcf/run_exp030_auto_rank.sh
#   DRY_RUN=1 PILOT=1 bash rpcf/run_exp030_auto_rank.sh
#
# n512 formal 必须显式关闭 pilot，并提供通过门槛的 pilot summary：
#   PILOT=0 EXP030_PILOT_SUMMARY=logs/exp030/<pilot>/analysis/summary.json \
#     bash rpcf/run_exp030_auto_rank.sh

DATASET="${EXP030_DATASET:-thubenchmark}"
MODEL="eegnet"
SEED="${EXP030_SEED:-42}"
FOLD="${EXP030_FOLD:-0}"
EPS="${EXP030_EPS:-0.03}"
CONDA_ENV="${CONDA_ENV:-torch}"
VARIANT="${EXP030_VARIANT:-auto_rank_ard_regrow}"
SMOKE="${SMOKE:-0}"
PILOT="${PILOT:-1}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-2}"
CHECKPOINT_EVERY="${EXP030_CHECKPOINT_EVERY:-4}"
EVAL_BATCH_SIZE="${EXP030_EVAL_BATCH_SIZE:-64}"
BOOTSTRAP_SAMPLES="${EXP030_BOOTSTRAP_SAMPLES:-10000}"

GPU_IDS_RAW="${EXP030_GPU_IDS:-0,1,2,3,4,5,6,7}"
GPU_IDLE_MAX_USED_MB="${EXP030_GPU_IDLE_MAX_USED_MB:-100}"
GPU_POLL_SECONDS="${EXP030_GPU_POLL_SECONDS:-60}"
PROCESSES_PER_GPU=2

if [[ "${SMOKE}" == "1" ]]; then
  MODE="smoke"
  SAMPLE_NUM="${SMOKE_SAMPLE_NUM:-2}"
  case "${VARIANT}" in
    auto_rank_masked_cv)
      CONFIG="PTR3d_rank_cv_smoke_r20_3d_interpolate.yaml"
      ;;
    auto_rank_spectral_round)
      CONFIG="PTR3d_rank_spectral_smoke_r20_3d_interpolate.yaml"
      ;;
    auto_rank_independent_cv)
      CONFIG="PTR3d_rank_sweep_cv_smoke_r20_3d_interpolate.yaml"
      ;;
    *)
      CONFIG="PTR3d_rank_ard_regrow_smoke_r40_3d_interpolate.yaml"
      ;;
  esac
  BOOTSTRAP_SAMPLES="${SMOKE_BOOTSTRAP_SAMPLES:-100}"
elif [[ "${PILOT}" == "1" ]]; then
  MODE="pilot"
  SAMPLE_NUM="${EXP030_PILOT_SAMPLE_NUM:-64}"
else
  MODE="formal"
  SAMPLE_NUM="${EXP030_SAMPLE_NUM:-512}"
fi

if [[ "${SMOKE}" != "1" ]]; then
  case "${VARIANT}" in
    auto_rank_ard)
      CONFIG="PTR3d_rank_ard_8_2048_r40_3d_interpolate.yaml"
      ;;
    auto_rank_ard_regrow)
      CONFIG="PTR3d_rank_ard_regrow_8_2048_r40_3d_interpolate.yaml"
      ;;
    auto_rank_ard_masked)
      CONFIG="PTR3d_rank_ard_masked_8_2048_r40_3d_interpolate.yaml"
      ;;
    auto_rank_masked_cv)
      CONFIG="PTR3d_rank_cv_8_2048_r40_3d_interpolate.yaml"
      ;;
    auto_rank_spectral_round)
      CONFIG="PTR3d_rank_spectral_8_2048_r40_3d_interpolate.yaml"
      ;;
    auto_rank_independent_cv)
      CONFIG="PTR3d_rank_sweep_cv_8_2048_r40_3d_interpolate.yaml"
      ;;
    *)
      echo "Unsupported EXP030_VARIANT=${VARIANT}" >&2
      exit 1
      ;;
  esac
fi

RUN_ID="${EXP030_RUN_ID:-exp030_${VARIANT}_${MODE}_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${EXP030_LOG_ROOT:-logs/exp030/${RUN_ID}}"
PURIFICATION_ROOT="${EXP030_PURIFICATION_ROOT:-purified_data/exp030}"
ANALYSIS_ROOT="${LOG_ROOT}/analysis"

MADRY_CHECKPOINT="${EXP030_MADRY_CHECKPOINT:-checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_exp025_eegnet_prep_seed42_20260706_1225_at_best.pth}"
RPCF_CHECKPOINT="${EXP030_RPCF_CHECKPOINT:-checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_exp025_layer_prefix_seed42_20260706_1225_eegnet_eegnet_budget_2_rpcf_at_best.pth}"
MADRY_ATTACK="${EXP030_MADRY_ATTACK:-ad_data/exp027/exp027_oracle_rank_seed42_20260716_1305_madry_at_autoattack.pth}"
RPCF_ATTACK="${EXP030_RPCF_ATTACK:-logs/exp025/exp025_layer_prefix_seed42_20260706_1225_eegnet/eegnet/budget_2/rpcf_at_autoattack.pth}"
MADRY_FIXED25="${EXP030_MADRY_FIXED25:-purified_data/exp027/eval/exp027_oracle_rank_seed42_20260716_1305_madry_at_rank25.pth}"
RPCF_FIXED25="${EXP030_RPCF_FIXED25:-logs/exp025/exp025_layer_prefix_seed42_20260706_1225_eegnet/eegnet/budget_2/rpcf_at_rank25.pth}"
ORACLE_ROOT="${EXP030_ORACLE_ROOT:-logs/exp027/exp027_oracle_rank_seed42_20260716_1305/oracle}"

MADRY_OUTPUT="${PURIFICATION_ROOT}/${RUN_ID}_madry_at_${VARIANT}_n${SAMPLE_NUM}.pth"
RPCF_OUTPUT="${PURIFICATION_ROOT}/${RUN_ID}_rpcf_at_${VARIANT}_n${SAMPLE_NUM}.pth"

if ! [[ "${START_STAGE}" =~ ^[1-2]$ && "${STOP_STAGE}" =~ ^[1-2]$ ]]; then
  echo "START_STAGE and STOP_STAGE must be in [1, 2]." >&2
  exit 1
fi
if (( START_STAGE > STOP_STAGE )); then
  echo "START_STAGE cannot exceed STOP_STAGE." >&2
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

mkdir -p "${LOG_ROOT}/purification" "${PURIFICATION_ROOT}" "${ANALYSIS_ROOT}"
printf '%s\n' "$$" > "${LOG_ROOT}/controller.pid"
cat > "${LOG_ROOT}/run_config.txt" <<EOF
EXPERIMENT_ID=EXP-030
RUN_ID=${RUN_ID}
MODE=${MODE}
VARIANT=${VARIANT}
CONFIG=${CONFIG}
DATASET=${DATASET}
MODEL=${MODEL}
SEED=${SEED}
FOLD=${FOLD}
EPS=${EPS}
SAMPLE_NUM=${SAMPLE_NUM}
GPU_IDS=${GPU_IDS[*]}
GPU_IDLE_MAX_USED_MB=${GPU_IDLE_MAX_USED_MB}
PROCESSES_PER_GPU=${PROCESSES_PER_GPU}
MADRY_CHECKPOINT=${MADRY_CHECKPOINT}
RPCF_CHECKPOINT=${RPCF_CHECKPOINT}
MADRY_ATTACK=${MADRY_ATTACK}
RPCF_ATTACK=${RPCF_ATTACK}
MADRY_OUTPUT=${MADRY_OUTPUT}
RPCF_OUTPUT=${RPCF_OUTPUT}
SMOKE=${SMOKE}
PILOT=${PILOT}
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

if [[ "${MODE}" == "formal" ]]; then
  PILOT_SUMMARY="${EXP030_PILOT_SUMMARY:-}"
  if [[ -z "${PILOT_SUMMARY}" ]]; then
    echo "Formal n512 requires EXP030_PILOT_SUMMARY." >&2
    exit 1
  fi
  require_artifact "${PILOT_SUMMARY}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    conda run -n "${CONDA_ENV}" python -c \
      'import json,sys; d=json.load(open(sys.argv[1])); rows=d["methods"]; ok=all(r["auto_adv_accuracy"] >= r["fixed_adv_accuracy"] and r["clean_delta_pp"] >= -1.0 for r in rows); raise SystemExit(0 if ok else "EXP-030 pilot gate failed")' \
      "${PILOT_SUMMARY}"
  fi
fi

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

launch_eval() {
  local method="$1"
  local checkpoint="$2"
  local attack="$3"
  local output="$4"
  local gpu="$5"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${output}" ]]; then
    echo "[$(date -Is)] Reuse ${method}: ${output}"
    return
  fi
  require_artifact "${checkpoint}"
  require_artifact "${attack}"
  echo "[$(date -Is)] Start ${method} ${VARIANT} n${SAMPLE_NUM} on physical GPU ${gpu}."
  local overwrite_args=()
  if [[ "${SKIP_EXISTING}" != "1" ]]; then
    overwrite_args=(--overwrite)
  fi
  (
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf 'DRY_RUN eval method=%s gpu=%s output=%s config=%s\n' \
        "${method}" "${gpu}" "${output}" "${CONFIG}"
    else
      CUDA_VISIBLE_DEVICES="${gpu}" conda run -n "${CONDA_ENV}" --no-capture-output \
        python -u -m rpcf.evaluate_auto_rank \
        --attack_path "${attack}" --checkpoint_path "${checkpoint}" \
        --dataset "${DATASET}" --model "${MODEL}" --method_tag "${method}" \
        --variant "${VARIANT}" --config "${CONFIG}" \
        --fold "${FOLD}" --seed "${SEED}" --eps "${EPS}" \
        --sample_num "${SAMPLE_NUM}" --batch_size "${EVAL_BATCH_SIZE}" \
        --checkpoint_every "${CHECKPOINT_EVERY}" --gpu_id 0 \
        --output_path "${output}" "${overwrite_args[@]}"
    fi
  ) > "${LOG_ROOT}/purification/${method}.log" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LABELS+=("${method}")
}

run_purification_pair() {
  local pending=0
  [[ "${SKIP_EXISTING}" == "1" && -f "${MADRY_OUTPUT}" ]] || pending=$((pending + 1))
  [[ "${SKIP_EXISTING}" == "1" && -f "${RPCF_OUTPUT}" ]] || pending=$((pending + 1))
  if (( pending == 0 )); then
    echo "[$(date -Is)] Reuse both auto-rank payloads."
    return
  fi
  # 新启动必须成对；若只有一个缺失，仍等待空闲卡但只恢复缺失任务。
  select_idle_gpu
  ACTIVE_PIDS=()
  ACTIVE_LABELS=()
  launch_eval madry_at "${MADRY_CHECKPOINT}" "${MADRY_ATTACK}" "${MADRY_OUTPUT}" "${SELECTED_GPU}"
  launch_eval rpcf_at "${RPCF_CHECKPOINT}" "${RPCF_ATTACK}" "${RPCF_OUTPUT}" "${SELECTED_GPU}"
  local failures=()
  local index status
  for index in "${!ACTIVE_PIDS[@]}"; do
    status=0
    wait "${ACTIVE_PIDS[$index]}" || status=$?
    echo "[$(date -Is)] Finished ${ACTIVE_LABELS[$index]}: status=${status}"
    if [[ "${status}" -ne 0 ]]; then
      failures+=("${ACTIVE_LABELS[$index]}:${status}")
    fi
  done
  if (( ${#failures[@]} > 0 )); then
    echo "EXP-030 purification failures: ${failures[*]}" >&2
    exit 1
  fi
}

run_analysis() {
  if [[ "${SKIP_EXISTING}" == "1" && -f "${ANALYSIS_ROOT}/summary.json" && -f "${ANALYSIS_ROOT}/comparison.md" ]]; then
    echo "[$(date -Is)] Reuse analysis: ${ANALYSIS_ROOT}/summary.json"
    return
  fi
  for path in \
    "${MADRY_OUTPUT}" "${RPCF_OUTPUT}" "${MADRY_FIXED25}" "${RPCF_FIXED25}" \
    "${ORACLE_ROOT}/summary.json" "${ORACLE_ROOT}/oracle_selected_rows.csv"; do
    require_artifact "${path}"
  done
  select_idle_gpu
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN analysis gpu=%s output=%s\n' "${SELECTED_GPU}" "${ANALYSIS_ROOT}"
    return
  fi
  local overwrite_args=()
  if [[ "${SKIP_EXISTING}" != "1" ]]; then
    overwrite_args=(--overwrite)
  fi
  CUDA_VISIBLE_DEVICES="${SELECTED_GPU}" conda run -n "${CONDA_ENV}" --no-capture-output \
    python -u -m rpcf.analyze_exp030_auto_rank \
    --method madry_at "${MADRY_OUTPUT}" "${MADRY_FIXED25}" \
    --method rpcf_at "${RPCF_OUTPUT}" "${RPCF_FIXED25}" \
    --oracle_summary "${ORACLE_ROOT}/summary.json" \
    --oracle_rows "${ORACLE_ROOT}/oracle_selected_rows.csv" \
    --batch_size "${EVAL_BATCH_SIZE}" --gpu_id 0 \
    --bootstrap_samples "${BOOTSTRAP_SAMPLES}" --seed "${SEED}" \
    --output_dir "${ANALYSIS_ROOT}" "${overwrite_args[@]}" \
    >> "${LOG_ROOT}/stage2_analysis.log" 2>&1
}

if should_run 1; then
  run_purification_pair
fi
if should_run 2; then
  run_analysis
fi

echo "[$(date -Is)] EXP-030 pipeline finished: ${RUN_ID}"
