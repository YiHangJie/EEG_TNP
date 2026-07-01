#!/usr/bin/env bash
set -euo pipefail

DATASET="thubenchmark"
MODEL="eegnet"
SEED="${EXP023_SEED:-42}"
FOLD="${EXP023_FOLD:-0}"
EPS="${EXP023_EPS:-0.03}"
PGD_STEPS="${PGD_STEPS:-10}"
PGD_ALPHA="${PGD_ALPHA:-0.006}"
SAMPLE_NUM="${SAMPLE_NUM:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
RANKS="${RANKS:-25 30}"
GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-torch}"
RUN_ID="${EXP023_RUN_ID:-exp023_bpda_pgd10_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${EXP023_LOG_ROOT:-logs/exp023/${RUN_ID}}"
TAG="${RUN_ID//[^A-Za-z0-9_.-]/_}"

START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
PARALLEL_RANKS="${PARALLEL_RANKS:-1}"

PROTOCOL="train_only_subject_no_ea_subject_split"

case "${SEED}" in
  42)
    RPCF_AT_TAG="exp021_rpcf_at_seed42_20260623_1303"
    BASELINE_SUMMARY_DEFAULT="logs/exp021/exp021_rpcf_at_seed42_20260623_1303/comparison/summary.json"
    ;;
  43)
    RPCF_AT_TAG="exp021_rpcf_at_seeds43-44_20260624_0955_seed43"
    BASELINE_SUMMARY_DEFAULT="logs/exp021/exp021_rpcf_at_seeds43-44_20260624_0955_seed43/comparison/summary.json"
    ;;
  44)
    RPCF_AT_TAG="exp021_rpcf_at_seeds43-44_20260624_0955_seed44"
    BASELINE_SUMMARY_DEFAULT="logs/exp021/exp021_rpcf_at_seeds43-44_20260624_0955_seed44/comparison/summary.json"
    ;;
  *)
    echo "Unsupported EXP-023 seed: ${SEED}. Supported seeds: 42, 43, 44." >&2
    exit 1
    ;;
esac

RPCF_AT_CHECKPOINT_DEFAULT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_rpcf_at_${RPCF_AT_TAG}_best.pth"
RPCF_AT_CHECKPOINT="${EXP023_RPCF_AT_CHECKPOINT:-${RPCF_AT_CHECKPOINT_DEFAULT}}"
BASELINE_SUMMARY="${EXP023_BASELINE_SUMMARY:-${BASELINE_SUMMARY_DEFAULT}}"
SUMMARY_DIR="${LOG_ROOT}/comparison"

if [[ "${SMOKE}" == "1" ]]; then
  SAMPLE_NUM="${SMOKE_SAMPLE_NUM:-2}"
  RANKS="${SMOKE_RANKS:-25 30}"
fi

if ! [[ "${START_STAGE}" =~ ^[1-2]$ && "${STOP_STAGE}" =~ ^[1-2]$ ]]; then
  echo "START_STAGE and STOP_STAGE must be in [1, 2]." >&2
  exit 1
fi
if (( START_STAGE > STOP_STAGE )); then
  echo "START_STAGE cannot be greater than STOP_STAGE." >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}" ad_data/exp023
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
mkdir -p "${NUMBA_CACHE_DIR}" "${XDG_CACHE_HOME}"

printf '%s\n' "$$" > "${LOG_ROOT}/controller.pid"
cat > "${LOG_ROOT}/run_config.txt" <<EOF
EXPERIMENT_ID=EXP-023
RUN_ID=${RUN_ID}
DATASET=${DATASET}
MODEL=${MODEL}
SEED=${SEED}
FOLD=${FOLD}
EPS=${EPS}
ATTACK=BPDA+PGD-${PGD_STEPS}
PGD_ALPHA=${PGD_ALPHA}
SAMPLE_NUM=${SAMPLE_NUM}
BATCH_SIZE=${BATCH_SIZE}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}
RANKS=${RANKS}
PARALLEL_RANKS=${PARALLEL_RANKS}
RPCF_AT_CHECKPOINT=${RPCF_AT_CHECKPOINT}
BASELINE_SUMMARY=${BASELINE_SUMMARY}
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

join_by_comma() {
  local IFS=,
  echo "$*"
}

rank_config() {
  local rank="$1"
  case "${rank}" in
    25) echo "PTR3d_8_2048_rank25_3d_interpolate.yaml" ;;
    30) echo "PTR3d_8_2048_rank30_3d_interpolate.yaml" ;;
    *) echo "Unsupported EXP-023 rank: ${rank}" >&2; exit 1 ;;
  esac
}

overwrite_args=()
if [[ "${SKIP_EXISTING}" != "1" ]]; then
  overwrite_args=(--overwrite)
fi

require_artifact "${RPCF_AT_CHECKPOINT}"

bpda_paths=()
parallel_pids=()
for rank in ${RANKS}; do
  config="$(rank_config "${rank}")"
  output_path="ad_data/exp023/${TAG}_rpcf_at_bpda_pgd${PGD_STEPS}_rank${rank}.pth"
  bpda_paths+=("${output_path}")
  if should_run 1; then
    if [[ "${SKIP_EXISTING}" == "1" && -f "${output_path}" ]]; then
      echo "[Stage 1] Reuse BPDA artifact rank${rank}: ${output_path}"
    elif [[ "${PARALLEL_RANKS}" == "1" && "${DRY_RUN}" != "1" ]]; then
      stage="stage1_bpda_pgd_rank${rank}"
      echo "[$(date -Is)] ${stage}: conda run -n ${CONDA_ENV} --no-capture-output python -u -m rpcf.evaluate_bpda_pgd ... --rank ${rank} --output_path ${output_path}"
      (
        conda run -n "${CONDA_ENV}" --no-capture-output python -u \
          -m rpcf.evaluate_bpda_pgd \
          --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
          --seed "${SEED}" --checkpoint_path "${RPCF_AT_CHECKPOINT}" \
          --rank "${rank}" --config "${config}" --eps "${EPS}" \
          --pgd_steps "${PGD_STEPS}" --pgd_alpha "${PGD_ALPHA}" \
          --sample_num "${SAMPLE_NUM}" --batch_size "${BATCH_SIZE}" \
          --eval_batch_size "${EVAL_BATCH_SIZE}" --gpu_id "${GPU_ID}" \
          --output_path "${output_path}" "${overwrite_args[@]}" \
          2>&1 | tee -a "${LOG_ROOT}/${stage}.log"
      ) &
      parallel_pids+=("$!")
    else
      run_logged "stage1_bpda_pgd_rank${rank}" \
        conda run -n "${CONDA_ENV}" --no-capture-output python -u \
        -m rpcf.evaluate_bpda_pgd \
        --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
        --seed "${SEED}" --checkpoint_path "${RPCF_AT_CHECKPOINT}" \
        --rank "${rank}" --config "${config}" --eps "${EPS}" \
        --pgd_steps "${PGD_STEPS}" --pgd_alpha "${PGD_ALPHA}" \
        --sample_num "${SAMPLE_NUM}" --batch_size "${BATCH_SIZE}" \
        --eval_batch_size "${EVAL_BATCH_SIZE}" --gpu_id "${GPU_ID}" \
        --output_path "${output_path}" "${overwrite_args[@]}"
    fi
  fi
done

if ((${#parallel_pids[@]} > 0)); then
  status=0
  for pid in "${parallel_pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if (( status != 0 )); then
    echo "One or more BPDA rank jobs failed." >&2
    exit 1
  fi
fi

if should_run 2; then
  eval_paths="$(join_by_comma "${bpda_paths[@]}")"
  if [[ "${SMOKE}" == "1" ]]; then
    run_logged stage2_summary_smoke \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.compare_exp023 \
      --bpda_paths "${eval_paths}" --output_dir "${SUMMARY_DIR}"
  else
    require_artifact "${BASELINE_SUMMARY}"
    run_logged stage2_summary \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.compare_exp023 \
      --bpda_paths "${eval_paths}" --baseline_summaries "${BASELINE_SUMMARY}" \
      --output_dir "${SUMMARY_DIR}"
  fi
fi

echo "[$(date -Is)] EXP-023 BPDA+PGD pipeline finished: ${RUN_ID}"
