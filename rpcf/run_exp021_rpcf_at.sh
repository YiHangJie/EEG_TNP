#!/usr/bin/env bash
set -euo pipefail

DATASET="thubenchmark"
MODEL="eegnet"
SEED="${EXP021_SEED:-42}"
FOLD="${EXP021_FOLD:-0}"
EPS="${EXP021_EPS:-0.03}"
GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-torch}"
RUN_ID="${EXP021_RUN_ID:-exp021_rpcf_at_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${EXP021_LOG_ROOT:-logs/exp021/${RUN_ID}}"
TAG="${RUN_ID//[^A-Za-z0-9_.-]/_}"

START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"

RPCF_EPOCHS="${RPCF_EPOCHS:-100}"
RPCF_BATCH_SIZE="${RPCF_BATCH_SIZE:-64}"
ONLINE_AT_BATCH_SIZE="${ONLINE_AT_BATCH_SIZE:-128}"
ONLINE_AT_TRAIN_SAMPLE_NUM="${ONLINE_AT_TRAIN_SAMPLE_NUM:-}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-8}"
ATTACK_SAMPLE_NUM="${ATTACK_SAMPLE_NUM:-}"
EVAL_SAMPLE_NUM="${EVAL_SAMPLE_NUM:-512}"

PROTOCOL="train_only_subject_no_ea_subject_split"

case "${SEED}" in
  42)
    BASE_RUN_ID="exp018_full_20260612_124131"
    SENSITIVITY_RUN_ID="exp018_rpcf_no_early_stop_20260614_2357"
    CONSISTANCY_RUN_ID="exp018_seed42_fold0_consistancy_six_rank_20260617_1020"
    AT_CHECKPOINT_DEFAULT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${BASE_RUN_ID}_at_best.pth"
    CONSISTANCY_CHECKPOINT_DEFAULT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${CONSISTANCY_RUN_ID}_consistancy_six_rank_best.pth"
    RPCF_CACHE_DEFAULT="purified_data/exp018/rpcf_train/${BASE_RUN_ID}_six_rank.pth"
    SENSITIVITY_PATH_DEFAULT="logs/exp018/${SENSITIVITY_RUN_ID}/sensitivity.json"
    AT_PURIFY_DEFAULT="purified_data/exp018/eval/${BASE_RUN_ID}_at_rank25-30.pth"
    CONSISTANCY_PURIFY_DEFAULT="purified_data/exp018/eval/${CONSISTANCY_RUN_ID}_consistancy_six_rank_rank25-30.pth"
    ;;
  43)
    BASE_RUN_ID="exp019_seed43_fold0_full_20260615_1821"
    CONSISTANCY_RUN_ID="exp019_seed43_fold0_consistancy_six_rank_20260616_1451"
    AT_CHECKPOINT_DEFAULT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${BASE_RUN_ID}_at_best.pth"
    CONSISTANCY_CHECKPOINT_DEFAULT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${CONSISTANCY_RUN_ID}_consistancy_six_rank_best.pth"
    RPCF_CACHE_DEFAULT="purified_data/exp019/rpcf_train/${BASE_RUN_ID}_six_rank.pth"
    SENSITIVITY_PATH_DEFAULT="logs/exp019/${BASE_RUN_ID}/sensitivity.json"
    AT_PURIFY_DEFAULT="purified_data/exp019/eval/${BASE_RUN_ID}_madry_at_rank25-30.pth"
    CONSISTANCY_PURIFY_DEFAULT="purified_data/exp019/eval/${CONSISTANCY_RUN_ID}_consistancy_six_rank_rank25-30.pth"
    ;;
  44)
    BASE_RUN_ID="exp019_seed44_fold0_full_sixrank_20260617_2114"
    AT_CHECKPOINT_DEFAULT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${BASE_RUN_ID}_at_best.pth"
    CONSISTANCY_CHECKPOINT_DEFAULT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${BASE_RUN_ID}_consistancy_six_rank_best.pth"
    RPCF_CACHE_DEFAULT="purified_data/exp019/rpcf_train/${BASE_RUN_ID}_six_rank.pth"
    SENSITIVITY_PATH_DEFAULT="logs/exp019/${BASE_RUN_ID}/sensitivity.json"
    AT_PURIFY_DEFAULT="purified_data/exp019/eval/${BASE_RUN_ID}_madry_at_rank25-30.pth"
    CONSISTANCY_PURIFY_DEFAULT="purified_data/exp019/eval/${BASE_RUN_ID}_consistancy_six_rank_rank25-30.pth"
    ;;
  *)
    echo "Unsupported EXP-021 seed: ${SEED}. Supported seeds: 42, 43, 44." >&2
    exit 1
    ;;
esac

AT_CHECKPOINT="${EXP021_AT_CHECKPOINT:-${AT_CHECKPOINT_DEFAULT}}"
CONSISTANCY_CHECKPOINT="${EXP021_CONSISTANCY_CHECKPOINT:-${CONSISTANCY_CHECKPOINT_DEFAULT}}"
RPCF_CACHE="${EXP021_RPCF_CACHE:-${RPCF_CACHE_DEFAULT}}"
SENSITIVITY_PATH="${EXP021_SENSITIVITY_PATH:-${SENSITIVITY_PATH_DEFAULT}}"
AT_PURIFY="${EXP021_AT_PURIFY:-${AT_PURIFY_DEFAULT}}"
CONSISTANCY_PURIFY="${EXP021_CONSISTANCY_PURIFY:-${CONSISTANCY_PURIFY_DEFAULT}}"

RPCF_AT_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_rpcf_at_${TAG}_best.pth"
RPCF_AT_ATTACK="ad_data/exp021/${TAG}_rpcf_at_autoattack.pth"
RPCF_AT_PURIFY="purified_data/exp021/eval/${TAG}_rpcf_at_rank25-30.pth"
HISTORY_PREFIX="${LOG_ROOT}/finetune_rpcf_at"
SUMMARY_DIR="${LOG_ROOT}/comparison"

if [[ "${SMOKE}" == "1" ]]; then
  RPCF_EPOCHS="${SMOKE_EPOCHS:-1}"
  ONLINE_AT_TRAIN_SAMPLE_NUM="${SMOKE_TRAIN_SAMPLE_NUM:-2}"
  ATTACK_SAMPLE_NUM="${SMOKE_ATTACK_SAMPLE_NUM:-2}"
  EVAL_SAMPLE_NUM="${SMOKE_EVAL_SAMPLE_NUM:-2}"
fi

if ! [[ "${START_STAGE}" =~ ^[1-4]$ && "${STOP_STAGE}" =~ ^[1-4]$ ]]; then
  echo "START_STAGE and STOP_STAGE must be in [1, 4]." >&2
  exit 1
fi
if (( START_STAGE > STOP_STAGE )); then
  echo "START_STAGE cannot be greater than STOP_STAGE." >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}" checkpoints ad_data/exp021 purified_data/exp021/eval
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
mkdir -p "${NUMBA_CACHE_DIR}" "${XDG_CACHE_HOME}"

printf '%s\n' "$$" > "${LOG_ROOT}/controller.pid"
cat > "${LOG_ROOT}/run_config.txt" <<EOF
EXPERIMENT_ID=EXP-021
RUN_ID=${RUN_ID}
DATASET=${DATASET}
MODEL=${MODEL}
SEED=${SEED}
FOLD=${FOLD}
EPS=${EPS}
METHOD=RPCF_AT_SELECTIVE
RPCF_EPOCHS=${RPCF_EPOCHS}
ONLINE_AT_BATCH_SIZE=${ONLINE_AT_BATCH_SIZE}
ONLINE_AT_PGD_STEPS=10
ONLINE_AT_STEP_SIZE=0.006
ONLINE_AT_TRAIN_SAMPLE_NUM=${ONLINE_AT_TRAIN_SAMPLE_NUM:-full}
EVAL_SAMPLE_NUM=${EVAL_SAMPLE_NUM}
ATTACK_BATCH_SIZE=${ATTACK_BATCH_SIZE}
AT_CHECKPOINT=${AT_CHECKPOINT}
CONSISTANCY_CHECKPOINT=${CONSISTANCY_CHECKPOINT}
RPCF_CACHE=${RPCF_CACHE}
SENSITIVITY_PATH=${SENSITIVITY_PATH}
AT_PURIFY=${AT_PURIFY}
CONSISTANCY_PURIFY=${CONSISTANCY_PURIFY}
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

overwrite_args=()
if [[ "${SKIP_EXISTING}" != "1" ]]; then
  overwrite_args=(--overwrite)
fi
online_subset_args=()
if [[ -n "${ONLINE_AT_TRAIN_SAMPLE_NUM}" ]]; then
  online_subset_args=(--online_train_sample_num "${ONLINE_AT_TRAIN_SAMPLE_NUM}")
fi
attack_subset_args=()
if [[ -n "${ATTACK_SAMPLE_NUM}" ]]; then
  attack_subset_args=(--sample_num "${ATTACK_SAMPLE_NUM}")
fi

for path in \
  "${AT_CHECKPOINT}" "${CONSISTANCY_CHECKPOINT}" "${RPCF_CACHE}" \
  "${SENSITIVITY_PATH}" "${AT_PURIFY}" "${CONSISTANCY_PURIFY}"; do
  require_artifact "${path}"
done

if should_run 1; then
  if [[ "${SKIP_EXISTING}" == "1" && -f "${RPCF_AT_CHECKPOINT}" ]]; then
    echo "[Stage 1] Reuse RPCF_AT checkpoint: ${RPCF_AT_CHECKPOINT}"
  else
    run_logged stage1_finetune_rpcf_at \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.finetune \
      --cache_path "${RPCF_CACHE}" --sensitivity_path "${SENSITIVITY_PATH}" \
      --checkpoint_path "${AT_CHECKPOINT}" \
      --output_checkpoint "${RPCF_AT_CHECKPOINT}" \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
      --seed "${SEED}" --epsilon "${EPS}" --epochs "${RPCF_EPOCHS}" \
      --batch_size "${RPCF_BATCH_SIZE}" --lr 0.0001 --weight_decay 0.0001 \
      --rank_temperature 0.5 --consistancy_temperature 2.0 \
      --pgd_steps 10 --online_madry_at \
      --online_at_batch_size "${ONLINE_AT_BATCH_SIZE}" \
      --online_at_pgd_steps 10 --online_at_step_size 0.006 \
      "${online_subset_args[@]}" --gpu_id "${GPU_ID}" \
      --history_prefix "${HISTORY_PREFIX}"
  fi
fi

if should_run 2; then
  require_artifact "${RPCF_AT_CHECKPOINT}"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${RPCF_AT_ATTACK}" ]]; then
    echo "[Stage 2] Reuse RPCF_AT attack: ${RPCF_AT_ATTACK}"
  else
    run_logged stage2_attack_rpcf_at \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.evaluate_attack \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
      --seed "${SEED}" --checkpoint_path "${RPCF_AT_CHECKPOINT}" \
      --method_tag rpcf_at --attack autoattack --eps "${EPS}" \
      --batch_size "${ATTACK_BATCH_SIZE}" --gpu_id "${GPU_ID}" \
      --output_path "${RPCF_AT_ATTACK}" \
      "${attack_subset_args[@]}" "${overwrite_args[@]}"
  fi
fi

if should_run 3; then
  require_artifact "${RPCF_AT_ATTACK}"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${RPCF_AT_PURIFY}" ]]; then
    echo "[Stage 3] Reuse RPCF_AT purification: ${RPCF_AT_PURIFY}"
  else
    run_logged stage3_purify_rpcf_at \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.evaluate_purification \
      --attack_path "${RPCF_AT_ATTACK}" \
      --checkpoint_path "${RPCF_AT_CHECKPOINT}" \
      --dataset "${DATASET}" --model "${MODEL}" --fold "${FOLD}" \
      --seed "${SEED}" --eps "${EPS}" --sample_num "${EVAL_SAMPLE_NUM}" \
      --ranks "25,30" \
      --configs "PTR3d_8_2048_rank25_3d_interpolate.yaml,PTR3d_8_2048_rank30_3d_interpolate.yaml" \
      --gpu_id "${GPU_ID}" --output_path "${RPCF_AT_PURIFY}" \
      "${overwrite_args[@]}"
  fi
fi

if should_run 4; then
  if [[ "${SMOKE}" == "1" ]]; then
    echo "[Stage 4] Skip fair summary in smoke mode because sample_num != 512."
  else
    require_artifact "${RPCF_AT_PURIFY}"
    require_artifact "${HISTORY_PREFIX}.json"
    run_logged stage4_summary \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.compare_exp018 \
      --dataset "${DATASET}" --model "${MODEL}" --seed "${SEED}" \
      --fold "${FOLD}" --eps "${EPS}" --ranks "25,30" --sample_num 512 \
      --gpu_id "${GPU_ID}" --at_checkpoint "${AT_CHECKPOINT}" \
      --consistancy_checkpoint "${CONSISTANCY_CHECKPOINT}" \
      --rpcf_checkpoint "${RPCF_AT_CHECKPOINT}" \
      --at_purification_path "${AT_PURIFY}" \
      --consistancy_purification_path "${CONSISTANCY_PURIFY}" \
      --rpcf_purification_path "${RPCF_AT_PURIFY}" \
      --sensitivity_path "${SENSITIVITY_PATH}" \
      --history_path "${HISTORY_PREFIX}.json" \
      --output_dir "${SUMMARY_DIR}" --experiment_id EXP-021
  fi
fi

echo "[$(date -Is)] EXP-021 RPCF_AT pipeline finished: ${RUN_ID}"
