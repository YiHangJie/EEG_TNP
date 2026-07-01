#!/usr/bin/env bash
set -euo pipefail

# EXP-023 baseline：评估标准 AT 模型在同一 n512 子集上的 PGD-10 性能。
# 默认用于接在 BPDA+PGD adaptive attack 之后运行，不重训模型。

SEED="${EXP023_SEED:-42}"
FOLD="${EXP023_FOLD:-0}"
DATASET="${EXP023_DATASET:-thubenchmark}"
MODEL="${EXP023_MODEL:-eegnet}"
EPS="${EXP023_EPS:-0.03}"
PGD_STEPS="${EXP023_PGD_STEPS:-10}"
PGD_ALPHA="${EXP023_PGD_ALPHA:-0.006}"
SAMPLE_NUM="${EXP023_SAMPLE_NUM:-512}"
BATCH_SIZE="${EXP023_BATCH_SIZE:-32}"
GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-torch}"
RUN_ID="${EXP023_BASELINE_RUN_ID:-exp023_baseline_pgd10_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${EXP023_BASELINE_LOG_ROOT:-logs/exp023/${RUN_ID}}"
WAIT_FOR_RUN_ID="${WAIT_FOR_RUN_ID:-}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-60}"
STRATEGIES="${EXP023_BASELINE_STRATEGIES:-madry trades fbf}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p "${LOG_ROOT}" ad_data log_attack

if [[ -n "${WAIT_FOR_RUN_ID}" ]]; then
  rank25_artifact="ad_data/exp023/${WAIT_FOR_RUN_ID}_rpcf_at_bpda_pgd${PGD_STEPS}_rank25.pth"
  rank30_artifact="ad_data/exp023/${WAIT_FOR_RUN_ID}_rpcf_at_bpda_pgd${PGD_STEPS}_rank30.pth"
  echo "[$(date -Is)] Waiting for EXP-023 BPDA artifacts:"
  echo "  ${rank25_artifact}"
  echo "  ${rank30_artifact}"
  while [[ ! -f "${rank25_artifact}" || ! -f "${rank30_artifact}" ]]; do
    sleep "${WAIT_INTERVAL_SECONDS}"
  done
  echo "[$(date -Is)] BPDA artifacts are ready; start baseline PGD-${PGD_STEPS}."
fi

run_logged() {
  local name="$1"
  shift
  local log_path="${LOG_ROOT}/${name}.log"
  echo "[$(date -Is)] ${name}: $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@" > "${log_path}" 2>&1
}

default_checkpoint() {
  local strategy="$1"
  local default_path="checkpoints/${DATASET}_${MODEL}_train_only_subject_no_ea_subject_split_${strategy}_eps${EPS}_${SEED}_fold${FOLD}_best.pth"
  if [[ "${strategy}" == "madry" ]]; then
    case "${SEED}" in
      43)
        default_path="checkpoints/${DATASET}_${MODEL}_train_only_subject_no_ea_subject_split_madry_eps${EPS}_${SEED}_fold${FOLD}_exp019_seed43_fold0_full_20260615_1821_at_best.pth"
        ;;
      44)
        default_path="checkpoints/${DATASET}_${MODEL}_train_only_subject_no_ea_subject_split_madry_eps${EPS}_${SEED}_fold${FOLD}_exp019_seed44_fold0_full_sixrank_20260617_2114_at_best.pth"
        ;;
    esac
  fi
  printf '%s\n' "${default_path}"
}

for strategy in ${STRATEGIES}; do
  checkpoint="$(default_checkpoint "${strategy}")"
  override_var="EXP023_BASELINE_${strategy^^}_CHECKPOINT"
  checkpoint="${!override_var:-${checkpoint}}"
  output_path="ad_data/${DATASET}_${MODEL}_no_ea_exp023_baseline_pgd${PGD_STEPS}_${strategy}_pgd_eps${EPS}_seed${SEED}_fold${FOLD}.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint for ${strategy}: ${checkpoint}" >&2
    exit 1
  fi
  if [[ "${SKIP_EXISTING}" == "1" && -f "${output_path}" ]]; then
    echo "[$(date -Is)] Skip existing ${strategy} PGD-${PGD_STEPS}: ${output_path}"
    continue
  fi
  run_logged "baseline_${strategy}_pgd${PGD_STEPS}" \
    conda run -n "${CONDA_ENV}" --no-capture-output python -u attack.py \
      --dataset "${DATASET}" --model "${MODEL}" --at_strategy "${strategy}" \
      --fold "${FOLD}" --attack pgd --eps "${EPS}" \
      --pgd_steps "${PGD_STEPS}" --pgd_alpha "${PGD_ALPHA}" \
      --attack_sample_num "${SAMPLE_NUM}" --batch_size "${BATCH_SIZE}" \
      --seed "${SEED}" --gpu_id "${GPU_ID}" --no_ea \
      --checkpoint_path "${checkpoint}" --save_adv \
      --adv_output_tag "exp023_baseline_pgd${PGD_STEPS}"
done

echo "[$(date -Is)] EXP-023 baseline PGD-${PGD_STEPS} finished: ${RUN_ID}"
