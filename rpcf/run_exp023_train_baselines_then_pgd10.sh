#!/usr/bin/env bash
set -euo pipefail

# EXP-023 baseline 补全：等待 BPDA 完成后，先训练缺失的 baseline 模型，
# 再评估 seed43/44 的 Madry/TRADES/FBF PGD-10。

RUN_ID="${EXP023_BASELINE_CHAIN_ID:-exp023_baseline_train_then_pgd10_seeds43-44_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${EXP023_BASELINE_CHAIN_LOG_ROOT:-logs/exp023/${RUN_ID}}"
DATASET="${EXP023_DATASET:-thubenchmark}"
MODEL="${EXP023_MODEL:-eegnet}"
FOLD="${EXP023_FOLD:-0}"
EPS="${EXP023_EPS:-0.03}"
GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-torch}"
SEEDS="${EXP023_BASELINE_SEEDS:-43 44}"
TRAIN_STRATEGIES="${EXP023_BASELINE_TRAIN_STRATEGIES:-trades fbf}"
EVAL_STRATEGIES="${EXP023_BASELINE_EVAL_STRATEGIES:-madry trades fbf}"
TRAIN_TAG="${EXP023_BASELINE_TRAIN_TAG:-exp023_baseline_seed43-44_20260630_0102}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-300}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p "${LOG_ROOT}" checkpoints log_train_AT ad_data log_attack
printf '%s\n' "$$" > "${LOG_ROOT}/controller.pid"

cat > "${LOG_ROOT}/run_config.txt" <<EOF
EXPERIMENT_ID=EXP-023
RUN_ID=${RUN_ID}
DATASET=${DATASET}
MODEL=${MODEL}
FOLD=${FOLD}
EPS=${EPS}
SEEDS=${SEEDS}
TRAIN_STRATEGIES=${TRAIN_STRATEGIES}
EVAL_STRATEGIES=${EVAL_STRATEGIES}
TRAIN_TAG=${TRAIN_TAG}
WAIT_FOR_PID=${WAIT_FOR_PID}
WAIT_INTERVAL_SECONDS=${WAIT_INTERVAL_SECONDS}
SKIP_EXISTING=${SKIP_EXISTING}
EOF

run_logged() {
  local stage="$1"
  shift
  local log_path="${LOG_ROOT}/${stage}.log"
  echo "[$(date -Is)] ${stage}: $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@" 2>&1 | tee -a "${log_path}"
}

checkpoint_for() {
  local seed="$1"
  local strategy="$2"
  case "${strategy}" in
    madry)
      case "${seed}" in
        43)
          printf '%s\n' "checkpoints/${DATASET}_${MODEL}_train_only_subject_no_ea_subject_split_madry_eps${EPS}_${seed}_fold${FOLD}_exp019_seed43_fold0_full_20260615_1821_at_best.pth"
          ;;
        44)
          printf '%s\n' "checkpoints/${DATASET}_${MODEL}_train_only_subject_no_ea_subject_split_madry_eps${EPS}_${seed}_fold${FOLD}_exp019_seed44_fold0_full_sixrank_20260617_2114_at_best.pth"
          ;;
        *)
          printf '%s\n' "checkpoints/${DATASET}_${MODEL}_train_only_subject_no_ea_subject_split_madry_eps${EPS}_${seed}_fold${FOLD}_best.pth"
          ;;
      esac
      ;;
    trades|fbf)
      printf '%s\n' "checkpoints/${DATASET}_${MODEL}_train_only_subject_no_ea_subject_split_${strategy}_eps${EPS}_${seed}_fold${FOLD}_${TRAIN_TAG}_best.pth"
      ;;
    *)
      echo "Unsupported baseline strategy: ${strategy}" >&2
      exit 1
      ;;
  esac
}

if [[ -n "${WAIT_FOR_PID}" ]]; then
  echo "[$(date -Is)] Waiting for PID ${WAIT_FOR_PID} before EXP-023 baseline training/eval."
  while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
    sleep "${WAIT_INTERVAL_SECONDS}"
  done
  echo "[$(date -Is)] PID ${WAIT_FOR_PID} exited; start EXP-023 baseline training/eval."
fi

for seed in ${SEEDS}; do
  for strategy in ${TRAIN_STRATEGIES}; do
    checkpoint="$(checkpoint_for "${seed}" "${strategy}")"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${checkpoint}" ]]; then
      echo "[$(date -Is)] Skip existing ${strategy} checkpoint seed${seed}: ${checkpoint}"
      continue
    fi
    run_logged "train_seed${seed}_${strategy}" \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u train_AT.py \
        --dataset "${DATASET}" --model "${MODEL}" --at_strategy "${strategy}" \
        --fold "${FOLD}" --epsilon "${EPS}" --seed "${seed}" --gpu_id "${GPU_ID}" \
        --no_ea --checkpoint_tag "${TRAIN_TAG}"
  done
done

for seed in ${SEEDS}; do
  madry_checkpoint="$(checkpoint_for "${seed}" madry)"
  trades_checkpoint="$(checkpoint_for "${seed}" trades)"
  fbf_checkpoint="$(checkpoint_for "${seed}" fbf)"
  run_logged "baseline_seed${seed}_pgd10" \
    env \
      EXP023_SEED="${seed}" \
      EXP023_BASELINE_RUN_ID="exp023_baseline_pgd10_seed${seed}_after_baseline_train_20260630_0102" \
      EXP023_BASELINE_STRATEGIES="${EVAL_STRATEGIES}" \
      EXP023_BASELINE_MADRY_CHECKPOINT="${madry_checkpoint}" \
      EXP023_BASELINE_TRADES_CHECKPOINT="${trades_checkpoint}" \
      EXP023_BASELINE_FBF_CHECKPOINT="${fbf_checkpoint}" \
      SKIP_EXISTING="${SKIP_EXISTING}" \
      bash rpcf/run_exp023_baseline_pgd10.sh
done

echo "[$(date -Is)] EXP-023 baseline train+PGD-10 pipeline finished: ${RUN_ID}"
