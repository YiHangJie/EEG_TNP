#!/usr/bin/env bash
set -euo pipefail

# EXP-026 单 backbone 超参数预筛。
# 手动运行：
#   EXP026_MODEL=eegnet EXP026_PRESCREEN_ID=exp026_prescreen_seed42_YYYYMMDD_HHMMSS \
#     CUDA_VISIBLE_DEVICES=0 GPU_ID=0 bash rpcf/run_exp026_prescreen_model.sh

source "$(dirname "$0")/exp026_common.sh"

MODEL="${EXP026_MODEL:-eegnet}"
RUN_ID="${EXP026_PRESCREEN_ID:-exp026_prescreen_seed42_$(date +%Y%m%d_%H%M%S)}"
ROOT="${EXP026_PRESCREEN_ROOT:-logs/exp026/${RUN_ID}}"
GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-torch}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
EPOCHS="${PRESCREEN_EPOCHS:-10}"
EVAL_BATCH_SIZE="${RPCF_EVAL_BATCH_SIZE:-2}"
ONLINE_AT_BATCH_SIZE="${ONLINE_AT_BATCH_SIZE:-16}"

if [[ "${SMOKE}" == "1" ]]; then
  EPOCHS=1
fi

resolve_exp026_artifacts "${MODEL}"
if [[ "${DRY_RUN}" != "1" ]]; then
  require_exp026_artifact "${AT_CHECKPOINT}"
  require_exp026_artifact "${RPCF_CACHE}"
  require_exp026_artifact "${SENSITIVITY_PATH}"
fi

MODEL_ROOT="${ROOT}/${MODEL}"
mkdir -p "${MODEL_ROOT}" checkpoints
cat > "${MODEL_ROOT}/run_config.txt" <<EOF
EXPERIMENT_ID=EXP-026
PHASE=prescreen
RUN_ID=${RUN_ID}
MODEL=${MODEL}
AT_CHECKPOINT=${AT_CHECKPOINT}
RPCF_CACHE=${RPCF_CACHE}
SENSITIVITY_PATH=${SENSITIVITY_PATH}
EPOCHS=${EPOCHS}
CACHE_HOLDOUT_FRACTION=0.2
BALANCED_BATCH=4x2
EOF

run_candidate() {
  local config_id="$1"
  local objective="$2"
  local weight="$3"
  local margin_weight="$4"
  local config_root="${MODEL_ROOT}/${config_id}"
  local history="${config_root}/finetune"
  local checkpoint="checkpoints/thubenchmark_${MODEL}_exp026_${RUN_ID}_${config_id}_best.pth"
  mkdir -p "${config_root}"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${history}.json" && -f "${checkpoint}" ]]; then
    echo "[$(date -Is)] Reuse ${MODEL}/${config_id}"
    return
  fi
  local feature_args=()
  if [[ "${objective}" != "none" ]]; then
    feature_args=(
      --feature_objective "${objective}"
      --feature_loss_weight "${weight}"
    )
  fi
  if [[ "${objective}" == "prototype" ]]; then
    feature_args+=(--prototype_margin 1.0 --prototype_margin_weight "${margin_weight}")
  elif [[ "${objective}" == "contrastive" ]]; then
    feature_args+=(--contrastive_temperature 0.1)
  fi
  local smoke_args=()
  if [[ "${SMOKE}" == "1" ]]; then
    smoke_args=(--online_train_sample_num 8 --max_cache_batches 1 --pgd_steps 1)
  fi
  local command=(
    conda run -n "${CONDA_ENV}" --no-capture-output python -u -m rpcf.finetune
    --cache_path "${RPCF_CACHE}" --sensitivity_path "${SENSITIVITY_PATH}"
    --checkpoint_path "${AT_CHECKPOINT}" --output_checkpoint "${checkpoint}"
    --dataset thubenchmark --model "${MODEL}" --fold 0 --seed 42
    --epsilon 0.03 --epochs "${EPOCHS}" --batch_size 8
    --eval_batch_size "${EVAL_BATCH_SIZE}" --lr 0.0001 --weight_decay 0.0001
    --rank_temperature 0.5 --consistancy_temperature 2.0
    --online_madry_at --online_at_batch_size "${ONLINE_AT_BATCH_SIZE}"
    --online_at_pgd_steps 10 --online_at_step_size 0.006
    --balanced_classes_per_batch 4 --balanced_samples_per_class 2
    --cache_holdout_fraction 0.2 --gpu_id "${GPU_ID}"
    --history_prefix "${history}"
    "${feature_args[@]}" "${smoke_args[@]}"
  )
  echo "[$(date -Is)] ${MODEL}/${config_id}: ${command[*]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  "${command[@]}" 2>&1 | tee -a "${config_root}/run.log"
}

run_candidate control none 0 0
for weight_id in 0p01 0p1 1p0; do
  case "${weight_id}" in
    0p01) weight=0.01 ;;
    0p1) weight=0.1 ;;
    1p0) weight=1.0 ;;
  esac
  run_candidate "cmmd_w${weight_id}" cmmd "${weight}" 0
  run_candidate "prototype_w${weight_id}_margin0" prototype "${weight}" 0
  run_candidate "prototype_w${weight_id}_margin0p1" prototype "${weight}" 0.1
  run_candidate "contrastive_w${weight_id}" contrastive "${weight}" 0
done

echo "[$(date -Is)] EXP-026 prescreen finished: ${MODEL}"
