#!/usr/bin/env bash
set -euo pipefail

# EXP-026 单 backbone：四配置训练、各自 white-box AutoAttack、rank25/30 净化与汇总。
# 手动运行：
#   EXP026_MODEL=eegnet EXP026_RUN_ID=exp026_formal_seed42_YYYYMMDD_HHMMSS \
#     EXP026_SELECTED_ENV=logs/exp026/<prescreen>/selection/selected_hparams.env \
#     CUDA_VISIBLE_DEVICES=0 GPU_ID=0 bash rpcf/run_exp026_backbone.sh

source "$(dirname "$0")/exp026_common.sh"

MODEL="${EXP026_MODEL:-eegnet}"
RUN_ID="${EXP026_RUN_ID:-exp026_formal_seed42_$(date +%Y%m%d_%H%M%S)}"
ROOT="${EXP026_ROOT:-logs/exp026/${RUN_ID}}"
MODEL_ROOT="${ROOT}/${MODEL}"
GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-torch}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-4}"
EPOCHS="${RPCF_EPOCHS:-100}"
EVAL_BATCH_SIZE="${RPCF_EVAL_BATCH_SIZE:-2}"
ONLINE_AT_BATCH_SIZE="${ONLINE_AT_BATCH_SIZE:-16}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-8}"
EVAL_SAMPLE_NUM="${EVAL_SAMPLE_NUM:-512}"
SELECTED_ENV="${EXP026_SELECTED_ENV:-}"

if ! [[ "${START_STAGE}" =~ ^[1-4]$ && "${STOP_STAGE}" =~ ^[1-4]$ ]]; then
  echo "START_STAGE/STOP_STAGE must be in [1,4]." >&2
  exit 1
fi
if (( START_STAGE > STOP_STAGE )); then
  echo "START_STAGE cannot exceed STOP_STAGE." >&2
  exit 1
fi

if [[ -n "${SELECTED_ENV}" && -f "${SELECTED_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${SELECTED_ENV}"
elif [[ "${DRY_RUN}" == "1" || "${SMOKE}" == "1" ]]; then
  CMMD_WEIGHT="${CMMD_WEIGHT:-0.1}"
  PROTOTYPE_WEIGHT="${PROTOTYPE_WEIGHT:-0.1}"
  PROTOTYPE_MARGIN="${PROTOTYPE_MARGIN:-1.0}"
  PROTOTYPE_MARGIN_WEIGHT="${PROTOTYPE_MARGIN_WEIGHT:-0.1}"
  CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.1}"
  CONTRASTIVE_TEMPERATURE="${CONTRASTIVE_TEMPERATURE:-0.1}"
else
  echo "EXP026_SELECTED_ENV must point to completed prescreen selection." >&2
  exit 1
fi

resolve_exp026_artifacts "${MODEL}"
if [[ "${DRY_RUN}" != "1" ]]; then
  require_exp026_artifact "${AT_CHECKPOINT}"
  require_exp026_artifact "${RPCF_CACHE}"
  require_exp026_artifact "${SENSITIVITY_PATH}"
fi

if [[ "${SMOKE}" == "1" ]]; then
  EPOCHS=1
  EVAL_SAMPLE_NUM=2
fi

TAG="${RUN_ID}_${MODEL}"
mkdir -p "${MODEL_ROOT}" checkpoints ad_data/exp026 purified_data/exp026/eval
printf '%s\n' "$$" > "${MODEL_ROOT}/controller.pid"
cat > "${MODEL_ROOT}/run_config.txt" <<EOF
EXPERIMENT_ID=EXP-026
PHASE=formal
RUN_ID=${RUN_ID}
MODEL=${MODEL}
AT_CHECKPOINT=${AT_CHECKPOINT}
RPCF_CACHE=${RPCF_CACHE}
SENSITIVITY_PATH=${SENSITIVITY_PATH}
CONTEXT_SUMMARY=${CONTEXT_SUMMARY}
SELECTED_ENV=${SELECTED_ENV}
CMMD_WEIGHT=${CMMD_WEIGHT}
PROTOTYPE_WEIGHT=${PROTOTYPE_WEIGHT}
PROTOTYPE_MARGIN=${PROTOTYPE_MARGIN}
PROTOTYPE_MARGIN_WEIGHT=${PROTOTYPE_MARGIN_WEIGHT}
CONTRASTIVE_WEIGHT=${CONTRASTIVE_WEIGHT}
CONTRASTIVE_TEMPERATURE=${CONTRASTIVE_TEMPERATURE}
EOF

methods=(balanced_control cmmd prototype contrastive)
overwrite_args=()
if [[ "${SKIP_EXISTING}" != "1" ]]; then
  overwrite_args=(--overwrite)
fi

checkpoint_for() {
  local method="$1"
  printf '%s\n' "checkpoints/thubenchmark_${MODEL}_exp026_${TAG}_${method}_best.pth"
}

history_for() {
  local method="$1"
  printf '%s\n' "${MODEL_ROOT}/${method}/finetune"
}

attack_for() {
  local method="$1"
  printf '%s\n' "ad_data/exp026/${TAG}_${method}_autoattack.pth"
}

purification_for() {
  local method="$1"
  printf '%s\n' "purified_data/exp026/eval/${TAG}_${method}_rank25-30.pth"
}

should_run() {
  local stage="$1"
  (( stage >= START_STAGE && stage <= STOP_STAGE ))
}

run_logged() {
  local log="$1"
  shift
  echo "[$(date -Is)] $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  "$@" 2>&1 | tee -a "${log}"
}

if should_run 1; then
  for method in "${methods[@]}"; do
    method_root="${MODEL_ROOT}/${method}"
    mkdir -p "${method_root}"
    checkpoint="$(checkpoint_for "${method}")"
    history="$(history_for "${method}")"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${checkpoint}" && -f "${history}.json" ]]; then
      echo "[$(date -Is)] Reuse ${MODEL}/${method} training artifacts."
      continue
    fi
    feature_args=()
    case "${method}" in
      balanced_control) ;;
      cmmd)
        feature_args=(--feature_objective cmmd --feature_loss_weight "${CMMD_WEIGHT}")
        ;;
      prototype)
        feature_args=(
          --feature_objective prototype --feature_loss_weight "${PROTOTYPE_WEIGHT}"
          --prototype_margin "${PROTOTYPE_MARGIN}"
          --prototype_margin_weight "${PROTOTYPE_MARGIN_WEIGHT}"
        )
        ;;
      contrastive)
        feature_args=(
          --feature_objective contrastive --feature_loss_weight "${CONTRASTIVE_WEIGHT}"
          --contrastive_temperature "${CONTRASTIVE_TEMPERATURE}"
        )
        ;;
    esac
    smoke_args=()
    if [[ "${SMOKE}" == "1" ]]; then
      smoke_args=(--online_train_sample_num 8 --max_cache_batches 1 --pgd_steps 1)
    fi
    run_logged "${method_root}/train.log" \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u -m rpcf.finetune \
      --cache_path "${RPCF_CACHE}" --sensitivity_path "${SENSITIVITY_PATH}" \
      --checkpoint_path "${AT_CHECKPOINT}" --output_checkpoint "${checkpoint}" \
      --dataset thubenchmark --model "${MODEL}" --fold 0 --seed 42 \
      --epsilon 0.03 --epochs "${EPOCHS}" --batch_size 8 \
      --eval_batch_size "${EVAL_BATCH_SIZE}" --lr 0.0001 --weight_decay 0.0001 \
      --rank_temperature 0.5 --consistancy_temperature 2.0 --pgd_steps 10 \
      --online_madry_at --online_at_batch_size "${ONLINE_AT_BATCH_SIZE}" \
      --online_at_pgd_steps 10 --online_at_step_size 0.006 \
      --balanced_classes_per_batch 4 --balanced_samples_per_class 2 \
      --gpu_id "${GPU_ID}" --history_prefix "${history}" \
      "${feature_args[@]}" "${smoke_args[@]}"
  done
fi

if should_run 2; then
  for method in "${methods[@]}"; do
    checkpoint="$(checkpoint_for "${method}")"
    output="$(attack_for "${method}")"
    [[ "${DRY_RUN}" == "1" || -f "${checkpoint}" ]] || { echo "Missing ${checkpoint}" >&2; exit 1; }
    if [[ "${SKIP_EXISTING}" == "1" && -f "${output}" ]]; then
      continue
    fi
    attack_subset=()
    [[ "${SMOKE}" == "1" ]] && attack_subset=(--sample_num 2)
    run_logged "${MODEL_ROOT}/${method}/attack.log" \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.evaluate_attack --dataset thubenchmark --model "${MODEL}" \
      --fold 0 --seed 42 --checkpoint_path "${checkpoint}" \
      --method_tag "${method}" --attack autoattack --eps 0.03 \
      --batch_size "${ATTACK_BATCH_SIZE}" --gpu_id "${GPU_ID}" \
      --output_path "${output}" "${attack_subset[@]}" "${overwrite_args[@]}"
  done
fi

if should_run 3; then
  for method in "${methods[@]}"; do
    checkpoint="$(checkpoint_for "${method}")"
    attack_path="$(attack_for "${method}")"
    output="$(purification_for "${method}")"
    [[ "${DRY_RUN}" == "1" || -f "${attack_path}" ]] || { echo "Missing ${attack_path}" >&2; exit 1; }
    if [[ "${SKIP_EXISTING}" == "1" && -f "${output}" ]]; then
      continue
    fi
    run_logged "${MODEL_ROOT}/${method}/purification.log" \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.evaluate_purification --attack_path "${attack_path}" \
      --checkpoint_path "${checkpoint}" --dataset thubenchmark --model "${MODEL}" \
      --fold 0 --seed 42 --eps 0.03 --sample_num "${EVAL_SAMPLE_NUM}" \
      --ranks 25,30 \
      --configs PTR3d_8_2048_rank25_3d_interpolate.yaml,PTR3d_8_2048_rank30_3d_interpolate.yaml \
      --gpu_id "${GPU_ID}" --output_path "${output}" "${overwrite_args[@]}"
  done
fi

if should_run 4; then
  attack_pairs=()
  purification_pairs=()
  history_pairs=()
  for method in "${methods[@]}"; do
    attack_pairs+=("${method}=$(attack_for "${method}")")
    purification_pairs+=("${method}=$(purification_for "${method}")")
    history_pairs+=("${method}=$(history_for "${method}").json")
  done
  run_logged "${MODEL_ROOT}/summary.log" \
    conda run -n "${CONDA_ENV}" --no-capture-output python -u \
    -m rpcf.compare_exp026 --dataset thubenchmark --model "${MODEL}" \
    --seed 42 --fold 0 --eps 0.03 --sample_num "${EVAL_SAMPLE_NUM}" \
    --attack_paths "${attack_pairs[@]}" \
    --purification_paths "${purification_pairs[@]}" \
    --history_paths "${history_pairs[@]}" \
    --sensitivity_path "${SENSITIVITY_PATH}" --context_summary "${CONTEXT_SUMMARY}" \
    --output_dir "${MODEL_ROOT}/comparison"
fi

echo "[$(date -Is)] EXP-026 backbone finished: ${MODEL}"
