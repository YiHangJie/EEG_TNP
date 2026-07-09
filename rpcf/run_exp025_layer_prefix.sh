#!/usr/bin/env bash
set -euo pipefail

# EXP-025：RPCF_AT 敏感层降序累加微调预算曲线。
#
# 推荐正式运行：
#   EXP025_RUN_ID=exp025_layer_prefix_seed42_$(date +%Y%m%d_%H%M%S) \
#   EXP025_MODELS="eegnet tsception conformer atcnet" \
#   CUDA_VISIBLE_DEVICES=0 GPU_ID=0 \
#   nohup setsid bash rpcf/run_exp025_layer_prefix.sh \
#     > logs/exp025/exp025_layer_prefix_seed42_YYYYMMDD_HHMMSS/controller.log 2>&1 &
#
# 常用续跑：
#   START_STAGE=2 STOP_STAGE=4 EXP025_RUN_ID=<same_run_id> bash rpcf/run_exp025_layer_prefix.sh

DATASET="${EXP025_DATASET:-thubenchmark}"
SEED="${EXP025_SEED:-42}"
FOLD="${EXP025_FOLD:-0}"
EPS="${EXP025_EPS:-0.03}"
GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-torch}"
RUN_ID="${EXP025_RUN_ID:-exp025_layer_prefix_seed${SEED}_fold${FOLD}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${EXP025_LOG_ROOT:-logs/exp025/${RUN_ID}}"
MODELS="${EXP025_MODELS:-eegnet tsception conformer atcnet}"
BUDGET_FILTER="${EXP025_BUDGET_FILTER:-}"

START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"

RPCF_EPOCHS="${RPCF_EPOCHS:-100}"
RPCF_BATCH_SIZE="${RPCF_BATCH_SIZE:-64}"
RPCF_EVAL_BATCH_SIZE="${RPCF_EVAL_BATCH_SIZE:-128}"
ONLINE_AT_BATCH_SIZE="${ONLINE_AT_BATCH_SIZE:-128}"
ONLINE_AT_TRAIN_SAMPLE_NUM="${ONLINE_AT_TRAIN_SAMPLE_NUM:-}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-32}"
ATTACK_SAMPLE_NUM="${ATTACK_SAMPLE_NUM:-}"
EVAL_SAMPLE_NUM="${EVAL_SAMPLE_NUM:-512}"
RPCF_LR="${RPCF_LR:-0.0001}"
RPCF_WEIGHT_DECAY="${RPCF_WEIGHT_DECAY:-0.0001}"
ATTACK="${EXP025_ATTACK:-autoattack}"
PROTOCOL="train_only_subject_no_ea_subject_split"

EVAL_RANKS="${EXP025_EVAL_RANKS:-25 30}"
TRAIN_CACHE_SAMPLE_NUM="${TRAIN_CACHE_SAMPLE_NUM:-512}"

if [[ "${SMOKE}" == "1" ]]; then
  RPCF_EPOCHS="${SMOKE_EPOCHS:-1}"
  RPCF_BATCH_SIZE="${SMOKE_RPCF_BATCH_SIZE:-2}"
  RPCF_EVAL_BATCH_SIZE="${SMOKE_EVAL_BATCH_SIZE:-2}"
  ONLINE_AT_BATCH_SIZE="${SMOKE_ONLINE_AT_BATCH_SIZE:-2}"
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

mkdir -p "${LOG_ROOT}" checkpoints ad_data/exp025 purified_data/exp025/eval
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
mkdir -p "${NUMBA_CACHE_DIR}" "${XDG_CACHE_HOME}"
printf '%s\n' "$$" > "${LOG_ROOT}/controller.pid"

run_logged() {
  local log_path="$1"
  shift
  echo "[$(date -Is)] $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  "$@" 2>&1 | tee -a "${log_path}"
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

exp024_tag_for_model() {
  local model="$1"
  case "${model}" in
    tsception) printf '%s\n' "exp024_parallel_seed42_20260702_153337_tsception" ;;
    conformer) printf '%s\n' "exp024_parallel_seed42_20260702_153337_conformer_retry2" ;;
    atcnet) printf '%s\n' "exp024_parallel_seed42_20260702_153337_atcnet_retry2" ;;
    deepconvnet|tcnet)
      printf '%s\n' "exp024_deepconvnet_tcnet_seed42_20260707_1435_retry1_${model}"
      ;;
    *) printf '%s\n' "" ;;
  esac
}

at_checkpoint_for_model() {
  local model="$1"
  local env_name="EXP025_AT_CHECKPOINT_${model^^}"
  local override="${!env_name:-}"
  if [[ -n "${override}" ]]; then
    printf '%s\n' "${override}"
    return
  fi
  if [[ "${model}" == "eegnet" ]]; then
    echo "Set EXP025_AT_CHECKPOINT_EEGNET to the EEGNet Madry AT checkpoint path." >&2
    return 1
  fi
  local tag
  tag="$(exp024_tag_for_model "${model}")"
  printf '%s\n' "checkpoints/${DATASET}_${model}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${tag}_madry_at_best.pth"
}

cache_for_model() {
  local model="$1"
  local env_name="EXP025_CACHE_${model^^}"
  local override="${!env_name:-}"
  if [[ -n "${override}" ]]; then
    printf '%s\n' "${override}"
    return
  fi
  if [[ "${model}" == "eegnet" ]]; then
    echo "Set EXP025_CACHE_EEGNET to the EEGNet six-rank RPCF train cache path." >&2
    return 1
  fi
  local tag
  tag="$(exp024_tag_for_model "${model}")"
  printf '%s\n' "purified_data/exp024/rpcf_train/${tag}_six_rank.pth"
}

sensitivity_for_model() {
  local model="$1"
  local env_name="EXP025_SENSITIVITY_${model^^}"
  local override="${!env_name:-}"
  if [[ -n "${override}" ]]; then
    printf '%s\n' "${override}"
    return
  fi
  if [[ "${model}" == "eegnet" ]]; then
    echo "Set EXP025_SENSITIVITY_EEGNET to the EEGNet sensitivity.json path." >&2
    return 1
  fi
  local tag
  tag="$(exp024_tag_for_model "${model}")"
  printf '%s\n' "logs/exp024/${tag}/sensitivity.json"
}

rank_config() {
  local rank="$1"
  printf '%s\n' "PTR3d_8_2048_rank${rank}_3d_interpolate.yaml"
}

budget_lines() {
  local sensitivity_path="$1"
  python3 -c 'import json,sys
path=sys.argv[1]
with open(path, encoding="utf-8") as f:
    data=json.load(f)
layers=data.get("layers") or {}
ordered=sorted(layers, key=lambda name: (-float(layers[name]["score"]), name))
if not ordered:
    raise SystemExit("no layers in sensitivity artifact")
for index in range(1, len(ordered)+1):
    print(f"{index}\t" + ",".join(ordered[:index]))' "${sensitivity_path}"
}

write_budget_config() {
  local output_path="$1"
  local model="$2"
  local budget_id="$3"
  local prefix_k="$4"
  local selected_layers="$5"
  local checkpoint_path="$6"
  local sensitivity_path="$7"
  python3 -c 'import json,sys
out,model,budget,k,layers,ckpt,sens=sys.argv[1:]
payload={
  "experiment_id":"EXP-025",
  "model":model,
  "budget_id":budget,
  "prefix_k":int(k),
  "selected_layers":[item for item in layers.split(",") if item],
  "checkpoint_path":ckpt,
  "sensitivity_path":sens,
  "layer_selection_rule":"score_prefix",
}
with open(out,"w",encoding="utf-8") as f:
    json.dump(payload,f,ensure_ascii=False,indent=2)' \
    "${output_path}" "${model}" "${budget_id}" "${prefix_k}" "${selected_layers}" \
    "${checkpoint_path}" "${sensitivity_path}"
}

cat > "${LOG_ROOT}/run_config.txt" <<EOF
EXPERIMENT_ID=EXP-025
RUN_ID=${RUN_ID}
DATASET=${DATASET}
MODELS=${MODELS}
SEED=${SEED}
FOLD=${FOLD}
EPS=${EPS}
ATTACK=${ATTACK}
EVAL_SAMPLE_NUM=${EVAL_SAMPLE_NUM}
EVAL_RANKS=${EVAL_RANKS}
EXP025_BUDGET_FILTER=${BUDGET_FILTER}
LOG_ROOT=${LOG_ROOT}
EOF

for model in ${MODELS}; do
  case "${model}" in
    eegnet|tsception|conformer|atcnet|deepconvnet|tcnet) ;;
    *) echo "Unsupported EXP-025 model: ${model}" >&2; exit 1 ;;
  esac
  model_root="${LOG_ROOT}/${model}"
  mkdir -p "${model_root}"
  at_checkpoint="$(at_checkpoint_for_model "${model}")"
  rpcf_cache="$(cache_for_model "${model}")"
  sensitivity_path="$(sensitivity_for_model "${model}")"
  require_artifact "${at_checkpoint}"
  require_artifact "${rpcf_cache}"
  require_artifact "${sensitivity_path}"

  echo "[$(date -Is)] EXP-025 ${model}: sensitivity=${sensitivity_path}"
  mapfile -t budgets < <(budget_lines "${sensitivity_path}")
  if [[ -n "${BUDGET_FILTER}" ]]; then
    case "${BUDGET_FILTER}" in
      max)
        budgets=("${budgets[$((${#budgets[@]} - 1))]}")
        ;;
      *)
        filtered_budgets=()
        IFS=',' read -r -a wanted_budgets <<< "${BUDGET_FILTER}"
        max_prefix="${budgets[$((${#budgets[@]} - 1))]%%$'\t'*}"
        for line in "${budgets[@]}"; do
          prefix_k="${line%%$'	'*}"
          for wanted in "${wanted_budgets[@]}"; do
            if [[ "${prefix_k}" == "${wanted}" || ( "${wanted}" == "max" && "${prefix_k}" == "${max_prefix}" ) ]]; then
              filtered_budgets+=("${line}")
              break
            fi
          done
        done
        budgets=("${filtered_budgets[@]}")
        ;;
    esac
  fi
  if [[ "${#budgets[@]}" -eq 0 ]]; then
    echo "No budgets selected for ${model}; EXP025_BUDGET_FILTER=${BUDGET_FILTER}" >&2
    exit 1
  fi
  for line in "${budgets[@]}"; do
    prefix_k="${line%%$'\t'*}"
    selected_layers="${line#*$'\t'}"
    budget_id="budget_${prefix_k}"
    budget_dir="${model_root}/${budget_id}"
    mkdir -p "${budget_dir}"
    budget_tag="${RUN_ID}_${model}_${budget_id}_rpcf_at"
    checkpoint_path="checkpoints/${DATASET}_${model}_${PROTOCOL}_madry_eps${EPS}_${SEED}_fold${FOLD}_${budget_tag}_best.pth"
    history_prefix="${budget_dir}/finetune_rpcf_at"
    attack_path="${budget_dir}/rpcf_at_${ATTACK}.pth"

    write_budget_config \
      "${budget_dir}/budget_config.json" "${model}" "${budget_id}" \
      "${prefix_k}" "${selected_layers}" "${checkpoint_path}" "${sensitivity_path}"

    if should_run 1; then
      if [[ "${SKIP_EXISTING}" == "1" && -f "${checkpoint_path}" && -f "${history_prefix}.json" ]]; then
        echo "[Stage 1] Reuse ${model} ${budget_id} checkpoint: ${checkpoint_path}"
      else
        run_logged "${budget_dir}/stage1_finetune.log" \
          conda run -n "${CONDA_ENV}" --no-capture-output python -u \
          -m rpcf.finetune \
          --cache_path "${rpcf_cache}" --sensitivity_path "${sensitivity_path}" \
          --selected_layers_override "${selected_layers}" \
          --layer_selection_rule score_prefix --prefix_k "${prefix_k}" \
          --checkpoint_path "${at_checkpoint}" --output_checkpoint "${checkpoint_path}" \
          --dataset "${DATASET}" --model "${model}" --fold "${FOLD}" \
          --seed "${SEED}" --epsilon "${EPS}" --epochs "${RPCF_EPOCHS}" \
          --batch_size "${RPCF_BATCH_SIZE}" --eval_batch_size "${RPCF_EVAL_BATCH_SIZE}" \
          --lr "${RPCF_LR}" --weight_decay "${RPCF_WEIGHT_DECAY}" \
          --rank_temperature 0.5 --consistancy_temperature 2.0 --pgd_steps 10 \
          --online_madry_at --online_at_batch_size "${ONLINE_AT_BATCH_SIZE}" \
          --online_at_pgd_steps 10 --online_at_step_size 0.006 \
          "${online_subset_args[@]}" --gpu_id "${GPU_ID}" \
          --history_prefix "${history_prefix}"
      fi
    fi

    if should_run 2; then
      require_artifact "${checkpoint_path}"
      if [[ "${SKIP_EXISTING}" == "1" && -f "${attack_path}" ]]; then
        echo "[Stage 2] Reuse ${model} ${budget_id} attack: ${attack_path}"
      else
        run_logged "${budget_dir}/stage2_attack.log" \
          conda run -n "${CONDA_ENV}" --no-capture-output python -u \
          -m rpcf.evaluate_attack \
          --dataset "${DATASET}" --model "${model}" --fold "${FOLD}" \
          --seed "${SEED}" --checkpoint_path "${checkpoint_path}" \
          --method_tag "exp025_${budget_id}_rpcf_at" --attack "${ATTACK}" \
          --eps "${EPS}" --batch_size "${ATTACK_BATCH_SIZE}" \
          --gpu_id "${GPU_ID}" --output_path "${attack_path}" \
          "${attack_subset_args[@]}" "${overwrite_args[@]}"
      fi
    fi

    if should_run 3; then
      require_artifact "${checkpoint_path}"
      require_artifact "${attack_path}"
      pids=()
      for rank in ${EVAL_RANKS}; do
        output_path="${budget_dir}/rpcf_at_rank${rank}.pth"
        if [[ "${SKIP_EXISTING}" == "1" && -f "${output_path}" ]]; then
          echo "[Stage 3] Reuse ${model} ${budget_id} rank${rank}: ${output_path}"
          continue
        fi
        run_logged "${budget_dir}/stage3_purify_rank${rank}.log" \
          conda run -n "${CONDA_ENV}" --no-capture-output python -u \
          -m rpcf.evaluate_purification \
          --attack_path "${attack_path}" --checkpoint_path "${checkpoint_path}" \
          --dataset "${DATASET}" --model "${model}" --fold "${FOLD}" \
          --seed "${SEED}" --eps "${EPS}" --sample_num "${EVAL_SAMPLE_NUM}" \
          --ranks "${rank}" --configs "$(rank_config "${rank}")" \
          --gpu_id "${GPU_ID}" --output_path "${output_path}" \
          "${overwrite_args[@]}" &
        pids+=("$!")
      done
      for pid in "${pids[@]}"; do
        wait "${pid}"
      done
    fi
  done

  if should_run 4; then
    run_logged "${model_root}/stage4_summary.log" \
      conda run -n "${CONDA_ENV}" --no-capture-output python -u \
      -m rpcf.compare_exp025 \
      --dataset "${DATASET}" --model "${model}" --seed "${SEED}" \
      --fold "${FOLD}" --eps "${EPS}" --sample_num "${EVAL_SAMPLE_NUM}" \
      --run_root "${model_root}" --output_dir "${model_root}/comparison"
  fi
done

echo "[$(date -Is)] EXP-025 layer prefix pipeline finished: ${RUN_ID}"
