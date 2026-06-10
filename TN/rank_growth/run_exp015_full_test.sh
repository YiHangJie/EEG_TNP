#!/usr/bin/env bash
set -euo pipefail

# EXP-015 full-matrix runner.
# 默认先用 EXP015_SMOKE=1 跑小规模验证；全量运行时设置 EXP015_SMOKE=0。

EXP015_SMOKE="${EXP015_SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONDA_ENV="${CONDA_ENV:-torch}"
GPU_IDS="${GPU_IDS:-0}"
MAX_JOBS="${MAX_JOBS:-1}"
INNER_MAX_JOBS="${INNER_MAX_JOBS:-1}"
POLL_SECONDS="${POLL_SECONDS:-2}"
START_PHASE="${START_PHASE:-1}"
STOP_PHASE="${STOP_PHASE:-4}"

SAMPLE_NUM="${SAMPLE_NUM:-512}"
EPOCHS="${EPOCHS:-400}"
PATIENCE="${PATIENCE:-20}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
BATCH_SIZE_EEGNET="${BATCH_SIZE_EEGNET:-128}"
BATCH_SIZE_CONFORMER="${BATCH_SIZE_CONFORMER:-64}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-32}"
ATTACK_SAMPLE_NUM="${ATTACK_SAMPLE_NUM:-}"
TRAIN_SAMPLE_NUM="${TRAIN_SAMPLE_NUM:-}"

RANK_GROWTH_CONFIG="${RANK_GROWTH_CONFIG:-PTR3d_rank_growth_js_mse_exp015_8_2048_r5-40_3d_interpolate.yaml}"
TRAIN_CONFIGS_CSV="${TRAIN_CONFIGS_CSV:-PTR3d_8_2048_rank25_3d_interpolate.yaml,PTR3d_8_2048_rank30_3d_interpolate.yaml}"
BASELINE_STRATEGIES_CSV="${BASELINE_STRATEGIES_CSV:-madry,trades,fbf}"

DATASETS_CSV="${DATASETS_CSV:-thubenchmark,seediv}"
MODELS_CSV="${MODELS_CSV:-eegnet,conformer}"
SEEDS_CSV="${SEEDS_CSV:-42,43,45}"
FOLDS_CSV="${FOLDS_CSV:-0,1,2}"
EPSES_CSV="${EPSES_CSV:-0.01,0.03,0.05}"

if [[ "${EXP015_SMOKE}" == "1" ]]; then
  SAMPLE_NUM="${SMOKE_SAMPLE_NUM:-2}"
  EPOCHS="${SMOKE_EPOCHS:-1}"
  PATIENCE="${SMOKE_PATIENCE:-1}"
  SEEDS_CSV="${SMOKE_SEEDS_CSV:-42}"
  FOLDS_CSV="${SMOKE_FOLDS_CSV:-0}"
  EPSES_CSV="${SMOKE_EPSES_CSV:-0.01}"
  BASELINE_STRATEGIES_CSV="${SMOKE_BASELINE_STRATEGIES_CSV:-madry}"
  TRAIN_SAMPLE_NUM="${SMOKE_TRAIN_SAMPLE_NUM:-64}"
  ATTACK_SAMPLE_NUM="${SMOKE_ATTACK_SAMPLE_NUM:-2}"
  MAX_JOBS="${SMOKE_MAX_JOBS:-1}"
fi

RUN_ID="${EXP015_RUN_ID:-exp015_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${EXP015_LOG_ROOT:-logs/exp015/${RUN_ID}}"
TASKS_CSV="${LOG_ROOT}/planned_tasks.csv"

split_csv() {
  local csv="$1"
  local -n out_ref="$2"
  local item
  IFS=',' read -r -a out_ref <<< "${csv}"
  for item in "${!out_ref[@]}"; do
    out_ref["${item}"]="${out_ref[${item}]//[[:space:]]/}"
  done
}

safe_token() {
  local value="$1"
  if [[ -z "${value}" ]]; then
    echo "none"
    return
  fi
  printf '%s' "${value}" | sed 's/[^A-Za-z0-9_.-]/_/g'
}

eps_tag() {
  printf '%s' "$1" | sed 's/\./p/g'
}

model_batch_size() {
  if [[ "$1" == "conformer" || "$1" == "conformer_ea_forward" ]]; then
    echo "${BATCH_SIZE_CONFORMER}"
  else
    echo "${BATCH_SIZE_EEGNET}"
  fi
}

clean_checkpoint_path() {
  local dataset="$1"
  local model="$2"
  local seed="$3"
  local fold="$4"
  echo "checkpoints/${dataset}_${model}_train_only_subject_no_ea_subject_split_clean_eps0_${seed}_fold${fold}_best.pth"
}

baseline_adv_path() {
  local dataset="$1"
  local model="$2"
  local strategy="$3"
  local eps="$4"
  local seed="$5"
  local fold="$6"
  echo "ad_data/${dataset}_${model}_no_ea_${strategy}_autoattack_eps${eps}_seed${seed}_fold${fold}.pth"
}

baseline_checkpoint_path() {
  local dataset="$1"
  local model="$2"
  local strategy="$3"
  local eps="$4"
  local seed="$5"
  local fold="$6"
  echo "checkpoints/${dataset}_${model}_train_only_subject_no_ea_subject_split_${strategy}_eps${eps}_${seed}_fold${fold}_best.pth"
}

abat_model_name() {
  if [[ "$1" == "conformer" ]]; then
    echo "conformer_ea_forward"
  else
    echo "eegnet_ea_forward"
  fi
}

abat_checkpoint_path() {
  local dataset="$1"
  local ea_model="$2"
  local eps="$3"
  local seed="$4"
  local fold="$5"
  echo "checkpoints/${dataset}_${ea_model}_train_only_subject_no_ea_subject_split_madry_eps${eps}_${seed}_fold${fold}_best.pth"
}

abat_adv_path() {
  local dataset="$1"
  local ea_model="$2"
  local eps="$3"
  local seed="$4"
  local fold="$5"
  echo "ad_data/${dataset}_${ea_model}_no_ea_ea_forward_madry_autoattack_eps${eps}_seed${seed}_fold${fold}.pth"
}

main_checkpoint_path() {
  local dataset="$1"
  local model="$2"
  local eps="$3"
  local seed="$4"
  local fold="$5"
  local run_tag="$6"
  echo "checkpoints/${dataset}_${model}_train_only_subject_no_ea_subject_split_madry_eps${eps}_${seed}_fold${fold}_${run_tag}_best.pth"
}

main_adv_path() {
  local dataset="$1"
  local model="$2"
  local eps="$3"
  local seed="$4"
  local fold="$5"
  local run_tag="$6"
  echo "ad_data/${dataset}_${model}_no_ea_${run_tag}_madry_autoattack_eps${eps}_seed${seed}_fold${fold}.pth"
}

main_purified_ad_path() {
  local dataset="$1"
  local model="$2"
  local eps="$3"
  local seed="$4"
  local fold="$5"
  local run_tag="$6"
  local config_stem="${RANK_GROWTH_CONFIG%.yaml}"
  echo "purified_data/attacked/${dataset}_${model}_no_ea_${run_tag}_autoattack_eps${eps}_seed${seed}_fold${fold}_${config_stem}_n${SAMPLE_NUM}_ad.pth"
}

wait_for_slots() {
  while [[ "$(jobs -rp | wc -l)" -ge "${MAX_JOBS}" ]]; do
    sleep "${POLL_SECONDS}"
  done
}

wait_for_phase() {
  local phase_name="$1"
  shift
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      echo "[$(date -Is)] ${phase_name} failed: pid=${pid}" >&2
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "[$(date -Is)] ${phase_name} failed; stop EXP-015." >&2
    exit 1
  fi
}

record_task() {
  local phase="$1"
  local method="$2"
  local dataset="$3"
  local model="$4"
  local seed="$5"
  local fold="$6"
  local eps="$7"
  local log_path="$8"
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${phase}" "${method}" "${dataset}" "${model}" "${seed}" "${fold}" "${eps}" "${log_path}" >> "${TASKS_CSV}"
}

run_clean_task() {
  local dataset="$1"
  local model="$2"
  local seed="$3"
  local fold="$4"
  local gpu="$5"
  local log_path="$6"
  local batch_size
  batch_size="$(model_batch_size "${model}")"
  (
    set -euo pipefail
    train_sample_args=()
    if [[ -n "${TRAIN_SAMPLE_NUM}" ]]; then
      train_sample_args=(--train_sample_num "${TRAIN_SAMPLE_NUM}")
    fi
    echo "[$(date -Is)] clean start dataset=${dataset} model=${model} seed=${seed} fold=${fold} gpu=${gpu}"
    echo "checkpoint=$(clean_checkpoint_path "${dataset}" "${model}" "${seed}" "${fold}")"
    if [[ "${DRY_RUN}" != "1" ]]; then
      conda run -n "${CONDA_ENV}" --no-capture-output python -u train_AT.py \
        --dataset "${dataset}" \
        --model "${model}" \
        --at_strategy clean \
        --fold "${fold}" \
        --epsilon 0 \
        --epochs "${EPOCHS}" \
        --batch_size "${batch_size}" \
        "${train_sample_args[@]}" \
        --lr "${LR}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --patience "${PATIENCE}" \
        --seed "${seed}" \
        --gpu_id "${gpu}" \
        --no_ea
    fi
    echo "[$(date -Is)] clean finished"
  ) > "${log_path}" 2>&1
}

run_main_task() {
  local dataset="$1"
  local model="$2"
  local seed="$3"
  local fold="$4"
  local eps="$5"
  local gpu="$6"
  local log_path="$7"
  local run_tag="consistancy_rank25-30_n${SAMPLE_NUM}_eps$(eps_tag "${eps}")"
  local clean_ckpt
  local batch_size
  local main_ckpt
  local main_adv
  local main_start_stage="1"
  clean_ckpt="$(clean_checkpoint_path "${dataset}" "${model}" "${seed}" "${fold}")"
  batch_size="$(model_batch_size "${model}")"
  main_ckpt="$(main_checkpoint_path "${dataset}" "${model}" "${eps}" "${seed}" "${fold}" "${run_tag}")"
  main_adv="$(main_adv_path "${dataset}" "${model}" "${eps}" "${seed}" "${fold}" "${run_tag}")"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${main_ckpt}" && -f "${main_adv}" ]]; then
    main_start_stage="4"
  elif [[ "${SKIP_EXISTING}" == "1" && -f "${main_ckpt}" ]]; then
    main_start_stage="3"
  fi
  (
    set -euo pipefail
    echo "[$(date -Is)] main start dataset=${dataset} model=${model} seed=${seed} fold=${fold} eps=${eps} gpu=${gpu}"
    echo "run_tag=${run_tag}"
    echo "pipeline_start_stage=${main_start_stage}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      DATASET="${dataset}" \
      MODEL="${model}" \
      GPU_ID="${gpu}" \
      GPU_IDS="${gpu}" \
      EPS="${eps}" \
      ATTACK="autoattack" \
      AT_STRATEGY="madry" \
      SEED="${seed}" \
      FOLD="${fold}" \
      SAMPLE_NUM="${SAMPLE_NUM}" \
      RUN_TAG="${run_tag}" \
      USE_EA="0" \
      OVERWRITE="0" \
      DRY_RUN="0" \
      START_STAGE="${main_start_stage}" \
      MAX_JOBS="${INNER_MAX_JOBS}" \
      TRAIN_ADV_CHECKPOINT_PATH="${clean_ckpt}" \
      TRAIN_ADV_BATCH_SIZE="${ATTACK_BATCH_SIZE}" \
      ATTACK_SAMPLE_NUM="${ATTACK_SAMPLE_NUM}" \
      TRAIN_SAMPLE_NUM="${TRAIN_SAMPLE_NUM}" \
      EPOCHS="${EPOCHS}" \
      BATCH_SIZE="${batch_size}" \
      LR="${LR}" \
      WEIGHT_DECAY="${WEIGHT_DECAY}" \
      PATIENCE="${PATIENCE}" \
      TRAIN_CONFIGS_CSV="${TRAIN_CONFIGS_CSV}" \
      PURIFY_CONFIGS_CSV="${RANK_GROWTH_CONFIG}" \
      CONDA_ENV="${CONDA_ENV}" \
      ACTIVATE_CONDA="1" \
      bash purify_aug_consistancy_pipeline.sh
    fi
    echo "[$(date -Is)] main finished"
  ) > "${log_path}" 2>&1
}

run_baseline_task() {
  local dataset="$1"
  local model="$2"
  local strategy="$3"
  local seed="$4"
  local fold="$5"
  local eps="$6"
  local gpu="$7"
  local log_path="$8"
  local batch_size
  local ckpt
  batch_size="$(model_batch_size "${model}")"
  ckpt="$(baseline_checkpoint_path "${dataset}" "${model}" "${strategy}" "${eps}" "${seed}" "${fold}")"
  (
    set -euo pipefail
    train_sample_args=()
    if [[ -n "${TRAIN_SAMPLE_NUM}" ]]; then
      train_sample_args=(--train_sample_num "${TRAIN_SAMPLE_NUM}")
    fi
    attack_sample_args=()
    if [[ -n "${ATTACK_SAMPLE_NUM}" ]]; then
      attack_sample_args=(--attack_sample_num "${ATTACK_SAMPLE_NUM}")
    fi
    echo "[$(date -Is)] baseline start strategy=${strategy} dataset=${dataset} model=${model} seed=${seed} fold=${fold} eps=${eps} gpu=${gpu}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      if [[ "${SKIP_EXISTING}" == "1" && -f "${ckpt}" ]]; then
        echo "[$(date -Is)] skip existing baseline checkpoint ${ckpt}"
      else
        conda run -n "${CONDA_ENV}" --no-capture-output python -u train_AT.py \
          --dataset "${dataset}" \
          --model "${model}" \
          --at_strategy "${strategy}" \
          --fold "${fold}" \
          --epsilon "${eps}" \
          --epochs "${EPOCHS}" \
          --batch_size "${batch_size}" \
          "${train_sample_args[@]}" \
          --lr "${LR}" \
          --weight_decay "${WEIGHT_DECAY}" \
          --patience "${PATIENCE}" \
          --seed "${seed}" \
          --gpu_id "${gpu}" \
          --no_ea
      fi
      conda run -n "${CONDA_ENV}" --no-capture-output python -u attack.py \
        --dataset "${dataset}" \
        --model "${model}" \
        --at_strategy "${strategy}" \
        --fold "${fold}" \
        --attack autoattack \
        --eps "${eps}" \
        --batch_size "${ATTACK_BATCH_SIZE}" \
        --seed "${seed}" \
        --gpu_id "${gpu}" \
        --no_ea \
        "${attack_sample_args[@]}" \
        --save_adv
    fi
    echo "[$(date -Is)] baseline finished"
  ) > "${log_path}" 2>&1
}

run_abat_task() {
  local dataset="$1"
  local base_model="$2"
  local seed="$3"
  local fold="$4"
  local eps="$5"
  local gpu="$6"
  local log_path="$7"
  local ea_model
  local batch_size
  local ckpt
  ea_model="$(abat_model_name "${base_model}")"
  batch_size="$(model_batch_size "${base_model}")"
  ckpt="$(abat_checkpoint_path "${dataset}" "${ea_model}" "${eps}" "${seed}" "${fold}")"
  (
    set -euo pipefail
    train_sample_args=()
    if [[ -n "${TRAIN_SAMPLE_NUM}" ]]; then
      train_sample_args=(--train_sample_num "${TRAIN_SAMPLE_NUM}")
    fi
    attack_sample_args=()
    if [[ -n "${ATTACK_SAMPLE_NUM}" ]]; then
      attack_sample_args=(--attack_sample_num "${ATTACK_SAMPLE_NUM}")
    fi
    echo "[$(date -Is)] abat start dataset=${dataset} model=${ea_model} seed=${seed} fold=${fold} eps=${eps} gpu=${gpu}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      if [[ "${SKIP_EXISTING}" == "1" && -f "${ckpt}" ]]; then
        echo "[$(date -Is)] skip existing ABAT checkpoint ${ckpt}"
      else
        conda run -n "${CONDA_ENV}" --no-capture-output python -u train_AT_ea_forward.py \
          --dataset "${dataset}" \
          --model "${ea_model}" \
          --at_strategy madry \
          --fold "${fold}" \
          --epsilon "${eps}" \
          --epochs "${EPOCHS}" \
          --batch_size "${batch_size}" \
          "${train_sample_args[@]}" \
          --lr "${LR}" \
          --weight_decay "${WEIGHT_DECAY}" \
          --patience "${PATIENCE}" \
          --seed "${seed}" \
          --gpu_id "${gpu}" \
          --no_ea
      fi
      conda run -n "${CONDA_ENV}" --no-capture-output python -u attack_ea_forward.py \
        --dataset "${dataset}" \
        --model "${ea_model}" \
        --at_strategy madry \
        --fold "${fold}" \
        --attack autoattack \
        --eps "${eps}" \
        --batch_size "${ATTACK_BATCH_SIZE}" \
        --seed "${seed}" \
        --gpu_id "${gpu}" \
        --no_ea \
        "${attack_sample_args[@]}" \
        --save_adv
    fi
    echo "[$(date -Is)] abat finished"
  ) > "${log_path}" 2>&1
}

split_csv "${DATASETS_CSV}" DATASETS
split_csv "${MODELS_CSV}" MODELS
split_csv "${SEEDS_CSV}" SEEDS
split_csv "${FOLDS_CSV}" FOLDS
split_csv "${EPSES_CSV}" EPSES
split_csv "${BASELINE_STRATEGIES_CSV}" BASELINE_STRATEGIES
split_csv "${GPU_IDS}" GPU_ID_LIST

if [[ "${#GPU_ID_LIST[@]}" -eq 0 ]]; then
  echo "GPU_IDS must contain at least one GPU id." >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"
printf 'phase,method,dataset,model,seed,fold,eps,log_path\n' > "${TASKS_CSV}"

echo "EXP-015 runner"
echo "RUN_ID=${RUN_ID}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "SMOKE=${EXP015_SMOKE}, DRY_RUN=${DRY_RUN}, SKIP_EXISTING=${SKIP_EXISTING}"
echo "DATASETS=${DATASETS[*]}"
echo "MODELS=${MODELS[*]}"
echo "SEEDS=${SEEDS[*]}, FOLDS=${FOLDS[*]}, EPSES=${EPSES[*]}"
echo "BASELINES=${BASELINE_STRATEGIES[*]}"
echo "SAMPLE_NUM=${SAMPLE_NUM}, EPOCHS=${EPOCHS}, PATIENCE=${PATIENCE}"
echo "TRAIN_SAMPLE_NUM=${TRAIN_SAMPLE_NUM:-full}"
echo "ATTACK_SAMPLE_NUM=${ATTACK_SAMPLE_NUM:-full}"
echo "GPU_IDS=${GPU_ID_LIST[*]}, MAX_JOBS=${MAX_JOBS}, INNER_MAX_JOBS=${INNER_MAX_JOBS}"
echo "RANK_GROWTH_CONFIG=${RANK_GROWTH_CONFIG}"

gpu_cursor=0
next_gpu() {
  NEXT_GPU="${GPU_ID_LIST[$((gpu_cursor % ${#GPU_ID_LIST[@]}))]}"
  gpu_cursor=$((gpu_cursor + 1))
}

if (( START_PHASE <= 1 && STOP_PHASE >= 1 )); then
  echo "[$(date -Is)] Phase 1 clean checkpoints"
  phase_pids=()
  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        for fold in "${FOLDS[@]}"; do
          log_path="${LOG_ROOT}/phase1_clean_${dataset}_${model}_seed${seed}_fold${fold}.log"
          record_task "1" "clean" "${dataset}" "${model}" "${seed}" "${fold}" "0" "${log_path}"
          if [[ "${SKIP_EXISTING}" == "1" && -f "$(clean_checkpoint_path "${dataset}" "${model}" "${seed}" "${fold}")" ]]; then
            echo "[$(date -Is)] skip existing clean ${dataset} ${model} seed=${seed} fold=${fold}"
            continue
          fi
          wait_for_slots
          next_gpu
          run_clean_task "${dataset}" "${model}" "${seed}" "${fold}" "${NEXT_GPU}" "${log_path}" &
          phase_pids+=("$!")
        done
      done
    done
  done
  wait_for_phase "Phase 1" "${phase_pids[@]}"
fi

if (( START_PHASE <= 2 && STOP_PHASE >= 2 )); then
  echo "[$(date -Is)] Phase 2 main consistancy + rank-growth JS_MSE"
  phase_pids=()
  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        for fold in "${FOLDS[@]}"; do
          for eps in "${EPSES[@]}"; do
            run_tag="consistancy_rank25-30_n${SAMPLE_NUM}_eps$(eps_tag "${eps}")"
            log_path="${LOG_ROOT}/phase2_main_${dataset}_${model}_seed${seed}_fold${fold}_eps$(eps_tag "${eps}").log"
            record_task "2" "main_js_mse" "${dataset}" "${model}" "${seed}" "${fold}" "${eps}" "${log_path}"
            if [[ "${SKIP_EXISTING}" == "1" && -f "$(main_purified_ad_path "${dataset}" "${model}" "${eps}" "${seed}" "${fold}" "${run_tag}")" ]]; then
              echo "[$(date -Is)] skip existing main ${dataset} ${model} seed=${seed} fold=${fold} eps=${eps}"
              continue
            fi
            wait_for_slots
            next_gpu
            run_main_task "${dataset}" "${model}" "${seed}" "${fold}" "${eps}" "${NEXT_GPU}" "${log_path}" &
            phase_pids+=("$!")
          done
        done
      done
    done
  done
  wait_for_phase "Phase 2" "${phase_pids[@]}"
fi

if (( START_PHASE <= 3 && STOP_PHASE >= 3 )); then
  echo "[$(date -Is)] Phase 3 ordinary AT baselines"
  phase_pids=()
  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for strategy in "${BASELINE_STRATEGIES[@]}"; do
        for seed in "${SEEDS[@]}"; do
          for fold in "${FOLDS[@]}"; do
            for eps in "${EPSES[@]}"; do
              log_path="${LOG_ROOT}/phase3_${strategy}_${dataset}_${model}_seed${seed}_fold${fold}_eps$(eps_tag "${eps}").log"
              record_task "3" "${strategy}" "${dataset}" "${model}" "${seed}" "${fold}" "${eps}" "${log_path}"
              if [[ "${SKIP_EXISTING}" == "1" && -f "$(baseline_adv_path "${dataset}" "${model}" "${strategy}" "${eps}" "${seed}" "${fold}")" ]]; then
                echo "[$(date -Is)] skip existing baseline ${strategy} ${dataset} ${model} seed=${seed} fold=${fold} eps=${eps}"
                continue
              fi
              wait_for_slots
              next_gpu
              run_baseline_task "${dataset}" "${model}" "${strategy}" "${seed}" "${fold}" "${eps}" "${NEXT_GPU}" "${log_path}" &
              phase_pids+=("$!")
            done
          done
        done
      done
    done
  done
  wait_for_phase "Phase 3" "${phase_pids[@]}"
fi

if (( START_PHASE <= 4 && STOP_PHASE >= 4 )); then
  echo "[$(date -Is)] Phase 4 ABAT baselines"
  phase_pids=()
  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        for fold in "${FOLDS[@]}"; do
          for eps in "${EPSES[@]}"; do
            ea_model="$(abat_model_name "${model}")"
            log_path="${LOG_ROOT}/phase4_abat_${dataset}_${ea_model}_seed${seed}_fold${fold}_eps$(eps_tag "${eps}").log"
            record_task "4" "abat" "${dataset}" "${ea_model}" "${seed}" "${fold}" "${eps}" "${log_path}"
            if [[ "${SKIP_EXISTING}" == "1" && -f "$(abat_adv_path "${dataset}" "${ea_model}" "${eps}" "${seed}" "${fold}")" ]]; then
              echo "[$(date -Is)] skip existing ABAT ${dataset} ${ea_model} seed=${seed} fold=${fold} eps=${eps}"
              continue
            fi
            wait_for_slots
            next_gpu
            run_abat_task "${dataset}" "${model}" "${seed}" "${fold}" "${eps}" "${NEXT_GPU}" "${log_path}" &
            phase_pids+=("$!")
          done
        done
      done
    done
  done
  wait_for_phase "Phase 4" "${phase_pids[@]}"
fi

echo "[$(date -Is)] EXP-015 runner complete"
echo "planned_tasks=${TASKS_CSV}"
