#!/usr/bin/env bash
set -euo pipefail

# EA-in-forward EEGNet + Madry AT training and attack script.
# Usage:
#   bash ea_forward_AT.sh
#   EPS_LIST=0.03,0.05 ATTACKS_CSV=autoattack GPU_ID=0 bash ea_forward_AT.sh
#   DRY_RUN=1 EPS_LIST=0.03 bash ea_forward_AT.sh
#   nohup bash -c 'WAIT_PID=505395 bash ea_forward_AT.sh' > logs/ea_forward_AT_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   nohup bash -c 'bash ea_forward_AT.sh' > logs/ea_forward_AT_$(date +%Y%m%d_%H%M%S).log 2>&1 &


DATASET="${DATASET:-thubenchmark}"
MODEL="${MODEL:-eegnet_ea_forward}"
AT_STRATEGY="${AT_STRATEGY:-madry}"
FOLD="${FOLD:-0}"
SEED="${SEED:-42}"
GPU_ID="${GPU_ID:-0}"
EPS_LIST="${EPS_LIST:-0.03,0.05,0.1}"
ATTACKS_CSV="${ATTACKS_CSV:-autoattack}"

EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-32}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
PATIENCE="${PATIENCE:-20}"
CLEAN_RATIO="${CLEAN_RATIO:-0.0}"
CHECKPOINT_TAG="${CHECKPOINT_TAG:-}"
SAVE_ADV="${SAVE_ADV:-1}"

CONDA_ENV="${CONDA_ENV:-torch}"
ACTIVATE_CONDA="${ACTIVATE_CONDA:-1}"
DRY_RUN="${DRY_RUN:-0}"
WAIT_PID="${WAIT_PID:-505395}"

PROTOCOL_TAG="train_only_subject_no_ea_subject_split"

if [[ -n "${WAIT_PID}" ]]; then
  while kill -0 "${WAIT_PID}" 2> /dev/null; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID ${WAIT_PID} 仍在运行..."
    sleep 30
  done
fi

if [[ "${MODEL}" != "eegnet_ea_forward" ]]; then
  echo "MODEL must be eegnet_ea_forward for EA-in-forward experiments, got: ${MODEL}" >&2
  exit 1
fi
if [[ "${AT_STRATEGY}" != "madry" ]]; then
  echo "AT_STRATEGY must be madry for the first EA-in-forward version, got: ${AT_STRATEGY}" >&2
  exit 1
fi

safe_token() {
  local value="$1"
  if [[ -z "${value}" ]]; then
    echo "none"
    return
  fi
  printf '%s' "${value}" | sed 's/[^A-Za-z0-9_.-]/_/g'
}

trim_token() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

if [[ "${ACTIVATE_CONDA}" == "1" ]] && command -v conda > /dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
mkdir -p "${NUMBA_CACHE_DIR}" "${XDG_CACHE_HOME}" logs log_train_AT log_attack checkpoints ad_data

run_cmd() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

build_checkpoint_path() {
  local eps="$1"
  local path="checkpoints/${DATASET}_${MODEL}_${PROTOCOL_TAG}_${AT_STRATEGY}_eps${eps}_${SEED}_fold${FOLD}"
  if [[ -n "${CHECKPOINT_TAG}" ]]; then
    path="${path}_$(safe_token "${CHECKPOINT_TAG}")"
  fi
  printf '%s_best.pth' "${path}"
}

IFS=',' read -r -a raw_eps_values <<< "${EPS_LIST}"
eps_values=()
for eps in "${raw_eps_values[@]}"; do
  eps="$(trim_token "${eps}")"
  if [[ -n "${eps}" ]]; then
    eps_values+=("${eps}")
  fi
done
if [[ "${#eps_values[@]}" -eq 0 ]]; then
  echo "EPS_LIST is empty." >&2
  exit 1
fi

IFS=',' read -r -a raw_attacks <<< "${ATTACKS_CSV}"
attacks=()
for attack_name in "${raw_attacks[@]}"; do
  attack_name="$(trim_token "${attack_name}")"
  if [[ -n "${attack_name}" ]]; then
    attacks+=("${attack_name}")
  fi
done
if [[ "${#attacks[@]}" -eq 0 ]]; then
  echo "ATTACKS_CSV is empty." >&2
  exit 1
fi

echo "======================================================"
echo "EA-forward EEGNet + AT"
echo "DATASET=${DATASET}, MODEL=${MODEL}, AT_STRATEGY=${AT_STRATEGY}"
echo "FOLD=${FOLD}, SEED=${SEED}, GPU_ID=${GPU_ID}"
echo "EPS_LIST=${eps_values[*]}, ATTACKS=${attacks[*]}"
echo "EPOCHS=${EPOCHS}, BATCH_SIZE=${BATCH_SIZE}, LR=${LR}, WEIGHT_DECAY=${WEIGHT_DECAY}, PATIENCE=${PATIENCE}"
echo "CHECKPOINT_TAG=${CHECKPOINT_TAG:-none}, SAVE_ADV=${SAVE_ADV}, DRY_RUN=${DRY_RUN}"
echo "======================================================"

for eps in "${eps_values[@]}"; do
  checkpoint_path="$(build_checkpoint_path "${eps}")"

  train_args=(
    python -u train_AT_ea_forward.py
    --dataset "${DATASET}"
    --model "${MODEL}"
    --at_strategy "${AT_STRATEGY}"
    --fold "${FOLD}"
    --epsilon "${eps}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --patience "${PATIENCE}"
    --clean_ratio "${CLEAN_RATIO}"
    --seed "${SEED}"
    --gpu_id "${GPU_ID}"
    --no_ea
  )
  if [[ -n "${CHECKPOINT_TAG}" ]]; then
    train_args+=(--checkpoint_tag "${CHECKPOINT_TAG}")
  fi

  echo "[Stage 1] Train EA-forward AT | eps=${eps}"
  run_cmd "${train_args[@]}"

  if [[ "${DRY_RUN}" != "1" && ! -f "${checkpoint_path}" ]]; then
    echo "Expected checkpoint not found after training: ${checkpoint_path}" >&2
    exit 1
  fi

  for attack_name in "${attacks[@]}"; do
    attack_args=(
      python -u attack_ea_forward.py
      --dataset "${DATASET}"
      --model "${MODEL}"
      --at_strategy "${AT_STRATEGY}"
      --fold "${FOLD}"
      --attack "${attack_name}"
      --eps "${eps}"
      --batch_size "${ATTACK_BATCH_SIZE}"
      --seed "${SEED}"
      --gpu_id "${GPU_ID}"
      --no_ea
      --checkpoint_path "${checkpoint_path}"
    )
    if [[ -n "${CHECKPOINT_TAG}" ]]; then
      attack_args+=(--checkpoint_tag "${CHECKPOINT_TAG}")
    fi
    if [[ "${SAVE_ADV}" == "1" ]]; then
      adv_tag="ea_forward"
      if [[ -n "${CHECKPOINT_TAG}" ]]; then
        adv_tag="ea_forward_$(safe_token "${CHECKPOINT_TAG}")"
      fi
      attack_args+=(--save_adv --adv_output_tag "${adv_tag}")
    fi

    echo "[Stage 2] Attack EA-forward AT | eps=${eps}, attack=${attack_name}"
    run_cmd "${attack_args[@]}"
  done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] EA-forward AT script complete."
