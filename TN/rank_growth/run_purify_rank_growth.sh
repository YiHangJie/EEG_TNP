#!/usr/bin/env bash
set -euo pipefail

# 运行 PTR_3d_rank_growth 在 purify.py 上的快速测试。
#
# 示例：
#   bash TN/rank_growth/run_purify_rank_growth.sh
#   DRY_RUN=1 SAMPLE_NUM=2 RANK_GROWTH_RANKS=5,10 bash TN/rank_growth/run_purify_rank_growth.sh
#   DRY_RUN=1 RANK_GROWTH_JS_THRESHOLD=0.01 bash TN/rank_growth/run_purify_rank_growth.sh
#   DRY_RUN=1 RANK_GROWTH_MAX_MSE_TO_INPUT=0.06 bash TN/rank_growth/run_purify_rank_growth.sh
#   SAMPLE_NUM=32 bash TN/rank_growth/run_purify_rank_growth.sh
# AT_STRATEGY=madry \
# CHECKPOINT_PATH=checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_consistancy_rank25-30_n512_eps0p03_bes
# t.pth \
# AD_DATA_PATH=ad_data/thubenchmark_eegnet_no_ea_consistancy_rank25-30_n512_eps0p03_madry_autoattack_eps0.03_seed42_fold0.pth \
# MODEL_TAG=consistancy_rank25-30_n512_eps0p03 \
# OUTPUT_TAG=rank_growth_consistancy \
# SAMPLE_NUM=512 \
# nohup bash TN/rank_growth/run_purify_rank_growth.sh > logs/run_purify_rank_growth.log 2>&1 &

DATASET="${DATASET:-thubenchmark}"
MODEL="${MODEL:-eegnet}"
ATTACK="${ATTACK:-autoattack}"
EPS="${EPS:-0.03}"
SEED="${SEED:-42}"
FOLD="${FOLD:-0}"
GPU_ID="${GPU_ID:-0}"
SAMPLE_NUM="${SAMPLE_NUM:-512}"
PURIFY_BATCH_SIZE="${PURIFY_BATCH_SIZE:-32}"
USE_EA="${USE_EA:-0}"
AT_STRATEGY="${AT_STRATEGY:-madry}"
CONFIG="${CONFIG:-PTR3d_rank_growth_8_2048_r5-40_3d_interpolate.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
AD_DATA_PATH="${AD_DATA_PATH:-}"
MODEL_TAG="${MODEL_TAG:-rank_growth}"
OUTPUT_TAG="${OUTPUT_TAG:-rank_growth}"
DRY_RUN="${DRY_RUN:-0}"
CONDA_ENV="${CONDA_ENV:-torch}"
ACTIVATE_CONDA="${ACTIVATE_CONDA:-1}"
RANK_GROWTH_RANKS="${RANK_GROWTH_RANKS:-}"
RANK_GROWTH_STEPS_PER_RANK="${RANK_GROWTH_STEPS_PER_RANK:-}"
RANK_GROWTH_JS_THRESHOLD="${RANK_GROWTH_JS_THRESHOLD:-}"
RANK_GROWTH_MAX_MSE_TO_INPUT="${RANK_GROWTH_MAX_MSE_TO_INPUT:-}"

if [[ "${ACTIVATE_CONDA}" == "1" ]] && command -v conda > /dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_cache}"
mkdir -p "${NUMBA_CACHE_DIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

CONFIG_ARG="${CONFIG}"
BASE_CONFIG_PATH="${CONFIG}"
if [[ "${CONFIG}" != /* && "${CONFIG}" != */* ]]; then
  BASE_CONFIG_PATH="configs/${DATASET}/${CONFIG}"
fi

if [[ -n "${RANK_GROWTH_RANKS}" || -n "${RANK_GROWTH_STEPS_PER_RANK}" || -n "${RANK_GROWTH_JS_THRESHOLD}" || -n "${RANK_GROWTH_MAX_MSE_TO_INPUT}" ]]; then
  if [[ ! -f "${BASE_CONFIG_PATH}" ]]; then
    echo "Base config not found: ${BASE_CONFIG_PATH}" >&2
    exit 1
  fi
  TMP_CONFIG_DIR="${TMP_CONFIG_DIR:-/tmp/eegap_rank_growth_configs}"
  mkdir -p "${TMP_CONFIG_DIR}"
  override_tag="r${RANK_GROWTH_RANKS//,/}-s${RANK_GROWTH_STEPS_PER_RANK:-default}-js${RANK_GROWTH_JS_THRESHOLD:-default}-mse${RANK_GROWTH_MAX_MSE_TO_INPUT:-none}"
  override_tag="${override_tag//[^A-Za-z0-9_.-]/_}"
  TMP_CONFIG="${TMP_CONFIG_DIR}/$(basename "${CONFIG%.yaml}")_override_${override_tag}.yaml"
  python - "${BASE_CONFIG_PATH}" "${TMP_CONFIG}" "${RANK_GROWTH_RANKS}" "${RANK_GROWTH_STEPS_PER_RANK}" "${RANK_GROWTH_JS_THRESHOLD}" "${RANK_GROWTH_MAX_MSE_TO_INPUT}" <<'PY'
import sys
import yaml

base_path, output_path, ranks_csv, steps, js_threshold, max_mse = sys.argv[1:7]
with open(base_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

if ranks_csv:
    config["rank_growth_ranks"] = [int(item.strip()) for item in ranks_csv.split(",") if item.strip()]
    config["max_rank"] = max(config["rank_growth_ranks"])
if steps:
    config["rank_growth_steps_per_rank"] = int(steps)
if js_threshold:
    config["rank_growth_js_threshold"] = float(js_threshold)
if max_mse:
    config["rank_growth_max_mse_to_input"] = float(max_mse)

with open(output_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
PY
  CONFIG_ARG="${TMP_CONFIG}"
fi

ea_arg=(--no_ea)
if [[ "${USE_EA}" == "1" ]]; then
  ea_arg=(--use_ea)
fi

cmd=(
  python -u purify.py
  --dataset "${DATASET}"
  --model "${MODEL}"
  --at_strategy "${AT_STRATEGY}"
  --fold "${FOLD}"
  --attack "${ATTACK}"
  --eps "${EPS}"
  --sample_num "${SAMPLE_NUM}"
  --batch_size "${PURIFY_BATCH_SIZE}"
  --seed "${SEED}"
  --gpu_id "${GPU_ID}"
  "${ea_arg[@]}"
  --config "${CONFIG_ARG}"
  --model_tag "${MODEL_TAG}"
  --output_tag "${OUTPUT_TAG}"
)

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  cmd+=(--checkpoint_path "${CHECKPOINT_PATH}")
fi
if [[ -n "${AD_DATA_PATH}" ]]; then
  cmd+=(--ad_data_path "${AD_DATA_PATH}")
fi

printf '[%s]' "$(date '+%Y-%m-%d %H:%M:%S')"
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" != "1" ]]; then
  "${cmd[@]}"
fi
