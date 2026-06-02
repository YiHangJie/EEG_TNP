#!/usr/bin/env bash
set -euo pipefail

# EXP-001: PTR rank growth 与固定 rank 净化对比。
# 该脚本只编排已有测试入口，核心参数与 docs/EXPERIMENTS.md 中 EXP-001 保持一致。

DATASET="${DATASET:-thubenchmark}"
MODEL="${MODEL:-eegnet}"
AT_STRATEGY="${AT_STRATEGY:-madry}"
ATTACK="${ATTACK:-autoattack}"
EPS="${EPS:-0.03}"
SEED="${SEED:-42}"
FOLD="${FOLD:-0}"
GPU_IDS="${GPU_IDS:-0}"
GPU_ID="${GPU_ID:-0}"
MAX_JOBS="${MAX_JOBS:-2}"
SAMPLE_NUM="${SAMPLE_NUM:-512}"
PURIFY_BATCH_SIZE="${PURIFY_BATCH_SIZE:-32}"
USE_EA="${USE_EA:-0}"
CONDA_ENV="${CONDA_ENV:-torch}"
ACTIVATE_CONDA="${ACTIVATE_CONDA:-1}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"

RUN_FIXED="${RUN_FIXED:-1}"
RUN_DYNAMIC="${RUN_DYNAMIC:-1}"
RANKS_CSV="${RANKS_CSV:-5,10,15,20,25,30,35,40}"

MODEL_TAG="${MODEL_TAG:-consistancy_rank25-30_n512_eps0p03}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_consistancy_rank25-30_n512_eps0p03_best.pth}"
AD_DATA_PATH="${AD_DATA_PATH:-ad_data/thubenchmark_eegnet_no_ea_consistancy_rank25-30_n512_eps0p03_madry_autoattack_eps0.03_seed42_fold0.pth}"

run_fixed_rank_sweep() {
  AT_STRATEGY="${AT_STRATEGY}" \
  CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
  AD_DATA_PATH="${AD_DATA_PATH}" \
  MODEL_TAG="${MODEL_TAG}" \
  OUTPUT_TAG=exp001_fixed \
  RANKS_CSV="${RANKS_CSV}" \
  SAMPLE_NUM="${SAMPLE_NUM}" \
  PURIFY_BATCH_SIZE="${PURIFY_BATCH_SIZE}" \
  MAX_JOBS="${MAX_JOBS}" \
  GPU_IDS="${GPU_IDS}" \
  USE_EA="${USE_EA}" \
  CONDA_ENV="${CONDA_ENV}" \
  ACTIVATE_CONDA="${ACTIVATE_CONDA}" \
  DRY_RUN="${DRY_RUN}" \
  OVERWRITE="${OVERWRITE}" \
  DATASET="${DATASET}" \
  MODEL="${MODEL}" \
  ATTACK="${ATTACK}" \
  EPS="${EPS}" \
  SEED="${SEED}" \
  FOLD="${FOLD}" \
  bash purify_aug_stage4_rank_sweep.sh
}

run_dynamic_rank_growth() {
  local js_threshold="$1"
  local output_tag="$2"
  local max_mse="${3:-}"

  env_args=(
    AT_STRATEGY="${AT_STRATEGY}"
    CHECKPOINT_PATH="${CHECKPOINT_PATH}"
    AD_DATA_PATH="${AD_DATA_PATH}"
    MODEL_TAG="${MODEL_TAG}"
    OUTPUT_TAG="${output_tag}"
    SAMPLE_NUM="${SAMPLE_NUM}"
    PURIFY_BATCH_SIZE="${PURIFY_BATCH_SIZE}"
    GPU_ID="${GPU_ID}"
    USE_EA="${USE_EA}"
    CONDA_ENV="${CONDA_ENV}"
    ACTIVATE_CONDA="${ACTIVATE_CONDA}"
    DRY_RUN="${DRY_RUN}"
    DATASET="${DATASET}"
    MODEL="${MODEL}"
    ATTACK="${ATTACK}"
    EPS="${EPS}"
    SEED="${SEED}"
    FOLD="${FOLD}"
    RANK_GROWTH_JS_THRESHOLD="${js_threshold}"
  )

  if [[ -n "${max_mse}" ]]; then
    env_args+=(RANK_GROWTH_MAX_MSE_TO_INPUT="${max_mse}")
  fi

  env "${env_args[@]}" bash TN/rank_growth/run_purify_rank_growth.sh
}

if [[ "${RUN_FIXED}" == "1" ]]; then
  run_fixed_rank_sweep
fi

if [[ "${RUN_DYNAMIC}" == "1" ]]; then
  run_dynamic_rank_growth 0.02 exp001_dynamic_js0p02
  run_dynamic_rank_growth 0.01 exp001_dynamic_js0p01
  run_dynamic_rank_growth 0.005 exp001_dynamic_js0p005
  run_dynamic_rank_growth 0.02 exp001_dynamic_js0p02_mse0p06 0.06
fi
