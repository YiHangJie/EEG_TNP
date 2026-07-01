#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-torch}"
CHAIN_ID="${EXP021_CHAIN_ID:-exp021_rpcf_at_seeds43-44_$(date +%Y%m%d_%H%M%S)}"
CHAIN_LOG_ROOT="${EXP021_CHAIN_LOG_ROOT:-logs/exp021/${CHAIN_ID}}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-8}"

mkdir -p "${CHAIN_LOG_ROOT}"
printf '%s\n' "$$" > "${CHAIN_LOG_ROOT}/controller.pid"

cat > "${CHAIN_LOG_ROOT}/run_config.txt" <<EOF
EXPERIMENT_ID=EXP-021-seeds43-44
CHAIN_ID=${CHAIN_ID}
SEEDS=43,44
GPU_ID=${GPU_ID}
CONDA_ENV=${CONDA_ENV}
ATTACK_BATCH_SIZE=${ATTACK_BATCH_SIZE}
EOF

for seed in 43 44; do
  run_id="${CHAIN_ID}_seed${seed}"
  echo "[$(date -Is)] Start EXP-021 RPCF_AT seed${seed}: ${run_id}"
  EXP021_SEED="${seed}" \
  EXP021_RUN_ID="${run_id}" \
  GPU_ID="${GPU_ID}" \
  CONDA_ENV="${CONDA_ENV}" \
  ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE}" \
    bash rpcf/run_exp021_rpcf_at.sh
  echo "[$(date -Is)] Finished EXP-021 RPCF_AT seed${seed}: ${run_id}"
done

echo "[$(date -Is)] EXP-021 seed43/44 chain finished: ${CHAIN_ID}"
