#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-42 43 44}"
FOLD="${FOLD:-0}"
GPU_ID="${GPU_ID:-0}"
RUN_TAG="${EXP020_RUN_TAG:-$(date +%Y%m%d_%H%M)}"
CHAIN_ID="${EXP020_CHAIN_ID:-exp020_eps0p05_seeds42-44_${RUN_TAG}}"
CHAIN_ROOT="${EXP020_CHAIN_ROOT:-logs/exp020/${CHAIN_ID}}"

mkdir -p "${CHAIN_ROOT}"
printf '%s\n' "$$" > "${CHAIN_ROOT}/controller.pid"

for seed in ${SEEDS}; do
  run_id="exp020_seed${seed}_fold${FOLD}_eps0p05_${RUN_TAG}"
  log_root="logs/exp020/${run_id}"
  mkdir -p "${log_root}"

  echo "[$(date -Is)] start ${run_id}"
  SEED="${seed}" FOLD="${FOLD}" EPS=0.05 GPU_ID="${GPU_ID}" \
    EXP020_RUN_ID="${run_id}" EXP020_LOG_ROOT="${log_root}" \
    bash rpcf/run_exp020.sh 2>&1 | tee -a "${log_root}/controller.log"
  echo "[$(date -Is)] completed ${run_id}"
done

echo "[$(date -Is)] EXP-020 all-seed pipeline finished: ${CHAIN_ID}"
