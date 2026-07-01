#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-42 43 44}"
GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-torch}"
RUN_TAG="${EXP022_RUN_TAG:-$(date +%Y%m%d_%H%M)}"
CHAIN_ID="${EXP022_CHAIN_ID:-exp022_rpcf_at_eps0p05_seeds42-44_${RUN_TAG}}"
CHAIN_ROOT="${EXP022_CHAIN_ROOT:-logs/exp022/${CHAIN_ID}}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-8}"

mkdir -p "${CHAIN_ROOT}"
printf '%s\n' "$$" > "${CHAIN_ROOT}/controller.pid"
cat > "${CHAIN_ROOT}/run_config.txt" <<EOF
EXPERIMENT_ID=EXP-022
CHAIN_ID=${CHAIN_ID}
SEEDS=${SEEDS}
GPU_ID=${GPU_ID}
CONDA_ENV=${CONDA_ENV}
ATTACK_BATCH_SIZE=${ATTACK_BATCH_SIZE}
EOF

for seed in ${SEEDS}; do
  run_id="exp022_seed${seed}_fold0_eps0p05_${RUN_TAG}"
  log_root="logs/exp022/${run_id}"
  mkdir -p "${log_root}"

  echo "[$(date -Is)] Start EXP-022 seed${seed}: ${run_id}"
  EXP022_SEED="${seed}" \
  EXP022_RUN_ID="${run_id}" \
  EXP022_LOG_ROOT="${log_root}" \
  GPU_ID="${GPU_ID}" \
  CONDA_ENV="${CONDA_ENV}" \
  ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE}" \
    bash rpcf/run_exp022_rpcf_at.sh
  echo "[$(date -Is)] Finished EXP-022 seed${seed}: ${run_id}"
done

echo "[$(date -Is)] EXP-022 all-seed pipeline finished: ${CHAIN_ID}"
