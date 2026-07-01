#!/usr/bin/env bash
set -euo pipefail

# EXP-024 串行调度：避免单 GPU 上多个 backbone 长实验并发。

RUN_TAG="${EXP024_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
CHAIN_ID="${EXP024_CHAIN_ID:-exp024_other_backbones_seed${EXP024_SEED:-42}_${RUN_TAG}}"
CHAIN_ROOT="${EXP024_CHAIN_ROOT:-logs/exp024/${CHAIN_ID}}"
MODELS="${EXP024_MODELS:-tsception atcnet conformer}"
SEED="${EXP024_SEED:-42}"
FOLD="${EXP024_FOLD:-0}"
EPS="${EXP024_EPS:-0.03}"
GPU_ID="${GPU_ID:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p "${CHAIN_ROOT}"
printf '%s\n' "$$" > "${CHAIN_ROOT}/controller.pid"
cat > "${CHAIN_ROOT}/run_config.txt" <<EOF
EXPERIMENT_ID=EXP-024
CHAIN_ID=${CHAIN_ID}
MODELS=${MODELS}
SEED=${SEED}
FOLD=${FOLD}
EPS=${EPS}
GPU_ID=${GPU_ID}
SKIP_EXISTING=${SKIP_EXISTING}
EOF

for model in ${MODELS}; do
  run_id="${CHAIN_ID}_${model}"
  log_root="logs/exp024/${run_id}"
  mkdir -p "${log_root}"
  echo "[$(date -Is)] Start EXP-024 ${model}: ${run_id}"
  EXP024_MODEL="${model}" \
  EXP024_SEED="${SEED}" \
  EXP024_FOLD="${FOLD}" \
  EXP024_EPS="${EPS}" \
  EXP024_RUN_ID="${run_id}" \
  EXP024_LOG_ROOT="${log_root}" \
  GPU_ID="${GPU_ID}" \
  DRY_RUN="${DRY_RUN}" \
  SKIP_EXISTING="${SKIP_EXISTING}" \
    bash rpcf/run_exp024_backbone.sh
  echo "[$(date -Is)] Finished EXP-024 ${model}: ${run_id}"
done

echo "[$(date -Is)] EXP-024 all-backbone chain finished: ${CHAIN_ID}"
