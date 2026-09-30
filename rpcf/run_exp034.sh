#!/usr/bin/env bash
# 建档：RUN_ID=exp034_full_20260928_v1 bash rpcf/run_exp034.sh plan
# 正式运行：RUN_ID=exp034_full_20260928_v1 GPU_IDS=0,1,2,3,4,5,6,7 nohup setsid bash rpcf/run_exp034.sh run > logs/exp034_full_20260928_v1.controller.log 2>&1 < /dev/null &
# 小样本：RUN_ID=exp034_smoke_20260928_v1 SMOKE=1 bash rpcf/run_exp034.sh plan
# 小样本运行：RUN_ID=exp034_smoke_20260928_v1 GPU_IDS=0 bash rpcf/run_exp034.sh run
# 汇总：RUN_ID=exp034_full_20260928_v1 bash rpcf/run_exp034.sh summarize
# 检查失败并续跑：RUN_ID=exp034_full_20260928_v1 RETRY_FAILED=1 nohup setsid bash rpcf/run_exp034.sh run >> logs/exp034_full_20260928_v1.controller.log 2>&1 < /dev/null &
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
RUN_ID="${RUN_ID:-exp034_full_20260928_v1}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PYTHON="${PYTHON:-/home/yihangjie/miniconda3/envs/torch/bin/python}"
action="${1:-plan}"
args=("$action" --run-id "$RUN_ID" --gpu-ids "$GPU_IDS")
if [[ "${SMOKE:-0}" == 1 && "$action" == plan ]]; then args+=(--smoke); fi
if [[ "${CPU:-0}" == 1 ]]; then args+=(--cpu); fi
if [[ "${RETRY_FAILED:-0}" == 1 ]]; then args+=(--retry-failed); fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec "$PYTHON" -u -m rpcf.exp034 "${args[@]}"
