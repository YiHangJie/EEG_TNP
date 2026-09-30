#!/usr/bin/env bash
# 建档：RUN_ID=exp035_full_20260928_v1 bash rpcf/run_exp035.sh plan
# 正式：RUN_ID=exp035_full_20260928_v1 nohup setsid bash rpcf/run_exp035.sh run > logs/exp035_full_20260928_v1.controller.log 2>&1 < /dev/null &
# 汇总：RUN_ID=exp035_full_20260928_v1 bash rpcf/run_exp035.sh summary --strict
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
RUN_ID="${RUN_ID:-exp035_full_20260928_v1}"
ACTION="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
exec /home/yihangjie/miniconda3/envs/torch/bin/python -u -m rpcf.exp035 "$ACTION" --run-id "$RUN_ID" "$@"
