#!/usr/bin/env bash
# 建档：RUN_ID=exp033_full_v1 bash rpcf/run_exp033.sh plan
# 小样本：RUN_ID=exp033_smoke_v1 bash rpcf/run_exp033.sh smoke --cpu
# 正式启动：RUN_ID=exp033_full_v1 nohup setsid bash rpcf/run_exp033.sh run > logs/exp033_full_v1.controller.log 2>&1 < /dev/null &
# 分组：RUN_ID=exp033_full_v1 bash rpcf/run_exp033.sh run --groups ablation,visualize
# 普通计算：RUN_ID=exp033_full_v1 nohup setsid bash rpcf/run_exp033.sh run --queue compute > logs/exp033_full_v1.compute.log 2>&1 < /dev/null &
# 计算结束后计时：RUN_ID=exp033_full_v1 nohup setsid bash rpcf/run_exp033.sh run --queue timing > logs/exp033_full_v1.timing.log 2>&1 < /dev/null &
# 默认仅建档；不会自动启动正式训练，也不接管 EXP-032。
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
action="${1:-plan}"
if (( $# > 0 )); then shift; fi
exec conda run -n torch --no-capture-output python -u -m rpcf.exp033 "$action" \
  --run-id "${RUN_ID:-exp033_full_v1}" "$@"
