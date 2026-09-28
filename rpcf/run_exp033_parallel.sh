#!/usr/bin/env bash
# 检查策略：RUN_ID=exp033_full_20260923_v6 bash rpcf/run_exp033_parallel.sh plan
# 接管：RUN_ID=exp033_full_20260923_v6 bash rpcf/run_exp033_parallel.sh capture --old-controller-pid PID --old-controller-ticks TICKS
# 后台运行：RUN_ID=exp033_full_20260923_v6 nohup setsid bash rpcf/run_exp033_parallel.sh run > logs/exp033_full_20260923_v6.parallel_v1.controller.log 2>&1 < /dev/null &
# 恢复：确认旧并行worker已结束后，沿用run-id执行run；真实失败需诊断后显式追加--retry-failed。
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
exec conda run -n torch --no-capture-output python -u -m rpcf.parallel_exp033 "$@" --run-id "${RUN_ID:-exp033_full_20260923_v6}"
