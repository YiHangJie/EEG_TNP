#!/usr/bin/env bash
# 启动：nohup setsid bash rpcf/run_exp032_pgd10.sh > logs/exp032_full_20260917_2020.pgd10_v1.controller.log 2>&1 < /dev/null &
# 预检：bash rpcf/run_exp032_pgd10.sh plan
# 独立修订目录保存状态；既有实验产物和日志保留，攻击协议不从环境变量覆盖。
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
exec conda run --no-capture-output -n torch python -u -m rpcf.revise_exp032_adaptive \
    "${1:-run}" --run-id exp032_full_20260917_2020
