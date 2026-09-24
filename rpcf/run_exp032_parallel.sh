#!/usr/bin/env bash
# 共卡参数：PAUSED_GPU_PEERS=1:3640736:451734835,2:3643011:451991114,3:3643222:451998320；仅用于已核实暂停的进程，须与冻结policy一致。
# 计划：PAUSED_GPU_PEERS=1:3640736:451734835,2:3643011:451991114,3:3643222:451998320 RUN_ID=exp032_full_20260917_2020 bash rpcf/run_exp032_parallel.sh plan
# 接管准备（核对 PID/start_ticks 后）：PAUSED_GPU_PEERS=1:3640736:451734835,2:3643011:451991114,3:3643222:451998320 RUN_ID=exp032_full_20260917_2020 bash rpcf/run_exp032_parallel.sh capture --old-controller-pid 3674865 --old-controller-ticks 456488681 --controller-log logs/exp032_full_20260917_2020.parallel_v4.controller.log
# 后台：PAUSED_GPU_PEERS=1:3640736:451734835,2:3643011:451991114,3:3643222:451998320 RUN_ID=exp032_full_20260917_2020 GPU_IDS=0,1,2,3,4,5,6,7 CPU_WORKERS=4 TNP_PER_GPU=2 nohup setsid bash rpcf/run_exp032_parallel.sh run > logs/exp032_full_20260917_2020.parallel_v5.controller.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${RUN_ID:?请设置已有EXP-032的run-id}"
action="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec conda run -n torch --no-capture-output python -u -m rpcf.parallel_exp032 "$action" \
  --run-id "$RUN_ID" --gpu-ids "${GPU_IDS:-0,1,2,3,4,5,6,7}" \
  --paused-gpu-peers "${PAUSED_GPU_PEERS:-}" --cpu-workers "${CPU_WORKERS:-4}" --tnp-per-gpu "${TNP_PER_GPU:-2}" "$@"
