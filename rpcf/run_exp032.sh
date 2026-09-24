#!/usr/bin/env bash
# 生成计划：RUN_ID=exp032_full_20260917 bash rpcf/run_exp032.sh plan
# 正式后台：RUN_ID=exp032_full_20260917 GPU_IDS=0,1,2,3,4,5 nohup setsid bash rpcf/run_exp032.sh run > logs/exp032_full_20260917.controller.log 2>&1 < /dev/null &
# 暂缓 clean-only+TNP 后续跑已有计划：RUN_ID=exp032_full_20260917_2020 DEFER_CLEAN_TNP=1 GPU_IDS=0,1,2,3,4,5 nohup setsid bash rpcf/run_exp032.sh run >> logs/exp032_full_20260917_2020.controller.log 2>&1 < /dev/null &
# smoke：RUN_ID=exp032_smoke_20260917 SMOKE=1 GPU_IDS=0 nohup setsid bash rpcf/run_exp032.sh run > logs/exp032_smoke_20260917.controller.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${RUN_ID:?请显式设置独立的 exp032_ run id}"
action="${1:-plan}"
entry="rpcf.exp032"
if [[ "${DEFER_CLEAN_TNP:-0}" == "1" || -f "logs/exp032/$RUN_ID/execution_scope.json" ]]; then
  entry="rpcf.resume_exp032"
fi
args=("$action" --run-id "$RUN_ID" --source-run "${SOURCE_RUN:-exp031_full_20260729_174215}")
if [[ "${DEFER_CLEAN_TNP:-0}" == "1" ]]; then args+=(--defer-clean-tnp); fi
if [[ "${SMOKE:-0}" == "1" ]]; then args+=(--smoke); fi
if [[ "$action" == "run" ]]; then
  args+=(--gpu-ids "${GPU_IDS:-0,1,2,3,4,5}" --start-stage "${START_STAGE:-0}" --stop-stage "${STOP_STAGE:-3}")
  if [[ -n "${TASK_ID:-}" ]]; then args+=(--task-id "$TASK_ID"); fi
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec conda run -n torch --no-capture-output python -u -m "$entry" "${args[@]}"
