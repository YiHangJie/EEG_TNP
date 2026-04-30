#!/usr/bin/env bash
set -euo pipefail

# ===== conda =====
source ~/miniconda3/etc/profile.d/conda.sh   # 按你自己的 conda 路径改
conda activate torch

# ===== args =====
dataset="${1:-thubenchmark}"
gpu_id="${2:-0}"

# 第3个参数：eps 列表（逗号分隔），例如 "0.01,0.02,0.05,0.1"
# 不提供则用默认 eps 列表
eps_csv="${3:-0.01,0.02,0.05,0.1,0.2}"

# 第4个参数：目标 PID，非0则等待该PID结束再开始
target_pid="${4:-0}"

# ===== wait for target pid =====
while [ "$target_pid" -ne 0 ] && kill -0 "$target_pid" 2>/dev/null; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $target_pid 仍在运行..."
  sleep 30
done

# ===== concurrency =====
train_max_jobs=1
attack_max_jobs=1
purify_max_jobs=2

# ===== parse eps list =====
IFS=',' read -r -a eps_list <<< "$eps_csv"

# ===== helper: run command safely with nohup =====
run_bg() {
  local cmd="$1"
  local log_file="$2"
  # 用 bash -lc 执行，避免字符串分词/环境问题
  nohup bash -lc "$cmd" > "$log_file" 2>&1 &
}

wait_for_slots() {
  local max_jobs="$1"
  while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do
    sleep 60
  done
}

# # ===== prepare log dir for clean training =====
# ts="$(date '+%Y%m%d_%H%M%S')"
# eps_tag="clean"
# log_dir="logs/${dataset}/gpu${gpu_id}/eps_${eps_tag}"
# mkdir -p "$log_dir"
# clean_train_command="python -u train_AT.py --dataset ${dataset} --model eegnet --at_strategy clean --gpu_id ${gpu_id}"
# log_file="${log_dir}/train_${ts}.log"
# echo "[$(date '+%Y-%m-%d %H:%M:%S')] CLEAN TRAIN: $clean_train_command"
# run_bg "$clean_train_command" "$log_file"
# sleep 60
# wait_for_slots "$train_max_jobs"
# wait

# ===== main loop over eps =====
for eps in "${eps_list[@]}"; do
  eps_tag="${eps//./p}"               # 0.01 -> 0p01 便于做文件名
  log_dir="logs/${dataset}/gpu${gpu_id}/eps_${eps_tag}"
  mkdir -p "$log_dir"

  echo "======================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START eps=${eps}  dataset=${dataset} gpu=${gpu_id}"
  echo "Logs: ${log_dir}"
  echo "======================================================"

  # # ---------- train commands ----------
  # train_commands=(
  #   "python -u train_AT.py --dataset ${dataset} --model eegnet --at_strategy madry  --epsilon ${eps} --gpu_id ${gpu_id}"
  #   "python -u train_AT.py --dataset ${dataset} --model eegnet --at_strategy fbf    --epsilon ${eps} --gpu_id ${gpu_id}"
  #   "python -u train_AT.py --dataset ${dataset} --model eegnet --at_strategy trades --epsilon ${eps} --gpu_id ${gpu_id}"
  # )

  # for cmd in "${train_commands[@]}"; do
  #   ts="$(date '+%Y%m%d_%H%M%S')"
  #   log_file="${log_dir}/train_${ts}.log"
  #   echo "[$(date '+%Y-%m-%d %H:%M:%S')] TRAIN: $cmd"
  #   run_bg "$cmd" "$log_file"
  #   sleep 60
  #   wait_for_slots "$train_max_jobs"
  # done
  # wait

  # ---------- attack commands ----------
  attack_commands=(
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy clean  --fold 0 --attack autoattack --eps ${eps} --gpu_id ${gpu_id}"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy madry  --fold 0 --attack autoattack --eps ${eps} --gpu_id ${gpu_id}"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy fbf    --fold 0 --attack autoattack --eps ${eps} --gpu_id ${gpu_id}"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy trades --fold 0 --attack autoattack --eps ${eps} --gpu_id ${gpu_id}"

    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy clean  --fold 0 --attack fgsm --eps ${eps} --gpu_id ${gpu_id}"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy madry  --fold 0 --attack fgsm --eps ${eps} --gpu_id ${gpu_id}"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy fbf    --fold 0 --attack fgsm --eps ${eps} --gpu_id ${gpu_id}"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy trades --fold 0 --attack fgsm --eps ${eps} --gpu_id ${gpu_id}"

    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy clean  --fold 0 --attack pgd --eps ${eps} --gpu_id ${gpu_id}"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy madry  --fold 0 --attack pgd --eps ${eps} --gpu_id ${gpu_id}"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy fbf    --fold 0 --attack pgd --eps ${eps} --gpu_id ${gpu_id}"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy trades --fold 0 --attack pgd --eps ${eps} --gpu_id ${gpu_id}"

    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy clean  --fold 0 --attack cw --eps ${eps} --gpu_id ${gpu_id}"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy madry  --fold 0 --attack cw --eps ${eps} --gpu_id ${gpu_id}"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy fbf    --fold 0 --attack cw --eps ${eps} --gpu_id ${gpu_id}"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy trades --fold 0 --attack cw --eps ${eps} --gpu_id ${gpu_id}"
  )

  for cmd in "${attack_commands[@]}"; do
    ts="$(date '+%Y%m%d_%H%M%S')"
    log_file="${log_dir}/attack_${ts}.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ATTACK: $cmd"
    run_bg "$cmd" "$log_file"
    sleep 60
    wait_for_slots "$attack_max_jobs"
  done
  wait

  # # ---------- purify commands (按需打开) ----------
  # purify_commands=(
  #   "python purify.py --config PTR3d_8_2048_rank5_3d_interpolate.yaml  --sample_num 512 --dataset ${dataset} --attack autoattack --eps ${eps} --gpu_id ${gpu_id}"
  #   "python purify.py --config PTR3d_8_2048_rank10_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --eps ${eps} --gpu_id ${gpu_id}"
  #   "python purify.py --config PTR3d_8_2048_rank15_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --eps ${eps} --gpu_id ${gpu_id}"
  #   "python purify.py --config PTR3d_8_2048_rank20_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --eps ${eps} --gpu_id ${gpu_id}"
  #   "python purify.py --config PTR3d_8_2048_rank25_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --eps ${eps} --gpu_id ${gpu_id}"
  #   "python purify.py --config PTR3d_8_2048_rank30_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --eps ${eps} --gpu_id ${gpu_id}"
  #   "python purify.py --config PTR3d_8_2048_rank35_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --eps ${eps} --gpu_id ${gpu_id}"
  #   "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --eps ${eps} --gpu_id ${gpu_id}"
  # )
  
  # for cmd in "${purify_commands[@]}"; do
  #   ts="$(date '+%Y%m%d_%H%M%S')"
  #   log_file="${log_dir}/purify_${ts}.log"
  #   echo "[$(date '+%Y-%m-%d %H:%M:%S')] PURIFY: $cmd"
  #   run_bg "$cmd" "$log_file"
  #   sleep 60
  #   wait_for_slots "$purify_max_jobs"
  # done
  # wait

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE eps=${eps}"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL DONE."