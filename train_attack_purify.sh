target_pid=3938974
while kill -0 $target_pid 2> /dev/null; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $target_pid 仍在运行..."
  sleep 30
done

conda init

conda activate torch

dataset="${1:-thubenchmark}"
gpu_id="${2:-0}"

train_commands=(
    # train
    # "python -u train_AT.py --dataset ${dataset} --model eegnet --at_strategy madry --gpu_id ${gpu_id}"
    # "python -u train_AT.py --dataset ${dataset} --model eegnet --at_strategy fbf --gpu_id ${gpu_id}"
    # "python -u train_AT.py --dataset ${dataset} --model eegnet --at_strategy trades --gpu_id ${gpu_id}" 
    # "python -u train_AT.py --dataset ${dataset} --model eegnet --at_strategy clean --gpu_id ${gpu_id}"
)

attack_commands=(
    # attack
    # "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy clean --fold 0 --attack autoattack --eps 0.1 --gpu_id ${gpu_id}"
    # "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy madry --fold 0 --attack autoattack --eps 0.1 --gpu_id ${gpu_id}"
    # "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy fbf --fold 0 --attack autoattack --eps 0.1 --gpu_id ${gpu_id}"
    # "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy trades --fold 0 --attack autoattack --eps 0.1 --gpu_id ${gpu_id}"

    # "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy clean --fold 0 --attack fgsm --eps 0.1 --gpu_id ${gpu_id}"
    # "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy madry --fold 0 --attack fgsm --eps 0.1 --gpu_id ${gpu_id}"
    # "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy fbf --fold 0 --attack fgsm --eps 0.1 --gpu_id ${gpu_id}"
    # "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy trades --fold 0 --attack fgsm --eps 0.1 --gpu_id ${gpu_id}"

    # "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy clean --fold 0 --attack pgd --eps 0.1 --gpu_id ${gpu_id}"
    # "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy madry --fold 0 --attack pgd --eps 0.1 --gpu_id ${gpu_id}"
    # "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy fbf --fold 0 --attack pgd --eps 0.1 --gpu_id ${gpu_id}"
    # "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy trades --fold 0 --attack pgd --eps 0.1 --gpu_id ${gpu_id}"

    # "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy clean --fold 0 --attack cw --eps 0.1 --gpu_id ${gpu_id}"
    # "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy madry --fold 0 --attack cw --eps 0.1 --gpu_id ${gpu_id}"
    # "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy fbf --fold 0 --attack cw --eps 0.1 --gpu_id ${gpu_id}"
    # "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy trades --fold 0 --attack cw --eps 0.1 --gpu_id ${gpu_id}"
)

purify_commands=(
    # purify
    # "python purify.py --config PTR3d_8_2048_rank5_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --gpu_id ${gpu_id}"
    # "python purify.py --config PTR3d_8_2048_rank10_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --gpu_id ${gpu_id}"
    # "python purify.py --config PTR3d_8_2048_rank15_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --gpu_id ${gpu_id}"
    # "python purify.py --config PTR3d_8_2048_rank20_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --gpu_id ${gpu_id}"
    # "python purify.py --config PTR3d_8_2048_rank25_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --gpu_id ${gpu_id}"
    # "python purify.py --config PTR3d_8_2048_rank30_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --gpu_id ${gpu_id}"
    # "python purify.py --config PTR3d_8_2048_rank35_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --gpu_id ${gpu_id}"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --gpu_id ${gpu_id}"
    "python purify.py --config PTR3d_8_2048_rank45_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --gpu_id ${gpu_id}"
    "python purify.py --config PTR3d_8_2048_rank50_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --gpu_id ${gpu_id}"
    "python purify.py --config PTR3d_8_2048_rank55_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --gpu_id ${gpu_id}"
    "python purify.py --config PTR3d_8_2048_rank60_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --gpu_id ${gpu_id}"
    # "python purify.py --config PTR3d_8_2048_rank30_3d_interpolate_TVft0.01.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --gpu_id ${gpu_id}"
    # "python purify.py --config PTR3d_8_2048_rank30_3d_interpolate_TVft0.02.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --gpu_id ${gpu_id}"
    # "python purify.py --config PTR3d_8_2048_rank30_3d_interpolate_TVft0.03.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --gpu_id ${gpu_id}"
    # "python purify.py --config PTR3d_8_2048_rank30_3d_interpolate_TVft0.04.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --gpu_id ${gpu_id}"
    # "python purify.py --config PTR3d_8_2048_rank30_3d_interpolate_TVft0.05.yaml --sample_num 512 --dataset ${dataset} --attack autoattack --gpu_id ${gpu_id}"
)

# Run each command in the array in the background, with at most max_jobs concurrent jobs
train_max_jobs=1
attack_max_jobs=1
purify_max_jobs=2

# for cmd in "${train_commands[@]}"; do
#   nohup $cmd &
#   sleep 60
#   # Wait until there are fewer than max_jobs active background jobs
#   while [ "$(jobs -rp | wc -l)" -ge "$train_max_jobs" ]; do
#     sleep 60
#   done
# done
# wait

# for cmd in "${attack_commands[@]}"; do
#   nohup $cmd &
#   sleep 60
#   # Wait until there are fewer than max_jobs active background jobs
#   while [ "$(jobs -rp | wc -l)" -ge "$attack_max_jobs" ]; do
#     sleep 60
#   done
# done
# wait

for cmd in "${purify_commands[@]}"; do
  nohup $cmd &
  sleep 60
  # Wait until there are fewer than max_jobs active background jobs
  while [ "$(jobs -rp | wc -l)" -ge "$purify_max_jobs" ]; do
    sleep 60
  done
done
wait

# nohup bash train_attack_purify.sh thubenchmark 0 > train_attack_purify_thubenchmark_gpu0.log 2>&1 &
# nohup bash train_attack_purify.sh bciciv2a 1 > train_attack_purify_bciciv2a_gpu1.log 2>&1 &
# nohup bash train_attack_purify.sh m3cv 2 > train_attack_purify_m3cv_gpu2.log 2>&1 &
# nohup bash train_attack_purify.sh seediv 3 > train_attack_purify_seediv_gpu3.log 2>&1 &