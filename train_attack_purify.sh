target_pid=3938974
while kill -0 $target_pid 2> /dev/null; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $target_pid 仍在运行..."
  sleep 30
done

conda init

conda activate torch

dataset="thubenchmark"

commands=(
    # train
    "python -u train_AT.py --dataset ${dataset} --model eegnet --at_strategy madry"
    "python -u train_AT.py --dataset ${dataset} --model eegnet --at_strategy fbf"
    "python -u train_AT.py --dataset ${dataset} --model eegnet --at_strategy trades" 
    "python -u train_AT.py --dataset ${dataset} --model eegnet --at_strategy clean"

    # attack
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy clean --fold 0 --attack autoattack --eps 0.1"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy madry --fold 0 --attack autoattack --eps 0.1"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy fbf --fold 0 --attack autoattack --eps 0.1"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy trades --fold 0 --attack autoattack --eps 0.1"

    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy clean --fold 0 --attack fgsm --eps 0.1"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy madry --fold 0 --attack fgsm --eps 0.1"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy fbf --fold 0 --attack fgsm --eps 0.1"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy trades --fold 0 --attack fgsm --eps 0.1"

    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy clean --fold 0 --attack pgd --eps 0.1"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy madry --fold 0 --attack pgd --eps 0.1"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy fbf --fold 0 --attack pgd --eps 0.1"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy trades --fold 0 --attack pgd --eps 0.1"

    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy clean --fold 0 --attack cw --eps 0.1"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy madry --fold 0 --attack cw --eps 0.1"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy fbf --fold 0 --attack cw --eps 0.1"
    "python -u attack.py --dataset ${dataset} --model eegnet --at_strategy trades --fold 0 --attack cw --eps 0.1"

    # purify
    "python purify.py --config PTR3d_8_2048_rank5_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack"
    "python purify.py --config PTR3d_8_2048_rank10_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack"
    "python purify.py --config PTR3d_8_2048_rank15_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack"
    "python purify.py --config PTR3d_8_2048_rank20_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack"
    "python purify.py --config PTR3d_8_2048_rank25_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack"
    "python purify.py --config PTR3d_8_2048_rank30_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack"
    "python purify.py --config PTR3d_8_2048_rank35_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack"
    "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate.yaml --sample_num 512 --dataset ${dataset} --attack autoattack"
)

# Run each command in the array in the background, with at most max_jobs concurrent jobs
max_jobs=2
for cmd in "${commands[@]}"; do
  nohup $cmd &
  sleep 60
  # Wait until there are fewer than max_jobs active background jobs
  while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do
    sleep 60
  done
done

# Wait for all remaining background jobs to finish
wait

