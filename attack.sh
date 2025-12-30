target_pid=3938974
while kill -0 $target_pid 2> /dev/null; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $target_pid 仍在运行..."
  sleep 30
done

conda init

conda activate torch

commands=(
    # "python -u attack.py --dataset thubenchmark --model eegnet --fold 0 --attack fgsm" 
    # "python -u attack.py --dataset thubenchmark --model eegnet --fold 0 --attack pgd" 
    # "python -u attack.py --dataset thubenchmark --model eegnet --fold 0 --attack cw" 
    # "python -u attack.py --dataset thubenchmark --model eegnet --fold 0 --attack autoattack"
    "python -u attack.py --dataset thubenchmark --model eegnet --fold 0 --attack autoattack --eps 0.05"
    "python -u attack.py --dataset thubenchmark --model eegnet --fold 0 --attack autoattack --eps 0.75"
    "python -u attack.py --dataset thubenchmark --model eegnet --fold 0 --attack autoattack --eps 0.1"
)

# Run each command in the array in the background, with at most max_jobs concurrent jobs
max_jobs=1
for cmd in "${commands[@]}"; do
  nohup $cmd &
  sleep 1
  # Wait until there are fewer than max_jobs active background jobs
  while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do
    sleep 5
  done
done

# Wait for all remaining background jobs to finish
wait