target_pid=1963494
while kill -0 $target_pid 2> /dev/null; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $target_pid 仍在运行..."
  sleep 30
done

conda init

conda activate torch

commands=(
    # "python -u train_AT.py --dataset bciciv2a --model eegnet --at_strategy madry"
    # "python -u train_AT.py --dataset bciciv2a --model eegnet --at_strategy fbf"
    # "python -u train_AT.py --dataset bciciv2a --model eegnet --at_strategy trades" 
    # "python -u train_AT.py --dataset bciciv2a --model eegnet --at_strategy clean"

    # "python -u train_AT.py --dataset thubenchmark --model eegnet --at_strategy madry"
    # "python -u train_AT.py --dataset thubenchmark --model eegnet --at_strategy fbf"
    # "python -u train_AT.py --dataset thubenchmark --model eegnet --at_strategy trades" 
    "python -u train_AT.py --dataset thubenchmark --model eegnet --at_strategy clean"

    # "python -u train_AT.py --dataset thubenchmark --model eegnet --at_strategy madry --epsilon 0.05"
    # "python -u train_AT.py --dataset thubenchmark --model eegnet --at_strategy fbf --epsilon 0.05"
    # "python -u train_AT.py --dataset thubenchmark --model eegnet --at_strategy trades --epsilon 0.05" 

    # "python -u train_AT.py --dataset thubenchmark --model eegnet --at_strategy madry --epsilon 0.01"
    # "python -u train_AT.py --dataset thubenchmark --model eegnet --at_strategy fbf --epsilon 0.01"
    # "python -u train_AT.py --dataset thubenchmark --model eegnet --at_strategy trades --epsilon 0.01" 

    "python -u train_AT.py --dataset thubenchmark --model eegnet --at_strategy madry --epsilon 0.03"
    "python -u train_AT.py --dataset thubenchmark --model eegnet --at_strategy fbf --epsilon 0.03"
    "python -u train_AT.py --dataset thubenchmark --model eegnet --at_strategy trades --epsilon 0.03" 
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