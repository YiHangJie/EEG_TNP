target_pid=3938974
while kill -0 $target_pid 2> /dev/null; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $target_pid 仍在运行..."
  sleep 30
done

conda init

conda activate torch

commands=(
    # "python -u train.py --dataset bciciv2a --model eegnet"

    # "python -u train.py --dataset bciciv2a --model tsception"

    # "python -u train.py --dataset bciciv2a --model atcnet"

    # "python -u train.py --dataset bciciv2a --model conformer"

    # "python -u train.py --dataset thubenchmark --model eegnet" 

    "python -u train.py --dataset thubenchmark --model tsception"

    # "python -u train.py --dataset thubenchmark --model atcnet"

    "python -u train.py --dataset thubenchmark --model conformer"

    # "python -u train.py --dataset seediv --model eegnet" 

    # "python -u train.py --dataset seediv --model tsception"

    # "python -u train.py --dataset seediv --model atcnet"

    # "python -u train.py --dataset seediv --model conformer"

    "python -u train.py --dataset m3cv --model eegnet" 

    "python -u train.py --dataset m3cv --model tsception"

    # "python -u train.py --dataset m3cv --model atcnet"

    "python -u train.py --dataset m3cv --model conformer"
)

# Run each command in the array in the background, with at most max_jobs concurrent jobs
max_jobs=1
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