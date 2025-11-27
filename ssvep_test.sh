target_pid=643259
while kill -0 $target_pid 2> /dev/null; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $target_pid 仍在运行..."
  sleep 30
done

conda init

conda activate torch

commands=(
    # "python -u train.py --dataset thubenchmark --model eegnet --lr 0.01 --weight_decay 0"
    # "python -u train.py --dataset thubenchmark --model eegnet --lr 0.001 --weight_decay 0"
    # "python -u train.py --dataset thubenchmark --model eegnet --lr 0.0001 --weight_decay 0"
    # "python -u train.py --dataset thubenchmark --model eegnet --lr 0.001 --weight_decay 0.1"
    # "python -u train.py --dataset thubenchmark --model eegnet --lr 0.001 --weight_decay 0.01"
    # "python -u train.py --dataset thubenchmark --model eegnet --lr 0.001 --weight_decay 0.001"
    # "python -u train.py --dataset thubenchmark --model eegnet --lr 0.001 --weight_decay 0.0001"
    # "python -u train.py --dataset thubenchmark --model eegnet --lr 0.001 --weight_decay 0.00001"

    # "python -u train.py --dataset thubenchmark --model tsception --lr 0.01 --weight_decay 0"
    # "python -u train.py --dataset thubenchmark --model tsception --lr 0.001 --weight_decay 0"
    # "python -u train.py --dataset thubenchmark --model tsception --lr 0.0001 --weight_decay 0"
    # "python -u train.py --dataset thubenchmark --model tsception --lr 0.001 --weight_decay 0.1"
    # "python -u train.py --dataset thubenchmark --model tsception --lr 0.001 --weight_decay 0.01"
    # "python -u train.py --dataset thubenchmark --model tsception --lr 0.001 --weight_decay 0.001"
    # "python -u train.py --dataset thubenchmark --model tsception --lr 0.001 --weight_decay 0.0001"
    # "python -u train.py --dataset thubenchmark --model tsception --lr 0.001 --weight_decay 0.00001"

    # "python -u train.py --dataset thubenchmark --model conformer --lr 0.01 --weight_decay 0"
    # "python -u train.py --dataset thubenchmark --model conformer --lr 0.001 --weight_decay 0"
    # "python -u train.py --dataset thubenchmark --model conformer --lr 0.0001 --weight_decay 0"
    # "python -u train.py --dataset thubenchmark --model conformer --lr 0.001 --weight_decay 0.1"
    # "python -u train.py --dataset thubenchmark --model conformer --lr 0.001 --weight_decay 0.01"
    # "python -u train.py --dataset thubenchmark --model conformer --lr 0.001 --weight_decay 0.001"
    # "python -u train.py --dataset thubenchmark --model conformer --lr 0.001 --weight_decay 0.0001"
    # "python -u train.py --dataset thubenchmark --model conformer --lr 0.001 --weight_decay 0.00001"

    "python -u train.py --dataset thubenchmark --model atcnet --lr 0.01 --weight_decay 0"
    "python -u train.py --dataset thubenchmark --model atcnet --lr 0.001 --weight_decay 0"
    "python -u train.py --dataset thubenchmark --model atcnet --lr 0.0001 --weight_decay 0"
    "python -u train.py --dataset thubenchmark --model atcnet --lr 0.001 --weight_decay 0.1"
    "python -u train.py --dataset thubenchmark --model atcnet --lr 0.001 --weight_decay 0.01"
    "python -u train.py --dataset thubenchmark --model atcnet --lr 0.001 --weight_decay 0.001"
    "python -u train.py --dataset thubenchmark --model atcnet --lr 0.001 --weight_decay 0.0001"
    "python -u train.py --dataset thubenchmark --model atcnet --lr 0.001 --weight_decay 0.00001"
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