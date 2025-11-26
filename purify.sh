target_pid=3938974
while kill -0 $target_pid 2> /dev/null; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $target_pid 仍在运行..."
  sleep 30
done

conda init

conda activate torch

commands=(
    # bciciv2a autoattack
    "python purify.py --config PTR_8_2048_rank10.yaml --sample_num 512 --dataset bciciv2a --attack autoattack"
    "python purify.py --config PTR_8_2048_rank12.yaml --sample_num 512 --dataset bciciv2a --attack autoattack"
    "python purify.py --config PTR_8_2048_rank14.yaml --sample_num 512 --dataset bciciv2a --attack autoattack"
    "python purify.py --config PTR_8_2048_rank16.yaml --sample_num 512 --dataset bciciv2a --attack autoattack"
    "python purify.py --config PTR_8_2048_rank18.yaml --sample_num 512 --dataset bciciv2a --attack autoattack"
    "python purify.py --config PTR_8_2048_rank20.yaml --sample_num 512 --dataset bciciv2a --attack autoattack"
    "python purify.py --config PTR_8_2048_rank22.yaml --sample_num 512 --dataset bciciv2a --attack autoattack"
    "python purify.py --config PTR_8_2048_rank24.yaml --sample_num 512 --dataset bciciv2a --attack autoattack"
    "python purify.py --config PTR_8_2048_rank26.yaml --sample_num 512 --dataset bciciv2a --attack autoattack"
    "python purify.py --config PTR_8_2048_rank28.yaml --sample_num 512 --dataset bciciv2a --attack autoattack"
    "python purify.py --config PTR_8_2048_rank30.yaml --sample_num 512 --dataset bciciv2a --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank40.yaml --sample_num 512 --dataset bciciv2a --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank50.yaml --sample_num 512 --dataset bciciv2a --attack autoattack"

    # # thubenchmark autoattack
    # "python purify.py --config PTR_8_2048_rank10.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank20.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank30.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank40.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank50.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # # seediv autoattack
    # "python purify.py --config PTR_8_2048_rank10.yaml --sample_num 512 --dataset seediv --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank20.yaml --sample_num 512 --dataset seediv --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank30.yaml --sample_num 512 --dataset seediv --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank40.yaml --sample_num 512 --dataset seediv --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank50.yaml --sample_num 512 --dataset seediv --attack autoattack"

    # # m3cv autoattack
    # "python purify.py --config PTR_8_2048_rank10.yaml --sample_num 512 --dataset m3cv --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank20.yaml --sample_num 512 --dataset m3cv --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank30.yaml --sample_num 512 --dataset m3cv --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank40.yaml --sample_num 512 --dataset m3cv --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank50.yaml --sample_num 512 --dataset m3cv --attack autoattack"
)

# Run each command in the array in the background, with at most max_jobs concurrent jobs
max_jobs=3
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