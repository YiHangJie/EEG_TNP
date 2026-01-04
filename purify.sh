target_pid=882405
while kill -0 $target_pid 2> /dev/null; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $target_pid 仍在运行..."
  sleep 30
done

conda init

conda activate torch

commands=(
    # thubenchmark autoattack
    # "python purify.py --config PTR3d_8_2048_rank5_3d_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank10_3d_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank15_3d_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank20_3d_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank25_3d_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank30_3d_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank35_3d_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank30_3d_interpolate_TVft0.001.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank30_3d_interpolate_TVft0.005.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank30_3d_interpolate_TVft0.01.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank30_3d_interpolate_TVft0.02.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank30_3d_interpolate_TVft0.03.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank30_3d_interpolate_TVft0.04.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank30_3d_interpolate_TVft0.05.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TVft0.01.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TVft0.02.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TVft0.03.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TVft0.04.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TVft0.05.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TVft0.06.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TVft0.07.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TVft0.08.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TVft0.09.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
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

# sudo /sbin/shutdown -h +1