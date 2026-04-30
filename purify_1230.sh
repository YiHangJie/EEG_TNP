target_pid=882405
while kill -0 $target_pid 2> /dev/null; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $target_pid 仍在运行..."
  sleep 30
done

conda init

conda activate torch

commands=(
    # thubenchmark autoattack
    # "python purify.py --config PTR_8_2048_rank5_fft.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank15_fft.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank20_fft.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank15_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank20_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank10_fft_c-1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_c0.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_c1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_c2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_c3.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_interpolate_c-1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_interpolate_c0.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_interpolate_c1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_interpolate_c2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_interpolate_c3.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank15_fft_c-1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank15_fft_c0.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank15_fft_c1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank15_fft_c2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank15_fft_c3.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank15_interpolate_c-1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank15_interpolate_c0.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank15_interpolate_c1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank15_interpolate_c2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank15_interpolate_c3.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank10_fft_stage1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_stage2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_stage3.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_stage4.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_stage5.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank10_fft_stage4_c0.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_stage5_c1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_stage6_c2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # "python purify.py --config PTR3d_8_2048_rank2_fft.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank3_fft.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank5_fft.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank6_fft.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank7_fft.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank8_fft.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank9_fft.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank10_fft.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank15_fft.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank20_fft.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank25_fft.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank30_fft.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank35_fft.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_fft.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank5_fft_c1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank5_fft_c2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank5_fft_c3.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank5_fft_stage1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank5_fft_stage2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank5_fft_stage3.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank5_fft_stage5.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # "python purify.py --config PTR3d_8_2048_rank2_3d_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank3_3d_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank5_3d_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank10_3d_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank15_3d_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank20_3d_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank25_3d_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank30_3d_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank35_3d_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TVeeg1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TVeeg2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TVeeg3.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TVeeg4.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TVeeg5.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TV0.1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TV0.2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TV0.3.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TV0.4.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TV0.5.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    "python purify.py --config PTR3d_8_2048_rank25_3d_interpolate_TVft0.01.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank25_3d_interpolate_TVft0.02.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank25_3d_interpolate_TVft0.03.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    "python purify.py --config PTR3d_8_2048_rank25_3d_interpolate_TVft0.04.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    "python purify.py --config PTR3d_8_2048_rank25_3d_interpolate_TVft0.05.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank25_3d_interpolate_TVft0.001.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TVft0.01.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TVft0.02.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TVft0.03.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TVft0.04.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TVft0.05.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_3d_interpolate_TVft0.001.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # "python purify.py --config PTR3d_8_2048_rank10_fft_c1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank15_fft_c1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank20_fft_c1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank25_fft_c1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank30_fft_c1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank10_fft_c2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank15_fft_c2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank20_fft_c2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank25_fft_c2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank30_fft_c2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank10_fft_dwt_stage1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_dwt_stage2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_dwt_stage3.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_dwt_stage5.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_dwt_stage6.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_dwt_c-1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_dwt_c1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_dwt_c2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # "python purify.py --config PTRtfs_8_2048_rank10.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTRtfs_8_2048_rank15.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTRtfs_8_2048_rank20.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTRtfs_8_2048_rank25.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTRtfs_8_2048_rank30.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTRtfs_8_2048_rank40_0.01.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTRtfs_8_2048_rank40_0.03.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank10_fft_c2_TV0.01.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_c2_TV0.02.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_c2_TV0.03.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_c2_TV0.04.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_c2_TV0.05.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_c2_TV0.06.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank10_fft_c2_stage1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_c2_stage2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_c2_stage3.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_c2_stage5.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank10_fft_c2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_c0_stage3.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank10_fft_c4_stage5.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # "python purify.py --config PTRtfs_8_2048_rank5.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTRtfs_8_2048_rank10.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTRtfs_8_2048_rank15.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTRtfs_8_2048_rank20.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTRtfs_8_2048_rank25.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTRtfs_8_2048_rank30.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTRtfs_8_2048_rank10_c1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTRtfs_8_2048_rank15_c1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTRtfs_8_2048_rank20_c1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTRtfs_8_2048_rank25_c1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTRtfs_8_2048_rank30_c1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # "python purify.py --config PTR_8_2048_rank20_fft_TV1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank20_fft_TV2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank20_fft_TV3.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank20_fft_TV4.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank20_fft_TV5.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank20_fft_TVeeg1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank20_fft_TVeeg2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank20_fft_TVeeg3.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank20_fft_TVeeg4.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR_8_2048_rank20_fft_TVeeg5.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"

    # "python purify.py --config PTR3d_8_2048_rank40_fft_TV1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_fft_TV2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_fft_TV3.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_fft_TV4.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_fft_TV5.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_fft_TV0.1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_fft_TV0.2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_fft_TV0.3.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_fft_TV0.4.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_fft_TV0.5.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_fft_TVeeg1.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_fft_TVeeg2.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_fft_TVeeg3.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_fft_TVeeg4.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
    # "python purify.py --config PTR3d_8_2048_rank40_fft_TVeeg5.yaml --sample_num 512 --dataset thubenchmark --attack autoattack"
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