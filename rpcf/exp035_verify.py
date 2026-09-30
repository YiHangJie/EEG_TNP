"""真实CUDA前置验收：缓存串行/分rank逐位对照、partial恢复、2048步RNG校验。"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def run(out):
    import torch
    import rpcf.generate_cache as engine
    from rpcf.exp035_cache import run_cache,plan_streams,advance_pairs,rng_equal
    from rpcf.exp033_common import config_for_rank,set_option,write_json,file_hash
    from rpcf.exp032_tnp import rng_state,restore_rng
    from utils.reproducibility import seed_everything
    from rpcf.exp031 import checkpoint
    from purify import purify
    from types import SimpleNamespace
    if torch.cuda.device_count()!=1: raise ValueError('Require one visible CUDA GPU')
    torch.set_num_threads(2)
    if out.exists(): raise FileExistsError('Use a fresh verification output directory')
    out.mkdir(parents=True)
    ranks=[15,20,25,30,35,40]
    manifest=dict(run_dir=str(out),smoke=True)
    configs=[config_for_rank(manifest,r) for r in ranks]
    cp=checkpoint('exp031_full_20260729_174215','thubenchmark','eegnet',42,'madry')
    common=['--dataset','thubenchmark','--model','eegnet','--fold','0','--seed','42',
            '--attack','autoattack','--eps','.03','--checkpoint_path',cp,'--sample_num','2',
            '--attack_batch_size','2','--ranks',','.join(map(str,ranks)),
            '--configs',','.join(configs),'--gpu_id','0','--checkpoint_every','1','--tag','exp035_equivalence']
    serial=out/'serial.pth';split=out/'split.pth'
    argv=sys.argv
    try:
        sys.argv=['rpcf.generate_cache',*common,'--output_path',str(serial),'--keep_work_dir']
        engine.main()
    finally: sys.argv=argv
    original=torch.load(serial,map_location='cpu',weights_only=False)
    args=[*common,'--output_path',str(split)]
    run_cache([*args,'--base_only'],'base',out/'base.log')
    base=torch.load(str(split)+'.work/base.pth',map_location='cpu',weights_only=False)
    for key in ('x','x_adv','labels'): assert torch.equal(original[key],base[key]),key
    plan_streams(split,configs,ranks)
    plan_streams(split,configs,ranks)  # 规划阶段恢复必须幂等。
    resumed=False
    for rank,config in zip(ranks,configs):
        ra=set_option(set_option(args,'--ranks',rank),'--configs',config)+['--rank_shard_only']
        if rank==25:
            save=engine.atomic_torch_save
            def interrupt_after_saved(value,path):
                save(value,path)
                if str(path).endswith('rank25.partial.pth') and value['completed']==1:
                    raise InterruptedError('intentional_resume_check')
            try:
                engine.atomic_torch_save=interrupt_after_saved
                try: run_cache(ra,'rank',out/f'rank{rank}.log')
                except InterruptedError: resumed=True
            finally: engine.atomic_torch_save=save
            seed_everything(999)  # 验证恢复使用记录状态，不依赖新进程的当前状态。
        run_cache(ra,'rank',out/f'rank{rank}.log')
    run_cache([*args,'--finalize_only'],'merge',out/'merge.log')
    sharded=torch.load(split,map_location='cpu',weights_only=False)
    for key in ('x','x_adv','labels','x_pur_by_rank','x_adv_pur_by_rank'):
        assert torch.equal(original[key],sharded[key]),f'Bitwise cache mismatch: {key}'
    assert original['source_indices']==sharded['source_indices'] and resumed
    checks=[]
    device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats(device)
    for rank in ranks:
        config=f'configs/thubenchmark/PTR3d_8_2048_rank{rank}_3d_interpolate.yaml'
        args=SimpleNamespace(dataset='thubenchmark',config=config,visualize=False,seed=42)
        seed_everything(47)
        before=rng_state()
        purify(args,0,base['x'][0],250,device,logging)
        purify(args,512,base['x_adv'][0],250,device,logging)
        actual=rng_state()
        restore_rng(before);advance_pairs(config,1)
        assert rng_equal(actual,rng_state()),f'2048-step RNG mismatch at rank {rank}'
        checks.append(dict(rank=rank,iterations=2048,full_rng_equal=True))
        print(f'FORMAL_RNG_PASS rank={rank}',flush=True)
    report=dict(result='PASS',device=torch.cuda.get_device_name(0),seed42_smoke_cache_bitwise_equal=True,
                samples=2,ranks=ranks,smoke_iterations=40,partial_resume_bitwise_equal=resumed,
                planner_restart_idempotent=True,formal_2048_step_rng_checks=checks,
                peak_memory_mib=torch.cuda.max_memory_allocated(device)/1024**2,
                serial_sha256=file_hash(serial),split_sha256=file_hash(split),
                note='Tensor fields match bitwise; file hashes differ because provenance/output paths differ.')
    write_json(out/'verification.json',report)
    print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args().output_dir)
