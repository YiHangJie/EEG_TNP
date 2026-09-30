"""对原训练缓存做保持串行 RNG 的 rank 调度拆分；不改净化算法。"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from rpcf.exp033_common import file_hash, fingerprint, read_json, write_json


def configuration(path):
    import yaml
    from TN.opt import Config
    cfg = Config()
    path = Path(path)
    if not path.is_file(): path = Path('configs/thubenchmark') / path
    for key,value in yaml.safe_load(path.read_text()).items(): setattr(cfg,key,value)
    if cfg.model != 'PTR_3d' or cfg.strategy != '3d_interpolate' or cfg.dtype != 'float32':
        raise ValueError('RNG advance is only validated for the frozen PTR_3d float32 path')
    return cfg


def advance_pairs(config, count, shape=(10,11,2048)):
    """沿用原构造函数随机调用；每例 clean+adv 各初始化两套 TN。"""
    import torch
    from TN.tn_utils import get_rr_template_qtr_eeg_3d
    cfg = configuration(config)
    torch.set_default_dtype(torch.float32)  # 与 BaseTNModel 一致。
    for _ in range(int(count)*4):
        get_rr_template_qtr_eeg_3d(*shape, cfg.stage, cfg.max_rank,
                                  dim=cfg.dimensions, sigma_init=cfg.sigma_init, device='cpu')


def plan_streams(cache_path, configs, ranks):
    """从真实 AutoAttack 完成点保存各 rank 的原始串行 RNG 起点。"""
    import torch
    from purify import interpolate
    from types import SimpleNamespace
    from rpcf.exp032_tnp import rng_state, restore_rng
    from rpcf.exp031_artifacts import atomic_torch_save
    if len(configs)!=len(ranks) or len(set(ranks))!=len(ranks): raise ValueError('Invalid rank/config pairing')
    if any(configuration(c).max_rank!=r for r,c in zip(ranks,configs)): raise ValueError('Config rank differs')
    work = Path(str(cache_path)+'.work')
    base = torch.load(work/'base.pth',map_location='cpu',weights_only=False)
    if '_exp035_rng' not in base: raise ValueError('Base is missing the captured original RNG')
    before = rng_state()
    try:
        restore_rng(base['_exp035_rng'])
        # interpolate 不消费随机数，得到真实公共表示而不是根据原始输入形状猜测。
        args = SimpleNamespace(dataset='thubenchmark',config=configs[0])
        shape = tuple(interpolate(args,base['x'][0],250).shape)
        if shape != (10,11,2048): raise ValueError(f'Unexpected representation: {shape}')
        identity = fingerprint(dict(base_sha256=file_hash(work/'base.pth'), ranks=ranks,
                                    configs={p:file_hash(Path(p) if Path(p).is_file() else Path('configs/thubenchmark')/p) for p in configs},
                                    samples=len(base['x']),shape=shape))
        paths = []
        for rank,config in zip(ranks,configs):
            dest = work / f'rank{rank}.rng.pth'
            if dest.exists():
                previous=torch.load(dest,map_location='cpu',weights_only=False)
                if previous['identity']!=identity or previous['rank']!=rank or not rng_equal(previous['rng'],rng_state()):
                    raise ValueError('Existing rank RNG differs; preserve it and use a new run')
            else: atomic_torch_save(dict(rng=rng_state(),identity=identity,rank=rank),dest)
            paths.append(str(dest))
            advance_pairs(config,len(base['x']),shape)
            print(f'CACHE_RNG rank={rank} exact_serial_stream_ready',flush=True)
        write_json(work/'rng_manifest.json',dict(identity=identity,ranks=ranks,configs=configs,
                                                sample_num=len(base['x']),shape=shape,paths=paths))
        return paths
    finally:
        restore_rng(before)


def run_cache(arguments, mode, log_path):
    """原 main 原样执行；仅在确定边界捕获/恢复 RNG，给新断点补随机状态。"""
    import torch
    import rpcf.generate_cache as engine
    from rpcf.exp032_tnp import rng_state, restore_rng
    expected={'base':'--base_only','rank':'--rank_shard_only','merge':'--finalize_only'}
    if mode not in expected or expected[mode] not in arguments or sum(flag in arguments for flag in expected.values())!=1:
        raise ValueError('Cache execution mode and command flags differ')
    if any(x.startswith('--shared_clean_path') for x in arguments): raise ValueError('RNG split requires clean+adv purification')
    if any(x.startswith('--ranks=') for x in arguments): raise ValueError('Use separate rank option/value tokens')
    if torch.cuda.device_count()!=1: raise ValueError('Require one visible GPU for compatible CUDA RNG snapshots')
    output = Path(arguments[arguments.index('--output_path')+1])
    work = Path(str(output)+'.work')
    rank = int(arguments[arguments.index('--ranks')+1]) if mode=='rank' else None
    stream = None
    if rank is not None:
        stream = torch.load(work/f'rank{rank}.rng.pth',map_location='cpu',weights_only=False)
        frozen = read_json(work/'rng_manifest.json')
        if stream['identity'] != frozen['identity'] or stream['rank'] != rank:
            raise ValueError('Rank stream identity differs')
    original_save, original_shared, original_log = engine.atomic_torch_save, engine.load_shared_clean_cache, logging.basicConfig
    original_argv = sys.argv
    def save(payload,path):
        p=Path(path)
        if p == work/'base.pth' or p.name.endswith('.partial.pth'):
            payload={**payload,'_exp035_rng':rng_state(),
                     '_exp035_stream_identity':stream['identity'] if stream else None}
        return original_save(payload,path)
    def shared(*args,**kwargs):
        result=original_shared(*args,**kwargs)
        if mode=='rank':
            partial=work/f'rank{rank}.partial.pth'
            if partial.exists():
                value=torch.load(partial,map_location='cpu',weights_only=False)
                if value.get('_exp035_stream_identity') != stream['identity']:
                    raise ValueError('Partial RNG does not belong to this base/config stream')
                if '_exp035_rng' not in value: raise ValueError('Partial is missing RNG; preserve it and use a new run')
                restore_rng(value['_exp035_rng'])
            else: restore_rng(stream['rng'])
        return result
    def log_config(*args,**kwargs):
        # 同seed六rank可同时启动，原秒级文件名会碰撞；仅重定向新增任务日志。
        kwargs['filename']=str(log_path)
        kwargs['filemode']='a'
        kwargs['force']=True
        return original_log(*args,**kwargs)
    try:
        engine.atomic_torch_save=save
        engine.load_shared_clean_cache=shared
        logging.basicConfig=log_config
        sys.argv=['rpcf.generate_cache',*arguments,'--keep_work_dir']
        engine.main()
    finally:
        engine.atomic_torch_save=original_save
        engine.load_shared_clean_cache=original_shared
        logging.basicConfig=original_log
        sys.argv=original_argv


def rng_equal(a,b):
    """对完整随机状态逐域比较，用于断点和并行数值验收。"""
    import torch
    import numpy as np
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray): return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict): return set(a)==set(b) and all(rng_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(rng_equal(x,y) for x,y in zip(a,b))
    return a==b
