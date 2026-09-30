"""EXP-035：保持 EXP-033 科学协议，仅扩展敏感性 seed 47–51。"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from rpcf.exp033_common import (ATTACK_PROTOCOL, WEIGHTS, loss_variants, read_json,
                               write_json, file_hash, fingerprint)

OLD_RUN = 'logs/exp033/exp033_full_20260923_v6'
SOURCE_RUN = 'logs/exp031/exp031_full_20260729_174215'
NEW_SEEDS = [47, 48, 49, 50, 51]
RANKS = [15, 20, 25, 30, 35, 40]


def templates():
    """冻结已完成的 seed42 命令模板；seed42–46 参数一致性另行审计。"""
    names = {
        'madry': 'train_standard_thubenchmark_eegnet_seed42_madry',
        'cache': 'rpcf_cache_thubenchmark_eegnet_seed42_madry',
        'caf': 'rpcf_at_thubenchmark_eegnet_seed42_rpcf_at',
        'madry_aa': 'attack_thubenchmark_eegnet_seed42_madry_autoattack',
        'madry_pgd': 'attack_thubenchmark_eegnet_seed42_madry_pgd',
        'default_pgd': 'attack_thubenchmark_eegnet_seed42_rpcf_at_pgd',
        'canonical': 'tnp_thubenchmark_eegnet_seed42_madry_autoattack',
        'default_tnp': 'tnp_thubenchmark_eegnet_seed42_rpcf_at_pgd',
    }
    result = {}
    for name, tid in names.items():
        p = Path(SOURCE_RUN) / 'status' / (tid + '.json')
        value = read_json(p)
        assert value['status'] == 'completed', p
        command = value['command']
        result[name] = dict(command=command[command.index('-m') + 1:],
                            source=str(p), sha256=file_hash(p))
    # EXP-031 的路径触发 batch 安全表及 S512 存储压缩；迁移目录必须显式保留。
    safe = Path(SOURCE_RUN) / 'actual_cache_attack_batch_sizes.json'
    result['effective_attack_batch'] = read_json(safe)['thubenchmark_eegnet']
    result['attack_batch_source'] = dict(path=str(safe), sha256=file_hash(safe))
    assert result['effective_attack_batch'] == 16
    return result


def build_tasks(manifest):
    """拆分资源依赖；cache rank 独立运行前必须有原随机流起点。"""
    root = Path(manifest['run_dir'])
    tasks = []
    def add(kind, seed, suffix='', deps=(), group='prerequisite', heavy=False, **kw):
        tid = '_'.join(str(x) for x in (kind, f'seed{seed}', suffix) if x != '')
        t = dict(task_id=tid, kind=kind, seed=seed, group=group, heavy=heavy,
                 dependencies=list(deps), output_path=str(root/'metrics'/f'{tid}.json'), **kw)
        tasks.append(t)
        return tid
    for seed in manifest['seeds']:
        prep = add('prepare', seed)
        madry = add('madry', seed, deps=[prep], heavy=True)
        base = add('cache_base', seed, deps=[madry], heavy=True)
        stream = add('cache_streams', seed, deps=[base])
        shards = [add('cache_rank', seed, rank, deps=[stream], rank=rank) for rank in RANKS]
        cache = add('cache_merge', seed, deps=shards, heavy=True)
        aa = add('madry_aa', seed, deps=[madry], heavy=True)
        canon = add('canonical', seed, deps=[aa])
        pgd = add('madry_pgd', seed, deps=[madry])
        # 训练只依赖同seed初始化和训练cache，无需等待测试净化。
        variants = loss_variants()[:2] if manifest['smoke'] else loss_variants()
        trains = {}
        for var in variants:
            trains[var['variant']] = add('train', seed, var['variant'], [cache],
                                         group='loss', heavy=True, **var)
        default_train = trains['default']
        da = add('default_pgd', seed, deps=[default_train], heavy=False)
        dt = add('default_tnp', seed, deps=[da, canon])
        src = add('sources', seed, deps=[cache, pgd, canon, default_train, da, dt])
        add('reference', seed, 'rank', [src], group='rank', method='rpcf_at')
        add('reference', seed, 'loss', [src], group='loss', method='rpcf_at', variant='default')
        for rank in ([15] if manifest['smoke'] else [15, 20, 35, 40]):
            add('tnp', seed, f'rank{rank}', [src], group='rank', method='rpcf_at', ranks=[rank])
        for var in variants[1:]:
            train = trains[var['variant']]
            attack = add('attack', seed, var['variant'], [train, pgd], group='loss',
                         training_task=train, **var)
            add('tnp', seed, var['variant'], [attack, src], group='loss', method='rpcf_at',
                ranks=[25, 30], attack_task=attack, training_task=train, **var)
    ids = [t['task_id'] for t in tasks]
    assert len(ids) == len(set(ids))
    assert not ({d for t in tasks for d in t['dependencies']} - set(ids))
    return tasks


def science_hashes():
    from rpcf.exp033 import science_hashes as original
    paths = set(original())
    paths.update(str(p) for p in Path('rpcf').glob('exp035*.py'))
    paths.update(['rpcf/run_exp035.sh', 'train_AT.py', 'rpcf/generate_cache.py',
                  'rpcf/evaluate_attack.py', 'rpcf/evaluate_purification.py',
                  'rpcf/parallel_exp033.py', 'rpcf/parallel_exp032.py', 'runtime_env.py'])
    return {p: file_hash(p) for p in sorted(paths)}


def create_plan(run_id, smoke=False):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Unsafe run id')
    root = Path('logs/exp035') / run_id
    old = Path(OLD_RUN)
    report = read_json(old/'summary/report.json')
    assert report['complete'] and not report['smoke'] and not report['errors']
    manifest = dict(experiment_id='EXP-035', run_id=run_id, run_dir=str(root), smoke=smoke,
                    seeds=[42] if smoke else NEW_SEEDS, retained_seeds=list(range(42,47)),
                    combined_seeds=list(range(42,52)), dataset='thubenchmark', model='eegnet',
                    fold=0, sample_num=2 if smoke else 512, old_run=OLD_RUN,
                    source_run='exp031_full_20260729_174215', external_run='exp032_full_20260917_2020',
                    groups=['rank','loss'], ranks=RANKS, loss_defaults=WEIGHTS,
                    attack_protocol=ATTACK_PROTOCOL, test_tuning=False,
                    templates=templates(), cache_rng='original_CPU_random_stream_per_rank',
                    training_resume='new_attempt_from_original_seed_and_Madry_checkpoint',
                    old_sources={str(p):file_hash(p) for p in
                                 (old/'summary/report.json', old/'summary/metrics_long.csv', old/'tasks.json')})
    tasks = build_tasks(manifest)
    manifest['task_count'] = len(tasks)
    manifest['counts'] = dict(Counter(t['kind'] for t in tasks))
    manifest['expected_new_rows'] = 10 if smoke else 275
    manifest['expected_combined_rows'] = None if smoke else 550
    frozen = {'manifest.json': manifest, 'tasks.json': tasks, 'source_sha256.json': science_hashes()}
    for name, value in frozen.items():
        p = root/name
        if p.exists() and read_json(p) != value:
            raise ValueError(f'Frozen {name} differs; use a new run id')
    for name,value in frozen.items():
        if not (root/name).exists(): write_json(root/name,value)
    return root, manifest, tasks


def verify_plan(root):
    manifest = read_json(root/'manifest.json')
    if read_json(root/'source_sha256.json') != science_hashes():
        raise ValueError('Frozen implementation changed; use a new run id')
    tasks = read_json(root/'tasks.json')
    if tasks != build_tasks(manifest): raise ValueError('DAG changed')
    for path,digest in manifest['old_sources'].items():
        if file_hash(path) != digest: raise ValueError(f'Old accepted source changed: {path}')
    return manifest,tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan','dry-run','run','summary','worker'])
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--gpu-ids',default='0,1,2,3,4,5,6,7')
    parser.add_argument('--max-workers',type=int,default=24)
    parser.add_argument('--light-per-gpu',type=int,default=3)
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--strict',action='store_true')
    parser.add_argument('--task-id')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id): parser.error('Unsafe run id')
    root=Path('logs/exp035')/args.run_id
    if args.action in ('plan','dry-run'):
        _,m,t=create_plan(args.run_id,args.smoke)
        print(dict(run_dir=str(root),counts=m['counts'],tasks=len(t),expected_new_rows=m['expected_new_rows']),flush=True)
    elif args.action=='worker':
        from rpcf.exp035_worker import execute
        execute(root,args.task_id)
    elif args.action=='summary':
        from rpcf.exp035_report import summarize
        summarize(root,strict=args.strict)
    else:
        from rpcf.exp035_scheduler import run
        run(root,[int(x) for x in args.gpu_ids.split(',')],args.max_workers,args.light_per_gpu,args.retry_failed)


if __name__=='__main__': main()
