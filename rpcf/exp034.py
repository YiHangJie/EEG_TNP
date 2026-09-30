"""EXP-034：补充净化器+Madry、PGD-10 baseline 和时间张量化 TR。

独立建档/调度/汇总，不改 EXP-031/032/033 的代码、产物或运行状态。
"""
import argparse
import csv
import fcntl
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from rpcf.exp033_common import read_json, write_json, file_hash, fingerprint
from rpcf.exp033 import idle_gpus

DATASETS = ('thubenchmark', 'seediv', 'bciciv2a')
MODELS = ('eegnet', 'deepconvnet', 'tsception', 'atcnet', 'conformer', 'tcnet')
SEEDS = (42, 43, 44, 45, 46)
RAW = ('clean', 'madry', 'trades', 'fbf', 'ea_forward')
ATTACKS = ('fgsm', 'pgd', 'pgd_l2', 'autoattack', 'cw')
SOURCE_RUN = 'exp031_full_20260729_174215'
EXTERNAL_RUN = 'exp032_full_20260917_2020'
STRUCTURE_RUN = 'exp033_full_20260923_v6'


def build_tasks(manifest):
    """主实验覆盖全矩阵；跨攻击/adaptive和新结构只覆盖THU/EEGNet。"""
    root, tasks = Path(manifest['run_dir']), []
    def add(kind, dataset, model, seed, suffix='', dependencies=(), **kw):
        tid = '_'.join(str(x) for x in (kind, dataset, model, f'seed{seed}', suffix) if x != '')
        task = dict(task_id=tid, kind=kind, dataset=dataset, model=model, seed=seed,
                    dependencies=list(dependencies), output_path=str(root/'metrics'/f'{tid}.json'), **kw)
        tasks.append(task)
        return tid
    for ds in manifest['datasets']:
        for model in manifest['models']:
            for seed in manifest['seeds']:
                focus = ds == 'thubenchmark' and model == 'eegnet'
                add('external', ds, model, seed, attacks=list(ATTACKS) if focus else ['pgd'],
                    adaptive=focus, classifier_method='madry', purifiers=['magnet', 'dcae'])
    for seed in manifest['seeds']:
        for method in RAW:
            add('raw', 'thubenchmark', 'eegnet', seed, method, method=method)
        cal = add('calibrate', 'thubenchmark', 'eegnet', seed)
        for budget in (25, 30):
            sid = add('structure', 'thubenchmark', 'eegnet', seed, budget,
                      dependencies=[cal], calibration_task=cal, budget_rank=budget, method='tr_time')
            add('timing', 'thubenchmark', 'eegnet', seed, budget,
                dependencies=[sid], calibration_task=cal, budget_rank=budget, method='tr_time')
    return tasks


def expected_keys(task):
    """逐任务精确验收方法/攻击/评估方式，不能以相同行数掩盖重复或缺失。"""
    if task['kind'] == 'external':
        keys = [(f'madry_{p}', a, 'nonadaptive_purification', None)
                for p in task['purifiers'] for a in task['attacks']]
        if task['adaptive']:
            keys += [(f'madry_{p}', 'pgd10', 'adaptive_exact_gradient', None) for p in task['purifiers']]
        return keys
    if task['kind'] == 'raw':
        return [(task['method'], 'pgd10', 'classifier_whitebox', None)]
    if task['kind'] in ('structure', 'timing'):
        return [('tr_time', 'pgd', 'timing' if task['kind'] == 'timing' else 'nonadaptive_purification', task['budget_rank'])]
    return []


def science_hashes():
    files = set(Path('rpcf').glob('exp034*.py'))
    files |= {Path(p) for p in ('rpcf/run_exp034.sh', 'rpcf/exp033.py', 'rpcf/exp033_common.py',
        'rpcf/exp033_structures.py', 'rpcf/exp033_worker.py', 'rpcf/exp032_common.py', 'rpcf/exp032_purifiers.py',
        'rpcf/exp032_external_pgd10.py', 'rpcf/exp032_evaluate.py', 'rpcf/exp032_tnp.py', 'rpcf/exp031.py',
        'rpcf/exp031_artifacts.py', 'rpcf/core.py', 'attack_ea_forward.py', 'data/subject_ea.py', 'data/load.py',
        'utils/reproducibility.py', 'utils/experiment_artifacts.py', 'purify.py', 'TN/opt.py',
        'configs/thubenchmark/PTR3d_8_2048_rank25_3d_interpolate.yaml')}
    files |= set(Path('models').rglob('*.py'))
    return {str(p): file_hash(p) for p in sorted(files)}


def plan(run_id, smoke=False):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Unsafe run-id')
    root = Path('logs/exp034')/run_id
    manifest = dict(experiment_id='EXP-034', run_id=run_id, run_dir=str(root), smoke=smoke,
        source_run=SOURCE_RUN, external_run=EXTERNAL_RUN, structure_run=STRUCTURE_RUN,
        datasets=['thubenchmark'] if smoke else list(DATASETS), models=['eegnet'] if smoke else list(MODELS),
        seeds=[42] if smoke else list(SEEDS), sample_num=2 if smoke else 512,
        validation_sample_num=2 if smoke else 32, fold=0, evaluation_batch_size=8, raw_evaluation_batch_size=32,
        adaptive_protocol=dict(norm='Linf', eps=.03, alpha=.006, steps=10, random_start=False,
            restarts=1, attack_batch_size=1, eot_enabled=False, extra_input_clamp=False, iterate_selection='last'),
        budget_targets={'25':14731,'30':19891}, test_tuning=False,
        deferred_reporting='Next user-requested result整理: rank/CE/KL sensitivity charts with five-seed mean and sample SD',
        storage='scalar metrics, predictions, per-sample norms/hash and RNG partials; no duplicated full attack tensors')
    tasks = build_tasks(manifest)
    manifest['counts'] = dict(Counter(t['kind'] for t in tasks))
    manifest['task_count'] = len(tasks)
    manifest['expected_metric_rows'] = sum(len(expected_keys(t)) for t in tasks)
    for name, val in [('manifest.json', manifest), ('tasks.json', tasks), ('source_sha256.json', science_hashes())]:
        p = root/name
        if p.exists() and read_json(p) != val:
            raise ValueError(f'Frozen {name} differs; use a new run-id')
        if not p.exists():
            write_json(p, val)
    return root, manifest, tasks


def verify(root):
    manifest, tasks = read_json(root/'manifest.json'), read_json(root/'tasks.json')
    if read_json(root/'source_sha256.json') != science_hashes() or tasks != build_tasks(manifest):
        raise ValueError('Frozen code/tasks changed; create another run-id')
    return manifest, tasks


def check_result(manifest, task, result):
    if result.get('task_fingerprint') != fingerprint(task) or result.get('smoke') != manifest['smoke']:
        raise ValueError('Result task identity differs')
    if result.get('experiment_id') != 'EXP-034' or result.get('seed') != task['seed']:
        raise ValueError('Result experiment/seed differs')
    if len(result['source_indices']) != manifest['sample_num'] or len(set(result['source_indices'])) != manifest['sample_num']:
        raise ValueError('Result sample identity missing/duplicated')
    if len(result['labels']) != manifest['sample_num']:
        raise ValueError('Result labels missing')
    rows = result['rows']
    keys = [(r['method'],r['attack'],r['evaluation'],r.get('budget_rank')) for r in rows]
    if Counter(keys) != Counter(expected_keys(task)):
        raise ValueError('Missing/duplicate/unexpected metric keys')
    for r in rows:
        if (r['dataset'], r['model'], r['seed'], r['fold']) != (task['dataset'], task['model'], task['seed'], 0):
            raise ValueError('Metric condition mismatch')
        n = min(8,manifest['sample_num'])*2 if r['evaluation']=='timing' else manifest['sample_num']
        if r['sample_num'] != n:
            raise ValueError('Metric sample count differs')
        if r['evaluation'] != 'timing':
            for pred, accuracy in [('clean_predictions','standard_accuracy'),('adv_predictions','robust_accuracy')]:
                ps = r[pred]
                if len(ps) != n or abs(sum(a==b for a,b in zip(ps,result['labels']))/n-r[accuracy]) > 1e-12:
                    raise ValueError('Accuracy inconsistent with predictions')
        if r['attack'] == 'pgd10':
            protocol = r['attack_protocol']
            if any(protocol.get(k) != v for k,v in manifest['adaptive_protocol'].items()):
                raise ValueError('PGD-10 protocol differs')
    for entry in result['sources'].values():
        p = Path(entry['path'])
        stat = p.stat()
        if (stat.st_size,stat.st_mtime_ns) != (entry['size'],entry['mtime_ns']) and file_hash(p) != entry['sha256']:
            raise ValueError(f'Frozen source changed: {p}')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.with_suffix('.csv.tmp').open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k:(__import__('json').dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v) for k,v in r.items()})
    path.with_suffix('.csv.tmp').replace(path)


def summarize(root):
    manifest, tasks = verify(root)
    rows, states, errors, pending = [], [], [], []
    for task in tasks:
        sp = root/'status'/f"{task['task_id']}.json"
        state = read_json(sp) if sp.exists() else {'status':'pending'}
        states.append(dict(task_id=task['task_id'],kind=task['kind'],status=state['status']))
        if state['status'] != 'completed':
            pending.append(task['task_id']); continue
        try:
            if file_hash(task['output_path']) != state.get('output_sha256'):
                raise ValueError('Output hash differs')
            result = read_json(task['output_path'])
            check_result(manifest, task, result)
            rows.extend(dict(r, task_id=task['task_id'], source_path=task['output_path']) for r in result['rows'])
        except Exception as exc:
            errors.append(dict(task_id=task['task_id'],error=str(exc)))
    grouped = defaultdict(list)
    for r in rows:
        key = tuple(r.get(k) for k in ('dataset','model','method','attack','evaluation','budget_rank'))
        grouped[key].append(r)
    summary = []
    metric_names = ['standard_accuracy','robust_accuracy','parameters','budget_deviation', 'clean_mse_mean',
        'adv_mse_to_clean_mean','removed_mse_mean','decomposition_seconds_mean','total_seconds_mean',
        'total_seconds_median','peak_memory_bytes','native_compression_ratio','tensor_compression_ratio']
    for key, vals in grouped.items():
        seeds = sorted(v['seed'] for v in vals)
        if len(seeds) != len(set(seeds)):
            errors.append(dict(group=key,error='duplicate seed'))
        out = dict(zip(('dataset','model','method','attack','evaluation','budget_rank'),key))
        out.update(seed_count=len(seeds), seeds=seeds, missing_seeds=sorted(set(manifest['seeds'])-set(seeds)),
                   complete=seeds==manifest['seeds'], sections=vals[0]['sections'])
        for name in metric_names:
            values = [v[name] for v in vals if name in v]
            if values:
                out.update({name+'_mean':statistics.mean(values),name+'_sample_std':statistics.stdev(values) if len(values)>1 else None})
        summary.append(out)
    complete = not pending and not errors and len(rows)==manifest['expected_metric_rows'] and all(x['complete'] for x in summary)
    report = dict(experiment_id='EXP-034', run_id=manifest['run_id'], smoke=manifest['smoke'],
        complete=complete, status=('Smoke passed' if manifest['smoke'] else 'Complete') if complete else 'Pending',
        task_count=len(tasks), completed_task_count=sum(s['status']=='completed' for s in states),
        metric_rows=len(rows), expected_metric_rows=manifest['expected_metric_rows'], errors=errors,pending_tasks=pending,
        scope='Only supplemental EXP-034 tasks; no claim that other historical gaps were run',
        deferred_reporting=manifest['deferred_reporting'])
    for filename, content in [('metrics_long.csv',rows),('metrics_grouped.csv',summary),('task_states.csv',states)]:
        write_csv(root/'summary'/filename,content)
    write_json(root/'summary/report.json',report)
    return report


def start_ticks(pid):
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
    except FileNotFoundError:
        return None


def run(root, gpu_ids, retry_failed=False, cpu=False):
    manifest, tasks = verify(root)
    if cpu and not manifest['smoke']:
        raise ValueError('Formal experiments require GPU')
    lock = (root/'controller.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    running, done, failed, pending = {}, set(), {}, []
    for task in tasks:
        sp = root/'status'/f"{task['task_id']}.json"
        state = read_json(sp) if sp.exists() else {}
        if state.get('status')=='running' and start_ticks(state['pid'])==state.get('start_ticks'):
            raise RuntimeError('Existing worker is still alive; do not duplicate it')
        if state.get('status')=='completed':
            if file_hash(task['output_path']) != state['output_sha256']:
                raise ValueError('Completed task changed')
            check_result(manifest,task,read_json(task['output_path']))
            done.add(task['task_id'])
        elif state.get('status')=='failed' and not retry_failed:
            failed[task['task_id']] = state.get('exit_code','prior failure')
        else:
            pending.append(task)
    try:
        while pending or running:
            for tid, rec in list(running.items()):
                process, task, gpu, started, handle = rec
                code = process.poll()
                if code is None: continue
                handle.close()
                state = dict(status='failed',exit_code=code,elapsed_seconds=time.time()-started)
                if code==0:
                    try:
                        check_result(manifest,task,read_json(task['output_path']))
                        state.update(status='completed',output_sha256=file_hash(task['output_path']))
                        done.add(tid)
                    except Exception as exc:
                        state.update(error=str(exc),exit_code=-1)
                if state['status']=='failed': failed[tid]=state['exit_code']
                write_json(root/'status'/f'{tid}.json',state)
                print(f"END {tid} {state['status']} elapsed={state['elapsed_seconds']:.1f}s",flush=True)
                del running[tid]
            candidates = [t for t in pending if set(t['dependencies'])<=done]
            # 所有普通任务结束后才派发计时，避免共享CPU负载影响净化总时间。
            compute_left = any(t['kind']!='timing' for t in pending) or any(rec[1]['kind']!='timing' for rec in running.values())
            candidates = [t for t in candidates if t['kind']!='timing' or not compute_left]
            if failed:
                candidates = []  # 失败后让在途任务收尾，避免静默继续改变实验覆盖。
            if candidates and shutil.disk_usage(root).free < 8*1024**3:
                raise RuntimeError('Free disk below 8GiB reserve')
            free = ([-1] if not running else []) if cpu else idle_gpus(gpu_ids)
            used = {rec[2] for rec in running.values()}
            free = [g for g in free if g not in used]
            if candidates and candidates[0]['kind']=='timing':
                free = free[:1] if not running else []  # 所有计时全局串行。
            for task,gpu in zip(candidates,free):
                tid=task['task_id']
                path=root/'workers'/f'{tid}.log';path.parent.mkdir(parents=True,exist_ok=True)
                handle=path.open('a',buffering=1)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu<0 else str(gpu),OMP_NUM_THREADS='2',
                         MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                cmd=[sys.executable,'-u','-m','rpcf.exp034_worker','--run-dir',str(root),'--task-id',tid]
                process=subprocess.Popen(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT)
                started=time.time()
                write_json(root/'status'/f'{tid}.json',dict(status='running',pid=process.pid,start_ticks=start_ticks(process.pid),
                    gpu=gpu,started_at_epoch=started,command=cmd,log_path=str(path)))
                running[tid]=(process,task,gpu,started,handle);pending.remove(task)
                print(f'START {tid} gpu={gpu} pid={process.pid}',flush=True)
            write_json(root/'runtime.json',dict(at_epoch=time.time(),controller_pid=os.getpid(),controller_start_ticks=start_ticks(os.getpid()),
                completed=len(done),pending=len(pending),failures=failed,
                running=[dict(task_id=tid,pid=r[0].pid,gpu=r[2],kind=r[1]['kind']) for tid,r in running.items()]))
            if failed and not running: break
            if not running and pending and not candidates and not failed:
                raise RuntimeError('Unsatisfied dependency cycle')
            if pending or running: time.sleep(3)
    finally:
        lock.close()
    report=summarize(root)
    print(report,flush=True)
    if not report['complete']: raise RuntimeError('EXP-034 incomplete; inspect status and worker logs')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','run','summarize'])
    p.add_argument('--run-id',required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--gpu-ids',default='0,1,2,3,4,5,6,7')
    p.add_argument('--retry-failed',action='store_true')
    args=p.parse_args()
    if args.action=='plan':
        _,m,_=plan(args.run_id,args.smoke);print(m,flush=True)
    else:
        root=Path('logs/exp034')/args.run_id
        if args.action=='summarize':print(summarize(root),flush=True)
        else:run(root,[int(x) for x in args.gpu_ids.split(',')],args.retry_failed,args.cpu)


if __name__=='__main__': main()
