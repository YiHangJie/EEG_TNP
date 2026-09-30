"""EXP-035 资源调度：全部空闲卡、轻任务共卡、重任务独占，记录真实退出码。"""
from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rpcf.exp033_common import read_json,write_json,file_hash,fingerprint
from rpcf.parallel_exp032 import proc_info,AdoptedProcess
from rpcf.parallel_exp033 import resources,identity_alive

PRIORITY={'prepare':0,'madry':1,'cache_base':2,'cache_streams':2,'cache_rank':3,'cache_merge':2,
          'madry_aa':4,'canonical':4,'madry_pgd':4,'default_pgd':4,'default_tnp':4,'sources':4,
          'reference':4,'train':5,'attack':6,'tnp':7}


def resource_records(running):
    return {tid:{**r,'task':{**r['task'],'kind':'train' if r['task']['heavy'] else 'light'}} for tid,r in running.items()}


def choose(task,running,gpus,ram,policy,draining=()):
    """不调整batch；未知进程拒派，预留含尚未兑现的内存/显存。"""
    heavy=task['heavy'];category='train' if heavy else 'light'
    if len(running)>=policy['max_workers'] or ram < policy['min_ram_gib']+policy['ram_reserve_gib'][category]: return None
    choices=[]
    for gpu_id in policy['gpu_ids']:
        gpu=gpus.get(gpu_id)
        if gpu is None or gpu['unknown']: continue
        peers=[r for r in running.values() if r['gpu']==gpu_id]
        if any(r['task']['heavy'] or r.get('isolated') for r in peers): continue
        if (heavy or task.get('isolated')) and peers: continue
        if not heavy and gpu_id in draining: continue
        if len(peers)>=policy['light_per_gpu']: continue
        if gpu['free_mib'] < policy['gpu_margin_mib']+policy['gpu_reserve_mib'][category]: continue
        if (not peers and gpu['utilization']>10) or (peers and gpu['utilization']>85): continue
        choices.append((len(peers),-gpu['free_mib'],gpu_id))
    return min(choices)[2] if choices else None


def supervise(root,tid,receipt):
    if receipt.exists(): raise FileExistsError(receipt)
    command=[sys.executable,'-u','-m','rpcf.exp035','worker','--run-id',root.name,'--task-id',tid]
    process=subprocess.Popen(command)
    rc=process.wait()
    write_json(receipt,dict(task_id=tid,returncode=rc,worker_pid=process.pid,command=command,at_epoch=time.time()))
    return rc


def completed(task,state):
    from rpcf.exp033_worker import verify_artifacts
    path=Path(task['output_path'])
    if file_hash(path)!=state['output_sha256']: raise ValueError(f'Completed output changed: {path}')
    value=read_json(path)
    if value['task_fingerprint']!=fingerprint(task): raise ValueError('Completed task identity differs')
    verify_artifacts(value)


def finish(root,record,rc):
    from rpcf.exp033_worker import verify_artifacts
    task=record['task'];tid=task['task_id'];error=None;actual={}
    receipt=Path(record['receipt'])
    try:
        if not receipt.exists(): raise RuntimeError(f'Missing true worker return code; supervisor={rc}')
        actual=read_json(receipt)
        if actual['task_id']!=tid or actual['returncode']!=0 or rc not in (0,None):
            raise RuntimeError(f'Worker returned {actual.get("returncode")}; supervisor={rc}')
        value=read_json(task['output_path'])
        if value['task_fingerprint']!=fingerprint(task): raise ValueError('Output task identity differs')
        verify_artifacts(value)
        digest=file_hash(task['output_path'])
    except Exception as exc: error=str(exc)
    state={k:v for k,v in record.items() if k not in ('task','process')}
    state.update(status='failed' if error else 'completed',returncode=actual.get('returncode',rc),
                 elapsed_seconds=time.time()-record['started'],at_epoch=time.time())
    if error: state['error']=error
    else: state['output_sha256']=digest
    write_json(root/'status'/f'{tid}.json',state)
    print(f'END {tid} {state["status"]} elapsed={state["elapsed_seconds"]:.1f}s',flush=True)
    return error


def run(root,gpu_ids,max_workers=24,light_per_gpu=3,retry_failed=False):
    from rpcf.exp035 import verify_plan
    m,tasks=verify_plan(root)
    if not gpu_ids or len(gpu_ids)!=len(set(gpu_ids)) or max_workers<1 or light_per_gpu<1:
        raise ValueError('Invalid resource policy')
    policy=dict(gpu_ids=gpu_ids,max_workers=max_workers,light_per_gpu=light_per_gpu,
                min_ram_gib=24,gpu_margin_mib=1024,gpu_reserve_mib={'light':2048,'train':8192},
                ram_reserve_gib={'light':4,'train':10},min_disk_gib=32,max_starts_per_tick=2,
                heavy_gpu_scope='all_available',batch_fallback=False)
    lock=(root/'controller.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    policy_path=root/'scheduler_policy.json'
    if policy_path.exists() and read_json(policy_path)!=policy: raise ValueError('Frozen policy differs')
    if not policy_path.exists(): write_json(policy_path,policy)
    pending=[];running={};done=set();failures=[]
    for task in tasks:
        p=root/'status'/f"{task['task_id']}.json"
        state=read_json(p) if p.exists() else {}
        if state.get('status')=='completed':
            completed(task,state);done.add(task['task_id']);continue
        if state.get('status')=='running':
            record={**state,'task':task}
            if identity_alive(state['pid'],state['start_ticks']):
                record['process']=AdoptedProcess(state['pid'],state['start_ticks'])
                running[task['task_id']]=record;continue
            error=finish(root,record,None)
            if error is None: done.add(task['task_id']);continue
            if not retry_failed: raise RuntimeError(error)
        if state.get('status')=='failed' and not retry_failed: raise RuntimeError(f'Inspect failed task: {task["task_id"]}')
        pending.append(task)
    pending.sort(key=lambda t:(PRIORITY[t['kind']],t.get('variant')!='default',t['seed'],t['task_id']))
    last_notice=0
    try:
        while pending or running:
            for tid,r in list(running.items()):
                if isinstance(r['process'],AdoptedProcess):
                    # 重启后的supervisor可能已由PID 1回收；以身份存活和真实receipt判断，
                    # 不调用依赖/proc仍存在的旧poll实现，也不误认复用后的PID。
                    if identity_alive(r['pid'],r['start_ticks']): continue
                    rc=None
                else:
                    rc=r['process'].poll()
                    if rc is None: continue
                error=finish(root,r,rc)
                del running[tid]
                if error: failures.append(dict(task_id=tid,error=error))
                else: done.add(tid)
            if not failures:
                if shutil.disk_usage(root).free/1024**3 < policy['min_disk_gib']:
                    failures.append(dict(error='Below 32GiB storage reserve; no new workers started'))
                else:
                    ready=[t for t in pending if set(t['dependencies'])<=done]
                    # 有训练待运行但所有卡都被轻任务占据时，暂不向最少peer的一卡补轻任务，待自然排空。
                    draining=[]
                    if any(t['heavy'] for t in ready):
                        heavy_count=sum(r['task']['heavy'] for r in running.values())
                        target=min(len(gpu_ids),6 if any(not t['heavy'] for t in ready) else len(gpu_ids))
                        if heavy_count<target:
                            choices=[(sum(r['gpu']==g for r in running.values()),g) for g in gpu_ids
                                     if not any(r['gpu']==g and r['task']['heavy'] for r in running.values())]
                            draining=[g for _,g in sorted(choices)[:target-heavy_count]]
                    starts=0
                    for task in ready:
                        if starts>=policy['max_starts_per_tick']: break
                        heavy_count=sum(r['task']['heavy'] for r in running.values())
                        if task['heavy'] and heavy_count>=6 and any(not t['heavy'] for t in ready): continue
                        gpus,ram=resources(resource_records(running),policy)
                        gpu=choose(task,running,gpus,ram,policy,draining)
                        if gpu is None: continue
                        tid=task['task_id'];logs=root/'worker_logs';logs.mkdir(exist_ok=True)
                        attempt=len(list(logs.glob(f'{tid}.attempt*.log')))+1
                        receipt=root/'receipts'/f'{tid}.attempt{attempt}.json'
                        cmd=[sys.executable,'-u','-m','rpcf.exp035_scheduler','--supervise',str(root),
                             '--task-id',tid,'--receipt',str(receipt)]
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'}
                        with (logs/f'{tid}.attempt{attempt}.log').open('x') as handle:
                            process=subprocess.Popen(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT)
                        record=dict(task=task,process=process,pid=process.pid,start_ticks=proc_info(process.pid)['start_ticks'],
                                    gpu=gpu,started=time.time(),attempt=attempt,receipt=str(receipt),command=cmd)
                        old=root/'status'/f'{tid}.json'
                        if old.exists(): write_json(root/'state_history'/f'{tid}.{time.time_ns()}.json',read_json(old))
                        write_json(old,{**{k:v for k,v in record.items() if k not in ('task','process')},'status':'running'})
                        running[tid]=record;pending.remove(task);starts+=1
                        print(f'START {tid} gpu={gpu} heavy={task["heavy"]}',flush=True)
            write_json(root/'runtime.json',dict(at_epoch=time.time(),controller_pid=os.getpid(),
                completed=len(done),task_count=len(tasks),pending=len(pending),failures=failures,
                running=[dict(task_id=tid,pid=r['pid'],gpu=r['gpu'],kind=r['task']['kind']) for tid,r in running.items()]))
            if failures and not running: raise RuntimeError(failures)
            if pending and not running and all(not set(t['dependencies'])<=done for t in pending): raise RuntimeError('DAG deadlock')
            if time.time()-last_notice>60:
                print(f'PROGRESS completed={len(done)}/{len(tasks)} running={len(running)} pending={len(pending)}',flush=True)
                last_notice=time.time()
            if pending or running: time.sleep(2)
        from rpcf.exp035_report import summarize
        summarize(root,strict=True)
    finally:
        # 控制器退出不杀工作进程；supervisor保存退出码，重启按PID身份/receipt接管。
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervise',type=Path,required=True)
    parser.add_argument('--task-id',required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    a=parser.parse_args()
    raise SystemExit(supervise(a.supervise,a.task_id,a.receipt))
