"""EXP-033 独立并行调度：保留在途任务，普通任务共卡，训练/计时独占。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from rpcf.exp033 import verify_plan
from rpcf.exp033_common import read_json, write_json, file_hash, fingerprint
from rpcf.parallel_exp032 import proc_info, AdoptedProcess, acquire_controller_lock, checked_old_controller

PRIORITY = {"sources": 0, "reference": 1, "calibrate": 2, "train": 3,
            "attack": 4, "structure": 5, "tnp": 6, "visualize": 7, "timing": 8}


def identity_alive(pid, ticks):
    try:
        info = proc_info(pid)
        return info["start_ticks"] == ticks and info["state"] != "Z"
    except FileNotFoundError:
        return False


def scheduler_hashes():
    return {p: file_hash(p) for p in ("rpcf/parallel_exp033.py", "rpcf/run_exp033_parallel.sh",
                                      "rpcf/parallel_exp032.py")}


def freeze_policy(root, args):
    """调度指纹独立冻结，不改原科学清单；后续扩容使用新 policy 版本。"""
    folder = root / "parallel_v1"
    folder.mkdir(exist_ok=True)
    policy = dict(gpu_ids=args.gpu_ids, light_per_gpu=args.light_per_gpu,
                  training_gpus=args.gpu_ids[:args.training_gpus], max_workers=args.max_workers,
                  min_ram_gib=args.min_ram_gib, gpu_margin_mib=1024,
                  gpu_reserve_mib={"light": 2048, "train": 8192},
                  ram_reserve_gib={"light": 4, "train": 10},
                  max_starts_per_tick=2, timing_after_compute=True, source_sha256=scheduler_hashes())
    path = folder / "policy.json"
    if path.exists() and read_json(path) != policy:
        raise ValueError("Frozen parallel policy changed; inspect before creating another revision")
    if not path.exists():
        write_json(path, policy)
    return folder, policy


def capture(root, folder, tasks, pid, ticks, gpu_ids):
    """只在旧控制器完成一轮登记后暂停，工作进程继续运行；失败自动恢复旧控制器。"""
    path = folder / "handoff.json"
    if path.exists():
        raise FileExistsError(path)
    old = proc_info(pid)
    command = [c.decode() for c in old["command"]]
    if old["start_ticks"] != ticks or "rpcf.exp033" not in command or root.name not in command:
        raise ValueError("Original controller identity mismatch")
    deadline = time.monotonic() + 15
    while Path(f"/proc/{pid}/wchan").read_text().strip() != "hrtimer_nanosleep":
        if time.monotonic() >= deadline:
            raise RuntimeError("Controller is not at a safe dispatch boundary; retry later")
        time.sleep(.05)
    paused = False
    try:
        os.kill(pid, signal.SIGSTOP)
        paused = True
        for _ in range(100):
            if proc_info(pid)["state"] in {"T", "t"}:
                break
            time.sleep(.02)
        else:
            raise RuntimeError("Controller did not pause")
        states = {p.stem: read_json(p) for p in (root / "status").glob("*.json")}
        by_pid = {s["pid"]: (tid, s) for tid, s in states.items() if s.get("status") == "running"}
        task_map = {t["task_id"]: t for t in tasks}
        children = [int(p) for p in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
        if set(children) != set(by_pid):
            raise RuntimeError("Worker registration was in transition; resume and retry capture")
        boot_epoch = time.time() - float(Path('/proc/uptime').read_text().split()[0])
        records = []
        for child_pid in children:
            tid, state = by_pid[child_pid]
            child = proc_info(child_pid)
            expected = [sys.executable, '-u', '-m', 'rpcf.exp033_worker', '--run-dir', str(root), '--task-id', tid]
            actual = [c.decode() for c in child['command']]
            if (tid not in task_map or state['gpu'] not in gpu_ids or child['ppid'] != pid
                    or state.get('command') != expected or (actual and actual != expected)):
                raise ValueError(f"Unexpected worker identity: {tid}")
            records.append(dict(task_id=tid, pid=child_pid, start_ticks=child['start_ticks'],
                                gpu=state['gpu'], attempt=state['attempt'],
                                started=boot_epoch + child['start_ticks']/os.sysconf('SC_CLK_TCK')))
        handoff = dict(controller={k: old[k] for k in ('pid', 'start_ticks', 'pgid')},
                       workers=records, run_id=root.name, at_epoch=time.time(),
                       previous_runtime=read_json(root/'runtime.json'), policy='pause_controller_only')
        write_json(path, handoff)
        print(f"CAPTURED controller={pid} workers={len(records)}", flush=True)
        return handoff
    except BaseException:
        if paused and not path.exists():
            os.kill(pid, signal.SIGCONT)
        raise


def owned_pids(running):
    """包含训练子进程和退出码包装器；不把未知 GPU 进程当成本实验进程。"""
    owners = {}
    for tid, record in running.items():
        todo = [record['pid']]
        while todo:
            pid = todo.pop()
            if pid in owners:
                continue
            owners[pid] = tid
            try:
                todo.extend(int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split())
            except FileNotFoundError:
                pass
    return owners


def resources(running, policy):
    owners = owned_pids(running)
    rss = {tid: 0 for tid in running}
    for pid, tid in owners.items():
        try:
            lines = Path(f'/proc/{pid}/status').read_text().splitlines()
            rss[tid] += next((int(s.split()[1])*1024 for s in lines if s.startswith('VmRSS:')), 0)
        except FileNotFoundError:
            pass
    query = lambda fields, mode: subprocess.check_output(
        ['nvidia-smi', f'--query-{mode}={fields}', '--format=csv,noheader,nounits'], text=True)
    gpus = {}
    uuids = {}
    for line in query('index,uuid,memory.free,utilization.gpu', 'gpu').splitlines():
        g, uuid, free, util = [s.strip() for s in line.split(',')]
        if int(g) in policy['gpu_ids']:
            uuids[uuid] = int(g)
            gpus[int(g)] = dict(free_mib=int(free), utilization=int(util), unknown=False, usage={})
    for line in query('gpu_uuid,pid,used_memory', 'compute-apps').splitlines():
        uuid, pid, memory = [s.strip() for s in line.split(',')]
        if uuid not in uuids:
            continue
        gpu = gpus[uuids[uuid]]
        tid = owners.get(int(pid))
        if tid is None or running[tid]['gpu'] != uuids[uuid] or not memory.isdigit():
            gpu['unknown'] = True
        else:
            gpu['usage'][tid] = gpu['usage'].get(tid, 0) + int(memory)
    available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
                     if s.startswith('MemAvailable:')) / 1024**3
    # 启动中的进程尚未分配内存，必须扣除未兑现的预留，不能只看瞬时空闲。
    for tid, record in running.items():
        category = 'train' if record['task']['kind'] == 'train' else 'light'
        available -= max(0, policy['ram_reserve_gib'][category] - rss[tid]/1024**3)
        gpu = gpus.get(record['gpu'])
        if gpu is not None:
            gpu['free_mib'] -= max(0, policy['gpu_reserve_mib'][category] - gpu['usage'].get(tid, 0))
    return gpus, available


def choose_gpu(task, running, gpus, ram_gib, policy, training_ready=False, isolated=False):
    """轻任务最多共卡三进程；训练/计时/显存回退任务必须独占。"""
    if len(running) >= policy['max_workers']:
        return None
    category = 'train' if task['kind'] == 'train' else 'light'
    if ram_gib < policy['min_ram_gib'] + policy['ram_reserve_gib'][category]:
        return None
    if task['kind'] == 'train' and sum(r['task']['kind'] == 'train' for r in running.values()) >= len(policy['training_gpus']):
        return None
    exclusive = task['kind'] in {'train', 'timing'} or isolated
    choices = []
    for gpu_id in policy['gpu_ids']:
        gpu = gpus.get(gpu_id)
        if gpu is None or gpu['unknown']:
            continue
        peers = [r for r in running.values() if r['gpu'] == gpu_id]
        if task['kind'] == 'train' and gpu_id not in policy['training_gpus']:
            continue
        if task['kind'] != 'train' and task['kind'] != 'timing' and training_ready and gpu_id in policy['training_gpus']:
            continue
        if any(r['task']['kind'] in {'train', 'timing'} or r.get('isolated') for r in peers):
            continue
        if (exclusive and peers) or len(peers) >= policy['light_per_gpu']:
            continue
        if gpu['free_mib'] < policy['gpu_margin_mib'] + policy['gpu_reserve_mib'][category]:
            continue
        if (not peers and gpu['utilization'] > 10) or (peers and gpu['utilization'] > 85):
            continue
        choices.append((len(peers), -gpu['free_mib'], gpu_id))
    return min(choices)[2] if choices else None


def archive_state(root, folder, tid):
    path = root/'status'/f'{tid}.json'
    if path.exists():
        write_json(folder/'state_history'/f'{tid}.{time.time_ns()}.json', read_json(path))


def execute_worker(root, task_id, receipt):
    """保留真实退出码，即使调度器中断也不按“输出存在”猜测执行成功。"""
    if receipt.exists():
        raise FileExistsError(receipt)
    _, tasks = verify_plan(root)
    task = next(t for t in tasks if t['task_id'] == task_id)
    command = [sys.executable, '-u', '-m', 'rpcf.exp033_worker', '--run-dir', str(root), '--task-id', task_id]
    process = subprocess.Popen(command)
    rc = process.wait()
    write_json(receipt, dict(task_id=task_id, task_fingerprint=fingerprint(task), returncode=rc,
                             child_pid=process.pid, at_epoch=time.time(), command=command))
    return rc


def recorded_exit(record, supervisor_rc):
    """新任务以包装器记录的真实worker退出码为准，保留被信号终止的负值。"""
    if record.get('adopted'):
        return supervisor_rc
    path = Path(record['receipt'])
    if not path.exists():
        if supervisor_rc == 0:
            raise RuntimeError('Successful supervisor has no worker receipt')
        return supervisor_rc
    receipt = read_json(path)
    if receipt.get('task_fingerprint') != fingerprint(record['task']):
        raise ValueError('Worker receipt fingerprint differs')
    if supervisor_rc != 0 and receipt['returncode'] == 0:
        return supervisor_rc
    return receipt['returncode']


def finalize(root, folder, record, rc):
    from rpcf.exp033_worker import verify_artifacts
    task = record['task']; tid = task['task_id']; error = None
    try:
        if rc != 0:
            raise RuntimeError(f'Worker exited {rc}')
        result = read_json(task['output_path'])
        if result.get('task_fingerprint') != fingerprint(task):
            raise ValueError('Task fingerprint differs')
        verify_artifacts(result)
        digest = file_hash(task['output_path'])
    except Exception as exc:
        error = str(exc)
    archive_state(root, folder, tid)
    state = dict(status='completed' if error is None else 'failed', returncode=rc,
                 elapsed_seconds=time.time()-record['started'], gpu=record['gpu'],
                 attempt=record['attempt'], pid=record['pid'], start_ticks=record['start_ticks'],
                 output_path=task['output_path'], at_epoch=time.time())
    if error is None:
        state['output_sha256'] = digest
    else:
        state['error'] = error
    write_json(root/'status'/f'{tid}.json', state)
    print(f"END {tid} {state['status']} rc={rc}", flush=True)
    return error is None


def launch(root, folder, task, gpu, isolated):
    tid = task['task_id']; log_dir = root/'worker_logs'; log_dir.mkdir(exist_ok=True)
    attempt = len(list(log_dir.glob(f'{tid}.attempt*.log'))) + 1
    log_path = log_dir/f'{tid}.attempt{attempt}.log'
    receipt = folder/'receipts'/f'{tid}.attempt{attempt}.json'
    command = [sys.executable, '-u', '-m', 'rpcf.parallel_exp033', 'worker', '--run-id', root.name,
               '--task-id', tid, '--receipt', str(receipt)]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2',
           'MKL_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'}
    with log_path.open('x') as handle:
        process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
    record = dict(task=task, pid=process.pid, start_ticks=proc_info(process.pid)['start_ticks'],
                  process=process, gpu=gpu, attempt=attempt, started=time.time(), adopted=False,
                  isolated=isolated, receipt=str(receipt))
    archive_state(root, folder, tid)
    state = {k: v for k, v in record.items() if k not in {'task', 'process'}}
    write_json(root/'status'/f'{tid}.json', dict(state, status='running', command=command, at_epoch=time.time()))
    print(f'START {tid} gpu={gpu} pid={process.pid} isolated={isolated}', flush=True)
    return record


def retire(handoff, root, folder, lock):
    old = handoff['controller']
    if identity_alive(old['pid'], old['start_ticks']):
        checked_old_controller(handoff)
        # 所有旧worker均已取得真实退出码；只终止旧控制器，不向其进程组发信号。
        os.kill(old['pid'], signal.SIGTERM)
        os.kill(old['pid'], signal.SIGCONT)
    acquire_controller_lock(lock)
    write_json(folder/'handoff_complete.json', dict(at_epoch=time.time(), old_controller=old,
                                                  controller_pid=os.getpid()))
    print('HANDOFF_COMPLETE', flush=True)


def run(root, folder, tasks, policy, retry_failed=False):
    from rpcf.exp033_worker import verify_artifacts
    handoff = read_json(folder/'handoff.json')
    retired = (folder/'handoff_complete.json').exists()
    if not retired:
        old = handoff['controller']
        if identity_alive(old['pid'], old['start_ticks']):
            checked_old_controller(handoff)
        elif any(not (root/'status'/f"{w['task_id']}.json").exists() or read_json(root/'status'/f"{w['task_id']}.json").get('status') not in {'completed', 'failed'} for w in handoff['workers']):
            raise RuntimeError('Original controller disappeared before adopted workers were finalized')
    task_map = {t['task_id']: t for t in tasks}
    adopted = {r['task_id']: r for r in handoff['workers']} if not retired else {}
    done, running, pending, failures = set(), {}, {}, {}
    isolated_path = folder/'isolated_retries.json'
    isolated = set(read_json(isolated_path)) if isolated_path.exists() else set()
    for task in tasks:
        tid = task['task_id']; path = root/'status'/f'{tid}.json'
        state = read_json(path) if path.exists() else {}
        if state.get('status') == 'completed':
            if file_hash(task['output_path']) != state.get('output_sha256'):
                raise ValueError(f'Completed output changed: {tid}')
            verify_artifacts(read_json(task['output_path']))
            done.add(tid)
        elif tid in adopted and state.get('status') != 'failed':
            r = adopted[tid]
            running[tid] = dict(r, task=task, process=AdoptedProcess(r['pid'], r['start_ticks']), adopted=True)
        else:
            if state.get('status') == 'running':
                if identity_alive(state['pid'], state.get('start_ticks', -1)):
                    raise RuntimeError(f'Existing parallel worker still live: {tid}; wait before resuming')
                if state.get('receipt') and Path(state['receipt']).exists():
                    receipt = read_json(state['receipt'])
                    if receipt['task_fingerprint'] != fingerprint(task):
                        raise ValueError('Receipt identity mismatch')
                    if finalize(root, folder, dict(state, task=task), receipt['returncode']):
                        done.add(tid)
                        continue
                    state = read_json(path)
            if state.get('status') == 'failed' and not retry_failed:
                failures[tid] = state.get('returncode')
            else:
                pending[tid] = task
    controller_lock = (root/'controller.lock').open('a')
    if retired:
        acquire_controller_lock(controller_lock)
    elif not any(r.get('adopted') for r in running.values()):
        retire(handoff, root, folder, controller_lock)
        retired = True
    stop = {'requested': False}
    def stopping(signum, frame):
        stop['requested'] = True
        print(f'DRAIN signal={signum}', flush=True)
    previous_handlers = {s: signal.signal(s, stopping) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        while pending or running:
            if not retired:
                checked_old_controller(handoff)
            for tid, record in list(running.items()):
                rc = record['process'].poll()
                if rc is None:
                    continue
                rc = recorded_exit(record, rc)
                ok = finalize(root, folder, record, rc)
                del running[tid]
                if ok:
                    done.add(tid)
                else:
                    log = root/'worker_logs'/f"{tid}.attempt{record['attempt']}.log"
                    with log.open('rb') as handle:
                        handle.seek(max(0, log.stat().st_size-65536))
                        tail = handle.read().decode(errors='replace').lower()
                    oom = 'out of memory' in tail or 'cuda error: memory allocation' in tail
                    if oom and record['task']['kind'] not in {'train', 'timing'} and tid not in isolated:
                        isolated.add(tid); write_json(isolated_path, sorted(isolated))
                        pending[tid] = record['task']
                        print(f'RETRY_ISOLATED {tid}', flush=True)
                    else:
                        failures[tid] = rc
            if not retired and not any(r.get('adopted') for r in running.values()):
                retire(handoff, root, folder, controller_lock)
                retired = True
            compute_remaining = any(t['kind'] != 'timing' for t in pending.values()) or any(r['task']['kind'] != 'timing' for r in running.values())
            ready = [t for t in pending.values() if set(t['dependencies']) <= done
                     and (t['kind'] != 'timing' or not compute_remaining)]
            ready.sort(key=lambda t: (PRIORITY[t['kind']], t['seed'], t['task_id']))
            training_ready = any(t['kind'] == 'train' for t in ready)
            if ready and not stop['requested']:
                gpus, ram = resources(running, policy)
                started = 0
                for task in ready:
                    gpu = choose_gpu(task, running, gpus, ram, policy, training_ready, task['task_id'] in isolated)
                    if gpu is None:
                        continue
                    tid = task['task_id']
                    running[tid] = launch(root, folder, task, gpu, tid in isolated)
                    del pending[tid]
                    cat = 'train' if task['kind'] == 'train' else 'light'
                    ram -= policy['ram_reserve_gib'][cat]
                    gpus[gpu]['free_mib'] -= policy['gpu_reserve_mib'][cat]
                    started += 1
                    if started >= policy['max_starts_per_tick']:
                        break
            runtime = dict(at_epoch=time.time(), controller_pid=os.getpid(), controller_start_ticks=proc_info(os.getpid())['start_ticks'],
                           completed=len(done), pending=len(pending), failures=failures, handoff_complete=retired,
                           draining=stop['requested'], scheduler='parallel_v1', policy_path=str(folder/'policy.json'),
                           running=[dict(task_id=tid, kind=r['task']['kind'], pid=r['pid'], start_ticks=r['start_ticks'],
                                         gpu=r['gpu'], adopted=r.get('adopted', False)) for tid, r in running.items()])
            write_json(folder/'runtime.json', runtime)
            write_json(root/'runtime.json', runtime)
            if not running and (stop['requested'] or (pending and not ready)):
                raise RuntimeError(f'Stopped or dependency blocked: failures={failures}')
            if pending or running:
                time.sleep(2)
        if failures:
            raise RuntimeError(f'Tasks failed: {failures}')
        from rpcf.exp033_report import summarize
        summarize(root, strict=True)
    finally:
        for s, handler in previous_handlers.items():
            signal.signal(s, handler)
        controller_lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'capture', 'run', 'worker'])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--gpu-ids', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--light-per-gpu', type=int, default=3)
    parser.add_argument('--training-gpus', type=int, default=3)
    parser.add_argument('--max-workers', type=int, default=24)
    parser.add_argument('--min-ram-gib', type=float, default=24)
    parser.add_argument('--old-controller-pid', type=int)
    parser.add_argument('--old-controller-ticks', type=int)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--task-id')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        parser.error('Invalid run-id')
    root = Path('logs/exp033')/args.run_id
    if args.action == 'worker':
        if not args.task_id or args.receipt is None:
            parser.error('worker needs task-id and receipt')
        raise SystemExit(execute_worker(root, args.task_id, args.receipt))
    args.gpu_ids = [int(g) for g in args.gpu_ids.split(',')]
    if (not args.gpu_ids or len(set(args.gpu_ids)) != len(args.gpu_ids) or min(args.gpu_ids) < 0
            or not 1 <= args.training_gpus <= len(args.gpu_ids) or args.light_per_gpu < 1
            or args.max_workers < 1 or args.min_ram_gib < 0):
        parser.error('Invalid resource limits')
    _, tasks = verify_plan(root)
    folder, policy = freeze_policy(root, args)
    with (folder/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'capture':
            capture(root, folder, tasks, args.old_controller_pid, args.old_controller_ticks, args.gpu_ids)
        elif args.action == 'run':
            run(root, folder, tasks, policy, args.retry_failed)
        else:
            print(json.dumps(dict(tasks=len(tasks), policy=policy), indent=2))


if __name__ == '__main__':
    main()
