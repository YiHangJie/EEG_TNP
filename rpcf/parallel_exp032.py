"""EXP-032 独立 CPU 队列、双 TNP 并发及保留在途 worker 的 Linux 接管。"""

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from rpcf.resume_exp032 import digest, prepare_scope, write_once
from rpcf.exp031 import (actual_batch, actual_cache_attack_batch, actual_rpcf_batch,
                        actual_rpcf_eval_batch, finalize_task, log_contains_oom,
                        render_command, reservation_active, start_task, task_complete)


def proc_info(pid):
    """start_ticks 防 PID 复用；僵尸的 exit_code 保留真实退出结果。"""
    path = Path(f"/proc/{pid}")
    tail = (path / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "state": tail[0], "ppid": int(tail[1]), "pgid": int(tail[2]),
            "start_ticks": int(tail[19]), "exit_code": int(tail[49]),
            "comm": (path / "comm").read_text().strip(),
            "command": (path / "cmdline").read_bytes().split(b"\0")[:-1]}


def normalise_command(command):
    """兼容conda的Python shebang包装，任务参数仍逐项严格比较。"""
    command = list(command)
    if (len(command) > 1 and Path(command[0]).name.startswith("python")
            and Path(command[1]).name == "conda"):
        command = command[1:]
    return [Path(command[0]).name, *command[1:]] if command else []


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checked_old_controller(handoff):
    old = handoff["controller"]
    current = proc_info(old["pid"])
    if current["start_ticks"] != old["start_ticks"] or current["state"] not in {"T", "t"}:
        raise RuntimeError("Old controller must retain its verified identity and remain paused")
    return current


def valid_worker_gpu(task, gpu, gpu_ids):
    """接管独立 CPU toy 时允许 -1，GPU 任务仍严格限制在授权卡池。"""
    return gpu in gpu_ids or (task.kind == "toy" and gpu == -1)


def capture_handoff(args, run_dir, tasks, active):
    """仅暂停旧 controller，worker 继续；退出后保持可读取真实 exit code。"""
    path = run_dir / "parallel_handoff_v5.json"
    if path.exists():
        raise FileExistsError(path)
    old = proc_info(args.old_controller_pid)
    command = [p.decode() for p in old["command"]]
    if (old["start_ticks"] != args.old_controller_ticks or args.run_id not in command
            or not {"rpcf.resume_exp032", "rpcf.parallel_exp032"}.intersection(command)):
        raise ValueError("Old controller identity mismatch")
    if old["state"] in {"T", "t", "Z"}:
        raise ValueError("Expected a running original controller")
    task_map = {t.task_id: t for t in tasks}
    paused = False
    try:
        os.kill(old["pid"], signal.SIGSTOP)
        paused = True
        for _ in range(100):
            if proc_info(old["pid"])["state"] in {"T", "t"}:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("Old controller did not pause")
        children = [int(p) for p in Path(f"/proc/{old['pid']}/task/{old['pid']}/children").read_text().split()]
        starts = {int(pid): (tid, int(gpu)) for tid, gpu, pid in re.findall(
            r"START task=(\S+) physical_gpu=(-?\d+) pid=(\d+)",
            Path(args.controller_log).read_text())}
        boot_epoch = time.time() - float(Path("/proc/uptime").read_text().split()[0])
        records = []
        for pid in children:
            child = proc_info(pid)
            if pid not in starts:
                if child["comm"] == "nvidia-smi":
                    continue
                raise RuntimeError(f"Unidentified child {pid}; retry capture after controller resumes")
            tid, gpu = starts[pid]
            if tid not in active or not valid_worker_gpu(task_map[tid], gpu, args.gpu_ids):
                raise ValueError(f"Unexpected active worker: {tid}/{gpu}")
            task = task_map[tid]
            rendered = render_command(task, run_dir)
            actual = [p.decode() for p in child["command"]]
            if actual and normalise_command(actual) != normalise_command(rendered):
                raise ValueError(f"Worker command differs from frozen task: {tid}")
            log_path = run_dir / "tasks" / f"{tid}.log"
            records.append({"task_id": tid, "pid": pid, "start_ticks": child["start_ticks"],
                            "gpu": gpu, "command": rendered,
                            "started": boot_epoch + child["start_ticks"] / os.sysconf("SC_CLK_TCK"),
                            "log_start_offset": max(0, log_path.read_bytes().rfind(b"COMMAND=")),
                            "actual_batch_size": actual_batch(run_dir, task),
                            "actual_cache_attack_batch_size": actual_cache_attack_batch(run_dir, task),
                            "actual_rpcf_batch_size": actual_rpcf_batch(run_dir, task),
                            "actual_rpcf_eval_batch_size": actual_rpcf_eval_batch(run_dir, task)})
        handoff = {"run_id": args.run_id, "created_at_epoch": time.time(),
                   "controller": {k: old[k] for k in ("pid", "start_ticks", "pgid")},
                   "workers": records, "policy": "pause_controller_only_keep_workers_running"}
        write_once(path, json.dumps(handoff, indent=2) + "\n")
        print(f"CAPTURED old_controller={old['pid']} live_workers={len(records)}", flush=True)
    except BaseException:
        if paused and not path.exists():
            os.kill(old["pid"], signal.SIGCONT)
        raise


class AdoptedProcess:
    """旧 parent 暂停期间不会 reap；用 /proc 中真实 wait status 接管在途任务。"""

    def __init__(self, pid, start_ticks):
        self.pid, self.start_ticks, self.returncode = pid, start_ticks, None

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        current = proc_info(self.pid)
        if current["start_ticks"] != self.start_ticks:
            raise RuntimeError(f"Adopted PID reused: {self.pid}")
        if current["state"] == "Z":
            self.returncode = os.waitstatus_to_exitcode(current["exit_code"])
        return self.returncode


def adopt_records(handoff, task_map, run_dir):
    running = {}
    for item in handoff["workers"]:
        task = task_map[item["task_id"]]
        if task_complete(run_dir, task):
            continue
        log_path = run_dir / "tasks" / f"{task.task_id}.log"
        record = {k: v for k, v in item.items() if k not in {"task_id", "pid", "start_ticks"}}
        peer_count = sum(task_map[peer["task_id"]].kind == "tnp" and peer["gpu"] == item["gpu"]
                         for peer in handoff["workers"])
        record.update(task=task, log_path=log_path, handle=log_path.open("a"),
                      process=AdoptedProcess(item["pid"], item["start_ticks"]),
                      adopted=True, process_start_ticks=item["start_ticks"],
                      tnp_capacity_at_start=max(1, peer_count), tnp_single_process_at_start=True)
        running[item["pid"]] = record
        print(f"ADOPT task={task.task_id} resource={'cpu' if task.kind == 'toy' else 'gpu'} "
              f"physical_gpu={item['gpu']} pid={item['pid']}", flush=True)
    return running


def gpu_free_memory(gpu_ids):
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    values = {}
    for line in result.stdout.splitlines():
        gpu, free, total = (int(p.strip()) for p in line.split(","))
        if gpu in gpu_ids:
            values[gpu] = (free, total)
    return values


def parse_paused_peers(value, gpu_ids):
    """共享授权绑定 GPU/PID/start_ticks，避免 PID 复用或扩展到未授权设备。"""
    peers = {}
    for item in filter(None, value.split(",")):
        gpu, pid, ticks = (int(part) for part in item.split(":"))
        if gpu not in gpu_ids or gpu in peers or pid <= 0 or ticks <= 0:
            raise ValueError("Invalid paused GPU peer identity")
        peers[gpu] = {"pid": pid, "start_ticks": ticks}
    return peers


def paused_peer_matches(expected, current, app_pids, utilization, max_utilization=5):
    """仅允许已核实的暂停进程；未知 context、恢复或身份变化都停止新派发。"""
    if utilization > max_utilization:
        return False
    if current is None:
        return not app_pids
    return (current["start_ticks"] == expected["start_ticks"]
            and current["state"] in {"T", "t"}
            and app_pids == {expected["pid"]})


def shareable_gpus(peers, running):
    """不向外部进程发送信号；仅在共享卡空出本实验槽位时重新核对。"""
    candidates = {gpu: peer for gpu, peer in peers.items()
                  if not any(r["gpu"] == gpu and r["task"].kind != "toy" for r in running.values())}
    if not candidates:
        return set()
    try:
        gpu_rows = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu",
                                   "--format=csv,noheader,nounits"], check=True, capture_output=True,
                                  text=True, timeout=10).stdout
        app_rows = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                   "--format=csv,noheader,nounits"], check=True, capture_output=True,
                                  text=True, timeout=10).stdout
        identities = {}
        for line in gpu_rows.splitlines():
            gpu, uuid, utilization = (part.strip() for part in line.split(","))
            identities[int(gpu)] = (uuid, int(utilization))
        apps = {}
        for line in app_rows.splitlines():
            if not line.strip():
                continue
            uuid, pid = (part.strip() for part in line.split(","))
            apps.setdefault(uuid, set()).add(int(pid))
        eligible = set()
        for gpu, peer in candidates.items():
            if gpu not in identities:
                continue
            uuid, utilization = identities[gpu]
            try:
                current = proc_info(peer["pid"])
            except FileNotFoundError:
                current = None
            if paused_peer_matches(peer, current, apps.get(uuid, set()), utilization):
                eligible.add(gpu)
        return eligible
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"SHARED_GPU_CHECK_SKIP error={error}", flush=True)
        return set()


def choose_gpu(task, running, gpu_ids, memory, tnp_capacity, min_free_mb,
               shared_gpu_ids=(), shared_ready=(), shared_min_free_mb=6144):
    """共享卡每卡仅一个非训练任务；普通卡沿用双 TNP/重任务独占规则。"""
    occupied = {g: [r for r in running.values() if r["task"].kind != "toy" and r["gpu"] == g]
                for g in gpu_ids}
    order = (sorted(gpu_ids, key=lambda g: (
        0 if g in shared_ready and not occupied[g] else 1 if occupied[g] else 2, gpu_ids.index(g)))
        if task.kind == "tnp" else gpu_ids)
    for gpu in order:
        if gpu not in memory:
            continue
        free, total = memory[gpu]
        on_gpu = occupied[gpu]
        if gpu in shared_gpu_ids:
            if (not on_gpu and gpu in shared_ready and free >= shared_min_free_mb
                    and task.kind in {"tnp", "attack", "external_eval"}):
                return gpu
            continue
        if not on_gpu:
            if total - free <= 768:
                return gpu
        elif (task.kind == "tnp" and all(r["task"].kind == "tnp" for r in on_gpu)
              and len(on_gpu) < tnp_capacity and free >= min_free_mb):
            return gpu
    return None


def acquire_controller_lock(handle, timeout=30.0):
    """进程退出与flock释放并非原子操作；有限等待，避免接管竞态退出。"""
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise TimeoutError("Controller lock still held; refusing concurrent dispatch")
            time.sleep(0.1)


def retire_controller(handoff, controller_lock, run_dir):
    """可重入地完成接管，兼容旧controller已退出但完成标记尚未写入的恢复。"""
    old = handoff["controller"]
    identity = (old["pid"], old["start_ticks"])
    if reservation_active(identity):
        checked_old_controller(handoff)
        os.kill(old["pid"], signal.SIGTERM)
        os.kill(old["pid"], signal.SIGCONT)
    acquire_controller_lock(controller_lock)
    atomic_json(run_dir / "parallel_handoff_completed_v5.json", {
        "old_controller": old["pid"], "old_start_ticks": old["start_ticks"],
        "at_epoch": time.time(), "controller_pid": os.getpid(),
        "completion_basis": "all_adopted_tasks_finalized_and_exclusive_lock_acquired"})
    print(f"HANDOFF_COMPLETE old_controller={old['pid']}", flush=True)


def run_scheduler(args, run_dir, tasks, scope, handoff):
    task_map = {t.task_id: t for t in tasks}
    active = set(scope["active_task_ids"])
    selected = [t for t in tasks if t.task_id in active]
    pending = {t.task_id: t for t in selected if not task_complete(run_dir, t)}
    retired = (run_dir / "parallel_handoff_completed_v5.json").exists()
    previous_state = run_dir / "parallel_runtime_state.json"
    original_pids = {r["pid"] for r in handoff["workers"]}
    if previous_state.exists():
        previous = json.loads(previous_state.read_text())
        for record in previous["running"]:
            if record["pid"] not in original_pids and reservation_active((record["pid"], record["start_ticks"])):
                raise RuntimeError(f"Previous parallel worker still active; refuse duplicate dispatch: {record}")
    running = {} if retired else adopt_records(handoff, task_map, run_dir)
    for record in running.values():
        pending.pop(record["task"].task_id, None)
    failures = {}
    controller_lock = (run_dir / "controller.lock").open("a")
    if retired:
        acquire_controller_lock(controller_lock)
    elif not running and not reservation_active((handoff["controller"]["pid"], handoff["controller"]["start_ticks"])):
        retire_controller(handoff, controller_lock, run_dir)
        retired = True
    fallback = run_dir / "parallel_tnp_single_process.flag"
    try:
        while pending or running:
            if not retired:
                checked_old_controller(handoff)
            for pid, record in list(running.items()):
                if record["process"].poll() is None:
                    continue
                # 原 flag 保留；并行版使用独立的 OOM 降档记录。
                completed, batch, cache_batch, rpcf_batch, _ = finalize_task(record, run_dir)
                del running[pid]
                if not completed:
                    task = record["task"]
                    tnp_oom = (task.kind == "tnp" and record["tnp_capacity_at_start"] > 1
                               and log_contains_oom(record["log_path"], record["log_start_offset"]))
                    if tnp_oom:
                        fallback.touch(exist_ok=True)
                    if any((batch, cache_batch, rpcf_batch, tnp_oom)):
                        pending[task.task_id] = task
                        print(f"RETRY task={task.task_id} tnp_single={tnp_oom}", flush=True)
                    else:
                        failures[task.task_id] = record["process"].returncode
                        print(f"FAILED task={task.task_id}", flush=True)
            if not retired and not any(r.get("adopted") for r in running.values()):
                retire_controller(handoff, controller_lock, run_dir)
                retired = True
            capacity = 1 if fallback.exists() else args.tnp_per_gpu
            memory = gpu_free_memory(args.gpu_ids)
            shared_ready = shareable_gpus(args.paused_peers, running)
            available_kb = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                                if line.startswith("MemAvailable:"))
            for tid, task in list(pending.items()):
                if task_complete(run_dir, task):
                    del pending[tid]
                    continue
                if not all(task_complete(run_dir, task_map[d]) for d in task.dependencies):
                    continue
                if available_kb < args.min_available_ram_gib * 1024 ** 2:
                    break
                if task.kind == "toy":
                    if sum(r["task"].kind == "toy" for r in running.values()) >= args.cpu_workers:
                        continue
                    gpu = -1
                else:
                    gpu = choose_gpu(task, running, args.gpu_ids, memory, capacity, args.min_tnp_free_mb,
                                     args.paused_peers, shared_ready, args.shared_min_free_mb)
                    if gpu is None:
                        continue
                if task.kind == "tnp":
                    for peer in running.values():
                        if peer["task"].kind == "tnp" and peer["gpu"] == gpu:
                            peer["tnp_capacity_at_start"] = max(peer["tnp_capacity_at_start"], capacity)
                record = start_task(task, run_dir, gpu)
                record.update(adopted=False, process_start_ticks=proc_info(record["process"].pid)["start_ticks"],
                              tnp_capacity_at_start=capacity,
                              tnp_single_process_at_start=True)
                running[record["process"].pid] = record
                del pending[tid]
                print(f"RESOURCE task={tid} pool={'cpu' if gpu == -1 else 'gpu'} gpu={gpu}", flush=True)
                # 预留正在加载的新进程资源，避免同一调度轮尚未分配显存时过量派发。
                available_kb -= (2 if gpu == -1 else 6) * 1024 ** 2
                if gpu != -1:
                    free, total = memory[gpu]
                    memory[gpu] = (max(0, free - 3072), total)
            runtime = {
                "at_epoch": time.time(), "controller_pid": os.getpid(), "handoff_complete": retired,
                "pending": len(pending), "failures": failures,
                "shared_ready_gpu_ids": sorted(shared_ready), "shared_gpu_ids": sorted(args.paused_peers),
                "running": [{"task_id": r["task"].task_id, "kind": r["task"].kind,
                             "resource": "cpu" if r["task"].kind == "toy" else "gpu", "gpu": r["gpu"],
                             "pid": r["process"].pid, "start_ticks": r["process_start_ticks"], "adopted": r["adopted"]} for r in running.values()]}
            runtime.update(execution_dir=str(run_dir), status_dir=str(run_dir / "status"))
            atomic_json(run_dir / "parallel_runtime_state.json", runtime)
            if getattr(args, "runtime_mirror", None):
                atomic_json(Path(args.runtime_mirror), runtime)
            if pending and not running and not any(all(task_complete(run_dir, task_map[d]) for d in t.dependencies)
                                                   for t in pending.values()):
                raise RuntimeError(f"DAG blocked by failures: {failures}")
            time.sleep(2)
    finally:
        controller_lock.close()
    if failures:
        raise RuntimeError(f"Tasks failed: {failures}")
    subprocess.run([sys.executable, "-u", "-m", getattr(args, "summary_module", "rpcf.summarize_exp032"), "--run-id", args.run_id], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "capture", "run"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run", default="exp031_full_20260729_174215")
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--paused-gpu-peers", default="", help="GPU:PID:start_ticks，逗号分隔")
    parser.add_argument("--shared-min-free-mb", type=int, default=6144)
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--tnp-per-gpu", type=int, choices=(1, 2), default=2)
    parser.add_argument("--min-tnp-free-mb", type=int, default=4096)
    parser.add_argument("--min-available-ram-gib", type=int, default=16)
    parser.add_argument("--old-controller-pid", type=int)
    parser.add_argument("--old-controller-ticks", type=int)
    parser.add_argument("--controller-log")
    args = parser.parse_args()
    if not args.run_id.startswith("exp032_") or "/" in args.run_id or args.cpu_workers < 1:
        raise ValueError("Invalid run/scheduling arguments")
    args.gpu_ids = [int(g) for g in args.gpu_ids.split(",")]
    if len(set(args.gpu_ids)) != len(args.gpu_ids) or not args.gpu_ids:
        raise ValueError("Invalid GPU list")
    args.paused_peers = parse_paused_peers(args.paused_gpu_peers, args.gpu_ids)
    if args.shared_min_free_mb < 6144:
        raise ValueError("Shared GPUs require at least 6144 MiB free")
    args.defer_clean_tnp = False
    run_dir = Path("logs/exp032") / args.run_id
    with (run_dir / "parallel_controller_v5.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks, scope = prepare_scope(args, run_dir)
        policy = {"version": 5, "run_id": args.run_id, "cpu_workers": args.cpu_workers,
                  "tnp_per_gpu": args.tnp_per_gpu, "gpu_ids": args.gpu_ids,
                  "paused_gpu_peers": args.paused_peers, "shared_min_free_mb": args.shared_min_free_mb,
                  "shared_gpu_task_limit": 1, "shared_gpu_kinds": ["tnp", "attack", "external_eval"],
                  "min_tnp_free_mb": args.min_tnp_free_mb, "min_available_ram_gib": args.min_available_ram_gib,
                  "gpu_exclusive_kinds": ["audit", "purifier_train", "train_standard", "attack", "external_eval"],
                  "sources_sha256": {p: digest(p) for p in ("rpcf/parallel_exp032.py", "rpcf/run_exp032_parallel.sh")}}
        write_once(run_dir / "parallel_policy_v5.json", json.dumps(policy, indent=2) + "\n")
        print(json.dumps(policy, indent=2), flush=True)
        if args.action == "capture":
            if not all((args.old_controller_pid, args.old_controller_ticks, args.controller_log)):
                parser.error("capture requires old controller PID, start ticks and controller log")
            capture_handoff(args, run_dir, tasks, set(scope["active_task_ids"]))
        elif args.action == "run":
            handoff = json.loads((run_dir / "parallel_handoff_v5.json").read_text())
            os.environ.setdefault("OMP_NUM_THREADS", "2")
            os.environ.setdefault("MKL_NUM_THREADS", "2")
            run_scheduler(args, run_dir, tasks, scope, handoff)


if __name__ == "__main__":
    main()
