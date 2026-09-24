"""EXP-033 独立任务清单和调度入口；默认只建档，不启动长实验。"""

import argparse
import fcntl
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from rpcf.exp033_common import (SOURCE_RUN, EXTERNAL_RUN, SEEDS, GROUPS, METHODS,
                               WEIGHTS, ATTACK_PROTOCOL, loss_variants, read_json,
                               write_json, file_hash, fingerprint)


def build_tasks(manifest):
    root = Path(manifest["run_dir"])
    tasks = []
    def add(kind, group, seed, suffix="", deps=(), **kw):
        tid = "_".join(str(x) for x in (kind, f"seed{seed}", suffix) if x != "")
        task = dict(task_id=tid, kind=kind, group=group, seed=seed,
                    output_path=str(root / "metrics" / f"{tid}.json"),
                    dependencies=list(deps), **kw)
        tasks.append(task)
        return tid
    for seed in manifest["seeds"]:
        source = add("sources", "shared", seed)
        cal = add("calibrate", "structure", seed, deps=[source])
        add("reference", "structure", seed, deps=[source], method="madry")
        for method in METHODS:
            for budget in (25, 30):
                add("structure", "structure", seed, f"{method}_{budget}", [cal], method=method, budget_rank=budget)
        for method in (*METHODS, "ptr"):
            for budget in (25, 30):
                add("timing", "structure", seed, f"{method}_{budget}", [cal], method=method,
                    budget_rank=budget, exclusive=True)
        add("reference", "rank", seed, "rank", [source], method="rpcf_at")
        for rank in ([15] if manifest["smoke"] else [15, 20, 35, 40]):
            add("tnp", "rank", seed, f"rank{rank}", [source], method="rpcf_at", ranks=[rank])
        add("reference", "loss", seed, "loss", [source], method="rpcf_at", variant="default")
        variants = loss_variants()[1:2] if manifest["smoke"] else loss_variants()[1:]
        for var in variants:
            train = add("train", "loss", seed, var["variant"], [source], **var)
            attack = add("attack", "loss", seed, var["variant"], [train], training_task=train, **var)
            add("tnp", "loss", seed, var["variant"], [attack], method="rpcf_at", ranks=[25, 30],
                attack_task=attack, training_task=train, **var)
        add("reference", "ablation", seed, "ablation", [source])
        ablation = add("tnp", "ablation", seed, "clean", [source], method="clean", ranks=[25, 30], capture=True)
        add("visualize", "visualize", seed, deps=[ablation], ablation_task=ablation)
    ids = [t["task_id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task IDs")
    if {d for t in tasks for d in t["dependencies"]} - set(ids):
        raise ValueError("Unknown dependency")
    return tasks


def select_tasks(tasks, groups):
    """选择模块时自动包含依赖，但不改变完整计划的验收范围。"""
    by_id = {t["task_id"]: t for t in tasks}
    selected = {t["task_id"] for t in tasks if t["group"] in groups}
    while True:
        expanded = selected | {d for tid in selected for d in by_id[tid]["dependencies"]}
        if expanded == selected:
            break
        selected = expanded
    return [t for t in tasks if t["task_id"] in selected]


def select_queue(tasks, queue):
    """计时队列仅消费已完成的依赖，普通队列不运行计时任务。"""
    if queue == "all": return tasks
    if queue == "compute": return [t for t in tasks if t["kind"] != "timing"]
    if queue == "timing": return [t for t in tasks if t["kind"] == "timing"]
    raise ValueError("Unknown queue")


def science_hashes():
    files = list(Path("rpcf").glob("exp033*.py"))
    files += [Path(p) for p in ("rpcf/run_exp033.sh", "rpcf/finetune.py", "rpcf/core.py", "rpcf/exp032_common.py",
                              "rpcf/exp032_tnp.py", "rpcf/exp031_artifacts.py", "utils/experiment_artifacts.py",
                              "rpcf/exp032_purifiers.py", "purify.py", "TN/PTR_3d.py",
                              "TN/tn_utils.py", "TN/BaseTNModel.py", "TN/utils.py", "TN/opt.py",
                              "utils/reproducibility.py", "attack/pgd.py", "data/subject_ea.py", "data/load.py")]
    files += list(Path("configs/thubenchmark").glob("PTR3d_8_2048_rank*_3d_interpolate.yaml"))
    return {str(p): file_hash(p) for p in sorted(set(files))}


def create_plan(run_id, smoke=False, source_run=SOURCE_RUN, external_run=EXTERNAL_RUN):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run-id must be a safe path token")
    root = Path("logs/exp033") / run_id
    manifest = dict(experiment_id="EXP-033", run_id=run_id, run_dir=str(root), smoke=smoke,
                    source_run=source_run, external_run=external_run, seeds=[42] if smoke else list(SEEDS),
                    dataset="thubenchmark", model="eegnet", sample_num=2 if smoke else 512,
                    fold=0, attack_protocol=ATTACK_PROTOCOL, validation_sample_num=2 if smoke else 32,
                    groups=list(GROUPS), loss_defaults=WEIGHTS, test_tuning=False,
                    budget_targets={"25": 14731, "30": 19891}, ordinary_tr_low_rank=[3, 22, 2, 3],
                    training_resume="restart_incomplete_attempt_from_original_seed_and_initial_checkpoint")
    tasks = build_tasks(manifest)
    manifest["counts"] = dict(Counter(t["kind"] for t in tasks))
    manifest["task_count"] = len(tasks)
    values = {"manifest.json": manifest, "tasks.json": tasks, "source_sha256.json": science_hashes()}
    for name, value in values.items():
        dest = root / name
        if dest.exists() and read_json(dest) != value:
            raise ValueError(f"Frozen {name} differs; use a new run-id")
    for name, value in values.items():
        if not (root / name).exists():
            write_json(root / name, value)
    return root, manifest, tasks


def verify_plan(root):
    if read_json(root / "source_sha256.json") != science_hashes():
        raise ValueError("Science code changed after plan freeze; create a new run-id")
    manifest, tasks = read_json(root / "manifest.json"), read_json(root / "tasks.json")
    if tasks != build_tasks(manifest):
        raise ValueError("Task manifest mismatch")
    return manifest, tasks


def idle_gpus(gpu_ids):
    """仅向无计算进程的 GPU 派发；不修改任何外部任务。"""
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    occupied = {line.split(",")[0].strip() for line in apps.splitlines() if "," in line}
    result = []
    for line in output.splitlines():
        index, uuid, free, utilization = [x.strip() for x in line.split(",")]
        if int(index) in gpu_ids and uuid not in occupied and int(free) >= 4096 and int(utilization) <= 10:
            result.append(int(index))
    return result


def run(root, groups, gpu_ids, cpu=False, retry_failed=False, queue="all"):
    manifest, all_tasks = verify_plan(root)
    if cpu and not manifest["smoke"]:
        raise ValueError("CPU runner is reserved for smoke; formal timing requires GPU")
    tasks = select_queue(select_tasks(all_tasks, groups), queue)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "controller.lock").open("a+")
    running, done, pending = {}, set(), []
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from rpcf.exp033_worker import verify_artifacts
        if queue == "timing":
            for dep in {d for task in tasks for d in task["dependencies"]}:
                dependency = next(t for t in all_tasks if t["task_id"] == dep)
                status_file = root / "status" / f"{dep}.json"
                if not status_file.exists() or read_json(status_file).get("status") != "completed":
                    raise ValueError("Timing dependencies are incomplete; run compute queue first")
                if file_hash(dependency["output_path"]) != read_json(status_file).get("output_sha256"):
                    raise ValueError("Timing dependency output changed")
                verify_artifacts(read_json(dependency["output_path"]))
                done.add(dep)
        for task in tasks:
            sp = root / "status" / f"{task['task_id']}.json"
            state = read_json(sp) if sp.exists() else {}
            if state.get("status") == "running" and Path(f"/proc/{state.get('pid')}").exists():
                raise RuntimeError(f"Worker still alive: {task['task_id']}; wait before restarting")
            if state.get("status") == "completed":
                if not Path(task["output_path"]).exists() or file_hash(task["output_path"]) != state.get("output_sha256"):
                    raise ValueError(f"Completed output missing/changed: {task['task_id']}")
                verify_artifacts(read_json(task["output_path"]))
                done.add(task["task_id"])
            elif state.get("status") == "failed" and not retry_failed:
                raise RuntimeError("Failed task present; inspect logs and use --retry-failed explicitly")
            else:
                pending.append(task)
    except BaseException:
        lock.close()
        raise
    if manifest["smoke"]:
        # 优先验证真实训练/净化链路，避免结构条件重复掩盖后续接口错误。
        priority = {"sources": 0, "train": 1, "attack": 2, "tnp": 3, "visualize": 4,
                    "calibrate": 5, "reference": 6, "structure": 7, "timing": 8}
        pending.sort(key=lambda task: priority[task["kind"]])
    try:
        while pending or running:
            for tid, record in list(running.items()):
                process, task, gpu, started, handle, attempt = record
                rc = process.poll()
                if rc is None:
                    continue
                handle.close()
                ok = rc == 0 and Path(task["output_path"]).exists()
                state = dict(status="completed" if ok else "failed", returncode=rc,
                             elapsed_seconds=time.time()-started, gpu=gpu, attempt=attempt,
                             output_path=task["output_path"], at_epoch=time.time())
                if ok:
                    state["output_sha256"] = file_hash(task["output_path"])
                    done.add(tid)
                write_json(root / "status" / f"{tid}.json", state)
                del running[tid]
                print(f"END {tid} {state['status']}", flush=True)
                if not ok:
                    raise RuntimeError(f"Worker failed: {tid}; remaining work is not marked completed")
            slots = [-1] if cpu and not running else [] if cpu else idle_gpus(gpu_ids)
            slots = [g for g in slots if all(record[2] != g for record in running.values())]
            for gpu in slots:
                ready = next((t for t in pending if set(t["dependencies"]) <= done), None)
                if ready is None:
                    break
                tid = ready["task_id"]
                log_dir = root / "worker_logs"
                log_dir.mkdir(exist_ok=True)
                attempt = len(list(log_dir.glob(f"{tid}.attempt*.log"))) + 1
                handle = (log_dir / f"{tid}.attempt{attempt}.log").open("a")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": "" if cpu else str(gpu),
                       "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
                command = [sys.executable, "-u", "-m", "rpcf.exp033_worker", "--run-dir", str(root), "--task-id", tid]
                process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
                write_json(root / "status" / f"{tid}.json", dict(status="running", pid=process.pid,
                           gpu=gpu, attempt=attempt, command=command, at_epoch=time.time()))
                running[tid] = (process, ready, gpu, time.time(), handle, attempt)
                pending.remove(ready)
                print(f"START {tid} gpu={gpu}", flush=True)
            write_json(root / "runtime.json", dict(at_epoch=time.time(), controller_pid=os.getpid(),
                       completed=len(done), pending=len(pending), running=[dict(task_id=k,pid=v[0].pid,gpu=v[2]) for k,v in running.items()]))
            if pending and not running and all(not set(t["dependencies"]) <= done for t in pending):
                raise RuntimeError("Unsatisfied dependency graph")
            if pending or running:
                time.sleep(2)
    finally:
        # 控制器异常时仅等待自己创建的在途任务，避免遗留匿名 worker。
        for process, task, gpu, started, handle, attempt in running.values():
            rc = process.wait()
            handle.close()
            ok = rc == 0 and Path(task["output_path"]).exists()
            state = dict(status="completed" if ok else "failed", returncode=rc,
                         elapsed_seconds=time.time()-started, gpu=gpu, attempt=attempt)
            if ok:
                state["output_sha256"] = file_hash(task["output_path"])
            write_json(root / "status" / f"{task['task_id']}.json", state)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "dry-run", "run", "smoke", "summary"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run", default=SOURCE_RUN)
    parser.add_argument("--external-run", default=EXTERNAL_RUN)
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--queue", choices=["all", "compute", "timing"], default="all")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("run-id must be a safe path token")
    groups = args.groups.split(",")
    if set(groups) - set(GROUPS):
        parser.error("Unknown group")
    root = Path("logs/exp033") / args.run_id
    if args.action == "summary":
        from rpcf.exp033_report import summarize
        result = summarize(root, strict=args.strict)
        print(result.get("status", result.get("completed")))
        return
    if (root / "manifest.json").exists():
        manifest, tasks = verify_plan(root)
        if args.smoke and not manifest["smoke"] or args.action == "smoke" and not manifest["smoke"]:
            raise ValueError("Smoke needs its own run-id")
    else:
        root, manifest, tasks = create_plan(args.run_id, args.smoke or args.action == "smoke", args.source_run, args.external_run)
    selected = select_queue(select_tasks(tasks, groups), args.queue)
    print(f"EXP-033 smoke={manifest['smoke']} planned={len(tasks)} selected={len(selected)} counts={dict(Counter(t['kind'] for t in selected))}", flush=True)
    if args.action in ("run", "smoke"):
        run(root, groups, [int(g) for g in args.gpu_ids.split(",")], args.cpu, args.retry_failed, args.queue)
        from rpcf.exp033_report import summarize
        summarize(root, strict=set(groups)==set(GROUPS) and args.queue=="all")


if __name__ == "__main__":
    main()
