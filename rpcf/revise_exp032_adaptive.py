"""隔离 EXP-032 的 PGD-10 修订；旧计划、日志与科学产物原样保留。"""

import argparse
import csv
import io
import json
import os
import signal
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from rpcf.exp031 import reservation_active, task_complete
from rpcf.exp032_external_pgd10 import ADAPTIVE_PROTOCOL, REVISION
from rpcf.parallel_exp032 import (acquire_controller_lock, atomic_json, parse_paused_peers,
                                  proc_info, run_scheduler)
from rpcf.resume_exp032 import digest, prepare_scope, write_once


SOURCES = ("rpcf/exp032_external_pgd10.py", "rpcf/revise_exp032_adaptive.py",
           "rpcf/summarize_exp032_pgd10.py", "rpcf/parallel_exp032.py",
           "rpcf/run_exp032_pgd10.sh")


def revision_dir(run_id):
    return Path("logs/exp032") / run_id / "execution_revisions" / REVISION


def revised_tasks(tasks, run_dir):
    """只改外部评估入口和输出；所有训练、普通攻击、TNP 和依赖保持原样。"""
    result = []
    for task in tasks:
        if task.kind != "external_eval":
            result.append(task)
            continue
        output = str(run_dir / "external" / Path(task.output_path).name)
        command = list(task.command)
        command[command.index("rpcf.exp032_external")] = "rpcf.exp032_external_pgd10"
        command[command.index("--output-path") + 1] = output
        for flag in ("--l2-steps", "--l2-restarts"):
            if flag in command:
                index = command.index(flag)
                del command[index:index + 2]
        result.append(replace(task, output_path=output, command=tuple(command)))
    return result


def validated_nonadaptive(payload, task, audit, sample_num):
    """按样本身份、预测和协议选择唯一非自适应行；绝不复用旧 adaptive 行。"""
    for key in ("source_indices", "labels"):
        if payload[key] != audit[key] or len(payload[key]) != sample_num:
            raise ValueError(f"Invalid reused {key}: {task.task_id}")
    if len(set(payload["source_indices"])) != sample_num:
        raise ValueError(f"Duplicate reused samples: {task.task_id}")
    rows = [row for row in payload["rows"] if row["evaluation"] == "nonadaptive_purification"]
    if len(rows) != 1:
        raise ValueError(f"Expected one nonadaptive row: {task.task_id}")
    row = rows[0]
    expected = dict(dataset=task.dataset, model=task.model, seed=task.seed, fold=0,
                    method=f"clean_{task.method}", attack=task.attack, scope="S", sample_num=sample_num)
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Reused row identity mismatch: {task.task_id}")
    for metric, field in (("standard_accuracy", "clean_predictions"), ("robust_accuracy", "adv_predictions")):
        predictions = row[field]
        if len(predictions) != sample_num:
            raise ValueError(f"Reused prediction count mismatch: {task.task_id}")
        value = sum(p == y for p, y in zip(predictions, payload["labels"])) / sample_num
        if abs(row[metric] - value) > 1e-12:
            raise ValueError(f"Reused accuracy mismatch: {task.task_id}")
    result = dict(payload, rows=rows, protocol_revision=REVISION)
    result["reuse"] = {"source_path": task.output_path, "source_sha256": digest(task.output_path),
                       "basis": "validated_nonadaptive_row_only", "old_adaptive_rows_excluded": True}
    return result


def prepare_revision(args, base, run_dir):
    original_tasks, original_scope = prepare_scope(args, base)
    tasks = revised_tasks(original_tasks, run_dir)
    for name in ("status", "tasks", "external"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(asdict(tasks[0])), lineterminator="\n")
    writer.writeheader()
    for task in tasks:
        row = asdict(task)
        for field in ("dependencies", "command"):
            row[field] = json.dumps(row[field])
        writer.writerow(row)
    write_once(run_dir / "planned_tasks.csv", buffer.getvalue())
    scope = dict(original_scope, name=REVISION, expected_rows_per_condition=62,
                 planned_tasks_sha256=digest(run_dir / "planned_tasks.csv"),
                 reason="User requested historical adaptive PGD-10 only; clean-only+TNP remains deferred.",
                 protocol_revision=REVISION, adaptive_protocol=ADAPTIVE_PROTOCOL,
                 base_execution_scope_sha256=digest(base / "execution_scope.json"))
    write_once(run_dir / "execution_scope.json", json.dumps(scope, indent=2) + "\n")
    manifest = json.loads((base / "manifest.json").read_text())
    manifest.update(protocol_revision=REVISION, base_run_dir=str(base),
                    base_manifest_sha256=digest(base / "manifest.json"))
    write_once(run_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    write_once(run_dir / "source_sha256.json", json.dumps({p: digest(p) for p in SOURCES}, indent=2) + "\n")
    # 只在首次迁移复制运行时 batch 配置；之后 OOM 降档由新执行目录独立记录。
    if not (run_dir / "migration_report.json").exists():
        for path in base.glob("actual*batch_sizes.json"):
            write_once(run_dir / path.name, path.read_text())
        for name in ("tnp_single_process.flag", "parallel_tnp_single_process.flag"):
            if (base / name).exists():
                write_once(run_dir / name, (base / name).read_text())
        migrated = Counter()
        active = set(scope["active_task_ids"])
        for old, new in zip(original_tasks, tasks):
            if old.task_id not in active or not task_complete(base, old):
                continue
            source_status = base / "status" / f"{old.task_id}.json"
            if old.kind != "external_eval":
                write_once(run_dir / "status" / source_status.name, source_status.read_text())
                migrated["unchanged_tasks"] += 1
            elif old.attack == "pgd":
                migrated["old_pgd_tasks_requiring_pgd10"] += 1
            else:
                payload = json.loads(Path(old.output_path).read_text())
                audit = json.loads((base / "audit" / f"{old.dataset}_{old.model}_seed{old.seed}.json").read_text())
                payload = validated_nonadaptive(payload, old, audit, 2 if manifest["smoke"] else 512)
                write_once(Path(new.output_path), json.dumps(payload, indent=2) + "\n")
                status = {"task_id": new.task_id, "status": "completed", "returncode": None,
                          "exit_code_observed": False, "completion_basis": "validated_nonadaptive_row_reuse",
                          "source_status": str(source_status), "source_status_sha256": digest(source_status),
                          "output_path": new.output_path, "protocol_revision": REVISION}
                write_once(run_dir / "status" / source_status.name, json.dumps(status, indent=2) + "\n")
                migrated["reused_nonadaptive_external_tasks"] += 1
        report = dict(migrated, active_tasks=len(active), deferred_tasks=len(scope["deferred_task_ids"]),
                      expected_rows_per_condition=62, adaptive_conditions=180 if not manifest["smoke"] else 2,
                      old_artifacts_preserved=True)
        write_once(run_dir / "migration_report.json", json.dumps(report, indent=2) + "\n")
    return tasks, scope


def validate_revision_scope(run_dir, tasks):
    """供汇总器检查修订范围，避免重新导入冻结的旧 adaptive 结果。"""
    scope = json.loads((run_dir / "execution_scope.json").read_text())
    if (scope["protocol_revision"] != REVISION or scope["adaptive_protocol"] != ADAPTIVE_PROTOCOL
            or scope["expected_rows_per_condition"] != 62
            or scope["planned_tasks_sha256"] != digest(run_dir / "planned_tasks.csv")):
        raise ValueError("Invalid PGD-10 revision scope")
    active, deferred = set(scope["active_task_ids"]), set(scope["deferred_task_ids"])
    if active & deferred or active | deferred != {t["task_id"] for t in tasks}:
        raise ValueError("Revision task coverage mismatch")
    for task in tasks:
        if task["task_id"] in active and not set(json.loads(task["dependencies"])).issubset(active):
            raise ValueError("Active task depends on deferred task")
    return scope


def retire_paused_original(args, base):
    """仅终止已核实且暂停的旧控制器；先确认其 GPU worker 已退出。"""
    runtime = json.loads((base / "parallel_runtime_state.json").read_text())
    for worker in runtime["running"]:
        if reservation_active((worker["pid"], worker["start_ticks"])):
            raise RuntimeError(f"Original worker still alive: {worker}")
    if reservation_active((args.old_controller_pid, args.old_controller_ticks)):
        old = proc_info(args.old_controller_pid)
        command = [part.decode() for part in old["command"]]
        if (old["start_ticks"] != args.old_controller_ticks or old["state"] not in {"T", "t"}
                or "rpcf.parallel_exp032" not in command or args.run_id not in command):
            raise RuntimeError("Refuse retiring an unverified controller")
        os.kill(old["pid"], signal.SIGTERM)
        os.kill(old["pid"], signal.SIGCONT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "run"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run", default="exp031_full_20260729_174215")
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--paused-gpu-peers", default="1:3640736:451734835,2:3643011:451991114,3:3643222:451998320")
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--tnp-per-gpu", type=int, choices=[1, 2], default=2)
    parser.add_argument("--shared-min-free-mb", type=int, default=6144)
    parser.add_argument("--min-tnp-free-mb", type=int, default=4096)
    parser.add_argument("--min-available-ram-gib", type=int, default=16)
    parser.add_argument("--old-controller-pid", type=int, default=3801846)
    parser.add_argument("--old-controller-ticks", type=int, default=458003150)
    args = parser.parse_args()
    if not args.run_id.startswith("exp032_") or "/" in args.run_id:
        raise ValueError("Invalid run id")
    args.gpu_ids = [int(g) for g in args.gpu_ids.split(",")]
    if not args.gpu_ids or len(set(args.gpu_ids)) != len(args.gpu_ids) or args.cpu_workers < 1:
        raise ValueError("Invalid resource policy")
    if args.shared_min_free_mb < 6144:
        raise ValueError("Shared GPUs require at least 6144 MiB free")
    args.paused_peers = parse_paused_peers(args.paused_gpu_peers, args.gpu_ids)
    args.defer_clean_tnp = False
    args.summary_module = "rpcf.summarize_exp032_pgd10"
    base, run_dir = Path("logs/exp032") / args.run_id, revision_dir(args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "revision.lock").open("a") as revision_lock:
        acquire_controller_lock(revision_lock, timeout=1)
        tasks, scope = prepare_revision(args, base, run_dir)
        policy = {k: getattr(args, k) for k in ("gpu_ids", "paused_peers", "cpu_workers", "tnp_per_gpu",
                  "shared_min_free_mb", "min_tnp_free_mb", "min_available_ram_gib")}
        write_once(run_dir / "resource_policy.json", json.dumps(policy, indent=2) + "\n")
        print((run_dir / "migration_report.json").read_text(), flush=True)
        if args.action == "plan":
            return
        # 新控制器持有原 run 的锁，禁止旧入口与修订入口同时派发。
        if not (run_dir / "parallel_handoff_completed_v5.json").exists():
            retire_paused_original(args, base)
        with (base / "controller.lock").open("a") as base_lock:
            acquire_controller_lock(base_lock)
            args.runtime_mirror = str(base / "parallel_runtime_state.json")
            pointer = {"revision": REVISION, "run_id": args.run_id, "execution_dir": str(run_dir),
                       "status_dir": str(run_dir / "status"), "controller_pid": os.getpid(),
                       "controller_start_ticks": proc_info(os.getpid())["start_ticks"],
                       "at_epoch": time.time(), "adaptive_protocol": ADAPTIVE_PROTOCOL,
                       "controller_log": f"logs/{args.run_id}.pgd10_v1.controller.log"}
            atomic_json(base / "current_execution.json", pointer)
            handoff = {"controller": {"pid": args.old_controller_pid, "start_ticks": args.old_controller_ticks},
                       "workers": []}
            os.environ.setdefault("OMP_NUM_THREADS", "2")
            os.environ.setdefault("MKL_NUM_THREADS", "2")
            run_scheduler(args, run_dir, tasks, scope, handoff)


if __name__ == "__main__":
    main()
