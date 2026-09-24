"""保留 EXP-032 原始计划，以独立执行范围暂缓 clean-only+TNP 并续跑。"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from rpcf.exp032 import tasks_for
from rpcf.exp031 import run_parallel


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scope_for(run_dir, rows):
    """仅允许暂缓 clean TNP，其他方法、攻击与依赖仍使用原计划。"""
    deferred = [r["task_id"] for r in rows if r["kind"] == "tnp" and r["method"] == "clean"]
    active = [r["task_id"] for r in rows if r["task_id"] not in set(deferred)]
    return {
        "version": 1, "name": "without_clean_tnp",
        "reason": "User requested deferring clean-only+TNP; other EXP-032 work continues.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "planned_tasks_sha256": digest(run_dir / "planned_tasks.csv"),
        "active_task_ids": active, "deferred_task_ids": deferred,
        "active_counts": dict(Counter(r["kind"] for r in rows if r["task_id"] in set(active))),
        "expected_rows_per_condition": 70,
    }


def read_scope(run_dir, rows):
    """调度与验收共用严格范围检查，避免漏跑被误报为完成。"""
    path = run_dir / "execution_scope.json"
    if not path.exists():
        return None
    scope = json.loads(path.read_text())
    expected = scope_for(run_dir, rows)
    for key in ("version", "name", "planned_tasks_sha256", "active_task_ids",
                "deferred_task_ids", "active_counts", "expected_rows_per_condition"):
        if scope.get(key) != expected[key]:
            raise ValueError(f"Invalid EXP-032 execution scope: {key}")
    active = set(scope["active_task_ids"])
    for row in rows:
        if row["task_id"] in active and not set(json.loads(row["dependencies"])).issubset(active):
            raise ValueError(f"Active task depends on deferred task: {row['task_id']}")
    return scope


def write_once(path, content):
    """同内容可续跑；拒绝改写已存在的计划、代码快照或范围记录。"""
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Existing scope artifact differs: {path}")
        return
    with path.open("x") as handle:
        handle.write(content)


def prepare_scope(args, run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["run_id"] != args.run_id or manifest["source_run"] != args.source_run:
        raise ValueError("Run/source identity differs from original manifest")
    with (run_dir / "planned_tasks.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    # 不调用原 write_plan，原始 manifest、计划与科学实现快照保持原样。
    tasks = tasks_for(SimpleNamespace(run_id=args.run_id, source_run=args.source_run, smoke=manifest["smoke"]))
    expected_rows = []
    for task in tasks:
        row = asdict(task)
        row["dependencies"] = json.dumps(row["dependencies"])
        row["command"] = json.dumps(row["command"])
        expected_rows.append({k: str(v) for k, v in row.items()})
    if rows != expected_rows or dict(Counter(t.kind for t in tasks)) != manifest["counts"]:
        raise ValueError("Original task plan differs from current frozen implementation")
    original = json.loads((run_dir / "source_sha256.json").read_text())
    for path, expected in original.items():
        if digest(path) != expected:
            raise ValueError(f"Scientific implementation changed: {path}")
    scope = read_scope(run_dir, rows)
    if scope is None:
        if not args.defer_clean_tnp:
            raise ValueError("First scope amendment requires --defer-clean-tnp")
        scope = scope_for(run_dir, rows)
    snapshot = {p: digest(p) for p in (
        "rpcf/resume_exp032.py", "rpcf/summarize_exp032.py", "rpcf/run_exp032.sh")}
    write_once(run_dir / "scope_sources_sha256.json", json.dumps(snapshot, indent=2) + "\n")
    for name, ids in (("active_tasks.csv", scope["active_task_ids"]),
                      ("deferred_tasks.csv", scope["deferred_task_ids"])):
        import io
        handle = io.StringIO(newline="")
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(r for r in rows if r["task_id"] in set(ids))
        write_once(run_dir / name, handle.getvalue())
    write_once(run_dir / "execution_scope.json", json.dumps(scope, indent=2) + "\n")
    read_scope(run_dir, rows)
    return tasks, scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run", default="exp031_full_20260729_174215")
    parser.add_argument("--defer-clean-tnp", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5")
    parser.add_argument("--start-stage", type=int, default=0)
    parser.add_argument("--stop-stage", type=int, default=3)
    parser.add_argument("--task-id")
    args = parser.parse_args()
    if not args.run_id.startswith("exp032_") or "/" in args.run_id:
        raise ValueError("Invalid run-id")
    run_dir = Path("logs/exp032") / args.run_id
    # 修改范围前必须由调用者停止旧 controller，锁防止同时调度。
    with (run_dir / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks, scope = prepare_scope(args, run_dir)
        active = set(scope["active_task_ids"])
        selected = [t for t in tasks if t.task_id in active
                    and args.start_stage <= t.stage <= args.stop_stage
                    and (args.task_id is None or t.task_id == args.task_id)]
        if not selected:
            raise ValueError("No active tasks selected; clean-only+TNP is deferred")
        print(json.dumps({"run_id": args.run_id, "scope": scope["name"],
                          "active_tasks": len(active), "deferred_tasks": len(scope["deferred_task_ids"]),
                          "active_counts": scope["active_counts"]}, indent=2), flush=True)
        if args.action == "run":
            os.environ.setdefault("EXP031_MAX_IDLE_MEMORY_MB", "256")
            os.environ.setdefault("OMP_NUM_THREADS", "2")
            os.environ.setdefault("MKL_NUM_THREADS", "2")
            run_parallel(tasks, selected, run_dir, [int(g) for g in args.gpu_ids.split(",")])
            subprocess.run([sys.executable, "-u", "-m", "rpcf.summarize_exp032",
                            "--run-id", args.run_id], check=True)


if __name__ == "__main__":
    main()
