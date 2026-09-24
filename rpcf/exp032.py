"""EXP-032 补充矩阵，复用 EXP-031 调度器但隔离所有新产物。"""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from rpcf.exp031 import (DATASETS, MODELS, SEEDS, ATTACKS, RAW_METHODS, Task,
                        attack_path, base_python, checkpoint, run_parallel, training_command)

SOURCE_RUN = "exp031_full_20260729_174215"
EXPECTED = {"audit": 90, "train_standard": 90, "purifier_train": 30,
            "attack": 900, "tnp": 630, "external_eval": 900, "toy": 45}


def new_attack_path(run, dataset, model, seed, method, attack):
    return f"ad_data/exp032/{run}/{dataset}_{model}_seed{seed}_{method}_{attack}.pth"


def tasks_for(args):
    tasks = []
    conditions = [(d, m, s) for d in DATASETS for m in MODELS for s in SEEDS]
    if args.smoke:
        conditions = [("thubenchmark", "eegnet", 42)]
    run = args.run_id
    n = 2 if args.smoke else 512

    def add(kind, stage, d, m, s, method, attack, output, deps, cmd):
        tid = "_".join(str(p) for p in (kind, d, m, f"seed{s}", method, attack) if p)
        tasks.append(Task(tid, stage, kind, d, m, s, method, attack, -1, output, tuple(deps), tuple(cmd)))
        return tid

    audits = {}
    for d, m, s in conditions:
        audits[d, m, s] = add("audit", 0, d, m, s, "", "",
            f"logs/exp032/{run}/audit/{d}_{m}_seed{s}.json", [],
            base_python("rpcf.exp032_evaluate", "audit", "--run-id", run, "--source-run", args.source_run,
                        "--dataset", d, "--model", m, "--seed", s, "--sample-num", n,
                        "--output-path", f"logs/exp032/{run}/audit/{d}_{m}_seed{s}.json"))
    purifier_ids = {}
    for d, s in sorted(set((d, s) for d, _, s in conditions)):
        for method in ("magnet", "dcae"):
            path = f"checkpoints/exp032/{run}/{d}_seed{s}_{method}.pth"
            extra = ("--epochs", "1", "--train-sample-num", "2", "--batch-size", "2") if args.smoke else (
                "--epochs", "100", "--batch-size", "256" if method == "magnet" else "128")
            purifier_ids[d, s, method] = add("purifier_train", 1, d, "", s, method, "", path, [],
                base_python("rpcf.exp032_external", "train", "--run-id", run, "--dataset", d,
                            "--seed", s, "--purifier", method, "--output-path", path, *extra))
    for d, m, s in conditions:
        clean_id = add("train_standard", 1, d, m, s, "clean", "", checkpoint(run, d, m, s, "clean").replace("_clean_eps0.03_", "_clean_eps0_"),
                       [audits[d, m, s]], training_command(run, d, m, s, "clean", args.smoke))
        for method in (*RAW_METHODS, "clean"):
            attacks = (*ATTACKS, "pgd_l2") if method == "clean" else ("pgd_l2",)
            for attack in attacks:
                output = new_attack_path(run, d, m, s, method, attack)
                extra = ("--l2-steps", "2", "--l2-restarts", "1") if args.smoke else ()
                aid = add("attack", 2, d, m, s, method, attack, output,
                          [clean_id] if method == "clean" else [audits[d, m, s]],
                          base_python("rpcf.exp032_evaluate", "attack", "--run-id", run,
                                      "--source-run", args.source_run, "--dataset", d, "--model", m,
                                      "--seed", s, "--method", method, "--attack", attack, "--sample-num", n,
                                      "--output-path", output, *extra))
                if method in {"clean", "madry", "rpcf_at"}:
                    pout = f"purified_data/exp032/{run}/{d}_{m}_seed{s}_{method}_{attack}.pth"
                    add("tnp", 3, d, m, s, method, attack, pout, [aid],
                        base_python("rpcf.exp032_tnp", "--run-id", run, "--source-run", args.source_run,
                                    "--dataset", d, "--model", m, "--seed", s, "--method", method,
                                    "--attack-path", output, "--output-path", pout))
                if method == "clean":
                    for purifier in ("magnet", "dcae"):
                        ep = f"logs/exp032/{run}/external/{d}_{m}_seed{s}_{purifier}_{attack}.json"
                        add("external_eval", 3, d, m, s, purifier, attack, ep,
                            [aid, purifier_ids[d, s, purifier]],
                            base_python("rpcf.exp032_external", "evaluate", "--run-id", run,
                                        "--source-run", args.source_run, "--dataset", d, "--model", m,
                                        "--seed", s, "--purifier", purifier, "--sample-num", n,
                                        "--batch-size", "8", "--attack", attack, "--attack-path", output,
                                        "--purifier-path", f"checkpoints/exp032/{run}/{d}_seed{s}_{purifier}.pth",
                                        "--output-path", ep, *extra))
        if m == "eegnet":
            for method in ("clean", "madry", "rpcf_at"):
                source = new_attack_path(run, d, m, s, "clean", "pgd") if method == "clean" else attack_path(args.source_run, d, m, s, method, "pgd")
                dep = f"attack_{d}_{m}_seed{s}_clean_pgd" if method == "clean" else audits[d, m, s]
                output = f"logs/exp032/{run}/toy/{d}_{m}_seed{s}_{method}.json"
                add("toy", 2, d, m, s, method, "pgd", output, [dep],
                    base_python("rpcf.exp032_toy", "--attack-path", source, "--dataset", d,
                                "--seed", s, "--method", method, "--sample-num", n, "--output-path", output))
    return tasks


def write_plan(args, tasks, run_dir):
    ids = [t.task_id for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task ids")
    missing = {dep for t in tasks for dep in t.dependencies} - set(ids)
    if missing:
        raise ValueError(f"Unknown dependencies: {missing}")
    counts = dict(Counter(t.kind for t in tasks))
    if not args.smoke and counts != EXPECTED:
        raise ValueError(f"Incorrect matrix coverage: {counts}")
    manifest = {"experiment_id": "EXP-032", "source_run": args.source_run,
                "run_id": args.run_id, "smoke": args.smoke, "counts": counts,
                "protocol": {"datasets": DATASETS, "models": MODELS, "seeds": SEEDS,
                             "sample_num": 2 if args.smoke else 512, "fold": 0,
                             "linf_eps": 0.03, "l2_eps": 1.0, "tnp_ranks": [25, 30],
                             "purifiers": ["magnet", "dcae"], "gan": "withdrawn_by_user",
                             "toy_backbone": "eegnet", "test_tuning": False,
                             "storage_policy": "adv_float32_clean_reference_tnp_scalar_and_preview",
                             "estimated_new_attack_gib": 112.5}}
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError("Existing run manifest differs; use a new run-id")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    with (run_dir / "planned_tasks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(tasks[0])))
        writer.writeheader()
        for task in tasks:
            row = asdict(task)
            row["dependencies"] = json.dumps(row["dependencies"])
            row["command"] = json.dumps(row["command"])
            writer.writerow(row)
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("rpcf").glob("exp032*.py")}
    source_snapshot = run_dir / "source_sha256.json"
    if source_snapshot.exists() and json.loads(source_snapshot.read_text()) != frozen:
        raise ValueError("Code changed after run snapshot; create a new run-id")
    source_snapshot.write_text(json.dumps(frozen, indent=2) + "\n")
    # clean-only 优先使用该 dataset/backbone 在 EXP-031 的实际训练 batch。
    previous = Path(f"logs/exp031/{args.source_run}/actual_batch_sizes.json")
    batch_path = run_dir / "actual_batch_sizes.json"
    if previous.exists() and not batch_path.exists():
        batch_path.write_text(previous.read_text())
    # 默认每 GPU 一个任务，避免大 DCAE/插值张量与 TNP 竞争显存。
    (run_dir / "tnp_single_process.flag").touch()
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "run"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run", default=SOURCE_RUN)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5")
    parser.add_argument("--start-stage", type=int, default=0)
    parser.add_argument("--stop-stage", type=int, default=3)
    parser.add_argument("--task-id")
    args = parser.parse_args()
    if not args.run_id.startswith("exp032_") or "/" in args.run_id:
        raise ValueError("run-id must start with exp032_ and contain no slash")
    tasks = tasks_for(args)
    run_dir = Path("logs/exp032") / args.run_id
    write_plan(args, tasks, run_dir)
    if args.action == "run":
        import fcntl
        with (run_dir / "controller.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            selected = [t for t in tasks if args.start_stage <= t.stage <= args.stop_stage
                        and (args.task_id is None or t.task_id == args.task_id)]
            if not selected:
                raise ValueError("No tasks selected")
            gpu_ids = [int(g) for g in args.gpu_ids.split(",")]
            os.environ.setdefault("EXP031_MAX_IDLE_MEMORY_MB", "256")
            os.environ.setdefault("OMP_NUM_THREADS", "2")
            os.environ.setdefault("MKL_NUM_THREADS", "2")
            run_parallel(tasks, selected, run_dir, gpu_ids)
            import subprocess
            import sys
            subprocess.run([sys.executable, "-u", "-m", "rpcf.summarize_exp032", "--run-id", args.run_id], check=True)


if __name__ == "__main__":
    main()
