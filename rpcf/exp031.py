"""EXP-031 全矩阵 DAG 规划、执行和完整性入口。

正式运行示例见 ``rpcf/run_exp031.sh``。本模块默认只生成计划，不会隐式启动训练。
"""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


EXPERIMENT_ID = "EXP-031"
DATASETS = ("thubenchmark", "seediv", "bciciv2a")
MODELS = ("eegnet", "deepconvnet", "tsception", "atcnet", "conformer", "tcnet")
SEEDS = (42, 43, 44, 45, 46)
ATTACKS = ("autoattack", "fgsm", "pgd", "cw")
RAW_METHODS = ("madry", "trades", "fbf", "ea_forward", "rpcf_at")
TNP_METHODS = ("madry", "rpcf_at")
TRAIN_RANKS = "15,20,25,30,35,40"
TRAIN_CONFIGS = ",".join(
    f"PTR3d_8_2048_rank{rank}_3d_interpolate.yaml"
    for rank in (15, 20, 25, 30, 35, 40)
)
EVAL_RANKS = "25,30"
EVAL_CONFIGS = ",".join(
    f"PTR3d_8_2048_rank{rank}_3d_interpolate.yaml" for rank in (25, 30)
)
PROTOCOL = "train_only_subject_no_ea_subject_split"
EXPECTED_FULL_COUNTS = {
    "train_standard": 270,
    "train_ea": 90,
    "rpcf_cache": 90,
    "rpcf_at": 90,
    "attack": 1800,
    "tnp": 720,
    "bpda": 10,
}
CACHE_ATTACK_BATCH_FALLBACK = (32, 16, 8, 4)
RPCF_BATCH_FALLBACK = (64, 32, 16, 8)
TASK_SCOPES = ("all", "thu_eegnet_closure")


@dataclass(frozen=True)
class Task:
    task_id: str
    stage: int
    kind: str
    dataset: str = ""
    model: str = ""
    seed: int = -1
    method: str = ""
    attack: str = ""
    rank: int = -1
    output_path: str = ""
    dependencies: tuple = ()
    command: tuple = ()


def token(*parts):
    return "_".join(str(part).replace("/", "-") for part in parts if part != "")


def checkpoint(run_id, dataset, model, seed, method):
    if method == "ea_forward":
        model_name = f"{model}_ea_forward"
        strategy = "madry"
    else:
        model_name = model
        strategy = "madry" if method == "rpcf_at" else method
    tag = token(run_id, method)
    return (
        f"checkpoints/{dataset}_{model_name}_{PROTOCOL}_{strategy}_eps0.03_"
        f"{seed}_fold0_{tag}_best.pth"
    )


def cache_path(run_id, dataset, model, seed):
    return f"purified_data/exp031/{run_id}/cache/{dataset}_{model}_seed{seed}.pth"


def attack_path(run_id, dataset, model, seed, method, attack):
    return (
        f"ad_data/exp031/{run_id}/{dataset}_{model}_seed{seed}_"
        f"{method}_{attack}.pth"
    )


def tnp_path(run_id, dataset, model, seed, method, attack):
    return (
        f"purified_data/exp031/{run_id}/eval/{dataset}_{model}_seed{seed}_"
        f"{method}_{attack}_rank25-30.pth"
    )


def base_python(module, *args):
    return (
        "conda", "run", "-n", "torch", "--no-capture-output",
        "python", "-u", "-m", module, *[str(arg) for arg in args],
    )


def condition_id(dataset, model, seed):
    return token(dataset, model, f"seed{seed}")


def make_task(run_id, kind, stage, dataset="", model="", seed=-1,
              method="", attack="", rank=-1, output_path="", deps=(), command=()):
    task_id = token(kind, dataset, model, f"seed{seed}" if seed >= 0 else "", method, attack,
                    f"rank{rank}" if rank >= 0 else "")
    return Task(task_id, stage, kind, dataset, model, seed, method, attack, rank,
                output_path, tuple(deps), tuple(command))


def training_command(run_id, dataset, model, seed, method, smoke):
    epochs = 1 if smoke else 400
    patience = 1 if smoke else 20
    sample_args = ("--train_sample_num", "2") if smoke else ()
    if method == "ea_forward":
        return base_python(
            "train_AT_ea_forward", "--dataset", dataset, "--model", f"{model}_ea_forward",
            "--at_strategy", "madry", "--fold", 0, "--epsilon", 0.03,
            "--pgd_steps", 10, "--pgd_step_size", 0.006, "--epochs", epochs,
            "--patience", patience, "--batch_size", "__BATCH_SIZE__", "--lr", 0.001,
            "--weight_decay", 0.0001, "--seed", seed, "--gpu_id", 0, "--no_ea",
            "--checkpoint_tag", token(run_id, method), *sample_args,
        )
    return base_python(
        "train_AT", "--dataset", dataset, "--model", model, "--at_strategy", method,
        "--fold", 0, "--epsilon", 0.03, "--pgd_steps", 10, "--pgd_step_size", 0.006,
        "--fbf_replays", 3, "--trades_beta", 0.1, "--epochs", epochs,
        "--patience", patience, "--batch_size", "__BATCH_SIZE__", "--lr", 0.001,
        "--weight_decay", 0.0001, "--seed", seed, "--gpu_id", 0, "--no_ea",
        "--checkpoint_tag", token(run_id, method), *sample_args,
    )


def build_tasks(run_id, smoke=False):
    tasks = []
    conditions = [(d, m, s) for d in DATASETS for m in MODELS for s in SEEDS]
    if smoke:
        conditions = [(d, m, 42) for d in DATASETS for m in MODELS]

    previous_data_task = None
    for dataset in DATASETS:
        marker = f"logs/exp031/{run_id}/data/{dataset}.json"
        tasks.append(make_task(
            run_id, "prepare_data", 0, dataset=dataset, output_path=marker,
            deps=(() if previous_data_task is None else (previous_data_task,)),
            command=base_python("rpcf.exp031", "prepare-data", "--dataset", dataset,
                                "--output-path", marker),
        ))
        previous_data_task = token("prepare_data", dataset)

    for dataset, model, seed in conditions:
        cid = condition_id(dataset, model, seed)
        data_dep = token("prepare_data", dataset)
        madry_id = token("train_standard", dataset, model, f"seed{seed}", "madry")
        for method in ("madry", "trades", "fbf"):
            previous_method = {"madry": None, "trades": "madry", "fbf": "trades"}[method]
            deps = (data_dep,) if previous_method is None else (
                data_dep,
                token("train_standard", dataset, model, f"seed{seed}", previous_method),
            )
            tasks.append(make_task(
                run_id, "train_standard", 1, dataset, model, seed, method,
                output_path=checkpoint(run_id, dataset, model, seed, method), deps=deps,
                command=training_command(run_id, dataset, model, seed, method, smoke),
            ))
        tasks.append(make_task(
            run_id, "train_ea", 1, dataset, model, seed, "ea_forward",
            output_path=checkpoint(run_id, dataset, model, seed, "ea_forward"),
            deps=(data_dep, token("train_standard", dataset, model, f"seed{seed}", "fbf")),
            command=training_command(run_id, dataset, model, seed, "ea_forward", smoke),
        ))

        cache = cache_path(run_id, dataset, model, seed)
        cache_deps = [
            token("train_ea", dataset, model, f"seed{seed}", "ea_forward")
        ]
        cache_args = []
        if model != "eegnet":
            canonical_id = token("rpcf_cache", dataset, "eegnet", f"seed{seed}", "madry")
            cache_deps.append(canonical_id)
            cache_args = ["--shared_clean_path", cache_path(run_id, dataset, "eegnet", seed)]
        tasks.append(make_task(
            run_id, "rpcf_cache", 2, dataset, model, seed, "madry",
            output_path=cache, deps=cache_deps,
            command=base_python(
                "rpcf.generate_cache", "--dataset", dataset, "--model", model,
                "--fold", 0, "--seed", seed, "--attack", "autoattack", "--eps", 0.03,
                "--checkpoint_path", checkpoint(run_id, dataset, model, seed, "madry"),
                "--sample_num", 2 if smoke else 512, "--attack_batch_size",
                2 if smoke else "__CACHE_ATTACK_BATCH_SIZE__",
                "--ranks", TRAIN_RANKS, "--configs", TRAIN_CONFIGS, "--gpu_id", 0,
                "--tag", token(run_id, cid), "--output_path", cache, *cache_args,
            ),
        ))

        rpcf_ckpt = checkpoint(run_id, dataset, model, seed, "rpcf_at")
        history = f"logs/exp031/{run_id}/history/{cid}_rpcf_at"
        tasks.append(make_task(
            run_id, "rpcf_at", 3, dataset, model, seed, "rpcf_at",
            output_path=rpcf_ckpt, deps=(token("rpcf_cache", dataset, model, f"seed{seed}", "madry"),),
            command=base_python(
                "rpcf.finetune", "--cache_path", cache, "--checkpoint_path",
                checkpoint(run_id, dataset, model, seed, "madry"), "--output_checkpoint", rpcf_ckpt,
                "--dataset", dataset, "--model", model, "--fold", 0, "--seed", seed,
                "--epsilon", 0.03, "--epochs", 1 if smoke else 100, "--batch_size",
                2 if smoke else "__RPCF_BATCH_SIZE__", "--eval_batch_size",
                2 if smoke else "__RPCF_EVAL_BATCH_SIZE__", "--lr", 0.0001,
                "--weight_decay", 0.0001, "--online_madry_at", "--online_at_batch_size",
                "__BATCH_SIZE__", "--online_at_pgd_steps", 10, "--online_at_step_size", 0.006,
                "--all_layers", "--static_rank_weights", "--feature_objective", "none",
                "--gpu_id", 0, "--history_prefix", history,
                *(("--online_train_sample_num", "2", "--max_cache_batches", "1") if smoke else ()),
            ),
        ))

    attack_conditions = conditions
    if smoke:
        # 18 条件轮转四攻击；每个方法和每种攻击至少出现一次。
        smoke_attack_specs = []
        for index, (dataset, model, seed) in enumerate(conditions):
            smoke_attack_specs.append((dataset, model, seed, RAW_METHODS[index % len(RAW_METHODS)],
                                       ATTACKS[index % len(ATTACKS)]))
            smoke_attack_specs.append((dataset, model, seed, TNP_METHODS[index % 2],
                                       ATTACKS[index % len(ATTACKS)]))
        smoke_attack_specs.extend(
            (dataset, "eegnet", 42, "madry", "autoattack")
            for dataset in DATASETS
        )
        smoke_attack_specs = list(dict.fromkeys(smoke_attack_specs))
    else:
        smoke_attack_specs = [
            (d, m, s, method, attack) for d, m, s in attack_conditions
            for method in RAW_METHODS for attack in ATTACKS
        ]
    for dataset, model, seed, method, attack in smoke_attack_specs:
        train_kind = "train_ea" if method == "ea_forward" else (
            "rpcf_at" if method == "rpcf_at" else "train_standard"
        )
        dep = token(train_kind, dataset, model, f"seed{seed}", method)
        attack_deps = [dep]
        if method in {"madry", "trades", "fbf"}:
            attack_deps.append(
                token("train_ea", dataset, model, f"seed{seed}", "ea_forward")
            )
        dep = attack_deps[-1]
        path = attack_path(run_id, dataset, model, seed, method, attack)
        if method == "ea_forward":
            command = base_python(
                "attack_ea_forward", "--dataset", dataset, "--model", f"{model}_ea_forward",
                "--at_strategy", "madry", "--fold", 0, "--attack", attack, "--eps", 0.03,
                "--batch_size", 2 if smoke else 32, "--seed", seed, "--gpu_id", 0, "--no_ea",
                "--checkpoint_path", checkpoint(run_id, dataset, model, seed, method),
                "--output_path", path, *(("--attack_sample_num", "2") if smoke else ()),
            )
        else:
            command = base_python(
                "rpcf.evaluate_attack", "--dataset", dataset, "--model", model, "--fold", 0,
                "--seed", seed, "--checkpoint_path", checkpoint(run_id, dataset, model, seed, method),
                "--method_tag", method, "--attack", attack, "--eps", 0.03,
                "--batch_size", 2 if smoke else 32, "--gpu_id", 0, "--output_path", path,
                *(("--sample_num", "2") if smoke else ()),
            )
        tasks.append(make_task(run_id, "attack", 4, dataset, model, seed, method, attack,
                               output_path=path, deps=(dep,), command=command))

    if smoke:
        tnp_specs = [
            (dataset, model, seed, TNP_METHODS[index % 2], ATTACKS[index % 4])
            for index, (dataset, model, seed) in enumerate(conditions)
        ]
        tnp_specs.extend(
            (dataset, "eegnet", 42, "madry", "autoattack") for dataset in DATASETS
        )
        tnp_specs = list(dict.fromkeys(tnp_specs))
    else:
        tnp_specs = [
            (d, m, s, method, attack) for d, m, s in conditions
            for method in TNP_METHODS for attack in ATTACKS
        ]
    for dataset, model, seed, method, attack in tnp_specs:
        source_attack = attack_path(run_id, dataset, model, seed, method, attack)
        output = tnp_path(run_id, dataset, model, seed, method, attack)
        dep = token("attack", dataset, model, f"seed{seed}", method, attack)
        shared_args = []
        shared_dep = []
        is_canonical = model == "eegnet" and method == "madry" and attack == "autoattack"
        if not is_canonical:
            canonical = tnp_path(run_id, dataset, "eegnet", seed, "madry", "autoattack")
            shared_args = ["--shared_clean_path", canonical]
            shared_dep = [token("tnp", dataset, "eegnet", f"seed{seed}", "madry", "autoattack")]
        tasks.append(make_task(
            run_id, "tnp", 5, dataset, model, seed, method, attack,
            output_path=output, deps=(dep, *shared_dep),
            command=base_python(
                "rpcf.evaluate_purification", "--attack_path", source_attack,
                "--checkpoint_path", checkpoint(run_id, dataset, model, seed, method),
                "--dataset", dataset, "--model", model, "--fold", 0, "--seed", seed,
                "--eps", 0.03, "--sample_num", 2 if smoke else 512,
                "--batch_size", 2 if smoke else 64, "--ranks", EVAL_RANKS,
                "--configs", EVAL_CONFIGS, "--gpu_id", 0, "--output_path", output,
                *shared_args,
            ),
        ))

    if not smoke:
        for seed in SEEDS:
            for rank in (25, 30):
                output = f"purified_data/exp031/{run_id}/bpda/thubenchmark_eegnet_seed{seed}_rank{rank}.pth"
                tasks.append(make_task(
                    run_id, "bpda", 6, "thubenchmark", "eegnet", seed, "rpcf_at",
                    "bpda_pgd", rank, output,
                    deps=(token("rpcf_at", "thubenchmark", "eegnet", f"seed{seed}", "rpcf_at"),),
                    command=base_python(
                        "rpcf.evaluate_bpda_pgd", "--experiment_id", EXPERIMENT_ID,
                        "--dataset", "thubenchmark", "--model", "eegnet", "--fold", 0,
                        "--seed", seed, "--checkpoint_path",
                        checkpoint(run_id, "thubenchmark", "eegnet", seed, "rpcf_at"),
                        "--rank", rank, "--config", f"PTR3d_8_2048_rank{rank}_3d_interpolate.yaml",
                        "--eps", 0.03, "--pgd_steps", 10, "--pgd_alpha", 0.006,
                        "--sample_num", 512, "--batch_size", 1, "--eval_batch_size", 64,
                        "--gpu_id", 0, "--output_path", output,
                    ),
                ))
    return tasks


def validate_plan(tasks, smoke=False):
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        duplicates = [key for key, count in Counter(ids).items() if count > 1]
        raise ValueError(f"Duplicate task ids: {duplicates[:10]}")
    known = set(ids)
    missing_deps = sorted({dep for task in tasks for dep in task.dependencies if dep not in known})
    if missing_deps:
        raise ValueError(f"Unknown dependencies: {missing_deps[:10]}")
    if not smoke:
        counts = Counter(task.kind for task in tasks)
        for kind, expected in EXPECTED_FULL_COUNTS.items():
            if counts[kind] != expected:
                raise ValueError(f"{kind}: planned {counts[kind]}, expected {expected}.")
    return Counter(task.kind for task in tasks)


def select_tasks(tasks, start_stage=0, stop_stage=7, task_id=None, task_scope="all"):
    """选择本次 runner 负责的任务；scope 只改变调度，不改变完整实验计划。"""
    if task_scope not in TASK_SCOPES:
        raise ValueError(f"Unknown task scope: {task_scope}")
    selected = [task for task in tasks if start_stage <= task.stage <= stop_stage]
    if task_scope == "thu_eegnet_closure":
        selected = [
            task for task in selected
            if task.dataset == "thubenchmark"
            and task.model == "eegnet"
            and task.kind in {"attack", "tnp", "bpda"}
        ]
    if task_id:
        selected = [task for task in selected if task.task_id == task_id]
        if not selected:
            raise ValueError(f"Unknown or filtered task id: {task_id}")
    return selected


def write_plan(tasks, run_dir, smoke=False):
    run_dir.mkdir(parents=True, exist_ok=True)
    counts = validate_plan(tasks, smoke=smoke)
    fields = list(asdict(tasks[0]).keys())
    with (run_dir / "planned_tasks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in tasks:
            row = asdict(task)
            row["dependencies"] = json.dumps(row["dependencies"])
            row["command"] = json.dumps(row["command"])
            writer.writerow(row)
    manifest = {
        "experiment_id": EXPERIMENT_ID, "run_id": run_dir.name, "smoke": smoke,
        "counts": dict(counts), "expected_full_counts": EXPECTED_FULL_COUNTS,
        "actual_batch_manifest": str(run_dir / "actual_batch_sizes.json"),
        "cache_attack_batch_manifest": str(
            run_dir / "actual_cache_attack_batch_sizes.json"
        ),
        "rpcf_batch_manifest": str(run_dir / "actual_rpcf_batch_sizes.json"),
        "protocol": {
            "fold": 0, "eps": 0.03, "baseline_batch": 128,
            "batch_fallback": [64, 32, 16], "baseline_epochs": 400,
            "cache_attack_batch": CACHE_ATTACK_BATCH_FALLBACK[0],
            "cache_attack_batch_fallback": list(CACHE_ATTACK_BATCH_FALLBACK[1:]),
            "rpcf_batch": RPCF_BATCH_FALLBACK[0],
            "rpcf_batch_fallback": list(RPCF_BATCH_FALLBACK[1:]),
            "rpcf_epochs": 100, "rpcf_all_layers": True,
            "static_rank_weights": True, "feature_objective": "none",
            "train_ranks": [15, 20, 25, 30, 35, 40], "eval_ranks": [25, 30],
        },
    }
    batch_path = run_dir / "actual_batch_sizes.json"
    if not batch_path.exists():
        batch_path.write_text(json.dumps(
            {token(dataset, model): 128 for dataset in DATASETS for model in MODELS},
            indent=2, sort_keys=True,
        ) + "\n", encoding="utf-8")
    cache_batch_path = run_dir / "actual_cache_attack_batch_sizes.json"
    if not cache_batch_path.exists():
        cache_batch_path.write_text(json.dumps(
            {
                token(dataset, model): (2 if smoke else CACHE_ATTACK_BATCH_FALLBACK[0])
                for dataset in DATASETS for model in MODELS
            },
            indent=2, sort_keys=True,
        ) + "\n", encoding="utf-8")
    rpcf_batch_path = run_dir / "actual_rpcf_batch_sizes.json"
    if not rpcf_batch_path.exists():
        rpcf_batch_path.write_text(json.dumps(
            {
                token(dataset, model): (2 if smoke else RPCF_BATCH_FALLBACK[0])
                for dataset in DATASETS for model in MODELS
            },
            indent=2, sort_keys=True,
        ) + "\n", encoding="utf-8")
    (run_dir / "run_config.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return counts


def task_status_path(run_dir, task):
    return run_dir / "status" / f"{task.task_id}.json"


def task_complete(run_dir, task):
    status_path = task_status_path(run_dir, task)
    if not status_path.exists() or not Path(task.output_path).exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "completed":
            return False
        if task.kind in {"train_standard", "train_ea", "rpcf_at"}:
            if int(status.get("actual_batch_size", -1)) != actual_batch(run_dir, task):
                return False
        if task.kind == "rpcf_at":
            return status_command_int(
                status, "actual_rpcf_batch_size", "--batch_size"
            ) == actual_rpcf_batch(run_dir, task)
        if task.kind == "rpcf_cache":
            return int(status.get("actual_cache_attack_batch_size", -1)) == (
                actual_cache_attack_batch(run_dir, task)
            )
        return True
    except (OSError, ValueError):
        return False


def actual_batch(run_dir, task):
    path = run_dir / "actual_batch_sizes.json"
    values = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return int(values.get(token(task.dataset, task.model), 128))


def actual_cache_attack_batch(run_dir, task):
    path = run_dir / "actual_cache_attack_batch_sizes.json"
    values = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return int(values.get(
        token(task.dataset, task.model), CACHE_ATTACK_BATCH_FALLBACK[0]
    ))


def actual_rpcf_batch(run_dir, task):
    path = run_dir / "actual_rpcf_batch_sizes.json"
    values = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return int(values.get(token(task.dataset, task.model), RPCF_BATCH_FALLBACK[0]))


def actual_rpcf_eval_batch(run_dir, task):
    return min(128, 2 * actual_rpcf_batch(run_dir, task))


def status_command_int(status, field, option):
    """读取新状态字段；旧 EXP-031 状态从已审计 command 兼容恢复。"""
    if status.get(field) is not None:
        return int(status[field])
    command = status.get("command") or []
    try:
        return int(command[command.index(option) + 1])
    except (ValueError, IndexError, TypeError):
        return -1


def render_command(task, run_dir):
    rendered = []
    for part in task.command:
        if part == "__BATCH_SIZE__":
            part = str(actual_batch(run_dir, task))
        elif part == "__CACHE_ATTACK_BATCH_SIZE__":
            part = str(actual_cache_attack_batch(run_dir, task))
        elif part == "__RPCF_BATCH_SIZE__":
            part = str(actual_rpcf_batch(run_dir, task))
        elif part == "__RPCF_EVAL_BATCH_SIZE__":
            part = str(actual_rpcf_eval_batch(run_dir, task))
        rendered.append(part)
    return rendered


def run_one(task, run_dir, gpu, dry_run=False):
    command = render_command(task, run_dir)
    log_path = run_dir / "tasks" / f"{task.task_id}.log"
    status_path = task_status_path(run_dir, task)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"GPU{gpu}: {shlex.join(command)}")
        return 0
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("NUMBA_CACHE_DIR", f"/tmp/exp031_numba_gpu{gpu}")
    env.setdefault("XDG_CACHE_HOME", f"/tmp/exp031_xdg_gpu{gpu}")
    started = time.time()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"COMMAND={shlex.join(command)}\n")
        handle.flush()
        result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
    status = {
        "task_id": task.task_id, "status": "completed" if result.returncode == 0 else "failed",
        "returncode": result.returncode, "physical_gpu": gpu, "gpu_id": 0,
        "elapsed_seconds": time.time() - started, "output_path": task.output_path,
        "actual_batch_size": actual_batch(run_dir, task),
        "actual_cache_attack_batch_size": actual_cache_attack_batch(run_dir, task),
        "actual_rpcf_batch_size": actual_rpcf_batch(run_dir, task),
        "actual_rpcf_eval_batch_size": actual_rpcf_eval_batch(run_dir, task),
        "command": command,
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return result.returncode


def log_contains_oom(log_path, start_offset=0):
    """只检查本次 attempt 的日志，避免追加日志中的历史 OOM 触发错误降档。"""
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as handle:
            handle.seek(max(int(start_offset), size - 1024 * 1024, 0))
            tail = handle.read().decode("utf-8", errors="ignore").lower()
    except OSError:
        return False
    return "out of memory" in tail or "cuda oom" in tail


def lower_manifest_batch(path, key, candidates, used_batch):
    """每轮最多降一档；同批并发 OOM 只复用已写入的下一档。"""
    values = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current = int(values.get(key, candidates[0]))
    if used_batch not in candidates or used_batch == candidates[-1]:
        return None
    desired = candidates[candidates.index(used_batch) + 1]
    if current in candidates and candidates.index(current) >= candidates.index(desired):
        return current
    values[key] = desired
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return desired


def lower_batch_after_oom(
    run_dir, task, log_path, used_batch=None, log_start_offset=0
):
    """OOM 时锁定下一档统一 batch；旧状态会因 batch 不一致自动失效。"""
    if task.kind not in {"train_standard", "train_ea"} or not log_contains_oom(
        log_path, log_start_offset
    ):
        return None
    path = run_dir / "actual_batch_sizes.json"
    used_batch = actual_batch(run_dir, task) if used_batch is None else used_batch
    return lower_manifest_batch(
        path, token(task.dataset, task.model), (128, 64, 32, 16), used_batch
    )


def lower_cache_attack_batch_after_oom(
    run_dir, task, log_path, used_batch=None, log_start_offset=0
):
    """RPCF cache OOM 时降低 AutoAttack 分块；攻击协议和样本集合保持不变。"""
    if task.kind != "rpcf_cache" or not log_contains_oom(log_path, log_start_offset):
        return None
    path = run_dir / "actual_cache_attack_batch_sizes.json"
    used_batch = (
        actual_cache_attack_batch(run_dir, task) if used_batch is None else used_batch
    )
    return lower_manifest_batch(
        path, token(task.dataset, task.model), CACHE_ATTACK_BATCH_FALLBACK, used_batch
    )


def lower_rpcf_batch_after_oom(
    run_dir, task, log_path, used_batch=None, log_start_offset=0
):
    """RPCF_AT 六-rank fine-tuning OOM 时统一降低 train/eval batch。"""
    if task.kind != "rpcf_at" or not log_contains_oom(log_path, log_start_offset):
        return None
    path = run_dir / "actual_rpcf_batch_sizes.json"
    used_batch = actual_rpcf_batch(run_dir, task) if used_batch is None else used_batch
    return lower_manifest_batch(
        path, token(task.dataset, task.model), RPCF_BATCH_FALLBACK, used_batch
    )

def start_task(task, run_dir, gpu):
    command = render_command(task, run_dir)
    log_path = run_dir / "tasks" / f"{task.task_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    task_status_path(run_dir, task).parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["GPU_ID"] = "0"
    env.setdefault("NUMBA_CACHE_DIR", f"/tmp/exp031_numba_gpu{gpu}")
    env.setdefault("XDG_CACHE_HOME", f"/tmp/exp031_xdg_gpu{gpu}")
    Path(env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    log_start_offset = log_path.stat().st_size if log_path.exists() else 0
    tnp_single_process_at_start = (run_dir / "tnp_single_process.flag").exists()
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"COMMAND={shlex.join(command)}\n")
    handle.flush()
    process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
    print(f"START task={task.task_id} physical_gpu={gpu} pid={process.pid}", flush=True)
    return {
        "task": task, "gpu": gpu, "command": command, "log_path": log_path,
        "handle": handle, "process": process, "started": time.time(),
        "actual_batch_size": actual_batch(run_dir, task),
        "actual_cache_attack_batch_size": actual_cache_attack_batch(run_dir, task),
        "actual_rpcf_batch_size": actual_rpcf_batch(run_dir, task),
        "actual_rpcf_eval_batch_size": actual_rpcf_eval_batch(run_dir, task),
        "log_start_offset": log_start_offset,
        "tnp_single_process_at_start": tnp_single_process_at_start,
    }


def finalize_task(record, run_dir):
    task = record["task"]
    process = record["process"]
    record["handle"].flush()
    record["handle"].close()
    returncode = process.returncode
    output_exists = Path(task.output_path).exists()
    completed = returncode == 0 and output_exists
    next_batch = None if completed else lower_batch_after_oom(
        run_dir, task, record["log_path"], record["actual_batch_size"],
        record["log_start_offset"]
    )
    next_cache_attack_batch = (
        None if completed else lower_cache_attack_batch_after_oom(
            run_dir, task, record["log_path"],
            record["actual_cache_attack_batch_size"],
            record["log_start_offset"],
        )
    )
    next_rpcf_batch = (
        None if completed else lower_rpcf_batch_after_oom(
            run_dir, task, record["log_path"], record["actual_rpcf_batch_size"],
            record["log_start_offset"],
        )
    )
    tnp_single_process = (
        not completed
        and task.kind == "tnp"
        and not record["tnp_single_process_at_start"]
        and log_contains_oom(record["log_path"], record["log_start_offset"])
    )
    if tnp_single_process:
        (run_dir / "tnp_single_process.flag").write_text("1\n", encoding="utf-8")
    status = {
        "task_id": task.task_id,
        "status": "completed" if completed else "failed",
        "returncode": returncode,
        "output_exists": output_exists,
        "physical_gpu": record["gpu"], "gpu_id": 0,
        "elapsed_seconds": time.time() - record["started"],
        "output_path": task.output_path,
        "actual_batch_size": record["actual_batch_size"],
        "actual_cache_attack_batch_size": record["actual_cache_attack_batch_size"],
        "actual_rpcf_batch_size": record["actual_rpcf_batch_size"],
        "actual_rpcf_eval_batch_size": record["actual_rpcf_eval_batch_size"],
        "oom_next_batch_size": next_batch,
        "oom_next_cache_attack_batch_size": next_cache_attack_batch,
        "oom_next_rpcf_batch_size": next_rpcf_batch,
        "oom_tnp_single_process": tnp_single_process,
        "command": record["command"],
    }
    task_status_path(run_dir, task).write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"END task={task.task_id} status={status['status']} returncode={returncode}",
        flush=True,
    )
    return completed, next_batch, next_cache_attack_batch, next_rpcf_batch, tnp_single_process


def parse_reserved_gpu_processes(value):
    """解析 `gpu:pid:start_ticks` 列表，防止 handoff 与旧 worker 抢占同一卡。"""
    reservations = {}
    if not value:
        return reservations
    for item in value.split(","):
        gpu, pid, start_ticks = (int(part) for part in item.split(":"))
        if gpu in reservations:
            raise ValueError(f"Duplicate reserved GPU: {gpu}")
        reservations[gpu] = (pid, start_ticks)
    return reservations


def reservation_active(reservation):
    """PID 与 Linux start time 同时匹配才视为原 worker 仍存活，避免 PID 复用。"""
    pid, expected_start_ticks = reservation
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = stat.rsplit(")", 1)[1].split()
        state = tail[0]
        start_ticks = int(tail[19])
        return state != "Z" and start_ticks == expected_start_ticks
    except (OSError, ValueError, IndexError):
        return False


def gpu_memory_used(gpu):
    """读取物理卡显存；nvidia-smi 不可用时返回 0，不阻塞可复现计划。"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            index, used = [part.strip() for part in line.split(",", 1)]
            if int(index) == gpu:
                return int(used)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0
    return 0


def run_parallel(tasks, selected, run_dir, gpu_ids, reserved_gpu_processes=None):
    """按 DAG 调度：训练/攻击一卡一任务，TNP 每卡最多两个隔离进程。"""
    task_map = {task.task_id: task for task in tasks}
    pending = {task.task_id: task for task in selected if not task_complete(run_dir, task)}
    running = {}
    idle_limit = int(os.environ.get("EXP031_MAX_IDLE_MEMORY_MB", "1024"))
    failure = None
    reserved_gpu_processes = reserved_gpu_processes or {}
    while pending or running:
        for pid, record in list(running.items()):
            if record["process"].poll() is None:
                continue
            completed, next_batch, next_cache_batch, next_rpcf_batch, tnp_single = (
                finalize_task(record, run_dir)
            )
            del running[pid]
            if not completed and failure is None:
                if next_batch:
                    suffix = f"; batch locked to {next_batch}, rerun stage"
                elif next_cache_batch:
                    suffix = (
                        f"; cache attack batch locked to {next_cache_batch}, rerun stage"
                    )
                elif next_rpcf_batch:
                    suffix = f"; RPCF batch locked to {next_rpcf_batch}, rerun stage"
                elif tnp_single:
                    suffix = "; TNP concurrency locked to one process/GPU, rerun task"
                else:
                    suffix = ""
                failure = f"Task failed: {record['task'].task_id}{suffix}"

        if failure is None:
            for task_id, task in list(pending.items()):
                if not all(task_complete(run_dir, task_map[dep]) for dep in task.dependencies):
                    continue
                chosen_gpu = None
                for gpu in gpu_ids:
                    reservation = reserved_gpu_processes.get(gpu)
                    if reservation and reservation_active(reservation):
                        continue
                    on_gpu = [record for record in running.values() if record["gpu"] == gpu]
                    capacity_ok = (
                        len(on_gpu) < (1 if (run_dir / "tnp_single_process.flag").exists() else 2)
                        and all(item["task"].kind == "tnp" for item in on_gpu)
                        if task.kind == "tnp" else not on_gpu
                    )
                    if not capacity_ok:
                        continue
                    if not on_gpu and gpu_memory_used(gpu) > idle_limit:
                        continue
                    chosen_gpu = gpu
                    break
                if chosen_gpu is None:
                    break
                record = start_task(task, run_dir, chosen_gpu)
                running[record["process"].pid] = record
                del pending[task_id]

        if failure and not running:
            raise RuntimeError(failure)
        if pending and not running and failure is None:
            unresolved = {
                task_id: [dep for dep in task.dependencies if not task_complete(run_dir, task_map[dep])]
                for task_id, task in list(pending.items())[:10]
            }
            ready_pending = any(
                all(task_complete(run_dir, task_map[dep]) for dep in task.dependencies)
                for task in pending.values()
            )
            if ready_pending:
                print("WAIT requested physical GPUs are busy or reserved", flush=True)
            elif unresolved:
                raise RuntimeError(f"DAG cannot progress; unresolved dependencies: {unresolved}")
        time.sleep(2)


def prepare_data(dataset, output_path):
    from rpcf.core import DATASET_LOADERS
    from data.subject_ea import prepare_subject_ea_forward_fold, prepare_subject_fold

    data, info = DATASET_LOADERS[dataset]()
    splits = []
    for seed in SEEDS:
        train, val, test, split_path = prepare_subject_fold(
            dataset, data, info, fold_id=0, seed=seed, use_ea=False
        )
        _, _, _, ea_split, matrices, subject_map = prepare_subject_ea_forward_fold(
            dataset, data, info, fold_id=0, seed=seed
        )
        splits.append({
            "seed": seed, "split_path": split_path, "ea_split_path": ea_split,
            "sizes": [len(train), len(val), len(test)],
            "ea_matrix_shape": list(matrices.shape), "subject_count": len(subject_map),
        })
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"dataset": dataset, "splits": splits}, indent=2) + "\n")


def parse_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--smoke", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--gpu-ids", default="0,1,2,3,4,5,6")
    run.add_argument("--start-stage", type=int, default=0)
    run.add_argument("--stop-stage", type=int, default=7)
    run.add_argument("--task-id", default=None)
    run.add_argument(
        "--task-scope", choices=TASK_SCOPES, default="all",
        help="只限制当前调度器负责的任务；planned_tasks.csv 始终保留完整矩阵。",
    )
    run.add_argument(
        "--reserved-gpu-processes", default="",
        help="handoff 保留项：逗号分隔的 gpu:pid:proc_start_ticks。",
    )
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    prep = sub.add_parser("prepare-data")
    prep.add_argument("--dataset", required=True, choices=DATASETS)
    prep.add_argument("--output-path", required=True)
    return parser.parse_args()


def main():
    args = parse_cli()
    if args.action == "prepare-data":
        prepare_data(args.dataset, args.output_path)
        return
    run_dir = Path("logs/exp031") / args.run_id
    tasks = build_tasks(args.run_id, smoke=args.smoke)
    counts = write_plan(tasks, run_dir, smoke=args.smoke)
    print(json.dumps(dict(counts), sort_keys=True))
    if args.action == "plan":
        return
    selected = select_tasks(
        tasks, args.start_stage, args.stop_stage, args.task_id, args.task_scope
    )
    selected_counts = Counter(task.kind for task in selected)
    print(json.dumps({"task_scope": args.task_scope, "selected": dict(selected_counts)},
                     sort_keys=True))
    gpu_ids = [int(value) for value in args.gpu_ids.split(",")]
    if not gpu_ids or any(gpu < 0 or gpu > 6 for gpu in gpu_ids):
        raise ValueError("EXP-031 physical GPU ids must be a non-empty subset of 0..6.")
    reserved_gpu_processes = parse_reserved_gpu_processes(args.reserved_gpu_processes)
    if any(gpu not in gpu_ids for gpu in reserved_gpu_processes):
        raise ValueError("Reserved GPUs must be included in --gpu-ids.")
    if reserved_gpu_processes:
        print(json.dumps({"reserved_gpu_processes": reserved_gpu_processes},
                         sort_keys=True))
    if not args.dry_run:
        run_parallel(tasks, selected, run_dir, gpu_ids, reserved_gpu_processes)
        return
    # dry-run 只打印稳定命令，不检查外部依赖产物，也不启动子进程。
    for index, task in enumerate(selected):
        if task_complete(run_dir, task):
            continue
        missing = [dep for dep in task.dependencies
                   if not task_complete(run_dir, next(item for item in tasks if item.task_id == dep))]
        if missing and not args.dry_run:
            raise RuntimeError(f"{task.task_id} has incomplete dependencies: {missing}")
        code = run_one(task, run_dir, gpu_ids[index % len(gpu_ids)], dry_run=args.dry_run)
        if code:
            raise RuntimeError(f"Task failed ({code}): {task.task_id}")


if __name__ == "__main__":
    main()
