"""EXP-033 的固定协议、产物来源与任务级断点工具。"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

DATASET = "thubenchmark"
MODEL = "eegnet"
SEEDS = (42, 43, 44, 45, 46)
SOURCE_RUN = "exp031_full_20260729_174215"
EXTERNAL_RUN = "exp032_full_20260917_2020"
GROUPS = ("structure", "rank", "loss", "ablation", "visualize")
METHODS = ("tr_dense", "tt_dense", "tt_time", "tucker", "svd")
WEIGHTS = {"clean_ce_weight": 1., "pur_ce_weight": .5,
           "adv_pur_ce_weight": 1., "lambda_pur": .2, "lambda_adv_pur": .5}
ATTACK_PROTOCOL = {"norm": "Linf", "eps": .03, "steps": 200,
                   "alpha": 2 / 255, "random_start": False}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """只在当前 run 中原子写入；不修改引用的旧实验。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def loss_variants():
    result = [{"variant": "default", "weights": dict(WEIGHTS),
               "scan_parameter": None, "scan_value": None}]
    for name, default in WEIGHTS.items():
        for multiplier in (0., .5, 2.):
            value = default * multiplier
            result.append({"variant": f"{name}_x{str(multiplier).replace('.', 'p')}",
                           "weights": {**WEIGHTS, name: value},
                           "scan_parameter": name, "scan_value": value})
    return result


def source_paths(manifest, seed):
    from rpcf.exp031 import checkpoint, attack_path, tnp_path, cache_path
    src, ext = manifest["source_run"], manifest["external_run"]
    paths = {}
    for method in ("madry", "rpcf_at"):
        paths[f"checkpoint_{method}"] = checkpoint(src, DATASET, MODEL, seed, method)
        paths[f"attack_{method}"] = attack_path(src, DATASET, MODEL, seed, method, "pgd")
        paths[f"tnp_{method}"] = tnp_path(src, DATASET, MODEL, seed, method, "pgd")
    paths["checkpoint_clean"] = checkpoint(ext, DATASET, MODEL, seed, "clean").replace("_clean_eps0.03_", "_clean_eps0_")
    paths["attack_clean"] = f"ad_data/exp032/{ext}/{DATASET}_{MODEL}_seed{seed}_clean_pgd.pth"
    paths["canonical_clean_tnp"] = tnp_path(src, DATASET, MODEL, seed, "madry", "autoattack")
    paths["cache"] = cache_path(src, DATASET, MODEL, seed)
    paths["training_status"] = f"logs/exp031/{src}/status/rpcf_at_{DATASET}_{MODEL}_seed{seed}_rpcf_at.json"
    paths["training_history"] = f"logs/exp031/{src}/history/{DATASET}_{MODEL}_seed{seed}_rpcf_at.json"
    paths["audit"] = f"logs/exp032/{ext}/audit/{DATASET}_{MODEL}_seed{seed}.json"
    for method in ("magnet", "dcae"):
        paths[f"purifier_{method}"] = f"checkpoints/exp032/{ext}/{DATASET}_seed{seed}_{method}.pth"
    return paths


def context(manifest, seed):
    """始终先取得规范 n512，smoke 只取其前缀，避免二次抽样。"""
    from rpcf.exp032_common import load_condition
    args = SimpleNamespace(dataset=DATASET, model=MODEL, seed=seed, fold=0,
                           sample_num=512, run_id=manifest["external_run"],
                           source_run=manifest["source_run"])
    data, info, train, val, clean, labels, indices, split = load_condition(args)
    n = manifest["sample_num"]
    return SimpleNamespace(args=args, data=data, info=info, train=train, val=val,
                           clean=clean[:n], labels=labels[:n], indices=indices[:n], split=split)


def load_attack(path, ctx, expected_checkpoint):
    from rpcf.exp032_common import load_payload, align_payload, norm_audit
    import torch
    payload = load_payload(path)
    meta = payload["meta"]
    for key, expected in {"dataset": DATASET, "model": MODEL, "seed": ctx.args.seed,
                          "fold": 0, "attack": "pgd"}.items():
        if meta.get(key) != expected:
            raise ValueError(f"Attack provenance {key}: {path}")
    if Path(meta["checkpoint_path"]).resolve() != Path(expected_checkpoint).resolve():
        raise ValueError(f"Attack checkpoint mismatch: {path}")
    if meta.get("checkpoint_sha256") and meta["checkpoint_sha256"] != file_hash(expected_checkpoint):
        raise ValueError("Attack checkpoint contents changed")
    for key, expected in ATTACK_PROTOCOL.items():
        if meta.get("attack_protocol", {}).get(key) != expected:
            raise ValueError(f"Attack protocol {key} mismatch: {path}")
    aligned = align_payload(payload, ctx.indices, ctx.labels, ctx.clean)
    if not torch.isfinite(aligned["adversarial"]).all() or float((aligned["adversarial"]-ctx.clean).abs().max()) > .030001:
        raise ValueError("Non-finite or out-of-budget attack")
    norm = norm_audit(ctx.clean, aligned["adversarial"], "Linf", .03)
    if norm.get("budget_violations", norm.get("violations", 0)):
        raise ValueError("Attack exceeds Linf budget")
    return aligned["adversarial"], norm


def load_model(path, ctx, device):
    from rpcf.core import load_model_checkpoint
    return load_model_checkpoint(MODEL, DATASET, ctx.info, path, device).eval()


def row(task, ctx, cp, ap, **extra):
    from rpcf.exp032_common import metric_row
    result = metric_row(ctx.args, extra.pop("method", task.get("method", "caf")), "pgd",
                        cp, ap, ctx.labels, **extra)
    result.update(group=task["group"], variant=task.get("variant", "default"))
    if "scan_parameter" in task:
        result.update(scan_parameter=task["scan_parameter"], scan_value=task["scan_value"])
    return result


def config_for_rank(manifest, rank):
    """smoke 使用独立短配置，原配置及正式 2048 步保持不变。"""
    path = Path(f"configs/{DATASET}/PTR3d_8_2048_rank{rank}_3d_interpolate.yaml")
    if not manifest["smoke"]:
        return str(path)
    import yaml
    dest = Path(manifest["run_dir"]) / "smoke_configs" / path.name
    values = yaml.safe_load(path.read_text())
    values.update(num_iterations=40, iterations_for_upsampling=[10, 20, 30], warmup_steps=1)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(yaml.safe_dump(values))
    return str(dest.resolve())


def set_option(command, flag, value):
    command = list(command)
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])
    return command


def training_arguments(manifest, task, attempt_dir):
    """复制实际完成命令，保留已验证 batch 和全部随机/优化设置。"""
    status = read_json(source_paths(manifest, task["seed"])["training_status"])
    original = status["command"]
    command = original[original.index("rpcf.finetune") + 1:]
    if any("__" in value for value in command):
        raise ValueError("Unresolved training command")
    for flag in ("--online_madry_at", "--all_layers", "--static_rank_weights"):
        if flag not in command:
            raise ValueError(f"Source CAF missing {flag}")
    command = set_option(command, "--output_checkpoint", Path(attempt_dir) / "checkpoint.pth")
    command = set_option(command, "--history_prefix", Path(attempt_dir) / "history")
    for name, value in task["weights"].items():
        command = set_option(command, "--" + name, value)
    if manifest["smoke"]:
        for flag, value in (("--epochs", 1), ("--batch_size", 2), ("--eval_batch_size", 16),
                            ("--online_at_batch_size", 2), ("--online_train_sample_num", 2),
                            ("--max_cache_batches", 1)):
            command = set_option(command, flag, value)
    return command
