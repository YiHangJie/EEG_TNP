import argparse
import json
import os

from runtime_env import configure_runtime_env

configure_runtime_env()

from rpcf.core import write_csv, write_json
from utils.experiment_artifacts import torch_load_cpu


def parse_args():
    parser = argparse.ArgumentParser(
        description="汇总 EXP-025 RPCF_AT 敏感层前缀预算曲线。"
    )
    parser.add_argument("--dataset", default="thubenchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--sample_num", type=int, default=512)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_meta(meta, path, args, expected_kind):
    if meta.get("kind") != expected_kind:
        raise ValueError(f"Unexpected kind in {path}: {meta.get('kind')}")
    for key, expected in (
        ("dataset", args.dataset),
        ("model", args.model),
        ("seed", args.seed),
        ("fold", args.fold),
    ):
        if str(meta.get(key)) != str(expected):
            raise ValueError(
                f"{path} metadata mismatch: {key}={meta.get(key)!r}, expected {expected!r}."
            )
    if abs(float(meta.get("eps")) - float(args.eps)) > 1e-12:
        raise ValueError(f"{path} eps={meta.get('eps')} but expected {args.eps}.")


def budget_sort_key(name):
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return name


def read_budget(args, budget_dir):
    config_path = os.path.join(budget_dir, "budget_config.json")
    history_path = os.path.join(budget_dir, "finetune_rpcf_at.json")
    attack_path = os.path.join(budget_dir, "rpcf_at_autoattack.pth")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing budget config: {config_path}")
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Missing history: {history_path}")
    if not os.path.exists(attack_path):
        raise FileNotFoundError(f"Missing attack artifact: {attack_path}")

    config = load_json(config_path)
    history = load_json(history_path)
    attack = torch_load_cpu(attack_path)
    attack_meta = attack.get("meta") or {}
    validate_meta(attack_meta, attack_path, args, "rpcf_attack_eval")

    rows = []
    base = {
        "model": args.model,
        "budget_id": config["budget_id"],
        "prefix_k": int(config["prefix_k"]),
        "layer_count": len(config["selected_layers"]),
        "selected_layers": ",".join(config["selected_layers"]),
        "trainable_ratio": history.get("trainable_stats", {}).get("trainable_ratio"),
        "best_epoch": history.get("best_epoch"),
        "val_clean_acc": (history.get("best_metric") or {}).get("clean_acc"),
        "val_robust_acc": (history.get("best_metric") or {}).get("robust_acc"),
        "attack_sample_num": attack_meta.get("sample_num"),
        "clean_accuracy": attack_meta.get("clean_accuracy"),
        "adv_accuracy": attack_meta.get("adv_accuracy"),
        "attack_mse": attack_meta.get("attack_mse"),
        "checkpoint_path": config.get("checkpoint_path"),
        "history_path": history_path,
        "attack_path": attack_path,
    }

    rank_paths = sorted(
        os.path.join(budget_dir, name)
        for name in os.listdir(budget_dir)
        if name.startswith("rpcf_at_rank") and name.endswith(".pth")
    )
    if not rank_paths:
        rows.append({**base, "rank": None})
        return rows

    for rank_path in rank_paths:
        payload = torch_load_cpu(rank_path)
        meta = payload.get("meta") or {}
        validate_meta(meta, rank_path, args, "rpcf_purification_eval")
        if str(meta.get("sample_num")) != str(args.sample_num):
            raise ValueError(
                f"{rank_path} sample_num={meta.get('sample_num')} but expected {args.sample_num}."
            )
        for metric in payload.get("metrics", []):
            rows.append(
                {
                    **base,
                    "rank": int(metric["rank"]),
                    "purified_clean_accuracy": metric["purified_clean_accuracy"],
                    "purified_adv_accuracy": metric["purified_adv_accuracy"],
                    "mean_clean_mse": metric["mean_clean_mse"],
                    "mean_adv_mse": metric["mean_adv_mse"],
                    "purification_path": rank_path,
                }
            )
    return rows


def format_percent(value):
    if value is None:
        return ""
    return f"{100 * float(value):.2f}%"


def write_markdown(path, rows):
    with open(path, "w", encoding="utf-8") as file:
        file.write("| Budget | k | Trainable | Rank | Clean | Adv | Pur clean | Pur adv | Layers |\n")
        file.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            rank = row.get("rank")
            file.write(
                f"| {row['budget_id']} | {row['prefix_k']} | "
                f"{format_percent(row.get('trainable_ratio'))} | "
                f"{rank if rank is not None else ''} | "
                f"{format_percent(row.get('clean_accuracy'))} | "
                f"{format_percent(row.get('adv_accuracy'))} | "
                f"{format_percent(row.get('purified_clean_accuracy'))} | "
                f"{format_percent(row.get('purified_adv_accuracy'))} | "
                f"{row['selected_layers']} |\n"
            )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    budget_dirs = [
        os.path.join(args.run_root, name)
        for name in os.listdir(args.run_root)
        if name.startswith("budget_") and os.path.isdir(os.path.join(args.run_root, name))
    ]
    budget_dirs.sort(key=lambda path: budget_sort_key(os.path.basename(path)))
    if not budget_dirs:
        raise FileNotFoundError(f"No budget_* directories found under {args.run_root}")

    rows = []
    for budget_dir in budget_dirs:
        rows.extend(read_budget(args, budget_dir))

    fieldnames = [
        "model",
        "budget_id",
        "prefix_k",
        "layer_count",
        "selected_layers",
        "trainable_ratio",
        "best_epoch",
        "val_clean_acc",
        "val_robust_acc",
        "attack_sample_num",
        "clean_accuracy",
        "adv_accuracy",
        "attack_mse",
        "rank",
        "purified_clean_accuracy",
        "purified_adv_accuracy",
        "mean_clean_mse",
        "mean_adv_mse",
        "checkpoint_path",
        "history_path",
        "attack_path",
        "purification_path",
    ]
    write_csv(os.path.join(args.output_dir, "budget_curve.csv"), fieldnames, rows)
    write_markdown(os.path.join(args.output_dir, "comparison.md"), rows)
    write_json(
        os.path.join(args.output_dir, "summary.json"),
        {
            "kind": "exp025_layer_prefix_comparison",
            "experiment_id": "EXP-025",
            "dataset": args.dataset,
            "model": args.model,
            "seed": args.seed,
            "fold": args.fold,
            "eps": args.eps,
            "sample_num": args.sample_num,
            "run_root": args.run_root,
            "rows": rows,
        },
    )
    print(os.path.join(args.output_dir, "comparison.md"))


if __name__ == "__main__":
    main()
