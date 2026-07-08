import argparse
import json
import os

from runtime_env import configure_runtime_env

configure_runtime_env()

import torch

from rpcf.core import write_csv, write_json
from utils.experiment_artifacts import torch_load_cpu


def parse_args():
    parser = argparse.ArgumentParser(
        description="汇总 EXP-024 其他 backbone 的 RPCF_AT 与 baseline 结果。"
    )
    parser.add_argument("--dataset", default="thubenchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--sample_num", type=int, default=512)
    parser.add_argument(
        "--attack_paths",
        nargs="+",
        required=True,
        help="method=path 列表，读取 rpcf.evaluate_attack 产物。",
    )
    parser.add_argument(
        "--purification_paths",
        nargs="*",
        default=[],
        help="method=path 列表，读取 rpcf.evaluate_purification 产物。",
    )
    parser.add_argument("--sensitivity_path", default=None)
    parser.add_argument("--history_path", default=None)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def parse_pairs(items):
    pairs = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected method=path, got {item!r}.")
        method, path = item.split("=", 1)
        method = method.strip()
        path = path.strip()
        if not method or not path:
            raise ValueError(f"Invalid method/path pair: {item!r}.")
        if method in pairs:
            raise ValueError(f"Duplicate method: {method}.")
        pairs[method] = path
    return pairs


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_attack(payload, path, args):
    if not isinstance(payload, dict) or "meta" not in payload:
        raise ValueError(f"Attack artifact missing meta: {path}")
    meta = payload["meta"]
    if meta.get("kind") != "rpcf_attack_eval":
        raise ValueError(f"Unexpected attack kind in {path}: {meta.get('kind')}")
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
    return meta


def validate_purification(payload, path, args):
    if not isinstance(payload, dict) or "meta" not in payload or "metrics" not in payload:
        raise ValueError(f"Purification artifact incomplete: {path}")
    meta = payload["meta"]
    if meta.get("kind") != "rpcf_purification_eval":
        raise ValueError(f"Unexpected purification kind in {path}: {meta.get('kind')}")
    for key, expected in (
        ("dataset", args.dataset),
        ("model", args.model),
        ("seed", args.seed),
        ("fold", args.fold),
        ("sample_num", args.sample_num),
    ):
        if str(meta.get(key)) != str(expected):
            raise ValueError(
                f"{path} metadata mismatch: {key}={meta.get(key)!r}, expected {expected!r}."
            )
    if abs(float(meta.get("eps")) - float(args.eps)) > 1e-12:
        raise ValueError(f"{path} eps={meta.get('eps')} but expected {args.eps}.")
    return meta


def write_markdown(path, full_rows, purification_rows):
    with open(path, "w", encoding="utf-8") as file:
        file.write("| Method | Attack | Clean acc | Adv acc | Attack MSE |\n")
        file.write("| --- | --- | ---: | ---: | ---: |\n")
        for row in full_rows:
            file.write(
                f"| {row['method']} | {row['attack']} | "
                f"{100 * row['clean_accuracy']:.2f}% | "
                f"{100 * row['adv_accuracy']:.2f}% | "
                f"{row['attack_mse']:.6f} |\n"
            )
        if purification_rows:
            file.write("\n| Method | Rank | Purified clean | Purified adv | Clean MSE | Adv MSE |\n")
            file.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
            for row in purification_rows:
                file.write(
                    f"| {row['method']} | {row['rank']} | "
                    f"{100 * row['purified_clean_accuracy']:.2f}% | "
                    f"{100 * row['purified_adv_accuracy']:.2f}% | "
                    f"{row['mean_clean_mse']:.6f} | "
                    f"{row['mean_adv_mse']:.6f} |\n"
                )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    attack_paths = parse_pairs(args.attack_paths)
    purification_paths = parse_pairs(args.purification_paths)

    full_rows = []
    artifacts = {}
    for method, path in attack_paths.items():
        payload = torch_load_cpu(path)
        meta = validate_attack(payload, path, args)
        full_rows.append(
            {
                "method": method,
                "attack": meta["attack"],
                "sample_num": meta["sample_num"],
                "clean_accuracy": meta["clean_accuracy"],
                "adv_accuracy": meta["adv_accuracy"],
                "attack_mse": meta["attack_mse"],
                "path": path,
            }
        )
        artifacts.setdefault(method, {})["attack"] = path

    purification_rows = []
    for method, path in purification_paths.items():
        payload = torch_load_cpu(path)
        meta = validate_purification(payload, path, args)
        artifacts.setdefault(method, {})["purification"] = path
        for metric in payload["metrics"]:
            purification_rows.append(
                {
                    "method": method,
                    "rank": int(metric["rank"]),
                    "sample_num": int(metric.get("sample_num", meta["sample_num"])),
                    "purified_clean_accuracy": metric["purified_clean_accuracy"],
                    "purified_adv_accuracy": metric["purified_adv_accuracy"],
                    "mean_clean_mse": metric["mean_clean_mse"],
                    "mean_adv_mse": metric["mean_adv_mse"],
                    "path": path,
                }
            )

    write_csv(
        os.path.join(args.output_dir, "full_test_attack.csv"),
        ["method", "attack", "sample_num", "clean_accuracy", "adv_accuracy", "attack_mse", "path"],
        full_rows,
    )
    write_csv(
        os.path.join(args.output_dir, "purification.csv"),
        [
            "method",
            "rank",
            "sample_num",
            "purified_clean_accuracy",
            "purified_adv_accuracy",
            "mean_clean_mse",
            "mean_adv_mse",
            "path",
        ],
        purification_rows,
    )
    write_markdown(
        os.path.join(args.output_dir, "comparison.md"), full_rows, purification_rows
    )

    rpcf_info = {}
    if args.sensitivity_path:
        sensitivity = load_json(args.sensitivity_path)
        rpcf_info["selected_layers"] = sensitivity.get("selected_layers")
        rpcf_info["selected_param_ratio"] = sensitivity.get("selected_param_ratio")
    if args.history_path:
        history = load_json(args.history_path)
        rpcf_info["best_epoch"] = history.get("best_epoch")
        rpcf_info["best_validation_metric"] = history.get("best_metric")
        rpcf_info["online_madry_at"] = history.get("online_madry_at")

    write_json(
        os.path.join(args.output_dir, "summary.json"),
        {
            "kind": "exp024_backbone_comparison",
            "experiment_id": "EXP-024",
            "dataset": args.dataset,
            "model": args.model,
            "seed": args.seed,
            "fold": args.fold,
            "eps": args.eps,
            "sample_num": args.sample_num,
            "full_test_attack": full_rows,
            "purification": purification_rows,
            "rpcf_at": rpcf_info,
            "artifacts": artifacts,
        },
    )
    print(os.path.join(args.output_dir, "comparison.md"))


if __name__ == "__main__":
    main()
