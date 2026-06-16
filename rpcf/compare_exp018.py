import argparse
import json
import os

from runtime_env import configure_runtime_env

configure_runtime_env()

import torch

from rpcf.core import (
    DATASET_LOADERS,
    evaluate_classifier,
    load_model_checkpoint,
    write_csv,
    write_json,
)
from utils.experiment_artifacts import torch_load_cpu
from utils.reproducibility import seed_everything


METHOD_ORDER = ("at_tnp", "consistancy", "rpcf")


def parse_args():
    parser = argparse.ArgumentParser(
        description="严格汇总 EXP-018 三条 seed42 公平重跑流程。"
    )
    parser.add_argument("--dataset", default="thubenchmark", choices=DATASET_LOADERS)
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--ranks", default="25,30")
    parser.add_argument("--sample_num", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--at_checkpoint", required=True)
    parser.add_argument("--consistancy_checkpoint", required=True)
    parser.add_argument("--rpcf_checkpoint", required=True)
    parser.add_argument("--at_purification_path", required=True)
    parser.add_argument("--consistancy_purification_path", required=True)
    parser.add_argument("--rpcf_purification_path", required=True)
    parser.add_argument("--sensitivity_path", required=True)
    parser.add_argument("--history_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment_id", default="EXP-018")
    return parser.parse_args()


def parse_ranks(value):
    ranks = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if ranks != [25, 30]:
        raise ValueError(f"EXP-018 最终评估固定为 ranks 25,30，收到 {ranks}。")
    return ranks


def validate_purification_payload(payload, path, expected, expected_ranks):
    required = {
        "clean",
        "adversarial",
        "clean_pur_by_rank",
        "adv_pur_by_rank",
        "labels",
        "source_indices",
        "ranks",
        "metrics",
        "meta",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"净化产物结构不完整: {path}")
    meta = payload["meta"]
    if meta.get("kind") != "rpcf_purification_eval":
        raise ValueError(f"净化产物 kind 非法: {path}")
    for key, expected_value in expected.items():
        actual = meta.get(key)
        if key == "eps":
            matches = actual is not None and abs(
                float(actual) - float(expected_value)
            ) <= 1e-12
        else:
            matches = str(actual) == str(expected_value)
        if not matches:
            raise ValueError(
                f"{path} 的 {key}={actual}，期望 {expected_value}。"
            )
    ranks = [int(rank) for rank in payload["ranks"]]
    if ranks != expected_ranks:
        raise ValueError(f"{path} ranks={ranks}，期望 {expected_ranks}。")
    attack_meta = meta.get("attack_meta", {})
    if attack_meta.get("attack") != "autoattack":
        raise ValueError(f"{path} 未使用 AutoAttack。")
    if int(attack_meta.get("attack_seed", -1)) != int(expected["seed"]):
        raise ValueError(f"{path} AutoAttack seed 未显式对齐。")
    if str(attack_meta.get("checkpoint_path")) != str(meta.get("checkpoint_path")):
        raise ValueError(f"{path} attack 与 purification checkpoint 不一致。")

    clean = torch.as_tensor(payload["clean"]).float()
    adversarial = torch.as_tensor(payload["adversarial"]).float()
    clean_pur = torch.as_tensor(payload["clean_pur_by_rank"]).float()
    adv_pur = torch.as_tensor(payload["adv_pur_by_rank"]).float()
    labels = torch.as_tensor(payload["labels"]).long().view(-1)
    source_indices = [int(index) for index in payload["source_indices"]]
    sample_num = clean.size(0)
    if clean.shape != adversarial.shape:
        raise ValueError(f"{path} clean/adversarial shape 不一致。")
    expected_rank_shape = (sample_num, len(ranks), *clean.shape[1:])
    if tuple(clean_pur.shape) != expected_rank_shape:
        raise ValueError(f"{path} clean_pur_by_rank shape 不合法。")
    if tuple(adv_pur.shape) != expected_rank_shape:
        raise ValueError(f"{path} adv_pur_by_rank shape 不合法。")
    if labels.numel() != sample_num or len(source_indices) != sample_num:
        raise ValueError(f"{path} labels/source_indices 长度不一致。")
    if len(set(source_indices)) != len(source_indices):
        raise ValueError(f"{path} source_indices 存在重复。")
    normalized = dict(payload)
    normalized.update(
        {
            "clean": clean,
            "adversarial": adversarial,
            "clean_pur_by_rank": clean_pur,
            "adv_pur_by_rank": adv_pur,
            "labels": labels,
            "source_indices": source_indices,
            "ranks": ranks,
        }
    )
    return normalized


def validate_shared_subset(payloads):
    reference_name = METHOD_ORDER[0]
    reference = payloads[reference_name]
    for method in METHOD_ORDER[1:]:
        current = payloads[method]
        if current["source_indices"] != reference["source_indices"]:
            raise ValueError(
                f"{method} 与 {reference_name} 的 source_indices 顺序不一致。"
            )
        if not torch.equal(current["labels"], reference["labels"]):
            raise ValueError(f"{method} 与 {reference_name} 的 labels 不一致。")
        if not torch.allclose(
            current["clean"], reference["clean"], atol=1e-6, rtol=0
        ):
            raise ValueError(f"{method} 与 {reference_name} 的 clean tensors 不一致。")
    return reference["source_indices"], reference["labels"], reference["clean"]


def tensor_metrics(model, data, labels, device, batch_size):
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, labels),
        batch_size=batch_size,
        shuffle=False,
    )
    return evaluate_classifier(model, loader, device)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    args = parse_args()
    ranks = parse_ranks(args.ranks)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    expected = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "fold": args.fold,
        "eps": args.eps,
        "selection_seed": args.seed + args.fold * 1000,
        "sample_num": args.sample_num,
    }
    paths = {
        "at_tnp": args.at_purification_path,
        "consistancy": args.consistancy_purification_path,
        "rpcf": args.rpcf_purification_path,
    }
    payloads = {
        method: validate_purification_payload(
            torch_load_cpu(path), path, expected, ranks
        )
        for method, path in paths.items()
    }
    source_indices, labels, clean = validate_shared_subset(payloads)

    _, info = DATASET_LOADERS[args.dataset]()
    checkpoints = {
        "at_tnp": args.at_checkpoint,
        "consistancy": args.consistancy_checkpoint,
        "rpcf": args.rpcf_checkpoint,
    }
    models = {
        method: load_model_checkpoint(
            args.model, args.dataset, info, checkpoint, device
        )
        for method, checkpoint in checkpoints.items()
    }
    for model in models.values():
        model.eval()

    rows = []
    full_test_rows = []
    for method in METHOD_ORDER:
        payload = payloads[method]
        model = models[method]
        attack_meta = payload["meta"]["attack_meta"]
        full_test_rows.append(
            {
                "method": method,
                "sample_num": attack_meta["sample_num"],
                "clean_accuracy": attack_meta["clean_accuracy"],
                "adv_accuracy": attack_meta["adv_accuracy"],
                "attack_mse": attack_meta["attack_mse"],
            }
        )
        clean_metric = tensor_metrics(model, clean, labels, device, args.batch_size)
        adv_metric = tensor_metrics(
            model, payload["adversarial"], labels, device, args.batch_size
        )
        for rank_position, rank in enumerate(ranks):
            clean_pur = payload["clean_pur_by_rank"][:, rank_position]
            adv_pur = payload["adv_pur_by_rank"][:, rank_position]
            clean_pur_metric = tensor_metrics(
                model, clean_pur, labels, device, args.batch_size
            )
            adv_pur_metric = tensor_metrics(
                model, adv_pur, labels, device, args.batch_size
            )
            rows.append(
                {
                    "method": method,
                    "rank": rank,
                    "sample_num": len(source_indices),
                    "clean_accuracy": clean_metric["accuracy"],
                    "adv_accuracy": adv_metric["accuracy"],
                    "attack_mse": torch.nn.functional.mse_loss(
                        payload["adversarial"], clean
                    ).item(),
                    "purified_clean_accuracy": clean_pur_metric["accuracy"],
                    "purified_adv_accuracy": adv_pur_metric["accuracy"],
                    "mean_clean_mse": torch.nn.functional.mse_loss(
                        clean_pur, clean
                    ).item(),
                    "mean_adv_mse": torch.nn.functional.mse_loss(
                        adv_pur, payload["adversarial"]
                    ).item(),
                }
            )

    write_csv(
        os.path.join(args.output_dir, "summary.csv"),
        [
            "method",
            "rank",
            "sample_num",
            "clean_accuracy",
            "adv_accuracy",
            "attack_mse",
            "purified_clean_accuracy",
            "purified_adv_accuracy",
            "mean_clean_mse",
            "mean_adv_mse",
        ],
        rows,
    )
    write_csv(
        os.path.join(args.output_dir, "full_test_attack.csv"),
        ["method", "sample_num", "clean_accuracy", "adv_accuracy", "attack_mse"],
        full_test_rows,
    )
    sensitivity = load_json(args.sensitivity_path)
    history = load_json(args.history_path)
    if not isinstance(sensitivity, dict) or "selected_layers" not in sensitivity:
        raise ValueError("RPCF sensitivity 产物缺少 selected_layers。")
    if not isinstance(history, dict) or "best_epoch" not in history:
        raise ValueError("RPCF history 产物缺少 best_epoch。")
    write_json(
        os.path.join(args.output_dir, "summary.json"),
        {
            "kind": (
                f"{str(args.experiment_id).strip().lower().replace('-', '')}"
                "_fair_comparison"
            ),
            "experiment_id": str(args.experiment_id),
            "dataset": args.dataset,
            "model": args.model,
            "seed": args.seed,
            "fold": args.fold,
            "eps": args.eps,
            "attack": "autoattack",
            "ranks": ranks,
            "sample_num": len(source_indices),
            "source_indices": source_indices,
            "full_test_attack": full_test_rows,
            "rows": rows,
            "rpcf": {
                "layer_selection_enabled": not bool(
                    history.get("all_layers", False)
                ),
                "effective_trainable_scope": (
                    "all_layers"
                    if history.get("all_layers", False)
                    else "selected_layers"
                ),
                "selected_layers": sensitivity["selected_layers"],
                "selected_param_ratio": sensitivity["selected_param_ratio"],
                "all_layers": bool(history.get("all_layers", False)),
                "trainable_params": history.get("trainable_stats", {}).get(
                    "trainable_params"
                ),
                "total_params": history.get("trainable_stats", {}).get(
                    "total_params"
                ),
                "trainable_param_ratio": history.get("trainable_stats", {}).get(
                    "trainable_ratio"
                ),
                "rank_schedule_enabled": not bool(
                    history.get("static_rank_weights", False)
                ),
                "static_rank_weights": bool(
                    history.get("static_rank_weights", False)
                ),
                "best_epoch": history["best_epoch"],
                "best_validation_metric": history["best_metric"],
            },
            "artifacts": {
                method: {
                    "checkpoint": checkpoints[method],
                    "purification": paths[method],
                }
                for method in METHOD_ORDER
            },
        },
    )
    print(os.path.join(args.output_dir, "summary.csv"))


if __name__ == "__main__":
    main()
