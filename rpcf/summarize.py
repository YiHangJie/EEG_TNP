import argparse
import json
import os

import torch

from rpcf.core import write_csv, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="汇总 RPCF 与 AT baseline 产物。")
    parser.add_argument("--at_attack_path", required=True)
    parser.add_argument("--rpcf_attack_path", required=True)
    parser.add_argument("--at_purification_path", required=True)
    parser.add_argument("--rpcf_purification_path", required=True)
    parser.add_argument("--sensitivity_path", required=True)
    parser.add_argument("--history_path", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    sensitivity = load_json(args.sensitivity_path)
    history = load_json(args.history_path)
    rows = []
    artifacts = {}
    for method, attack_path, purification_path in (
        ("at", args.at_attack_path, args.at_purification_path),
        ("rpcf", args.rpcf_attack_path, args.rpcf_purification_path),
    ):
        attack = torch.load(attack_path, map_location="cpu")
        purification = torch.load(purification_path, map_location="cpu")
        attack_meta = attack["meta"]
        artifacts[method] = {
            "attack_path": attack_path,
            "purification_path": purification_path,
            "attack_meta": attack_meta,
            "purification_meta": purification["meta"],
        }
        rows.append(
            {
                "method": method,
                "rank": "",
                "clean_accuracy": attack_meta["clean_accuracy"],
                "adv_accuracy": attack_meta["adv_accuracy"],
                "attack_mse": attack_meta["attack_mse"],
                "purified_clean_accuracy": "",
                "purified_adv_accuracy": "",
                "mean_clean_mse": "",
                "mean_adv_mse": "",
            }
        )
        for metric in purification["metrics"]:
            rows.append(
                {
                    "method": method,
                    "rank": metric["rank"],
                    "clean_accuracy": attack_meta["clean_accuracy"],
                    "adv_accuracy": attack_meta["adv_accuracy"],
                    "attack_mse": attack_meta["attack_mse"],
                    "purified_clean_accuracy": metric["purified_clean_accuracy"],
                    "purified_adv_accuracy": metric["purified_adv_accuracy"],
                    "mean_clean_mse": metric["mean_clean_mse"],
                    "mean_adv_mse": metric["mean_adv_mse"],
                }
            )
    write_csv(
        os.path.join(args.output_dir, "summary.csv"),
        [
            "method",
            "rank",
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
    layer_rows = []
    selected = set(sensitivity["selected_layers"])
    for layer_name, detail in sensitivity["layers"].items():
        layer_rows.append(
            {
                "layer": layer_name,
                "selected": layer_name in selected,
                "pur_mean": detail["pur_mean"],
                "advpur_mean": detail["advpur_mean"],
                "score": detail["score"],
            }
        )
    write_csv(
        os.path.join(args.output_dir, "layers.csv"),
        ["layer", "selected", "pur_mean", "advpur_mean", "score"],
        layer_rows,
    )
    summary = {
        "kind": "rpcf_summary",
        "artifacts": artifacts,
        "selected_layers": sensitivity["selected_layers"],
        "selected_param_ratio": sensitivity["selected_param_ratio"],
        "best_epoch": history["best_epoch"],
        "best_validation_metric": history["best_metric"],
        "rows": rows,
    }
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    print(os.path.join(args.output_dir, "summary.csv"))


if __name__ == "__main__":
    main()
