import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from utils.experiment_artifacts import torch_load_cpu


def parse_args():
    parser = argparse.ArgumentParser(
        description="汇总 EXP-023 BPDA+PGD-10 adaptive attack 三 seed 结果。"
    )
    parser.add_argument("--bpda_paths", required=True, help="逗号分隔的 EXP-023 artifact。")
    parser.add_argument(
        "--baseline_summaries",
        default="",
        help="可选：逗号分隔的 EXP-021 comparison/summary.json，用于对照非 adaptive 结果。",
    )
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def parse_csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def mean_std(values):
    values = [float(value) for value in values]
    mean = sum(values) / len(values)
    if len(values) <= 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def validate_bpda(payload, path):
    required = {
        "clean",
        "adversarial",
        "labels",
        "source_indices",
        "clean_pur",
        "adv_pur",
        "rank",
        "config",
        "metrics",
        "meta",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"BPDA artifact 结构不完整: {path}")
    meta = payload["meta"]
    if meta.get("kind") != "rpcf_bpda_pgd_eval":
        raise ValueError(f"BPDA artifact kind 非法: {path}")
    if meta.get("attack") != "bpda_pgd" or not meta.get("bpda_identity"):
        raise ValueError(f"BPDA artifact 攻击元数据非法: {path}")
    clean = torch.as_tensor(payload["clean"]).float()
    adversarial = torch.as_tensor(payload["adversarial"]).float()
    clean_pur = torch.as_tensor(payload["clean_pur"]).float()
    adv_pur = torch.as_tensor(payload["adv_pur"]).float()
    labels = torch.as_tensor(payload["labels"]).long().view(-1)
    source_indices = [int(index) for index in payload["source_indices"]]
    if clean.shape != adversarial.shape or clean.shape != clean_pur.shape:
        raise ValueError(f"BPDA artifact clean/adversarial/clean_pur shape 不一致: {path}")
    if clean.shape != adv_pur.shape:
        raise ValueError(f"BPDA artifact adv_pur shape 不一致: {path}")
    if labels.numel() != clean.size(0) or len(source_indices) != clean.size(0):
        raise ValueError(f"BPDA artifact labels/source_indices 长度不一致: {path}")
    normalized = dict(payload)
    normalized.update(
        {
            "clean": clean,
            "adversarial": adversarial,
            "clean_pur": clean_pur,
            "adv_pur": adv_pur,
            "labels": labels,
            "source_indices": source_indices,
            "rank": int(payload["rank"]),
        }
    )
    return normalized


def baseline_rows(paths):
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as file:
            summary = json.load(file)
        for row in summary.get("rows", []):
            if row.get("method") != "rpcf":
                continue
            rows.append(
                {
                    "seed": int(summary["seed"]),
                    "rank": int(row["rank"]),
                    "nonadaptive_purified_adv_accuracy": float(
                        row["purified_adv_accuracy"]
                    ),
                    "nonadaptive_purified_clean_accuracy": float(
                        row["purified_clean_accuracy"]
                    ),
                    "baseline_summary": path,
                }
            )
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percent(value):
    return f"{100 * value:.2f}%"


def write_markdown(path, aggregate_rows, has_baseline):
    lines = [
        "| Rank | BPDA purified clean | BPDA purified adv | Attack MSE |"
        + (" Non-adaptive purified adv | Δ adv |" if has_baseline else ""),
        "| ---: | ---: | ---: | ---: |" + (" ---: | ---: |" if has_baseline else ""),
    ]
    for row in aggregate_rows:
        values = [
            str(row["rank"]),
            f"{percent(row['bpda_purified_clean_accuracy_mean'])}±"
            f"{100 * row['bpda_purified_clean_accuracy_std']:.2f}%",
            f"{percent(row['bpda_purified_adv_accuracy_mean'])}±"
            f"{100 * row['bpda_purified_adv_accuracy_std']:.2f}%",
            f"{row['attack_mse_mean']:.8f}±{row['attack_mse_std']:.8f}",
        ]
        if has_baseline:
            values.extend(
                [
                    f"{percent(row['nonadaptive_purified_adv_accuracy_mean'])}±"
                    f"{100 * row['nonadaptive_purified_adv_accuracy_std']:.2f}%",
                    f"{100 * row['delta_adv_accuracy_mean']:+.2f} pp",
                ]
            )
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    artifacts = [
        validate_bpda(torch_load_cpu(path), path) for path in parse_csv(args.bpda_paths)
    ]
    rows = []
    protocol = None
    for artifact in artifacts:
        meta = artifact["meta"]
        current = {
            "dataset": meta["dataset"],
            "model": meta["model"],
            "fold": int(meta["fold"]),
            "eps": float(meta["eps"]),
            "pgd_steps": int(meta["pgd_steps"]),
            "pgd_alpha": float(meta["pgd_alpha"]),
            "sample_num": int(meta["sample_num"]),
        }
        if protocol is None:
            protocol = current
        elif current != protocol:
            raise ValueError(f"Protocol mismatch: {current} != {protocol}")
        metrics = artifact["metrics"]
        rows.append(
            {
                "seed": int(meta["seed"]),
                "rank": int(artifact["rank"]),
                "sample_num": int(meta["sample_num"]),
                "bpda_purified_clean_accuracy": float(
                    metrics["purified_clean_accuracy"]
                ),
                "bpda_purified_adv_accuracy": float(
                    metrics["bpda_purified_adv_accuracy"]
                ),
                "attack_mse": float(metrics["attack_mse"]),
                "mean_clean_mse": float(metrics["mean_clean_mse"]),
                "mean_adv_mse": float(metrics["mean_adv_mse"]),
            }
        )

    baselines = baseline_rows(parse_csv(args.baseline_summaries))
    baseline_map = {
        (row["seed"], row["rank"]): row for row in baselines
    }
    has_baseline = bool(baseline_map)
    for row in rows:
        baseline = baseline_map.get((row["seed"], row["rank"]))
        if baseline:
            row.update(baseline)
            row["delta_adv_accuracy"] = (
                row["bpda_purified_adv_accuracy"]
                - row["nonadaptive_purified_adv_accuracy"]
            )

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["rank"]].append(row)
    aggregate_rows = []
    for rank in sorted(grouped):
        group = grouped[rank]
        aggregate = {"rank": rank, "seeds": [row["seed"] for row in group]}
        for key in (
            "bpda_purified_clean_accuracy",
            "bpda_purified_adv_accuracy",
            "attack_mse",
            "mean_clean_mse",
            "mean_adv_mse",
        ):
            mean, std = mean_std(row[key] for row in group)
            aggregate[f"{key}_mean"] = mean
            aggregate[f"{key}_std"] = std
        if has_baseline and all("delta_adv_accuracy" in row for row in group):
            for key in (
                "nonadaptive_purified_adv_accuracy",
                "nonadaptive_purified_clean_accuracy",
                "delta_adv_accuracy",
            ):
                mean, std = mean_std(row[key] for row in group)
                aggregate[f"{key}_mean"] = mean
                aggregate[f"{key}_std"] = std
        aggregate_rows.append(aggregate)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "per_seed.csv",
        rows,
        [
            "seed",
            "rank",
            "sample_num",
            "bpda_purified_clean_accuracy",
            "bpda_purified_adv_accuracy",
            "attack_mse",
            "mean_clean_mse",
            "mean_adv_mse",
            "nonadaptive_purified_clean_accuracy",
            "nonadaptive_purified_adv_accuracy",
            "delta_adv_accuracy",
            "baseline_summary",
        ],
    )
    aggregate_fieldnames = sorted(
        {key for row in aggregate_rows for key in row.keys()},
        key=lambda key: (key != "rank", key),
    )
    write_csv(output_dir / "aggregate.csv", aggregate_rows, aggregate_fieldnames)
    write_markdown(output_dir / "comparison.md", aggregate_rows, has_baseline)
    summary = {
        "kind": "exp023_bpda_pgd_comparison",
        **(protocol or {}),
        "rows": rows,
        "aggregate": aggregate_rows,
        "artifacts": {
            "bpda_paths": parse_csv(args.bpda_paths),
            "baseline_summaries": parse_csv(args.baseline_summaries),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output_dir / "comparison.md")
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
