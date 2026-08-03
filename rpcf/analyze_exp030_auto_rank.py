"""EXP-030 自动定秩结果与 fixed rank25 / EXP-027 oracle 的严格比较。"""

import argparse
import csv
import json
import os
from collections import Counter

from runtime_env import configure_runtime_env

configure_runtime_env()

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/eegap_matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rpcf.analyze_exp027_oracle_rank import (
    load_method_payloads,
    paired_bootstrap_difference,
)
from rpcf.core import DATASET_LOADERS, load_model_checkpoint


PROTOCOL_KEYS = ("dataset", "model", "fold", "seed", "eps", "sample_num")


def parse_args():
    parser = argparse.ArgumentParser(
        description="EXP-030：比较自动定秩、fixed rank25 与 oracle。"
    )
    parser.add_argument(
        "--method",
        action="append",
        nargs=3,
        required=True,
        metavar=("METHOD", "AUTO_PAYLOAD", "FIXED25_PAYLOAD"),
    )
    parser.add_argument("--oracle_summary", required=True)
    parser.add_argument("--oracle_rows", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _matches(key, actual, expected):
    if key == "eps":
        return (
            actual is not None
            and expected is not None
            and abs(float(actual) - float(expected)) <= 1e-12
        )
    return str(actual) == str(expected)


def load_auto_payload(path):
    payload = torch_load_cpu(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: auto-rank payload must be a dict.")
    required = {
        "clean",
        "adversarial",
        "clean_pur",
        "adv_pur",
        "labels",
        "source_indices",
        "clean_mses",
        "adv_mses",
        "clean_rank_diagnostics",
        "adv_rank_diagnostics",
        "meta",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{path}: missing keys {missing}.")
    meta = payload["meta"]
    if meta.get("kind") != "exp030_auto_rank_eval":
        raise ValueError(f"{path}: unexpected meta.kind={meta.get('kind')}.")
    if meta.get("rank_inference_uses_labels") is not False:
        raise ValueError(f"{path}: rank inference must not use labels.")
    if meta.get("rank_inference_uses_classifier_logits") is not False:
        raise ValueError(f"{path}: rank inference must not use classifier logits.")

    out = dict(payload)
    for key in ("clean", "adversarial", "clean_pur", "adv_pur"):
        out[key] = torch.as_tensor(payload[key]).detach().cpu().float()
    out["labels"] = torch.as_tensor(payload["labels"]).detach().cpu().long().view(-1)
    out["source_indices"] = [int(value) for value in payload["source_indices"]]
    count = out["labels"].numel()
    expected_shape = out["clean"].shape
    for key in ("adversarial", "clean_pur", "adv_pur"):
        if out[key].shape != expected_shape:
            raise ValueError(f"{path}: {key} shape mismatch.")
    for key in (
        "source_indices",
        "clean_mses",
        "adv_mses",
        "clean_rank_diagnostics",
        "adv_rank_diagnostics",
    ):
        if len(out[key]) != count:
            raise ValueError(f"{path}: {key} length mismatch.")
    return out


def align_auto_to_fixed(auto, fixed, path):
    auto_sources = [int(value) for value in auto["source_indices"]]
    fixed_sources = [int(value) for value in fixed["source_indices"]]
    if not set(auto_sources).issubset(set(fixed_sources)):
        raise ValueError(f"{path}: source_indices are not covered by fixed payload.")
    position = {source: index for index, source in enumerate(fixed_sources)}
    order = [position[source] for source in auto_sources]
    index = torch.as_tensor(order, dtype=torch.long)
    for key in (
        "clean",
        "adversarial",
        "clean_pur_by_rank",
        "adv_pur_by_rank",
        "labels",
    ):
        fixed[key] = fixed[key].index_select(0, index)
    fixed["source_indices"] = auto_sources
    if not torch.equal(auto["labels"], fixed["labels"]):
        raise ValueError(f"{path}: labels mismatch after source alignment.")
    if not torch.equal(auto["clean"], fixed["clean"]):
        raise ValueError(f"{path}: clean tensor mismatch with fixed payload.")
    if not torch.equal(auto["adversarial"], fixed["adversarial"]):
        raise ValueError(f"{path}: adversarial tensor mismatch with fixed payload.")
    for key in PROTOCOL_KEYS:
        if key == "sample_num":
            continue
        if not _matches(key, auto["meta"].get(key), fixed["meta"].get(key)):
            raise ValueError(f"{path}: meta.{key} mismatch with fixed payload.")
    for key in ("checkpoint_path", "attack_path"):
        if str(auto["meta"].get(key)) != str(fixed["meta"].get(key)):
            raise ValueError(f"{path}: meta.{key} mismatch with fixed payload.")
    return auto, fixed


def predictions(model, data, labels, batch_size, device):
    predictions_out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, data.size(0), batch_size):
            logits = model(data[start : start + batch_size].to(device))
            predictions_out.append(logits.argmax(dim=1).detach().cpu())
    prediction = torch.cat(predictions_out)
    return prediction, prediction.eq(labels)


def _read_oracle_rows(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["method"], int(row["source_index"]))
            rows[key] = row
    return rows


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _diagnostic_rows(method, source_indices, input_kind, diagnostics):
    sample_rows = []
    trajectory_rows = []
    distribution = Counter()
    for sample_id, (source, diag) in enumerate(zip(source_indices, diagnostics)):
        ranks = [int(value) for value in diag["final_rank_vector"]]
        distribution.update(ranks)
        sample_rows.append(
            {
                "method": method,
                "sample_id": sample_id,
                "source_index": source,
                "input_kind": input_kind,
                "rank_vector": json.dumps(ranks),
                "mean_rank": float(np.mean(ranks)),
                "min_rank": min(ranks),
                "max_rank": max(ranks),
                "effective_parameters": int(diag["effective_parameters"]),
                "purification_time_sec": float(diag["purification_time_sec"]),
            }
        )
        for stage in diag["stage_trajectory"]:
            trajectory_rows.append(
                {
                    "method": method,
                    "sample_id": sample_id,
                    "source_index": source,
                    "input_kind": input_kind,
                    "stage_index": int(stage["stage_index"]),
                    "resolution": int(stage["resolution"]),
                    "rank_vector_before": json.dumps(stage["rank_vector_before"]),
                    "rank_vector_after": json.dumps(stage["rank_vector_after"]),
                    "pruned_count": int(stage["pruned_count"]),
                    "regrown_count": int(stage["regrown_count"]),
                    "mse": float(stage["mse"]),
                    "normalized_residual": float(stage["normalized_residual"]),
                    "heldout_mse": stage.get("heldout_mse"),
                }
            )
    return sample_rows, trajectory_rows, distribution


def analyze_method(
    method,
    auto_path,
    fixed_path,
    oracle_method,
    oracle_rows,
    args,
    device,
):
    fixed = load_method_payloads(method, [fixed_path], expected_ranks=[25])
    auto, fixed = align_auto_to_fixed(
        load_auto_payload(auto_path), fixed, auto_path
    )
    dataset, info = DATASET_LOADERS[auto["meta"]["dataset"]]()
    del dataset
    model = load_model_checkpoint(
        auto["meta"]["model"],
        auto["meta"]["dataset"],
        info,
        auto["meta"]["checkpoint_path"],
        device,
    )
    _, fixed_clean_correct = predictions(
        model,
        fixed["clean_pur_by_rank"][:, 0],
        fixed["labels"],
        args.batch_size,
        device,
    )
    _, fixed_adv_correct = predictions(
        model,
        fixed["adv_pur_by_rank"][:, 0],
        fixed["labels"],
        args.batch_size,
        device,
    )
    auto_clean_prediction, auto_clean_correct = predictions(
        model, auto["clean_pur"], auto["labels"], args.batch_size, device
    )
    auto_adv_prediction, auto_adv_correct = predictions(
        model, auto["adv_pur"], auto["labels"], args.batch_size, device
    )
    del model

    fixed_clean = float(fixed_clean_correct.float().mean())
    fixed_adv = float(fixed_adv_correct.float().mean())
    auto_clean = float(auto_clean_correct.float().mean())
    auto_adv = float(auto_adv_correct.float().mean())
    oracle_clean = float(oracle_method["oracle_clean_accuracy"])
    oracle_adv = float(oracle_method["oracle_adv_accuracy"])
    denominator = oracle_adv - fixed_adv
    recovery = (auto_adv - fixed_adv) / denominator if denominator > 0 else None
    bootstrap = paired_bootstrap_difference(
        auto_adv_correct.numpy().astype(np.int64),
        fixed_adv_correct.numpy().astype(np.int64),
        args.bootstrap_samples,
        args.seed,
    )
    clean_delta_pp = (auto_clean - fixed_clean) * 100.0
    if (
        recovery is not None
        and recovery >= 0.5
        and bootstrap["ci_low"] > 0
        and clean_delta_pp >= -1.0
    ):
        decision = "strong_support"
    elif recovery is not None and recovery >= 0.25:
        decision = "weak_signal"
    else:
        decision = "not_supported"

    sample_rows = []
    for sample_id, source in enumerate(auto["source_indices"]):
        oracle_row = oracle_rows[(method, int(source))]
        sample_rows.append(
            {
                "method": method,
                "sample_id": sample_id,
                "source_index": source,
                "label": int(auto["labels"][sample_id]),
                "fixed_clean_correct": int(fixed_clean_correct[sample_id]),
                "fixed_adv_correct": int(fixed_adv_correct[sample_id]),
                "auto_clean_prediction": int(auto_clean_prediction[sample_id]),
                "auto_adv_prediction": int(auto_adv_prediction[sample_id]),
                "auto_clean_correct": int(auto_clean_correct[sample_id]),
                "auto_adv_correct": int(auto_adv_correct[sample_id]),
                "oracle_clean_correct": int(oracle_row["oracle_clean_correct"]),
                "oracle_adv_correct": int(oracle_row["oracle_adv_correct"]),
                "auto_adv_rescued": int(
                    auto_adv_correct[sample_id] and not fixed_adv_correct[sample_id]
                ),
                "auto_adv_lost": int(
                    fixed_adv_correct[sample_id] and not auto_adv_correct[sample_id]
                ),
            }
        )

    rank_rows = []
    trajectory_rows = []
    distribution_rows = []
    for input_kind, diagnostics in (
        ("clean", auto["clean_rank_diagnostics"]),
        ("adversarial", auto["adv_rank_diagnostics"]),
    ):
        current_rank_rows, current_trajectory, distribution = _diagnostic_rows(
            method, auto["source_indices"], input_kind, diagnostics
        )
        rank_rows.extend(current_rank_rows)
        trajectory_rows.extend(current_trajectory)
        total = sum(distribution.values())
        for rank in sorted(distribution):
            distribution_rows.append(
                {
                    "method": method,
                    "input_kind": input_kind,
                    "rank": rank,
                    "count": distribution[rank],
                    "fraction": distribution[rank] / total,
                }
            )

    metric = {
        "method": method,
        "variant": auto["meta"]["variant"],
        "sample_num": len(auto["source_indices"]),
        "fixed_rank": 25,
        "fixed_clean_accuracy": fixed_clean,
        "fixed_adv_accuracy": fixed_adv,
        "auto_clean_accuracy": auto_clean,
        "auto_adv_accuracy": auto_adv,
        "oracle_clean_accuracy": oracle_clean,
        "oracle_adv_accuracy": oracle_adv,
        "robust_gain_pp": (auto_adv - fixed_adv) * 100.0,
        "robust_gain_ci_low_pp": bootstrap["ci_low"] * 100.0,
        "robust_gain_ci_high_pp": bootstrap["ci_high"] * 100.0,
        "clean_delta_pp": clean_delta_pp,
        "headroom_recovery": recovery,
        "oracle_regret_pp": (oracle_adv - auto_adv) * 100.0,
        "mean_clean_mse": float(np.mean(auto["clean_mses"])),
        "mean_adv_mse": float(np.mean(auto["adv_mses"])),
        "mean_clean_rank": float(
            np.mean([row["mean_rank"] for row in auto["clean_rank_diagnostics"]])
        ),
        "mean_adv_rank": float(
            np.mean([row["mean_rank"] for row in auto["adv_rank_diagnostics"]])
        ),
        "adv_rescued_count": int(
            (auto_adv_correct & ~fixed_adv_correct).sum().item()
        ),
        "adv_lost_count": int(
            (fixed_adv_correct & ~auto_adv_correct).sum().item()
        ),
        "decision": decision,
    }
    return {
        "metric": metric,
        "bootstrap": bootstrap,
        "sample_rows": sample_rows,
        "rank_rows": rank_rows,
        "trajectory_rows": trajectory_rows,
        "distribution_rows": distribution_rows,
        "meta": auto["meta"],
    }


def _plots(output_dir, metrics, distribution_rows):
    methods = [row["method"] for row in metrics]
    x = np.arange(len(methods))
    width = 0.24
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(x - width, [row["fixed_adv_accuracy"] for row in metrics], width, label="fixed r25")
    axis.bar(x, [row["auto_adv_accuracy"] for row in metrics], width, label="auto rank")
    axis.bar(x + width, [row["oracle_adv_accuracy"] for row in metrics], width, label="oracle")
    axis.set_xticks(x, methods)
    axis.set_ylabel("Purified adversarial accuracy")
    axis.set_ylim(0.75, 0.95)
    axis.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fixed_auto_oracle.png"), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4), squeeze=False)
    for axis, method in zip(axes[0], methods):
        for input_kind, marker in (("clean", "o"), ("adversarial", "s")):
            rows = [
                row for row in distribution_rows
                if row["method"] == method and row["input_kind"] == input_kind
            ]
            axis.plot(
                [row["rank"] for row in rows],
                [row["fraction"] for row in rows],
                marker=marker,
                label=input_kind,
            )
        axis.set_title(method)
        axis.set_xlabel("Bond rank")
        axis.set_ylabel("Fraction")
        axis.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "rank_distribution.png"), dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    summary_path = os.path.join(args.output_dir, "summary.json")
    comparison_path = os.path.join(args.output_dir, "comparison.md")
    if (
        os.path.exists(summary_path)
        and os.path.exists(comparison_path)
        and not args.overwrite
    ):
        print(summary_path)
        return
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    with open(args.oracle_summary, encoding="utf-8") as handle:
        oracle_summary = json.load(handle)
    oracle_by_method = {
        row["method"]: row for row in oracle_summary["methods"]
    }
    oracle_rows = _read_oracle_rows(args.oracle_rows)

    results = []
    for method, auto_path, fixed_path in args.method:
        if method not in oracle_by_method:
            raise ValueError(f"Oracle summary does not contain method={method}.")
        results.append(
            analyze_method(
                method,
                auto_path,
                fixed_path,
                oracle_by_method[method],
                oracle_rows,
                args,
                device,
            )
        )
    metrics = [result["metric"] for result in results]
    sample_rows = [row for result in results for row in result["sample_rows"]]
    rank_rows = [row for result in results for row in result["rank_rows"]]
    trajectory_rows = [
        row for result in results for row in result["trajectory_rows"]
    ]
    distribution_rows = [
        row for result in results for row in result["distribution_rows"]
    ]

    _write_csv(
        os.path.join(args.output_dir, "auto_rank_metrics.csv"),
        list(metrics[0]),
        metrics,
    )
    _write_csv(
        os.path.join(args.output_dir, "headroom_comparison.csv"),
        list(metrics[0]),
        metrics,
    )
    _write_csv(
        os.path.join(args.output_dir, "sample_outcomes.csv"),
        list(sample_rows[0]),
        sample_rows,
    )
    _write_csv(
        os.path.join(args.output_dir, "sample_rank_vectors.csv"),
        list(rank_rows[0]),
        rank_rows,
    )
    _write_csv(
        os.path.join(args.output_dir, "stage_rank_trajectories.csv"),
        list(trajectory_rows[0]),
        trajectory_rows,
    )
    _write_csv(
        os.path.join(args.output_dir, "rank_distribution.csv"),
        list(distribution_rows[0]),
        distribution_rows,
    )
    bootstrap_payload = {
        result["metric"]["method"]: result["bootstrap"] for result in results
    }
    with open(
        os.path.join(args.output_dir, "bootstrap_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(bootstrap_payload, handle, indent=2)

    strong = [row["method"] for row in metrics if row["decision"] == "strong_support"]
    if set(strong) == {"madry_at", "rpcf_at"}:
        overall = "general_auto_rank_support"
    elif strong == ["rpcf_at"]:
        overall = "rpcf_conditional_support"
    elif strong:
        overall = "partial_support"
    elif any(row["decision"] == "weak_signal" for row in metrics):
        overall = "weak_signal"
    else:
        overall = "not_supported"
    summary = {
        "kind": "exp030_auto_rank_summary",
        "experiment_id": "EXP-030",
        "methods": metrics,
        "overall_decision": overall,
        "bootstrap": bootstrap_payload,
        "oracle_summary": args.oracle_summary,
        "oracle_rows": args.oracle_rows,
        "limitations": [
            "Labels and classifier logits are used only after rank inference for evaluation.",
            "This non-adaptive experiment does not establish robustness to attacks through the purifier.",
        ],
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    lines = [
        "# EXP-030 Auto-rank vs Fixed Rank25 vs Oracle",
        "",
        "| Method | Fixed adv | Auto adv | Oracle adv | Gain | 95% CI | Recovery | Clean Δ | Mean adv rank | Decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in metrics:
        recovery_text = (
            "N/A"
            if row["headroom_recovery"] is None
            else f"{row['headroom_recovery']:.1%}"
        )
        lines.append(
            f"| {row['method']} | {row['fixed_adv_accuracy']:.2%} | "
            f"{row['auto_adv_accuracy']:.2%} | {row['oracle_adv_accuracy']:.2%} | "
            f"{row['robust_gain_pp']:+.2f} pp | "
            f"[{row['robust_gain_ci_low_pp']:+.2f}, {row['robust_gain_ci_high_pp']:+.2f}] pp | "
            f"{recovery_text} | {row['clean_delta_pp']:+.2f} pp | "
            f"{row['mean_adv_rank']:.2f} | {row['decision']} |"
        )
    lines.extend(
        [
            "",
            f"Overall decision: `{overall}`.",
            "",
            "> 自动定秩不读取标签或分类器输出；oracle 仅作为不可部署的理论参照。",
        ]
    )
    with open(
        comparison_path,
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    _plots(args.output_dir, metrics, distribution_rows)
    print(summary_path)


if __name__ == "__main__":
    main()
