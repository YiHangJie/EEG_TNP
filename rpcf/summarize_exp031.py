"""EXP-031 严格汇总器：任何缺失、协议漂移或配对错位都会阻止 completed。"""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from rpcf.exp031 import EXPECTED_FULL_COUNTS, EXPERIMENT_ID, status_command_int, token
from rpcf.exp031_artifacts import EXP031_ARTIFACT_SAMPLE_NUM
from utils.reproducibility import stable_subset_indices


def read_tasks(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = int(row["seed"])
        row["rank"] = int(row["rank"])
        row["stage"] = int(row["stage"])
        row["dependencies"] = json.loads(row["dependencies"])
        row["command"] = json.loads(row["command"])
    return rows


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def close(actual, expected, tolerance=1e-12):
    return actual is not None and abs(float(actual) - float(expected)) <= tolerance


def validate_attack(row, payload):
    if not isinstance(payload, dict):
        raise ValueError("attack payload is not a dict")
    required = {"clean", "adversarial", "labels", "source_indices", "meta"}
    if required - set(payload):
        raise ValueError(f"attack missing keys {sorted(required - set(payload))}")
    meta = payload["meta"]
    expected_model = f"{row['model']}_ea_forward" if row["method"] == "ea_forward" else row["model"]
    for key, expected in {
        "dataset": row["dataset"], "model": expected_model, "fold": 0,
        "seed": row["seed"], "attack": row["attack"],
    }.items():
        if meta.get(key) != expected:
            raise ValueError(f"attack {key}={meta.get(key)!r}, expected {expected!r}")
    if not close(meta.get("eps"), 0.03):
        raise ValueError("attack epsilon mismatch")
    if meta.get("selection_strategy") != "full_test_split":
        raise ValueError("formal attack does not cover full test split")
    clean = torch.as_tensor(payload["clean"])
    adv = torch.as_tensor(payload["adversarial"])
    labels = torch.as_tensor(payload["labels"]).view(-1)
    if clean.shape != adv.shape or clean.size(0) != labels.numel():
        raise ValueError("attack tensor shape mismatch")
    sources = [int(value) for value in payload["source_indices"]]
    if len(sources) != labels.numel():
        raise ValueError("attack source-index length mismatch")
    evaluation_num = int(meta.get("evaluation_sample_num", labels.numel()))
    evaluated_sources = [int(value) for value in meta.get("evaluated_source_indices", sources)]
    if evaluated_sources != list(range(evaluation_num)):
        raise ValueError("attack did not audit exact full-test source coverage")
    expected_num = min(evaluation_num, EXP031_ARTIFACT_SAMPLE_NUM)
    if labels.numel() != expected_num or int(meta.get("artifact_sample_num", -1)) != expected_num:
        raise ValueError("attack compact artifact sample count mismatch")
    expected_positions = list(range(evaluation_num))
    if evaluation_num > EXP031_ARTIFACT_SAMPLE_NUM:
        expected_positions, _ = stable_subset_indices(
            evaluation_num, EXP031_ARTIFACT_SAMPLE_NUM, row["seed"], 0
        )
    if sources != [evaluated_sources[position] for position in expected_positions]:
        raise ValueError("attack compact artifact uses the wrong deterministic subset")
    if not 1 <= int(meta.get("actual_attack_batch_size", -1)) <= 32:
        raise ValueError("attack actual batch audit field missing or invalid")
    for metric in ("clean_accuracy", "adv_accuracy"):
        value = float(meta[metric])
        if not 0 <= value <= 1:
            raise ValueError(f"{metric} outside [0,1]")
    if float(meta.get("attack_l2_mean", -1)) < 0 or float(meta.get("attack_mse", -1)) < 0:
        raise ValueError("attack L2/MSE audit field missing or negative")
    protocol = meta.get("attack_protocol", {})
    if row["attack"] == "fgsm" and not (
        protocol.get("norm") == "Linf" and close(protocol.get("eps"), 0.03)
        and protocol.get("steps") == 1
    ):
        raise ValueError("FGSM protocol mismatch")
    if row["attack"] == "autoattack" and not (
        protocol.get("norm") == "Linf" and close(protocol.get("eps"), 0.03)
        and protocol.get("version") == "standard"
    ):
        raise ValueError("AutoAttack protocol mismatch")
    if row["attack"] == "pgd" and protocol != {
        "norm": "Linf", "eps": 0.03, "steps": 200,
        "alpha": 2 / 255, "random_start": False,
    }:
        raise ValueError("PGD protocol mismatch")
    if row["attack"] == "cw" and not (
        protocol.get("norm") == "L2" and protocol.get("steps") == 200
        and close(protocol.get("lr"), 0.1) and protocol.get("c") == 10000
        and protocol.get("kappa") == 1 and protocol.get("eps_is_constraint") is False
    ):
        raise ValueError("CW protocol mismatch")
    return meta


def validate_tnp(row, payload, attack_payload):
    required = {
        "clean", "adversarial", "clean_pur_by_rank", "adv_pur_by_rank",
        "labels", "source_indices", "ranks", "metrics", "raw_subset_metrics", "meta",
    }
    if not isinstance(payload, dict) or required - set(payload):
        raise ValueError("TNP payload structure mismatch")
    if list(payload["ranks"]) != [25, 30]:
        raise ValueError("TNP ranks must be [25,30]")
    meta = payload["meta"]
    for key, expected in {
        "dataset": row["dataset"], "model": row["model"], "fold": 0, "seed": row["seed"],
    }.items():
        if meta.get(key) != expected:
            raise ValueError(f"TNP {key} mismatch")
    positions = [int(value) for value in meta.get("selected_positions", [])]
    expected_sources = [attack_payload["source_indices"][index] for index in positions]
    if list(payload["source_indices"]) != expected_sources:
        raise ValueError("TNP source indices are not derived from its attack payload")
    index = torch.as_tensor(positions, dtype=torch.long)
    expected_labels = torch.as_tensor(attack_payload["labels"]).index_select(0, index).long()
    if not torch.equal(torch.as_tensor(payload["labels"]).long(), expected_labels):
        raise ValueError("TNP labels are not aligned with attack payload")
    if not torch.equal(torch.as_tensor(payload["clean"]).float(),
                       torch.as_tensor(attack_payload["clean"]).index_select(0, index).float()):
        raise ValueError("TNP clean tensor is not aligned with attack payload")
    for metric in payload["metrics"]:
        for key in ("purified_clean_accuracy", "purified_adv_accuracy"):
            if not 0 <= float(metric[key]) <= 1:
                raise ValueError(f"TNP {key} outside [0,1]")
    for key in ("standard_accuracy", "robust_accuracy"):
        value = float(payload["raw_subset_metrics"][key])
        if not 0 <= value <= 1:
            raise ValueError(f"TNP raw subset {key} outside [0,1]")
    return meta


def validate_rpcf_history(row):
    command = row["command"]
    prefix = command[command.index("--history_prefix") + 1]
    path = Path(f"{prefix}.json")
    if not path.exists():
        raise FileNotFoundError(path)
    history = json.loads(path.read_text(encoding="utf-8"))
    if history.get("all_layers") is not True:
        raise ValueError("RPCF_AT all_layers is not true")
    if history.get("static_rank_weights") is not True:
        raise ValueError("RPCF_AT static_rank_weights is not true")
    if history.get("feature_objective") != "none":
        raise ValueError("RPCF_AT top-level feature_objective is not none")
    if history.get("feature_alignment", {}).get("objective") != "none":
        raise ValueError("RPCF_AT feature objective is not none")
    if history.get("sensitivity_path") is not None:
        raise ValueError("all-layer RPCF_AT unexpectedly consumed sensitivity")
    if history.get("online_madry_at") is not True:
        raise ValueError("RPCF_AT online Madry AT is disabled")
    if history.get("online_at", {}).get("pgd_steps") != 10 or not close(
        history.get("online_at", {}).get("step_size"), 0.006
    ):
        raise ValueError("RPCF_AT online PGD protocol mismatch")
    for epoch in history.get("history", []):
        weights = [float(value) for value in epoch.get("rank_weights", [])]
        if len(weights) != 6 or any(not close(value, 1 / 6, 1e-8) for value in weights):
            raise ValueError("RPCF_AT rank weights are not uniformly 1/6")


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(long_rows):
    groups = defaultdict(list)
    for row in long_rows:
        key = (
            row["dataset"], row["model"], row["method"], row["attack"],
            row["metric"], row["rank"],
        )
        groups[key].append(float(row["value"]))
    rows = []
    for key, values in sorted(groups.items()):
        if len(values) != 5:
            raise ValueError(f"five-seed aggregate count is {len(values)}, expected 5: {key}")
        rows.append({
            "dataset": key[0], "model": key[1], "method": key[2], "attack": key[3],
            "metric": key[4], "rank": key[5], "seed_count": len(values),
            "mean": statistics.mean(values), "sample_std": statistics.stdev(values),
            "mean_pm_sample_std": f"{statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}",
        })
    return rows


def paired_deltas(long_rows):
    lookup = {
        (r["dataset"], r["model"], r["seed"], r["method"], r["attack"], r["metric"], r["rank"]):
        float(r["value"]) for r in long_rows
    }
    pairs = []
    suffixes = sorted({(r["dataset"], r["model"], r["seed"], r["attack"], r["metric"], r["rank"])
                       for r in long_rows if r["method"] == "rpcf_at"})
    for dataset, model, seed, attack, metric, rank in suffixes:
        rpcf_key = (dataset, model, seed, "rpcf_at", attack, metric, rank)
        madry_key = (dataset, model, seed, "madry", attack, metric, rank)
        if madry_key not in lookup:
            continue
        pairs.append({
            "dataset": dataset, "model": model, "seed": seed, "attack": attack,
            "metric": metric, "rank": rank,
            "rpcf_at": lookup[rpcf_key], "madry": lookup[madry_key],
            "rpcf_at_minus_madry": lookup[rpcf_key] - lookup[madry_key],
        })
    return pairs


def tnp_minus_raw_deltas(long_rows):
    """计算同一 checkpoint/attack/subset 定义下 rank25/30 相对 raw 的变化。"""
    lookup = {
        (row["dataset"], row["model"], row["seed"], row["method"],
         row["attack"], row["metric"], str(row["rank"])): float(row["value"])
        for row in long_rows
    }
    rows = []
    metric_pairs = (
        ("purified_standard_accuracy", "subset_standard_accuracy"),
        ("purified_robust_accuracy", "subset_robust_accuracy"),
    )
    for row in long_rows:
        if str(row["rank"]) not in {"25", "30"}:
            continue
        for purified_metric, raw_metric in metric_pairs:
            if row["metric"] != purified_metric:
                continue
            raw_key = (
                row["dataset"], row["model"], row["seed"], row["method"],
                row["attack"], raw_metric, "subset_raw",
            )
            if raw_key not in lookup:
                raise ValueError(f"Missing paired raw metric for TNP row: {raw_key}")
            rows.append({
                "dataset": row["dataset"], "model": row["model"], "seed": row["seed"],
                "method": row["method"], "attack": row["attack"], "rank": row["rank"],
                "metric": purified_metric, "purified": float(row["value"]),
                "raw": lookup[raw_key], "purified_minus_raw": float(row["value"]) - lookup[raw_key],
            })
    return rows


def audit_task_artifacts(tasks, run_dir):
    """返回缺失/失败/批量协议漂移列表；供汇总入口与单元测试共用。"""
    errors = []
    batch_manifest_path = run_dir / "actual_batch_sizes.json"
    batch_manifest = (
        json.loads(batch_manifest_path.read_text()) if batch_manifest_path.exists() else {}
    )
    cache_batch_manifest_path = run_dir / "actual_cache_attack_batch_sizes.json"
    cache_batch_manifest = (
        json.loads(cache_batch_manifest_path.read_text())
        if cache_batch_manifest_path.exists() else {}
    )
    rpcf_batch_manifest_path = run_dir / "actual_rpcf_batch_sizes.json"
    rpcf_batch_manifest = (
        json.loads(rpcf_batch_manifest_path.read_text())
        if rpcf_batch_manifest_path.exists() else {}
    )
    for row in tasks:
        status_path = run_dir / "status" / f"{row['task_id']}.json"
        if not status_path.exists():
            errors.append(f"missing status: {row['task_id']}")
        else:
            status = json.loads(status_path.read_text())
            if status.get("status") != "completed":
                errors.append(f"non-completed status: {row['task_id']}")
            if row["kind"] in {"train_standard", "train_ea", "rpcf_at"}:
                expected_batch = batch_manifest.get(token(row["dataset"], row["model"]))
                actual_batch = int(status.get("actual_batch_size", -1))
                if expected_batch is None or actual_batch != int(expected_batch):
                    errors.append(f"actual batch manifest mismatch: {row['task_id']}")
            if row["kind"] == "rpcf_cache":
                expected_cache_batch = cache_batch_manifest.get(
                    token(row["dataset"], row["model"])
                )
                actual_cache_batch = int(
                    status.get("actual_cache_attack_batch_size", -1)
                )
                if (
                    expected_cache_batch is None
                    or actual_cache_batch != int(expected_cache_batch)
                ):
                    errors.append(
                        f"cache attack batch manifest mismatch: {row['task_id']}"
                    )
            if row["kind"] == "rpcf_at":
                expected_rpcf_batch = rpcf_batch_manifest.get(
                    token(row["dataset"], row["model"])
                )
                actual_rpcf_batch = status_command_int(
                    status, "actual_rpcf_batch_size", "--batch_size"
                )
                if (
                    expected_rpcf_batch is None
                    or actual_rpcf_batch != int(expected_rpcf_batch)
                ):
                    errors.append(f"RPCF batch manifest mismatch: {row['task_id']}")
        if not Path(row["output_path"]).exists():
            errors.append(f"missing output: {row['output_path']}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run_dir = Path("logs/exp031") / args.run_id
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = read_tasks(run_dir / "planned_tasks.csv")
    errors = audit_task_artifacts(tasks, run_dir)
    counts = defaultdict(int)
    task_audit_rows = []
    for row in tasks:
        counts[row["kind"]] += 1
        status_path = run_dir / "status" / f"{row['task_id']}.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            task_audit_rows.append({
                **{key: row[key] for key in (
                    "task_id", "kind", "dataset", "model", "seed", "method", "attack", "rank",
                )},
                "status": status.get("status"), "returncode": status.get("returncode"),
                "elapsed_seconds": status.get("elapsed_seconds"),
                "physical_gpu": status.get("physical_gpu"),
                "actual_batch_size": status.get("actual_batch_size"),
                "actual_cache_attack_batch_size": status.get(
                    "actual_cache_attack_batch_size"
                ),
                "actual_rpcf_batch_size": status_command_int(
                    status, "actual_rpcf_batch_size", "--batch_size"
                ),
                "actual_rpcf_eval_batch_size": status_command_int(
                    status, "actual_rpcf_eval_batch_size", "--eval_batch_size"
                ),
                "output_path": row["output_path"],
            })
    for kind, expected in EXPECTED_FULL_COUNTS.items():
        if counts[kind] != expected:
            errors.append(f"planned count {kind}={counts[kind]}, expected {expected}")

    long_rows = []
    attack_payloads = {}
    for row in (item for item in tasks if item["kind"] == "attack" and Path(item["output_path"]).exists()):
        try:
            payload = load(row["output_path"])
            meta = validate_attack(row, payload)
            attack_payloads[(row["dataset"], row["model"], row["seed"], row["method"], row["attack"])] = payload
            audit = {
                "attack_mse": meta.get("attack_mse"), "attack_l2_mean": meta.get("attack_l2_mean"),
            }
            long_rows.extend([
                {**{key: row[key] for key in ("dataset", "model", "seed", "method", "attack")},
                 "metric": "standard_accuracy", "rank": "raw", "value": meta["clean_accuracy"], **audit},
                {**{key: row[key] for key in ("dataset", "model", "seed", "method", "attack")},
                 "metric": "robust_accuracy", "rank": "raw", "value": meta["adv_accuracy"], **audit},
            ])
        except Exception as exc:  # 汇总器必须收集全部缺失后一次报告。
            errors.append(f"attack {row['task_id']}: {exc}")

    tnp_payloads = {}
    for row in (item for item in tasks if item["kind"] == "tnp" and Path(item["output_path"]).exists()):
        key = (row["dataset"], row["model"], row["seed"], row["method"], row["attack"])
        try:
            payload = load(row["output_path"])
            validate_tnp(row, payload, attack_payloads[key])
            tnp_payloads[key] = payload
            raw_subset = payload["raw_subset_metrics"]
            base = {
                name: row[name]
                for name in ("dataset", "model", "seed", "method", "attack")
            }
            raw_audit = {
                "attack_mse": payload["meta"]["attack_meta"].get("attack_mse"),
                "attack_l2_mean": payload["meta"]["attack_meta"].get("attack_l2_mean"),
                "tnp_clean_mse": None, "tnp_adv_mse": None,
            }
            long_rows.extend([
                {**base, "metric": "subset_standard_accuracy", "rank": "subset_raw",
                 "value": raw_subset["standard_accuracy"], **raw_audit},
                {**base, "metric": "subset_robust_accuracy", "rank": "subset_raw",
                 "value": raw_subset["robust_accuracy"], **raw_audit},
            ])
            for metric in payload["metrics"]:
                rank = int(metric["rank"])
                audit = {"attack_mse": payload["meta"]["attack_meta"].get("attack_mse"),
                         "attack_l2_mean": payload["meta"]["attack_meta"].get("attack_l2_mean"),
                         "tnp_clean_mse": metric["mean_clean_mse"], "tnp_adv_mse": metric["mean_adv_mse"]}
                long_rows.extend([
                    {**{name: row[name] for name in ("dataset", "model", "seed", "method", "attack")},
                     "metric": "purified_standard_accuracy", "rank": rank,
                     "value": metric["purified_clean_accuracy"], **audit},
                    {**{name: row[name] for name in ("dataset", "model", "seed", "method", "attack")},
                     "metric": "purified_robust_accuracy", "rank": rank,
                     "value": metric["purified_adv_accuracy"], **audit},
                ])
        except Exception as exc:
            errors.append(f"tnp {row['task_id']}: {exc}")

    for key, rpcf_payload in tnp_payloads.items():
        dataset, model, seed, method, attack = key
        if method != "rpcf_at":
            continue
        madry = tnp_payloads.get((dataset, model, seed, "madry", attack))
        if madry is None or list(madry["source_indices"]) != list(rpcf_payload["source_indices"]) \
                or not torch.equal(torch.as_tensor(madry["labels"]), torch.as_tensor(rpcf_payload["labels"])):
            errors.append(f"Madry/RPCF TNP pairing mismatch: {key}")

    for row in (item for item in tasks if item["kind"] == "rpcf_at"):
        try:
            validate_rpcf_history(row)
        except Exception as exc:
            errors.append(f"rpcf history {row['task_id']}: {exc}")

    bpda_rows = []
    for row in (item for item in tasks if item["kind"] == "bpda" and Path(item["output_path"]).exists()):
        payload = load(row["output_path"])
        meta, metrics = payload["meta"], payload["metrics"]
        if meta.get("experiment_id") != EXPERIMENT_ID or meta.get("pgd_steps") != 10 \
                or not close(meta.get("pgd_alpha"), 0.006):
            errors.append(f"BPDA protocol mismatch: {row['task_id']}")
        value = float(metrics["bpda_purified_adv_accuracy"])
        if not 0 <= value <= 1:
            errors.append(f"BPDA accuracy outside [0,1]: {row['task_id']}")
        bpda_rows.append({"dataset": row["dataset"], "model": row["model"], "seed": row["seed"],
                          "rank": row["rank"], **metrics})

    aggregate_rows = []
    delta_rows = []
    tnp_delta_rows = []
    if not errors:
        try:
            aggregate_rows = aggregate(long_rows)
            delta_rows = paired_deltas(long_rows)
            tnp_delta_rows = tnp_minus_raw_deltas(long_rows)
        except Exception as exc:
            errors.append(f"aggregate: {exc}")
    write_csv(output_dir / "conditions_long.csv", long_rows)
    write_csv(output_dir / "task_audit.csv", task_audit_rows)
    write_csv(output_dir / "five_seed_mean_std.csv", aggregate_rows)
    write_csv(output_dir / "rpcf_at_minus_madry_paired.csv", delta_rows)
    write_csv(output_dir / "tnp_minus_raw_paired.csv", tnp_delta_rows)
    write_csv(output_dir / "cross_dataset_backbone.csv", aggregate_rows)
    write_csv(output_dir / "bpda.csv", bpda_rows)
    completeness = {
        "experiment_id": EXPERIMENT_ID, "run_id": args.run_id,
        "completed": not errors, "errors": errors, "planned_counts": dict(counts),
        "expected_counts": EXPECTED_FULL_COUNTS, "long_row_count": len(long_rows),
        "five_seed_group_count": len(aggregate_rows), "paired_delta_count": len(delta_rows),
        "tnp_minus_raw_count": len(tnp_delta_rows),
    }
    (output_dir / "completeness.json").write_text(
        json.dumps(completeness, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if errors:
        raise SystemExit(f"EXP-031 is incomplete: {len(errors)} validation errors; see completeness.json")
    print(output_dir / "completeness.json")


if __name__ == "__main__":
    main()
