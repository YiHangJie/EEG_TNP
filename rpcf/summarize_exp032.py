"""严格验收 EXP-032 任务覆盖和同一 n512 双指标，运行中输出 Pending。"""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from rpcf.exp031 import DATASETS, MODELS, SEEDS
from rpcf.resume_exp032 import read_scope


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_dir = Path("logs/exp032") / args.run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    tasks = list(csv.DictReader((run_dir / "planned_tasks.csv").open()))
    scope = read_scope(run_dir, tasks)
    deferred = set(scope["deferred_task_ids"]) if scope else set()
    active_tasks = [t for t in tasks if t["task_id"] not in deferred]
    expected_rows = scope["expected_rows_per_condition"] if scope else 80
    states, rows, errors, identities, audits = [], [], [], {}, []
    for task in tasks:
        status_file = run_dir / "status" / (task["task_id"] + ".json")
        status = json.loads(status_file.read_text()) if status_file.exists() else {"status": "pending"}
        output = Path(task["output_path"])
        if task["task_id"] in deferred:
            states.append({"task_id": task["task_id"], "kind": task["kind"],
                           "status": "deferred", "output_path": str(output)})
            continue
        finished = status.get("status") == "completed" and output.exists()
        states.append({"task_id": task["task_id"], "kind": task["kind"],
                       "status": "completed" if finished else status.get("status", "pending"),
                       "output_path": str(output)})
        if not finished:
            continue
        metric_file = Path(str(output) + ".metrics.json") if task["kind"] in {"attack", "tnp"} else output
        if task["kind"] not in {"audit", "attack", "tnp", "external_eval"}:
            continue
        if not metric_file.exists():
            errors.append(f"Missing metrics: {metric_file}")
            continue
        payload = json.loads(metric_file.read_text())
        for check in payload.get("checks", []):
            audits.append({"dataset": task["dataset"], "model": task["model"], "seed": int(task["seed"]),
                           "method": check["method"], "checkpoint_path": check["checkpoint_path"],
                           "checkpoint_sha256": check["checkpoint_sha256"],
                           "archive_full_clean_span": check["archive_full_clean_span"],
                           "state_changes_after_clean": json.dumps(check["state_changes_after_clean"]),
                           "state_changes_after_all": json.dumps(check["state_changes_after_all"]),
                           "actual_batch_size": check.get("actual_batch_size"),
                           "training_options": json.dumps(check.get("training_options", {}))})
        identity = (payload["source_indices"], payload["labels"])
        key = (task["dataset"], int(task["seed"]))
        if key in identities and identities[key] != identity:
            errors.append(f"Cross-method identity mismatch: {metric_file}")
        identities[key] = identity
        if len(set(identity[0])) != len(identity[0]):
            errors.append(f"Duplicate sample ids: {metric_file}")
        for row in payload["rows"]:
            if row["sample_num"] != len(identity[0]) or row["scope"] != "S":
                errors.append(f"Invalid metric scope/count: {metric_file}")
            for metric, prediction in (("standard_accuracy", "clean_predictions"), ("robust_accuracy", "adv_predictions")):
                if len(row[prediction]) != len(identity[1]):
                    errors.append(f"Prediction count mismatch: {metric_file}")
                accuracy = sum(p == y for p, y in zip(row[prediction], identity[1])) / len(identity[1])
                if abs(accuracy - row[metric]) > 1e-12:
                    errors.append(f"Prediction metric mismatch: {metric_file}")
            rows.append(row)
    unique = [(r["dataset"], r["model"], r["seed"], r["method"], r["attack"], r["evaluation"]) for r in rows]
    if len(unique) != len(set(unique)):
        errors.append("Duplicate condition/method/attack/evaluation rows")
    expected_conditions = [("thubenchmark", "eegnet", 42)] if manifest["smoke"] else [
        (d, m, s) for d in DATASETS for m in MODELS for s in SEEDS]
    completed_tasks = sum(r["status"] == "completed" for r in states)
    all_tasks = completed_tasks == len(active_tasks)
    counts = Counter((r["dataset"], r["model"], r["seed"]) for r in rows)
    if all_tasks and any(counts[c] != expected_rows for c in expected_conditions):
        errors.append(f"Expected {expected_rows} rows per condition, got {dict(counts)}")
    # 预先固定的方法×攻击×评估假设，禁止仅凭总行数验收。
    expected_keys = set()
    for method in ("madry", "trades", "fbf", "ea_forward", "rpcf_at", "clean"):
        expected_keys.update((method, attack, "classifier_whitebox") for attack in ("fgsm", "pgd", "autoattack", "cw", "pgd_l2"))
    tnp_methods = ("madry", "rpcf_at") if scope else ("madry", "rpcf_at", "clean")
    for method in tnp_methods:
        for rank in (25, 30):
            expected_keys.update((f"{method}_tnp_r{rank}", attack, "nonadaptive_purification") for attack in ("fgsm", "pgd", "autoattack", "cw", "pgd_l2"))
    for method in ("clean_magnet", "clean_dcae"):
        for evaluation in ("nonadaptive_purification", "adaptive_exact_gradient"):
            expected_keys.update((method, attack, evaluation) for attack in ("fgsm", "pgd", "autoattack", "cw", "pgd_l2"))
    if all_tasks:
        for condition in expected_conditions:
            actual = {(r["method"], r["attack"], r["evaluation"]) for r in rows
                      if (r["dataset"], r["model"], r["seed"]) == condition}
            if actual != expected_keys:
                errors.append(f"Coverage mismatch: {condition}")
    out = run_dir / ("summary_without_clean_tnp" if scope else "summary")
    out.mkdir(exist_ok=True)
    write_csv(out / "tasks.csv", states)
    write_csv(out / "baseline_audit.csv", audits)
    scalar = [{k: r[k] for k in ("dataset", "model", "seed", "fold", "method", "attack", "evaluation",
                                "sample_num", "scope", "standard_accuracy", "robust_accuracy")} for r in rows]
    write_csv(out / "conditions_long.csv", scalar)
    groups = defaultdict(list)
    for row in scalar:
        groups[tuple(row[k] for k in ("dataset", "model", "method", "attack", "evaluation"))].append(row)
    means = []
    for key, values in sorted(groups.items()):
        result = dict(zip(("dataset", "model", "method", "attack", "evaluation"), key))
        result["seed_count"] = len(values)
        for metric in ("standard_accuracy", "robust_accuracy"):
            numbers = [v[metric] * 100 for v in values]
            result[metric + "_mean_percent"] = statistics.mean(numbers)
            result[metric + "_std_percent"] = statistics.stdev(numbers) if len(numbers) > 1 else None
        means.append(result)
    write_csv(out / "five_seed_mean_std.csv", means)
    by_condition = {(r["dataset"], r["model"], r["seed"], r["method"], r["attack"], r["evaluation"]): r for r in scalar}
    paired = []
    for row in scalar:
        if "_tnp_r" not in row["method"]:
            continue
        base = row["method"].split("_tnp_r")[0]
        other = by_condition.get((row["dataset"], row["model"], row["seed"], base, row["attack"], "classifier_whitebox"))
        if other:
            paired.append({"dataset": row["dataset"], "model": row["model"], "seed": row["seed"],
                           "method": row["method"], "attack": row["attack"],
                           "clean_delta_pp": 100 * (row["standard_accuracy"] - other["standard_accuracy"]),
                           "robust_delta_pp": 100 * (row["robust_accuracy"] - other["robust_accuracy"])})
    write_csv(out / "tnp_minus_raw_paired.csv", paired)
    completed = all_tasks and not errors and not manifest["smoke"]
    report = {"experiment_id": "EXP-032", "run_id": args.run_id, "completed": completed,
              "smoke": manifest["smoke"], "smoke_passed": all_tasks and not errors if manifest["smoke"] else None,
              "status": ("Completed active scope" if scope else "Completed") if completed else "Failed validation" if errors else "Pending",
              "execution_scope": scope["name"] if scope else "full",
              "full_plan_completed": completed and not deferred,
              "planned_tasks": len(tasks), "deferred_tasks": len(deferred),
              "deferred_reason": scope["reason"] if scope else None,
              "completed_tasks": completed_tasks, "expected_tasks": len(active_tasks),
              "metric_rows": len(rows), "expected_metric_rows": expected_rows * len(expected_conditions),
              "errors": errors}
    (out / "completeness.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
