"""EXP-031 全矩阵流式严格汇总：逐文件校验，不常驻保存大型 tensor。

运行：python -u -m rpcf.summarize_exp031_streaming --run-id exp031_full_20260729_174215
"""

import argparse
import gc
import json
from collections import Counter
from pathlib import Path

import torch

from rpcf.exp031 import EXPECTED_FULL_COUNTS, EXPERIMENT_ID, status_command_int
from rpcf.summarize_exp031 import (
    aggregate,
    audit_task_artifacts,
    close,
    paired_deltas,
    read_tasks,
    tnp_minus_raw_deltas,
    validate_attack,
    validate_rpcf_history,
    validate_tnp,
    write_csv,
)


def load_mmap(path):
    """只读映射张量存储，保证一次只保留一个 TNP 和其对应攻击产物。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def task_key(row):
    return tuple(row[name] for name in ("dataset", "model", "seed", "method", "attack"))


def base(row):
    return {name: row[name] for name in ("dataset", "model", "seed", "method", "attack")}


def status_audit(tasks, run_dir):
    errors = audit_task_artifacts(tasks, run_dir)
    rows = []
    for row in tasks:
        path = run_dir / "status" / f"{row['task_id']}.json"
        if not path.exists():
            continue
        status = json.loads(path.read_text(encoding="utf-8"))
        if status.get("returncode") != 0:
            errors.append(f"nonzero return code: {row['task_id']}")
        output = Path(row["output_path"])
        if output.exists() and output.stat().st_size == 0:
            errors.append(f"zero-byte output: {row['task_id']}")
        rows.append({
            **{name: row[name] for name in (
                "task_id", "kind", "dataset", "model", "seed", "method", "attack", "rank",
            )},
            "status": status.get("status"),
            "returncode": status.get("returncode"),
            "elapsed_seconds": status.get("elapsed_seconds"),
            "physical_gpu": status.get("physical_gpu"),
            "actual_batch_size": status.get("actual_batch_size"),
            "actual_cache_attack_batch_size": status.get("actual_cache_attack_batch_size"),
            "actual_rpcf_batch_size": status_command_int(
                status, "actual_rpcf_batch_size", "--batch_size"
            ),
            "actual_rpcf_eval_batch_size": status_command_int(
                status, "actual_rpcf_eval_batch_size", "--eval_batch_size"
            ),
            "output_path": row["output_path"],
        })
    return errors, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    run_dir = Path("logs/exp031") / args.run_id
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = read_tasks(run_dir / "planned_tasks.csv")
    errors, audit_rows = status_audit(tasks, run_dir)
    counts = Counter(row["kind"] for row in tasks)
    for kind, expected in EXPECTED_FULL_COUNTS.items():
        if counts[kind] != expected:
            errors.append(f"planned count {kind}={counts[kind]}, expected {expected}")
    expected_total = sum(EXPECTED_FULL_COUNTS.values()) + 3  # 三个串行 prepare_data 任务。
    if len(tasks) != expected_total or counts["prepare_data"] != 3:
        errors.append(f"planned total/prepare_data={len(tasks)}/{counts['prepare_data']}, expected {expected_total}/3")
    if len({row["task_id"] for row in tasks}) != len(tasks):
        errors.append("duplicate task_id in planned_tasks.csv")
    print(f"TASK_AUDIT {len(audit_rows)}/{len(tasks)} errors={len(errors)}", flush=True)

    for row in (item for item in tasks if item["kind"] == "rpcf_at"):
        try:
            validate_rpcf_history(row)
        except Exception as exc:
            errors.append(f"rpcf history {row['task_id']}: {exc}")
    print(f"RPCF_HISTORY checked=90 errors={len(errors)}", flush=True)

    long_rows = []
    attack_rows = {task_key(row): row for row in tasks if row["kind"] == "attack"}
    for index, row in enumerate((item for item in tasks if item["kind"] == "attack"), 1):
        payload = None
        try:
            payload = load_mmap(row["output_path"])
            meta = validate_attack(row, payload)
            fields = {
                "attack_mse": meta.get("attack_mse"),
                "attack_l2_mean": meta.get("attack_l2_mean"),
                "tnp_clean_mse": None,
                "tnp_adv_mse": None,
            }
            long_rows.extend([
                {**base(row), "metric": "standard_accuracy", "rank": "raw",
                 "value": meta["clean_accuracy"], **fields},
                {**base(row), "metric": "robust_accuracy", "rank": "raw",
                 "value": meta["adv_accuracy"], **fields},
            ])
        except Exception as exc:
            errors.append(f"attack {row['task_id']}: {exc}")
        finally:
            del payload
        if index % 300 == 0:
            gc.collect()
            print(f"ATTACK_VALIDATED {index}/1800 errors={len(errors)}", flush=True)

    # 每个 dataset/model/seed/attack 只保留 Madry 与 RPCF 的 source-index/label 身份信息。
    identities = {}
    for index, row in enumerate((item for item in tasks if item["kind"] == "tnp"), 1):
        payload = None
        attack_payload = None
        try:
            payload = load_mmap(row["output_path"])
            attack_payload = load_mmap(attack_rows[task_key(row)]["output_path"])
            validate_tnp(row, payload, attack_payload)
            meta = payload["meta"]
            attack_meta = meta.get("attack_meta", {})
            if attack_meta.get("method_tag") != row["method"] or attack_meta.get("attack") != row["attack"]:
                raise ValueError("embedded attack method/attack mismatch")
            if len(payload["metrics"]) != 2 or {int(item["rank"]) for item in payload["metrics"]} != {25, 30}:
                raise ValueError("TNP metric ranks must be exactly 25 and 30")
            positions = [int(value) for value in meta.get("selected_positions", [])]
            if len(positions) != len(payload["labels"]) or len(positions) != len(set(positions)):
                raise ValueError("TNP selected positions count/uniqueness mismatch")
            expected_shape = (len(positions), 2, *payload["clean"].shape[1:])
            for name in ("clean_pur_by_rank", "adv_pur_by_rank"):
                if tuple(payload[name].shape) != expected_shape:
                    raise ValueError(f"{name} shape mismatch")
            pair_key = (row["dataset"], row["model"], row["seed"], row["attack"])
            identities.setdefault(pair_key, {})[row["method"]] = (
                tuple(int(value) for value in payload["source_indices"]),
                torch.as_tensor(payload["labels"]).long().clone(),
            )
            raw = payload["raw_subset_metrics"]
            fields = {
                "attack_mse": attack_meta.get("attack_mse"),
                "attack_l2_mean": attack_meta.get("attack_l2_mean"),
                "tnp_clean_mse": None,
                "tnp_adv_mse": None,
            }
            long_rows.extend([
                {**base(row), "metric": "subset_standard_accuracy", "rank": "subset_raw",
                 "value": raw["standard_accuracy"], **fields},
                {**base(row), "metric": "subset_robust_accuracy", "rank": "subset_raw",
                 "value": raw["robust_accuracy"], **fields},
            ])
            for metric in payload["metrics"]:
                rank_fields = {
                    **fields,
                    "tnp_clean_mse": metric["mean_clean_mse"],
                    "tnp_adv_mse": metric["mean_adv_mse"],
                }
                long_rows.extend([
                    {**base(row), "metric": "purified_standard_accuracy", "rank": int(metric["rank"]),
                     "value": metric["purified_clean_accuracy"], **rank_fields},
                    {**base(row), "metric": "purified_robust_accuracy", "rank": int(metric["rank"]),
                     "value": metric["purified_adv_accuracy"], **rank_fields},
                ])
        except Exception as exc:
            errors.append(f"tnp {row['task_id']}: {exc}")
        finally:
            del payload, attack_payload
        if index % 20 == 0:
            gc.collect()
            print(f"TNP_VALIDATED {index}/720 errors={len(errors)}", flush=True)

    for key, pair in identities.items():
        madry, rpcf = pair.get("madry"), pair.get("rpcf_at")
        if madry is None or rpcf is None or madry[0] != rpcf[0] or not torch.equal(madry[1], rpcf[1]):
            errors.append(f"Madry/RPCF TNP pairing mismatch: {key}")

    bpda_rows = []
    for row in (item for item in tasks if item["kind"] == "bpda"):
        payload = None
        try:
            payload = load_mmap(row["output_path"])
            meta, metrics = payload["meta"], payload["metrics"]
            if meta.get("experiment_id") != EXPERIMENT_ID or meta.get("dataset") != row["dataset"] \
                    or meta.get("model") != row["model"] or meta.get("seed") != row["seed"] \
                    or meta.get("rank") != row["rank"] or meta.get("pgd_steps") != 10 \
                    or not close(meta.get("pgd_alpha"), 0.006) or not close(meta.get("eps"), 0.03):
                raise ValueError("BPDA protocol mismatch")
            for name in ("purified_clean_accuracy", "bpda_purified_adv_accuracy"):
                if not 0 <= float(metrics[name]) <= 1:
                    raise ValueError(f"{name} outside [0,1]")
            bpda_rows.append({"dataset": row["dataset"], "model": row["model"],
                              "seed": row["seed"], "rank": row["rank"], **metrics})
        except Exception as exc:
            errors.append(f"bpda {row['task_id']}: {exc}")
        finally:
            del payload

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
    write_csv(output_dir / "task_audit.csv", audit_rows)
    write_csv(output_dir / "five_seed_mean_std.csv", aggregate_rows)
    write_csv(output_dir / "rpcf_at_minus_madry_paired.csv", delta_rows)
    write_csv(output_dir / "tnp_minus_raw_paired.csv", tnp_delta_rows)
    write_csv(output_dir / "cross_dataset_backbone.csv", aggregate_rows)
    write_csv(output_dir / "bpda.csv", bpda_rows)
    completeness = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": args.run_id,
        "completed": not errors,
        "errors": errors,
        "planned_counts": dict(counts),
        "expected_counts": EXPECTED_FULL_COUNTS,
        "long_row_count": len(long_rows),
        "five_seed_group_count": len(aggregate_rows),
        "paired_delta_count": len(delta_rows),
        "tnp_minus_raw_count": len(tnp_delta_rows),
        "bpda_row_count": len(bpda_rows),
        "streaming_tensor_validation": True,
    }
    (output_dir / "completeness.json").write_text(
        json.dumps(completeness, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if errors:
        raise SystemExit(f"EXP-031 incomplete: {len(errors)} validation errors; see completeness.json")
    print(output_dir / "completeness.json", flush=True)


if __name__ == "__main__":
    main()
