"""严格汇总 EXP-031 的 THUbenchmark/EEGNet 五-seed 闭环结果。

该入口只将已完成的聚焦闭环标记为 completed，不会把仍在运行的完整 EXP-031
矩阵误标为完成。大体积 torch artifact 逐个用 mmap 加载并及时释放，避免汇总阶段
占满内存。
"""

import argparse
import gc
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import torch

from rpcf.exp031 import EXPERIMENT_ID
from rpcf.summarize_exp031 import (
    audit_task_artifacts,
    read_tasks,
    validate_attack,
    validate_rpcf_history,
    validate_tnp,
    write_csv,
)


FOCUS_COUNTS = {"attack": 100, "tnp": 40, "bpda": 10}
SEEDS = (42, 43, 44, 45, 46)
METHODS = ("madry", "trades", "fbf", "ea_forward", "rpcf_at")
ATTACKS = ("autoattack", "fgsm", "pgd", "cw")


def load_artifact(path):
    """用只读 mmap 解包 tensor storage；旧 torch 版本则退回普通 CPU 加载。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def mean_std(values):
    values = [float(value) for value in values]
    if len(values) != 5:
        raise ValueError(f"five-seed count={len(values)}, expected 5")
    return statistics.mean(values), statistics.stdev(values)


def aggregate_rows(rows, value_key="value"):
    groups = defaultdict(list)
    group_fields = ("method", "attack", "metric", "rank")
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(float(row[value_key]))
    output = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        mean, std = mean_std(values)
        output.append({
            **dict(zip(group_fields, key)),
            "seed_count": len(values),
            "mean": mean,
            "sample_std": std,
            "mean_pm_sample_std": f"{mean:.6f} ± {std:.6f}",
        })
    return output


def aggregate_deltas(rows, delta_key):
    groups = defaultdict(list)
    fields = ("attack", "metric", "rank")
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(float(row[delta_key]))
    output = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        mean, std = mean_std(values)
        output.append({
            **dict(zip(fields, key)),
            "seed_count": len(values),
            "mean_delta": mean,
            "sample_std": std,
            "mean_delta_pp": 100 * mean,
            "sample_std_pp": 100 * std,
        })
    return output


def metric_lookup(rows):
    return {
        (row["seed"], row["method"], row["attack"], row["metric"], str(row["rank"])):
        float(row["value"])
        for row in rows
    }


def paired_rows(rows):
    lookup = metric_lookup(rows)
    output = []
    for key, rpcf_value in sorted(lookup.items(), key=lambda item: tuple(map(str, item[0]))):
        seed, method, attack, metric, rank = key
        if method != "rpcf_at":
            continue
        madry_key = (seed, "madry", attack, metric, rank)
        if madry_key not in lookup:
            continue
        output.append({
            "seed": seed, "attack": attack, "metric": metric, "rank": rank,
            "rpcf_at": rpcf_value, "madry": lookup[madry_key],
            "rpcf_at_minus_madry": rpcf_value - lookup[madry_key],
        })
    return output


def tnp_minus_raw_rows(rows):
    lookup = metric_lookup(rows)
    output = []
    for row in rows:
        if str(row["rank"]) not in {"25", "30"}:
            continue
        raw_metric = {
            "purified_standard_accuracy": "subset_standard_accuracy",
            "purified_robust_accuracy": "subset_robust_accuracy",
        }.get(row["metric"])
        if raw_metric is None:
            continue
        raw_key = (row["seed"], row["method"], row["attack"], raw_metric, "subset_raw")
        if raw_key not in lookup:
            raise ValueError(f"missing paired subset raw metric: {raw_key}")
        purified = float(row["value"])
        raw = lookup[raw_key]
        output.append({
            "seed": row["seed"], "method": row["method"], "attack": row["attack"],
            "metric": row["metric"], "rank": row["rank"], "purified": purified,
            "raw": raw, "purified_minus_raw": purified - raw,
        })
    return output


def bpda_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        for metric in ("purified_clean_accuracy", "bpda_purified_adv_accuracy"):
            groups[(row["rank"], metric)].append(float(row[metric]))
    output = []
    for (rank, metric), values in sorted(groups.items()):
        mean, std = mean_std(values)
        output.append({
            "rank": rank, "metric": metric, "seed_count": len(values),
            "mean": mean, "sample_std": std,
            "mean_pm_sample_std": f"{mean:.6f} ± {std:.6f}",
        })
    return output


def pct(mean, std):
    return f"{100 * mean:.2f} ± {100 * std:.2f}"


def render_markdown(aggregate, paired_summary, tnp_delta_summary, bpda_aggregate):
    agg = {(r["method"], r["attack"], r["metric"], str(r["rank"])): r for r in aggregate}
    lines = [
        "# EXP-031 THUbenchmark / EEGNet 五 seed 聚焦汇总", "",
        "全部数值为 mean ± sample std（%）。raw 来自完整 test split；TNP 与 BPDA 为确定性 n512 子集。", "",
        "## Raw full-test", "",
        "| method | clean | AutoAttack | FGSM | PGD-200 | CW-L2 |", "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        clean = agg[(method, "autoattack", "standard_accuracy", "raw")]
        robust = [agg[(method, attack, "robust_accuracy", "raw")] for attack in ATTACKS]
        lines.append("| " + " | ".join([
            method, pct(clean["mean"], clean["sample_std"]),
            *[pct(row["mean"], row["sample_std"]) for row in robust],
        ]) + " |")

    lines += ["", "## EEG_TNP purified robust accuracy", "",
              "| method | attack | rank25 | rank30 |", "| --- | --- | ---: | ---: |"]
    for method in ("madry", "rpcf_at"):
        for attack in ATTACKS:
            values = [agg[(method, attack, "purified_robust_accuracy", str(rank))] for rank in (25, 30)]
            lines.append(f"| {method} | {attack} | {pct(values[0]['mean'], values[0]['sample_std'])} | {pct(values[1]['mean'], values[1]['sample_std'])} |")

    pair = {(r["attack"], r["metric"], str(r["rank"])): r for r in paired_summary}
    lines += ["", "## RPCF_AT − Madry 严格配对差值", "",
              "| attack | raw | TNP rank25 | TNP rank30 |", "| --- | ---: | ---: | ---: |"]
    for attack in ATTACKS:
        raw = pair[(attack, "robust_accuracy", "raw")]
        rank25 = pair[(attack, "purified_robust_accuracy", "25")]
        rank30 = pair[(attack, "purified_robust_accuracy", "30")]
        lines.append(f"| {attack} | {raw['mean_delta_pp']:.2f} ± {raw['sample_std_pp']:.2f} | {rank25['mean_delta_pp']:.2f} ± {rank25['sample_std_pp']:.2f} | {rank30['mean_delta_pp']:.2f} ± {rank30['sample_std_pp']:.2f} |")

    bpda = {(str(r["rank"]), r["metric"]): r for r in bpda_aggregate}
    lines += ["", "## Adaptive BPDA+PGD-10（RPCF_AT + TNP）", "",
              "| rank | purified clean | BPDA purified robust |", "| ---: | ---: | ---: |"]
    for rank in (25, 30):
        clean = bpda[(str(rank), "purified_clean_accuracy")]
        robust = bpda[(str(rank), "bpda_purified_adv_accuracy")]
        lines.append(f"| {rank} | {pct(clean['mean'], clean['sample_std'])} | {pct(robust['mean'], robust['sample_std'])} |")

    tnp_delta = {(r["method"], r["attack"], str(r["rank"])): r for r in tnp_delta_summary if r["metric"] == "purified_robust_accuracy"}
    lines += ["", "## TNP 相对同子集 raw robust 变化", "",
              "| method | attack | rank25 delta | rank30 delta |", "| --- | --- | ---: | ---: |"]
    for method in ("madry", "rpcf_at"):
        for attack in ATTACKS:
            r25 = tnp_delta[(method, attack, "25")]
            r30 = tnp_delta[(method, attack, "30")]
            lines.append(f"| {method} | {attack} | {r25['mean_delta_pp']:.2f} ± {r25['sample_std_pp']:.2f} | {r30['mean_delta_pp']:.2f} ± {r30['sample_std_pp']:.2f} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run_dir = Path("logs/exp031") / args.run_id
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "summary_thu_eegnet"
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = read_tasks(run_dir / "planned_tasks.csv")
    focus = [row for row in tasks if row["dataset"] == "thubenchmark" and row["model"] == "eegnet" and row["kind"] in FOCUS_COUNTS]
    rpcf_tasks = [row for row in tasks if row["dataset"] == "thubenchmark" and row["model"] == "eegnet" and row["kind"] == "rpcf_at"]
    errors = audit_task_artifacts(focus + rpcf_tasks, run_dir)
    counts = Counter(row["kind"] for row in focus)
    for kind, expected in FOCUS_COUNTS.items():
        if counts[kind] != expected:
            errors.append(f"focused count {kind}={counts[kind]}, expected {expected}")
    if sorted(row["seed"] for row in rpcf_tasks) != list(SEEDS):
        errors.append("focused RPCF_AT training rows do not cover seeds42-46 exactly")
    for row in rpcf_tasks:
        try:
            validate_rpcf_history(row)
        except Exception as exc:
            errors.append(f"rpcf history {row['task_id']}: {exc}")

    audit_rows = []
    for row in focus:
        status = json.loads((run_dir / "status" / f"{row['task_id']}.json").read_text())
        audit_rows.append({
            "task_id": row["task_id"], "kind": row["kind"], "seed": row["seed"],
            "method": row["method"], "attack": row["attack"], "rank": row["rank"],
            "status": status.get("status"), "returncode": status.get("returncode"),
            "elapsed_seconds": status.get("elapsed_seconds"),
            "physical_gpu": status.get("physical_gpu"), "output_path": row["output_path"],
        })

    long_rows = []
    attack_rows = {(r["seed"], r["method"], r["attack"]): r for r in focus if r["kind"] == "attack"}
    for key, row in sorted(attack_rows.items(), key=lambda item: tuple(map(str, item[0]))):
        try:
            payload = load_artifact(row["output_path"])
            meta = validate_attack(row, payload)
            base = {"seed": row["seed"], "method": row["method"], "attack": row["attack"]}
            audit = {"attack_mse": meta["attack_mse"], "attack_l2_mean": meta["attack_l2_mean"], "tnp_clean_mse": None, "tnp_adv_mse": None}
            long_rows += [
                {**base, "metric": "standard_accuracy", "rank": "raw", "value": meta["clean_accuracy"], **audit},
                {**base, "metric": "robust_accuracy", "rank": "raw", "value": meta["adv_accuracy"], **audit},
            ]
        except Exception as exc:
            errors.append(f"attack {row['task_id']}: {exc}")
        finally:
            if "payload" in locals():
                del payload
            gc.collect()

    tnp_rows = {(r["seed"], r["method"], r["attack"]): r for r in focus if r["kind"] == "tnp"}
    for seed in SEEDS:
        for attack in ATTACKS:
            pairing = {}
            for method in ("madry", "rpcf_at"):
                row = tnp_rows[(seed, method, attack)]
                attack_row = attack_rows[(seed, method, attack)]
                try:
                    attack_payload = load_artifact(attack_row["output_path"])
                    payload = load_artifact(row["output_path"])
                    validate_tnp(row, payload, attack_payload)
                    attack_meta = payload["meta"].get("attack_meta", {})
                    if attack_meta.get("method_tag") != method or attack_meta.get("attack") != attack:
                        raise ValueError("TNP embedded method/attack metadata mismatch")
                    pairing[method] = (
                        [int(value) for value in payload["source_indices"]],
                        torch.as_tensor(payload["labels"]).long().clone(),
                    )
                    base = {"seed": seed, "method": method, "attack": attack}
                    audit = {"attack_mse": attack_meta.get("attack_mse"), "attack_l2_mean": attack_meta.get("attack_l2_mean"), "tnp_clean_mse": None, "tnp_adv_mse": None}
                    raw = payload["raw_subset_metrics"]
                    long_rows += [
                        {**base, "metric": "subset_standard_accuracy", "rank": "subset_raw", "value": raw["standard_accuracy"], **audit},
                        {**base, "metric": "subset_robust_accuracy", "rank": "subset_raw", "value": raw["robust_accuracy"], **audit},
                    ]
                    for metric in payload["metrics"]:
                        rank = int(metric["rank"])
                        rank_audit = {**audit, "tnp_clean_mse": metric["mean_clean_mse"], "tnp_adv_mse": metric["mean_adv_mse"]}
                        long_rows += [
                            {**base, "metric": "purified_standard_accuracy", "rank": rank, "value": metric["purified_clean_accuracy"], **rank_audit},
                            {**base, "metric": "purified_robust_accuracy", "rank": rank, "value": metric["purified_adv_accuracy"], **rank_audit},
                        ]
                except Exception as exc:
                    errors.append(f"tnp {row['task_id']}: {exc}")
                finally:
                    if "payload" in locals():
                        del payload
                    if "attack_payload" in locals():
                        del attack_payload
                    gc.collect()
            if set(pairing) == {"madry", "rpcf_at"} and (
                pairing["madry"][0] != pairing["rpcf_at"][0]
                or not torch.equal(pairing["madry"][1], pairing["rpcf_at"][1])
            ):
                errors.append(f"Madry/RPCF TNP pairing mismatch: seed={seed}, attack={attack}")

    bpda_rows = []
    for row in sorted((r for r in focus if r["kind"] == "bpda"), key=lambda r: (r["seed"], r["rank"])):
        try:
            payload = load_artifact(row["output_path"])
            meta, metrics = payload["meta"], payload["metrics"]
            if meta.get("experiment_id") != EXPERIMENT_ID or meta.get("dataset") != "thubenchmark" or meta.get("model") != "eegnet" or meta.get("seed") != row["seed"] or meta.get("rank") != row["rank"] or meta.get("pgd_steps") != 10 or abs(float(meta.get("pgd_alpha", -1)) - 0.006) > 1e-12 or abs(float(meta.get("eps", -1)) - 0.03) > 1e-12:
                raise ValueError("BPDA protocol mismatch")
            for key in ("purified_clean_accuracy", "bpda_purified_adv_accuracy"):
                if not 0 <= float(metrics[key]) <= 1:
                    raise ValueError(f"{key} outside [0,1]")
            bpda_rows.append({"seed": row["seed"], "rank": row["rank"], **metrics})
        except Exception as exc:
            errors.append(f"bpda {row['task_id']}: {exc}")
        finally:
            if "payload" in locals():
                del payload
            gc.collect()

    aggregate = paired = paired_summary = tnp_delta = tnp_delta_summary = bpda_aggregate = []
    if not errors:
        try:
            aggregate = aggregate_rows(long_rows)
            paired = paired_rows(long_rows)
            paired_summary = aggregate_deltas(paired, "rpcf_at_minus_madry")
            tnp_delta = tnp_minus_raw_rows(long_rows)
            tnp_delta_summary = []
            groups = defaultdict(list)
            for row in tnp_delta:
                groups[(row["method"], row["attack"], row["metric"], str(row["rank"]))].append(row["purified_minus_raw"])
            for (method, attack, metric, rank), values in sorted(groups.items()):
                mean, std = mean_std(values)
                tnp_delta_summary.append({"method": method, "attack": attack, "metric": metric, "rank": rank, "seed_count": len(values), "mean_delta": mean, "sample_std": std, "mean_delta_pp": 100 * mean, "sample_std_pp": 100 * std})
            bpda_aggregate = bpda_summary(bpda_rows)
        except Exception as exc:
            errors.append(f"aggregate: {exc}")

    write_csv(output_dir / "conditions_long.csv", long_rows)
    write_csv(output_dir / "task_audit.csv", audit_rows)
    write_csv(output_dir / "five_seed_mean_std.csv", aggregate)
    write_csv(output_dir / "rpcf_at_minus_madry_paired.csv", paired)
    write_csv(output_dir / "rpcf_at_minus_madry_summary.csv", paired_summary)
    write_csv(output_dir / "tnp_minus_raw_paired.csv", tnp_delta)
    write_csv(output_dir / "tnp_minus_raw_summary.csv", tnp_delta_summary)
    write_csv(output_dir / "bpda.csv", bpda_rows)
    write_csv(output_dir / "bpda_summary.csv", bpda_aggregate)
    if not errors:
        (output_dir / "summary.md").write_text(
            render_markdown(aggregate, paired_summary, tnp_delta_summary, bpda_aggregate), encoding="utf-8"
        )
    completeness = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": args.run_id,
        "scope": "thubenchmark/eegnet/fold0/seeds42-46",
        "scope_completed": not errors,
        "full_exp031_completed": False,
        "errors": errors,
        "planned_counts": dict(counts),
        "expected_counts": FOCUS_COUNTS,
        "rpcf_training_history_count": len(rpcf_tasks),
        "long_row_count": len(long_rows),
        "five_seed_group_count": len(aggregate),
        "paired_delta_count": len(paired),
        "tnp_minus_raw_count": len(tnp_delta),
        "bpda_row_count": len(bpda_rows),
    }
    (output_dir / "completeness.json").write_text(json.dumps(completeness, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(f"focused EXP-031 summary incomplete: {len(errors)} errors")
    print(output_dir / "completeness.json")


if __name__ == "__main__":
    main()
