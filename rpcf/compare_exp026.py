import argparse
import csv
import gc
import json
import os

import torch

from utils.experiment_artifacts import torch_load_cpu


METHOD_ORDER = ("balanced_control", "cmmd", "prototype", "contrastive")


def parse_args():
    parser = argparse.ArgumentParser(description="严格汇总 EXP-026 单 backbone 结果。")
    parser.add_argument("--dataset", default="thubenchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--sample_num", type=int, default=512)
    parser.add_argument("--attack_paths", nargs="+", required=True)
    parser.add_argument("--purification_paths", nargs="+", required=True)
    parser.add_argument("--history_paths", nargs="+", required=True)
    parser.add_argument("--sensitivity_path", required=True)
    parser.add_argument("--context_summary", default=None)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def parse_pairs(items, expected_methods=METHOD_ORDER):
    pairs = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected method=path, got {item!r}.")
        method, path = item.split("=", 1)
        if method in pairs:
            raise ValueError(f"Duplicate method: {method}.")
        pairs[method] = path
    if set(pairs) != set(expected_methods):
        raise ValueError(
            f"Methods must be {list(expected_methods)}, got {sorted(pairs)}."
        )
    return pairs


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_common(meta, path, args, kind):
    if meta.get("kind") != kind:
        raise ValueError(f"Unexpected kind in {path}: {meta.get('kind')!r}.")
    for key, expected in (
        ("dataset", args.dataset),
        ("model", args.model),
        ("seed", args.seed),
        ("fold", args.fold),
    ):
        if str(meta.get(key)) != str(expected):
            raise ValueError(
                f"{path}: {key}={meta.get(key)!r}, expected {expected!r}."
            )
    if abs(float(meta.get("eps")) - float(args.eps)) > 1e-12:
        raise ValueError(f"{path}: eps mismatch.")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def context_rows(payload, model):
    if not payload:
        return []
    if payload.get("kind") == "exp024_backbone_comparison":
        rows = []
        for row in payload.get("full_test_attack", []):
            if row.get("method") in ("madry_at", "rpcf_at"):
                rows.append(
                    {
                        "source": "EXP-024",
                        "method": row["method"],
                        "rank": None,
                        "clean_accuracy": row["clean_accuracy"],
                        "adv_accuracy": row["adv_accuracy"],
                    }
                )
        for row in payload.get("purification", []):
            if row.get("method") in ("madry_at", "rpcf_at"):
                rows.append(
                    {
                        "source": "EXP-024",
                        "method": row["method"],
                        "rank": row["rank"],
                        "purified_clean_accuracy": row["purified_clean_accuracy"],
                        "purified_adv_accuracy": row["purified_adv_accuracy"],
                    }
                )
        return rows
    if payload.get("kind") == "exp025_layer_prefix_comparison" and model == "eegnet":
        return [
            {
                "source": "EXP-025",
                "method": "rpcf_at_top40_budget2",
                "rank": row["rank"],
                "clean_accuracy": row["clean_accuracy"],
                "adv_accuracy": row["adv_accuracy"],
                "purified_clean_accuracy": row["purified_clean_accuracy"],
                "purified_adv_accuracy": row["purified_adv_accuracy"],
            }
            for row in payload.get("rows", [])
            if row.get("budget_id") == "budget_2"
        ]
    return []


def main():
    args = parse_args()
    attacks = parse_pairs(args.attack_paths)
    purifications = parse_pairs(args.purification_paths)
    histories = parse_pairs(args.history_paths)
    os.makedirs(args.output_dir, exist_ok=True)

    history_payloads = {}
    for method in METHOD_ORDER:
        payload = load_json(histories[method])
        for key, expected in (
            ("dataset", args.dataset),
            ("model", args.model),
            ("seed", args.seed),
            ("fold", args.fold),
        ):
            if str(payload.get(key)) != str(expected):
                raise ValueError(f"History mismatch {key}: {histories[method]}")
        sampler = payload.get("cache_sampler") or {}
        if sampler.get("kind") != "class_balanced":
            raise ValueError(f"EXP-026 history is not class-balanced: {histories[method]}")
        expected_objective = "none" if method == "balanced_control" else method
        feature = payload.get("feature_alignment") or {}
        if feature.get("objective") != expected_objective:
            raise ValueError(f"Objective mismatch for {method}: {feature}")
        history_payloads[method] = payload

    full_rows = []
    for method in METHOD_ORDER:
        payload = torch_load_cpu(attacks[method])
        meta = payload.get("meta") or {}
        validate_common(meta, attacks[method], args, "rpcf_attack_eval")
        expected_checkpoint = history_payloads[method]["output_checkpoint"]
        if os.path.normpath(meta.get("checkpoint_path", "")) != os.path.normpath(
            expected_checkpoint
        ):
            raise ValueError(f"Attack checkpoint mismatch for {method}.")
        full_rows.append(
            {
                "method": method,
                "sample_num": int(meta["sample_num"]),
                "clean_accuracy": float(meta["clean_accuracy"]),
                "adv_accuracy": float(meta["adv_accuracy"]),
                "attack_mse": float(meta["attack_mse"]),
                "path": attacks[method],
            }
        )
        del payload
        gc.collect()

    purification_rows = []
    shared_indices = None
    shared_labels = None
    for method in METHOD_ORDER:
        payload = torch_load_cpu(purifications[method])
        meta = payload.get("meta") or {}
        validate_common(meta, purifications[method], args, "rpcf_purification_eval")
        if int(meta.get("sample_num")) != args.sample_num:
            raise ValueError(f"Purification sample_num mismatch for {method}.")
        if [int(rank) for rank in payload.get("ranks", [])] != [25, 30]:
            raise ValueError(f"Purification ranks mismatch for {method}.")
        indices = [int(index) for index in payload.get("source_indices", [])]
        labels = torch.as_tensor(payload.get("labels")).cpu().long()
        if shared_indices is None:
            shared_indices = indices
            shared_labels = labels
        elif indices != shared_indices or not torch.equal(labels, shared_labels):
            raise ValueError(f"Purification subset/labels mismatch for {method}.")
        attack_meta = meta.get("attack_meta") or {}
        expected_checkpoint = history_payloads[method]["output_checkpoint"]
        if attack_meta and os.path.normpath(
            attack_meta.get("checkpoint_path", "")
        ) != os.path.normpath(expected_checkpoint):
            raise ValueError(f"Purification attack checkpoint mismatch for {method}.")
        for metric in payload.get("metrics", []):
            purification_rows.append(
                {
                    "method": method,
                    "rank": int(metric["rank"]),
                    "sample_num": args.sample_num,
                    "purified_clean_accuracy": float(
                        metric["purified_clean_accuracy"]
                    ),
                    "purified_adv_accuracy": float(metric["purified_adv_accuracy"]),
                    "mean_clean_mse": float(metric["mean_clean_mse"]),
                    "mean_adv_mse": float(metric["mean_adv_mse"]),
                    "path": purifications[method],
                }
            )
        del payload
        gc.collect()

    full_by_method = {row["method"]: row for row in full_rows}
    pur_by_method = {
        method: [row for row in purification_rows if row["method"] == method]
        for method in METHOD_ORDER
    }
    control_full = full_by_method["balanced_control"]
    control_pur_adv = sum(
        row["purified_adv_accuracy"] for row in pur_by_method["balanced_control"]
    ) / 2
    decisions = []
    for method in ("cmmd", "prototype", "contrastive"):
        mean_pur_adv = sum(
            row["purified_adv_accuracy"] for row in pur_by_method[method]
        ) / 2
        delta_pur_adv = mean_pur_adv - control_pur_adv
        delta_full_adv = (
            full_by_method[method]["adv_accuracy"] - control_full["adv_accuracy"]
        )
        decisions.append(
            {
                "method": method,
                "mean_rank25_30_purified_adv_accuracy": mean_pur_adv,
                "delta_purified_adv_vs_control": delta_pur_adv,
                "delta_full_autoattack_vs_control": delta_full_adv,
                "passes_predefined_rule": (
                    delta_pur_adv >= 0.005 and delta_full_adv >= -0.01
                ),
            }
        )

    context_payload = (
        load_json(args.context_summary)
        if args.context_summary and os.path.exists(args.context_summary)
        else None
    )
    context = context_rows(context_payload, args.model)
    write_csv(os.path.join(args.output_dir, "full_test_attack.csv"), full_rows)
    write_csv(os.path.join(args.output_dir, "purification.csv"), purification_rows)
    write_csv(os.path.join(args.output_dir, "decision.csv"), decisions)
    sensitivity = load_json(args.sensitivity_path)
    summary = {
        "kind": "exp026_backbone_comparison",
        "experiment_id": "EXP-026",
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "fold": args.fold,
        "eps": args.eps,
        "sample_num": args.sample_num,
        "source_indices": shared_indices,
        "full_test_attack": full_rows,
        "purification": purification_rows,
        "decision": decisions,
        "selected_layers": sensitivity.get("selected_layers"),
        "histories": {method: histories[method] for method in METHOD_ORDER},
        "context_summary": args.context_summary,
        "context": context,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "comparison.md"), "w", encoding="utf-8") as file:
        file.write("| Method | Full clean | Full AA | Rank25 pur adv | Rank30 pur adv | Pass |\n")
        file.write("| --- | ---: | ---: | ---: | ---: | --- |\n")
        decision_by_method = {row["method"]: row for row in decisions}
        for method in METHOD_ORDER:
            rank_rows = sorted(pur_by_method[method], key=lambda row: row["rank"])
            passed = decision_by_method.get(method, {}).get("passes_predefined_rule")
            pass_text = "baseline" if passed is None else ("yes" if passed else "no")
            file.write(
                f"| {method} | {100*full_by_method[method]['clean_accuracy']:.2f}% | "
                f"{100*full_by_method[method]['adv_accuracy']:.2f}% | "
                f"{100*rank_rows[0]['purified_adv_accuracy']:.2f}% | "
                f"{100*rank_rows[1]['purified_adv_accuracy']:.2f}% | {pass_text} |\n"
            )
    print(os.path.join(args.output_dir, "comparison.md"))


if __name__ == "__main__":
    main()
