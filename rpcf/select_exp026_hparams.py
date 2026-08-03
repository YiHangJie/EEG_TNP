import argparse
import csv
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="按三代表 backbone 的 holdout/PGD delta 选择 EXP-026 全局参数。"
    )
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--models", nargs="+", default=["eegnet", "tsception", "conformer"]
    )
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def final_metrics(payload, path):
    history = payload.get("history") or []
    if not history:
        raise ValueError(f"History is empty: {path}")
    row = history[-1]
    required = {
        "robust_acc",
        "clean_acc",
        "holdout_rank25_30_adv_pur_acc",
        "holdout_rank25_30_clean_pur_acc",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"History lacks screening metrics {missing}: {path}")
    feature = payload.get("feature_alignment") or {}
    return {
        "objective": feature.get("objective", "none"),
        "weight": float(feature.get("loss_weight", 0.0)),
        "margin": float(feature.get("prototype_margin", 1.0)),
        "margin_weight": float(feature.get("prototype_margin_weight", 0.0)),
        "temperature": float(feature.get("contrastive_temperature", 0.1)),
        "robust_acc": float(row["robust_acc"]),
        "clean_acc": float(row["clean_acc"]),
        "holdout_adv_pur": float(row["holdout_rank25_30_adv_pur_acc"]),
        "holdout_clean_pur": float(row["holdout_rank25_30_clean_pur_acc"]),
        "path": str(path),
    }


def main():
    args = parse_args()
    root = Path(args.root)
    by_model = {}
    for model in args.models:
        configs = {}
        for path in sorted((root / model).glob("*/finetune.json")):
            configs[path.parent.name] = final_metrics(load_json(path), path)
        if "control" not in configs:
            raise FileNotFoundError(f"Missing balanced control history for {model}.")
        by_model[model] = configs

    config_ids = set.intersection(*(set(configs) for configs in by_model.values()))
    rows = []
    for config_id in sorted(config_ids):
        model_rows = [by_model[model][config_id] for model in args.models]
        objective = model_rows[0]["objective"]
        if any(row["objective"] != objective for row in model_rows):
            raise ValueError(f"Objective mismatch for config {config_id}.")
        deltas = []
        for model, row in zip(args.models, model_rows):
            control = by_model[model]["control"]
            deltas.append(
                {
                    "model": model,
                    "delta_adv_pur": row["holdout_adv_pur"]
                    - control["holdout_adv_pur"],
                    "delta_robust": row["robust_acc"] - control["robust_acc"],
                    "delta_clean_pur": row["holdout_clean_pur"]
                    - control["holdout_clean_pur"],
                    "delta_clean": row["clean_acc"] - control["clean_acc"],
                }
            )
        count = len(deltas)
        aggregate = {
            key: sum(item[key] for item in deltas) / count
            for key in (
                "delta_adv_pur",
                "delta_robust",
                "delta_clean_pur",
                "delta_clean",
            )
        }
        aggregate["score"] = 0.5 * aggregate["delta_adv_pur"] + 0.5 * aggregate[
            "delta_robust"
        ]
        rows.append(
            {
                "config_id": config_id,
                "objective": objective,
                "weight": model_rows[0]["weight"],
                "margin": model_rows[0]["margin"],
                "margin_weight": model_rows[0]["margin_weight"],
                "temperature": model_rows[0]["temperature"],
                **aggregate,
                "per_model": deltas,
            }
        )

    selected = {}
    for objective in ("cmmd", "prototype", "contrastive"):
        candidates = [row for row in rows if row["objective"] == objective]
        if not candidates:
            raise ValueError(f"No screening candidates found for {objective}.")
        winner = max(
            candidates,
            key=lambda row: (
                row["score"],
                row["delta_adv_pur"],
                row["delta_clean_pur"],
                row["delta_clean"],
                -row["weight"],
                -row["margin_weight"],
            ),
        )
        selected[objective] = winner

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)
    with open(output_dir / "screening_candidates.csv", "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "config_id",
            "objective",
            "weight",
            "margin",
            "margin_weight",
            "temperature",
            "score",
            "delta_adv_pur",
            "delta_robust",
            "delta_clean_pur",
            "delta_clean",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    payload = {
        "kind": "exp026_hparam_selection",
        "experiment_id": "EXP-026",
        "models": args.models,
        "score_rule": "mean_models(0.5*delta_holdout_rank25_30_adv_pur + 0.5*delta_val_pgd)",
        "tie_break": [
            "delta_holdout_adv_pur",
            "delta_holdout_clean_pur",
            "delta_val_clean",
            "smaller_weight",
            "margin_off",
        ],
        "selected": selected,
        "candidates": rows,
    }
    with open(output_dir / "selected_hparams.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    with open(output_dir / "selected_hparams.env", "w", encoding="utf-8") as file:
        for objective, prefix in (
            ("cmmd", "CMMD"),
            ("prototype", "PROTOTYPE"),
            ("contrastive", "CONTRASTIVE"),
        ):
            row = selected[objective]
            file.write(f"{prefix}_WEIGHT={row['weight']}\n")
            file.write(f"{prefix}_MARGIN={row['margin']}\n")
            file.write(f"{prefix}_MARGIN_WEIGHT={row['margin_weight']}\n")
            file.write(f"{prefix}_TEMPERATURE={row['temperature']}\n")
    print(output_dir / "selected_hparams.json")


if __name__ == "__main__":
    main()
