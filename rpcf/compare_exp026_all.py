import argparse
import csv
import json
import os


METHODS = ("cmmd", "prototype", "contrastive")


def parse_args():
    parser = argparse.ArgumentParser(description="汇总 EXP-026 六 backbone 结果。")
    parser.add_argument("--root", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    args = parse_args()
    summaries = {}
    for model in args.models:
        path = os.path.join(args.root, model, "comparison", "summary.json")
        payload = load_json(path)
        if payload.get("kind") != "exp026_backbone_comparison":
            raise ValueError(f"Unexpected summary kind: {path}")
        for key, expected in (
            ("model", model),
            ("dataset", "thubenchmark"),
            ("seed", 42),
            ("fold", 0),
            ("eps", 0.03),
            ("sample_num", 512),
        ):
            if str(payload.get(key)) != str(expected):
                raise ValueError(f"Protocol mismatch {key} in {path}.")
        summaries[model] = payload

    rows = []
    for model, payload in summaries.items():
        full = {row["method"]: row for row in payload["full_test_attack"]}
        pur = {}
        for row in payload["purification"]:
            pur.setdefault(row["method"], []).append(row)
        decisions = {row["method"]: row for row in payload["decision"]}
        for method in ("balanced_control", *METHODS):
            rank_rows = sorted(pur[method], key=lambda row: row["rank"])
            decision = decisions.get(method, {})
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "full_clean_accuracy": full[method]["clean_accuracy"],
                    "full_autoattack_accuracy": full[method]["adv_accuracy"],
                    "rank25_purified_clean_accuracy": rank_rows[0][
                        "purified_clean_accuracy"
                    ],
                    "rank25_purified_adv_accuracy": rank_rows[0][
                        "purified_adv_accuracy"
                    ],
                    "rank30_purified_clean_accuracy": rank_rows[1][
                        "purified_clean_accuracy"
                    ],
                    "rank30_purified_adv_accuracy": rank_rows[1][
                        "purified_adv_accuracy"
                    ],
                    "delta_purified_adv_vs_control": decision.get(
                        "delta_purified_adv_vs_control", 0.0
                    ),
                    "delta_full_autoattack_vs_control": decision.get(
                        "delta_full_autoattack_vs_control", 0.0
                    ),
                    "passes_predefined_rule": decision.get(
                        "passes_predefined_rule"
                    ),
                }
            )

    aggregate = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        aggregate.append(
            {
                "method": method,
                "backbone_count": len(method_rows),
                "mean_delta_purified_adv_vs_control": sum(
                    row["delta_purified_adv_vs_control"] for row in method_rows
                )
                / len(method_rows),
                "mean_delta_full_autoattack_vs_control": sum(
                    row["delta_full_autoattack_vs_control"] for row in method_rows
                )
                / len(method_rows),
                "passing_backbone_count": sum(
                    bool(row["passes_predefined_rule"]) for row in method_rows
                ),
            }
        )

    os.makedirs(args.output_dir, exist_ok=True)
    for name, values in (("all_backbones.csv", rows), ("aggregate.csv", aggregate)):
        with open(os.path.join(args.output_dir, name), "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(
            {
                "kind": "exp026_all_backbone_comparison",
                "experiment_id": "EXP-026",
                "models": args.models,
                "rows": rows,
                "aggregate": aggregate,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    with open(os.path.join(args.output_dir, "comparison.md"), "w", encoding="utf-8") as file:
        file.write("| Method | Mean Δ pur adv | Mean Δ full AA | Passed backbones |\n")
        file.write("| --- | ---: | ---: | ---: |\n")
        for row in aggregate:
            file.write(
                f"| {row['method']} | {100*row['mean_delta_purified_adv_vs_control']:+.2f} pp | "
                f"{100*row['mean_delta_full_autoattack_vs_control']:+.2f} pp | "
                f"{row['passing_backbone_count']}/{row['backbone_count']} |\n"
            )
    print(os.path.join(args.output_dir, "comparison.md"))


if __name__ == "__main__":
    main()
