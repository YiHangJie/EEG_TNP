import argparse
import csv
import json
from pathlib import Path


METHOD_MAP = {
    "at_tnp": "madry_at",
    "consistancy": "consistancy_six_rank",
    "rpcf": "rpcf_selective",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="汇总 EXP-020 普通 RPCF 与 EXP-022 RPCF_AT 的公平比较结果。"
    )
    parser.add_argument("--base_summary", required=True)
    parser.add_argument("--rpcf_at_summary", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_summary(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_protocol(base, rpcf_at):
    for key in ("dataset", "model", "seed", "fold", "eps", "ranks", "sample_num"):
        if base.get(key) != rpcf_at.get(key):
            raise ValueError(
                f"Protocol mismatch for {key}: {base.get(key)!r} != "
                f"{rpcf_at.get(key)!r}"
            )
    if base.get("source_indices") != rpcf_at.get("source_indices"):
        raise ValueError("Purification source_indices mismatch.")


def select_full_rows(summary, methods):
    rows = []
    for row in summary["full_test_attack"]:
        if row["method"] not in methods:
            continue
        item = dict(row)
        item["method"] = methods[row["method"]]
        rows.append(item)
    return rows


def select_purification_rows(summary, methods):
    rows = []
    for row in summary["rows"]:
        if row["method"] not in methods:
            continue
        item = dict(row)
        item["method"] = methods[row["method"]]
        rows.append(item)
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, full_rows, purification_rows):
    full_map = {row["method"]: row for row in full_rows}
    pur_map = {
        (row["method"], int(row["rank"])): row for row in purification_rows
    }
    methods = (
        "madry_at",
        "consistancy_six_rank",
        "rpcf_selective",
        "rpcf_at",
    )
    lines = [
        "| Method | Full clean | Full AutoAttack | Rank25 clean | "
        "Rank25 adversarial | Rank30 clean | Rank30 adversarial |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        full = full_map[method]
        rank25 = pur_map[(method, 25)]
        rank30 = pur_map[(method, 30)]
        values = [
            method,
            f"{100 * full['clean_accuracy']:.2f}%",
            f"{100 * full['adv_accuracy']:.2f}%",
            f"{100 * rank25['purified_clean_accuracy']:.2f}%",
            f"{100 * rank25['purified_adv_accuracy']:.2f}%",
            f"{100 * rank30['purified_clean_accuracy']:.2f}%",
            f"{100 * rank30['purified_adv_accuracy']:.2f}%",
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    base = load_summary(args.base_summary)
    rpcf_at = load_summary(args.rpcf_at_summary)
    validate_protocol(base, rpcf_at)

    full_rows = select_full_rows(base, METHOD_MAP)
    purification_rows = select_purification_rows(base, METHOD_MAP)
    full_rows += select_full_rows(rpcf_at, {"rpcf": "rpcf_at"})
    purification_rows += select_purification_rows(
        rpcf_at, {"rpcf": "rpcf_at"}
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_csv = output_dir / "full_test_attack.csv"
    purification_csv = output_dir / "purification.csv"
    markdown = output_dir / "comparison.md"
    summary_json = output_dir / "summary.json"

    write_csv(
        full_csv,
        full_rows,
        ["method", "sample_num", "clean_accuracy", "adv_accuracy", "attack_mse"],
    )
    write_csv(
        purification_csv,
        purification_rows,
        [
            "method",
            "rank",
            "sample_num",
            "clean_accuracy",
            "adv_accuracy",
            "attack_mse",
            "purified_clean_accuracy",
            "purified_adv_accuracy",
            "mean_clean_mse",
            "mean_adv_mse",
        ],
    )
    write_markdown(markdown, full_rows, purification_rows)

    payload = {
        "kind": "exp022_rpcf_at_comparison",
        "dataset": base["dataset"],
        "model": base["model"],
        "seed": base["seed"],
        "fold": base["fold"],
        "eps": base["eps"],
        "ranks": base["ranks"],
        "sample_num": base["sample_num"],
        "source_indices": base["source_indices"],
        "full_test_attack": full_rows,
        "rows": purification_rows,
        "artifacts": {
            "base_summary": args.base_summary,
            "rpcf_at_summary": args.rpcf_at_summary,
        },
    }
    summary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(markdown)
    print(summary_json)


if __name__ == "__main__":
    main()
