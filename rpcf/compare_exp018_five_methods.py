import argparse
import csv
import json
import os


METHOD_SPECS = (
    ("madry_at", "Madry AT", "base", "at_tnp"),
    ("consistancy", "consistancy", "base", "consistancy"),
    ("rpcf_selective", "RPCF selective", "selective", "rpcf"),
    ("rpcf_all_layers", "RPCF all-layers", "all_layers", "rpcf"),
    (
        "rpcf_rank_weight_uniform",
        "RPCF rank-weight uniform",
        "uniform",
        "rpcf",
    ),
)
PROTOCOL_KEYS = (
    "kind",
    "dataset",
    "model",
    "seed",
    "fold",
    "eps",
    "attack",
    "ranks",
    "sample_num",
)
LONG_FIELDS = (
    "method_id",
    "method",
    "rank",
    "full_test_sample_num",
    "full_test_clean_accuracy",
    "full_test_autoattack_accuracy",
    "full_test_attack_mse",
    "purification_sample_num",
    "subset_clean_accuracy",
    "subset_adv_accuracy",
    "subset_attack_mse",
    "purified_clean_accuracy",
    "purified_adv_accuracy",
    "mean_clean_mse",
    "mean_adv_mse",
    "source_summary",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="汇总 EXP-018 的五种方法，不重新运行训练、攻击或净化。"
    )
    parser.add_argument(
        "--base_summary",
        default="logs/exp018/exp018_full_20260612_124131/summary.json",
    )
    parser.add_argument(
        "--selective_summary",
        default=(
            "logs/exp018/exp018_rpcf_no_early_stop_20260614_2357/"
            "summary.json"
        ),
    )
    parser.add_argument(
        "--all_layers_summary",
        default=(
            "logs/exp018/exp018_rpcf_all_layers_20260615_0933/"
            "summary.json"
        ),
    )
    parser.add_argument(
        "--uniform_summary",
        default=(
            "logs/exp018/exp018_rpcf_static_ranks_20260615_1155/"
            "summary.json"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="logs/exp018/exp018_five_method_comparison",
    )
    parser.add_argument(
        "--experiment_id",
        default="EXP-018",
        help="写入汇总 JSON 的实验编号，例如 EXP-019。",
    )
    return parser.parse_args()


def load_summary(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"EXP-018 summary not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    validate_summary_payload(payload, path)
    return payload


def validate_summary_payload(payload, path):
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    missing = [key for key in PROTOCOL_KEYS if key not in payload]
    if missing:
        raise ValueError(f"{path} missing protocol fields: {missing}")
    if not str(payload["kind"]).endswith("_fair_comparison"):
        raise ValueError(f"{path} has unsupported kind={payload['kind']!r}.")
    ranks = [int(rank) for rank in payload["ranks"]]
    if not ranks or len(set(ranks)) != len(ranks):
        raise ValueError(f"{path} has invalid ranks={ranks}.")
    source_indices = payload.get("source_indices")
    if not isinstance(source_indices, list):
        raise ValueError(f"{path} missing source_indices list.")
    if len(source_indices) != int(payload["sample_num"]):
        raise ValueError(
            f"{path} source_indices length does not match sample_num."
        )
    if len(set(int(index) for index in source_indices)) != len(source_indices):
        raise ValueError(f"{path} source_indices contains duplicates.")
    for field in ("rows", "full_test_attack"):
        if not isinstance(payload.get(field), list):
            raise ValueError(f"{path} missing {field} list.")
    for row in payload["rows"]:
        if int(row.get("sample_num", -1)) != int(payload["sample_num"]):
            raise ValueError(
                f"{path} contains a purification row with inconsistent "
                "sample_num."
            )
        if int(row.get("rank", -1)) not in ranks:
            raise ValueError(f"{path} contains an unexpected rank row.")
    full_test_sizes = {
        int(row.get("sample_num", -1)) for row in payload["full_test_attack"]
    }
    if len(full_test_sizes) != 1 or next(iter(full_test_sizes)) <= 0:
        raise ValueError(
            f"{path} full_test_attack rows have inconsistent sample_num."
        )


def _protocol_value(payload, key):
    value = payload[key]
    if key == "ranks":
        return tuple(int(rank) for rank in value)
    if key == "eps":
        return float(value)
    return value


def _rows_by_key(rows, key_fields, path, field):
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path} contains a non-object {field} row.")
        try:
            key = tuple(row[name] for name in key_fields)
        except KeyError as error:
            raise ValueError(
                f"{path} {field} row missing field: {error.args[0]}"
            ) from error
        if key in indexed:
            raise ValueError(f"{path} contains duplicate {field} key={key}.")
        indexed[key] = row
    return indexed


def validate_shared_protocol(summaries, paths):
    reference_name = "base"
    reference = summaries[reference_name]
    for name, payload in summaries.items():
        for key in PROTOCOL_KEYS:
            actual = _protocol_value(payload, key)
            expected = _protocol_value(reference, key)
            if actual != expected:
                raise ValueError(
                    f"{paths[name]} protocol mismatch for {key}: "
                    f"{actual!r} != {expected!r}."
                )
        if [int(index) for index in payload["source_indices"]] != [
            int(index) for index in reference["source_indices"]
        ]:
            raise ValueError(
                f"{paths[name]} source_indices do not match "
                f"{paths[reference_name]}."
            )

    base_rows = _rows_by_key(
        reference["rows"], ("method", "rank"), paths["base"], "rows"
    )
    base_full = _rows_by_key(
        reference["full_test_attack"],
        ("method",),
        paths["base"],
        "full_test_attack",
    )
    for name in ("selective", "all_layers", "uniform"):
        current_rows = _rows_by_key(
            summaries[name]["rows"],
            ("method", "rank"),
            paths[name],
            "rows",
        )
        current_full = _rows_by_key(
            summaries[name]["full_test_attack"],
            ("method",),
            paths[name],
            "full_test_attack",
        )
        for method in ("at_tnp", "consistancy"):
            for rank in reference["ranks"]:
                key = (method, rank)
                if current_rows.get(key) != base_rows.get(key):
                    raise ValueError(
                        f"{paths[name]} baseline row {key} differs from "
                        f"{paths['base']}."
                    )
            key = (method,)
            if current_full.get(key) != base_full.get(key):
                raise ValueError(
                    f"{paths[name]} full-test baseline {method} differs from "
                    f"{paths['base']}."
                )
        rpcf_full = current_full.get(("rpcf",))
        expected_full_size = int(base_full[("at_tnp",)]["sample_num"])
        if (
            rpcf_full is None
            or int(rpcf_full["sample_num"]) != expected_full_size
        ):
            raise ValueError(
                f"{paths[name]} RPCF full-test sample_num does not match "
                f"{paths['base']}."
            )

    _validate_rpcf_variant(summaries["selective"], "selective", paths["selective"])
    _validate_rpcf_variant(summaries["all_layers"], "all_layers", paths["all_layers"])
    _validate_rpcf_variant(summaries["uniform"], "uniform", paths["uniform"])


def _validate_rpcf_variant(payload, variant, path):
    rpcf = payload.get("rpcf")
    if not isinstance(rpcf, dict):
        raise ValueError(f"{path} missing rpcf metadata.")
    all_layers = bool(rpcf.get("all_layers", False))
    static_weights = bool(rpcf.get("static_rank_weights", False))
    if variant == "selective" and (all_layers or static_weights):
        raise ValueError(
            f"{path} is not selective RPCF with dynamic rank schedule."
        )
    if variant == "all_layers" and not all_layers:
        raise ValueError(f"{path} is not an RPCF all-layers run.")
    if variant == "uniform" and (all_layers or not static_weights):
        raise ValueError(
            f"{path} is not selective RPCF with uniform rank weights."
        )


def build_five_method_rows(summaries, paths):
    long_rows = []
    for method_id, display_name, source_name, source_method in METHOD_SPECS:
        payload = summaries[source_name]
        summary_rows = _rows_by_key(
            payload["rows"],
            ("method", "rank"),
            paths[source_name],
            "rows",
        )
        full_rows = _rows_by_key(
            payload["full_test_attack"],
            ("method",),
            paths[source_name],
            "full_test_attack",
        )
        full_key = (source_method,)
        if full_key not in full_rows:
            raise ValueError(
                f"{paths[source_name]} missing full-test method={source_method}."
            )
        full = full_rows[full_key]
        for rank in payload["ranks"]:
            row_key = (source_method, rank)
            if row_key not in summary_rows:
                raise ValueError(
                    f"{paths[source_name]} missing method/rank={row_key}."
                )
            row = summary_rows[row_key]
            long_rows.append(
                {
                    "method_id": method_id,
                    "method": display_name,
                    "rank": int(rank),
                    "full_test_sample_num": int(full["sample_num"]),
                    "full_test_clean_accuracy": float(full["clean_accuracy"]),
                    "full_test_autoattack_accuracy": float(full["adv_accuracy"]),
                    "full_test_attack_mse": float(full["attack_mse"]),
                    "purification_sample_num": int(row["sample_num"]),
                    "subset_clean_accuracy": float(row["clean_accuracy"]),
                    "subset_adv_accuracy": float(row["adv_accuracy"]),
                    "subset_attack_mse": float(row["attack_mse"]),
                    "purified_clean_accuracy": float(
                        row["purified_clean_accuracy"]
                    ),
                    "purified_adv_accuracy": float(
                        row["purified_adv_accuracy"]
                    ),
                    "mean_clean_mse": float(row["mean_clean_mse"]),
                    "mean_adv_mse": float(row["mean_adv_mse"]),
                    "source_summary": paths[source_name],
                }
            )
    return long_rows


def build_wide_rows(long_rows, ranks):
    grouped = {}
    for row in long_rows:
        method_id = row["method_id"]
        if method_id not in grouped:
            grouped[method_id] = {
                "method_id": method_id,
                "method": row["method"],
                "full_test_sample_num": row["full_test_sample_num"],
                "full_test_clean_accuracy": row["full_test_clean_accuracy"],
                "full_test_autoattack_accuracy": row[
                    "full_test_autoattack_accuracy"
                ],
            }
        rank = int(row["rank"])
        grouped[method_id][
            f"rank{rank}_purified_clean_accuracy"
        ] = row["purified_clean_accuracy"]
        grouped[method_id][
            f"rank{rank}_purified_adv_accuracy"
        ] = row["purified_adv_accuracy"]

    ordered = []
    for method_id, _, _, _ in METHOD_SPECS:
        row = grouped[method_id]
        for rank in ranks:
            for metric in ("clean", "adv"):
                field = f"rank{rank}_purified_{metric}_accuracy"
                if field not in row:
                    raise ValueError(f"Wide table missing field={field}.")
        ordered.append(row)
    return ordered


def render_markdown(wide_rows, ranks):
    headers = ["Method", "Full clean", "Full AutoAttack"]
    for rank in ranks:
        headers.extend([f"Rank {rank} clean", f"Rank {rank} adversarial"])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for row in wide_rows:
        values = [
            row["method"],
            _format_percent(row["full_test_clean_accuracy"]),
            _format_percent(row["full_test_autoattack_accuracy"]),
        ]
        for rank in ranks:
            values.extend(
                [
                    _format_percent(
                        row[f"rank{rank}_purified_clean_accuracy"]
                    ),
                    _format_percent(
                        row[f"rank{rank}_purified_adv_accuracy"]
                    ),
                ]
            )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _format_percent(value):
    return f"{100.0 * float(value):.2f}%"


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir, summaries, paths, experiment_id="EXP-018"):
    validate_shared_protocol(summaries, paths)
    ranks = [int(rank) for rank in summaries["base"]["ranks"]]
    long_rows = build_five_method_rows(summaries, paths)
    wide_rows = build_wide_rows(long_rows, ranks)
    wide_fields = list(wide_rows[0])

    os.makedirs(output_dir, exist_ok=True)
    long_path = os.path.join(output_dir, "five_methods_long.csv")
    wide_path = os.path.join(output_dir, "five_methods_wide.csv")
    markdown_path = os.path.join(output_dir, "five_methods_table.md")
    json_path = os.path.join(output_dir, "five_methods_summary.json")
    write_csv(long_path, LONG_FIELDS, long_rows)
    write_csv(wide_path, wide_fields, wide_rows)
    with open(markdown_path, "w", encoding="utf-8") as file:
        file.write(render_markdown(wide_rows, ranks))
    with open(json_path, "w", encoding="utf-8") as file:
        experiment_token = str(experiment_id).strip().lower().replace("-", "")
        json.dump(
            {
                "kind": f"{experiment_token}_five_method_comparison",
                "experiment_id": str(experiment_id),
                "protocol": {
                    key: summaries["base"][key] for key in PROTOCOL_KEYS
                },
                "source_indices": summaries["base"]["source_indices"],
                "sources": paths,
                "method_order": [
                    {"method_id": method_id, "method": display_name}
                    for method_id, display_name, _, _ in METHOD_SPECS
                ],
                "long_rows": long_rows,
                "wide_rows": wide_rows,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")
    return {
        "long_csv": long_path,
        "wide_csv": wide_path,
        "markdown": markdown_path,
        "json": json_path,
    }


def main():
    args = parse_args()
    paths = {
        "base": args.base_summary,
        "selective": args.selective_summary,
        "all_layers": args.all_layers_summary,
        "uniform": args.uniform_summary,
    }
    summaries = {name: load_summary(path) for name, path in paths.items()}
    outputs = write_outputs(
        args.output_dir,
        summaries,
        paths,
        experiment_id=args.experiment_id,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
