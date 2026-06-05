import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/eegap_matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline paired-delta analysis for rank-growth frequency metrics."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, default="rank_growth_pair_delta_bootstrap")
    parser.add_argument("--bootstrap_iters", type=int, default=5000)
    parser.add_argument("--ci", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot_format", type=str, default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--plot_dpi", type=int, default=180)
    return parser.parse_args()


def torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_csv_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def logits_margin(logits, label):
    """计算 true-class logit 和最强错误类 logit 的 margin。"""
    logits = logits.detach().cpu().float().view(-1)
    label = int(label)
    true_logit = float(logits[label].item())
    mask = torch.ones(logits.numel(), dtype=torch.bool)
    mask[label] = False
    other_max = float(logits[mask].max().item())
    return true_logit - other_max


def build_trace_lookup(traces):
    lookup = {}
    for source_type, source_traces in traces.items():
        for trace in source_traces:
            sample_id = int(trace["sample_id"])
            rank_records = {}
            for record in trace["eval_records"]:
                rank_records[int(record["rank"])] = record
            lookup[(source_type, sample_id)] = rank_records
    return lookup


def build_enhanced_rows(input_dir):
    predictions = torch_load_cpu(input_dir / "rank_growth_predictions.pt")
    labels = predictions["labels"]
    trace_lookup = build_trace_lookup(predictions["traces"])
    frequency_rows = read_csv_rows(input_dir / "rank_growth_incremental_frequency.csv")

    by_key = defaultdict(dict)
    for row in frequency_rows:
        key = (
            int(row["sample_id"]),
            int(row["source_index"]),
            int(row["rank_prev"]),
            int(row["rank_next"]),
        )
        by_key[key][row["source_type"]] = row

    enhanced_rows = []
    for (sample_id, source_index, rank_prev, rank_next), source_rows in sorted(by_key.items()):
        clean_row = source_rows.get("clean")
        adv_row = source_rows.get("adv")
        if clean_row is None or adv_row is None:
            continue

        label = int(labels[sample_id].item())
        clean_records = trace_lookup[("clean", sample_id)]
        adv_records = trace_lookup[("adv", sample_id)]
        clean_margin_prev = logits_margin(clean_records[rank_prev]["logits"], label)
        clean_margin_next = logits_margin(clean_records[rank_next]["logits"], label)
        adv_margin_prev = logits_margin(adv_records[rank_prev]["logits"], label)
        adv_margin_next = logits_margin(adv_records[rank_next]["logits"], label)

        clean_margin_delta = clean_margin_next - clean_margin_prev
        adv_margin_delta = adv_margin_next - adv_margin_prev
        delta_margin = adv_margin_delta - clean_margin_delta

        clean_hf_ratio = float(clean_row["incremental_high_freq_ratio"])
        adv_hf_ratio = float(adv_row["incremental_high_freq_ratio"])
        delta_hf_ratio = adv_hf_ratio - clean_hf_ratio
        adv_correct_prev = int(adv_row["correct_prev"])
        adv_correct_next = int(adv_row["correct_next"])

        enhanced_rows.append({
            "sample_id": sample_id,
            "source_index": source_index,
            "label": label,
            "rank_prev": rank_prev,
            "rank_next": rank_next,
            "rank_pair": f"{rank_prev}->{rank_next}",
            "clean_incremental_high_freq_ratio": clean_hf_ratio,
            "adv_incremental_high_freq_ratio": adv_hf_ratio,
            "delta_hf_ratio": delta_hf_ratio,
            "clean_margin_prev": clean_margin_prev,
            "clean_margin_next": clean_margin_next,
            "clean_margin_delta": clean_margin_delta,
            "adv_margin_prev": adv_margin_prev,
            "adv_margin_next": adv_margin_next,
            "adv_margin_delta": adv_margin_delta,
            "delta_margin": delta_margin,
            "adv_hf_ratio_higher": int(delta_hf_ratio > 0.0),
            "adv_margin_improved": int(adv_margin_delta > 0.0),
            "hf_up_margin_down": int(delta_hf_ratio > 0.0 and delta_margin < 0.0),
            "hf_up_top1_bad": int(
                delta_hf_ratio > 0.0 and adv_correct_prev == 1 and adv_correct_next == 0
            ),
            "adv_correct_prev": adv_correct_prev,
            "adv_correct_next": adv_correct_next,
        })
    return enhanced_rows, predictions.get("meta", {})


def bootstrap_mean_ci(values, rng, bootstrap_iters, ci):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean()) if values.size else float("nan")
    if values.size == 0:
        return mean, float("nan"), float("nan")
    if bootstrap_iters <= 0:
        return mean, float("nan"), float("nan")
    sample_indices = rng.integers(0, values.size, size=(bootstrap_iters, values.size))
    boot_means = values[sample_indices].mean(axis=1)
    alpha = (100.0 - ci) / 2.0
    low, high = np.percentile(boot_means, [alpha, 100.0 - alpha])
    return mean, float(low), float(high)


def safe_corr(a_values, b_values):
    a_values = np.asarray(a_values, dtype=float)
    b_values = np.asarray(b_values, dtype=float)
    if a_values.size < 2 or b_values.size < 2:
        return float("nan")
    if np.std(a_values) == 0.0 or np.std(b_values) == 0.0:
        return float("nan")
    return float(np.corrcoef(a_values, b_values)[0, 1])


def build_summary_rows(enhanced_rows, bootstrap_iters, ci, seed):
    grouped = defaultdict(list)
    for row in enhanced_rows:
        grouped[(int(row["rank_prev"]), int(row["rank_next"]))].append(row)

    rng = np.random.default_rng(seed)
    summary_rows = []
    for (rank_prev, rank_next), rows in sorted(grouped.items()):
        delta_hf = [row["delta_hf_ratio"] for row in rows]
        delta_margin = [row["delta_margin"] for row in rows]
        hf_mean, hf_low, hf_high = bootstrap_mean_ci(delta_hf, rng, bootstrap_iters, ci)
        margin_mean, margin_low, margin_high = bootstrap_mean_ci(delta_margin, rng, bootstrap_iters, ci)
        summary_rows.append({
            "rank_prev": rank_prev,
            "rank_next": rank_next,
            "rank_pair": f"{rank_prev}->{rank_next}",
            "count": len(rows),
            "delta_hf_ratio_mean": hf_mean,
            "delta_hf_ratio_ci_low": hf_low,
            "delta_hf_ratio_ci_high": hf_high,
            "adv_hf_ratio_higher_rate": float(np.mean([
                int(row["adv_hf_ratio_higher"]) for row in rows
            ])),
            "delta_margin_mean": margin_mean,
            "delta_margin_ci_low": margin_low,
            "delta_margin_ci_high": margin_high,
            "adv_margin_improve_rate": float(np.mean([
                int(row["adv_margin_improved"]) for row in rows
            ])),
            "corr_delta_hf_margin": safe_corr(delta_hf, delta_margin),
            "hf_up_margin_down_rate": float(np.mean([
                int(row["hf_up_margin_down"]) for row in rows
            ])),
            "hf_up_top1_bad_rate": float(np.mean([
                int(row["hf_up_top1_bad"]) for row in rows
            ])),
        })
    return summary_rows


def plot_with_ci(summary_rows, metric_key, low_key, high_key, ylabel, title, output_path, dpi):
    rank_pairs = [row["rank_pair"] for row in summary_rows]
    values = np.asarray([float(row[metric_key]) for row in summary_rows], dtype=float)
    lows = np.asarray([float(row[low_key]) for row in summary_rows], dtype=float)
    highs = np.asarray([float(row[high_key]) for row in summary_rows], dtype=float)
    yerr = np.vstack([values - lows, highs - values])

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(rank_pairs))
    ax.axhline(0.0, color="#4d4d4d", linestyle="--", linewidth=1.0)
    ax.errorbar(x, values, yerr=yerr, marker="o", linewidth=2, capsize=4)
    ax.set_title(title)
    ax.set_xlabel("Adjacent rank pair")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(rank_pairs)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_rate(summary_rows, metric_key, ylabel, title, output_path, dpi):
    rank_pairs = [row["rank_pair"] for row in summary_rows]
    values = np.asarray([float(row[metric_key]) for row in summary_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(rank_pairs))
    ax.plot(x, values, marker="o", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Adjacent rank pair")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(rank_pairs)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.bootstrap_iters < 0:
        raise ValueError("--bootstrap_iters must be non-negative.")
    if not 0.0 < args.ci < 100.0:
        raise ValueError("--ci must be in (0, 100).")
    if args.plot_dpi <= 0:
        raise ValueError("--plot_dpi must be positive.")

    input_dir = Path(args.input_dir)
    enhanced_rows, meta = build_enhanced_rows(input_dir)
    summary_rows = build_summary_rows(
        enhanced_rows,
        bootstrap_iters=args.bootstrap_iters,
        ci=args.ci,
        seed=args.seed,
    )

    enhanced_path = input_dir / f"{args.output_prefix}_rows.csv"
    summary_path = input_dir / f"{args.output_prefix}_summary.csv"
    write_csv(
        enhanced_path,
        enhanced_rows,
        [
            "sample_id",
            "source_index",
            "label",
            "rank_prev",
            "rank_next",
            "rank_pair",
            "clean_incremental_high_freq_ratio",
            "adv_incremental_high_freq_ratio",
            "delta_hf_ratio",
            "clean_margin_prev",
            "clean_margin_next",
            "clean_margin_delta",
            "adv_margin_prev",
            "adv_margin_next",
            "adv_margin_delta",
            "delta_margin",
            "adv_hf_ratio_higher",
            "adv_margin_improved",
            "hf_up_margin_down",
            "hf_up_top1_bad",
            "adv_correct_prev",
            "adv_correct_next",
        ],
    )
    write_csv(
        summary_path,
        summary_rows,
        [
            "rank_prev",
            "rank_next",
            "rank_pair",
            "count",
            "delta_hf_ratio_mean",
            "delta_hf_ratio_ci_low",
            "delta_hf_ratio_ci_high",
            "adv_hf_ratio_higher_rate",
            "delta_margin_mean",
            "delta_margin_ci_low",
            "delta_margin_ci_high",
            "adv_margin_improve_rate",
            "corr_delta_hf_margin",
            "hf_up_margin_down_rate",
            "hf_up_top1_bad_rate",
        ],
    )

    plots_dir = input_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_with_ci(
        summary_rows,
        metric_key="delta_hf_ratio_mean",
        low_key="delta_hf_ratio_ci_low",
        high_key="delta_hf_ratio_ci_high",
        ylabel="Adv - clean high-frequency ratio",
        title="Paired high-frequency ratio delta with bootstrap CI",
        output_path=plots_dir / f"{args.output_prefix}_delta_hf_ratio_ci.{args.plot_format}",
        dpi=args.plot_dpi,
    )
    plot_with_ci(
        summary_rows,
        metric_key="delta_margin_mean",
        low_key="delta_margin_ci_low",
        high_key="delta_margin_ci_high",
        ylabel="Adv - clean margin delta",
        title="Paired margin delta with bootstrap CI",
        output_path=plots_dir / f"{args.output_prefix}_delta_margin_ci.{args.plot_format}",
        dpi=args.plot_dpi,
    )
    plot_rate(
        summary_rows,
        metric_key="hf_up_margin_down_rate",
        ylabel="Rate",
        title="HF-up and margin-down rate",
        output_path=plots_dir / f"{args.output_prefix}_hf_up_margin_down_rate.{args.plot_format}",
        dpi=args.plot_dpi,
    )

    meta_path = input_dir / f"{args.output_prefix}_meta.json"
    analysis_meta = {
        "input_dir": str(input_dir),
        "output_prefix": args.output_prefix,
        "bootstrap_iters": args.bootstrap_iters,
        "ci": args.ci,
        "seed": args.seed,
        "source_meta": meta,
        "outputs": {
            "rows": enhanced_path.name,
            "summary": summary_path.name,
            "plots_dir": "plots",
        },
    }
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(analysis_meta, file, ensure_ascii=False, indent=2)

    print(f"Saved enhanced paired-delta rows to: {enhanced_path}")
    print(f"Saved enhanced paired-delta summary to: {summary_path}")
    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()
