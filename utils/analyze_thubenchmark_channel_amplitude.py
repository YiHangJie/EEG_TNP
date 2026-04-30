#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import scipy.io as scio
from torcheeg.datasets.constants.ssvep import TSUBENCHMARK_CHANNEL_LIST

from data.ea_utils import EA_R_inv_sqrt
from data.load import load_thubenchmark


def init_accumulator(num_channels: int) -> dict:
    return {
        "sum_abs": np.zeros(num_channels, dtype=np.float64),
        "sum_sq": np.zeros(num_channels, dtype=np.float64),
        "sum_p2p": np.zeros(num_channels, dtype=np.float64),
        "count_values": 0,
        "count_samples": 0,
    }


def update_accumulator(acc: dict, samples: np.ndarray) -> None:
    samples = np.asarray(samples, dtype=np.float64)
    acc["sum_abs"] += np.abs(samples).sum(axis=(0, 2))
    acc["sum_sq"] += np.square(samples).sum(axis=(0, 2))
    acc["sum_p2p"] += (samples.max(axis=2) - samples.min(axis=2)).sum(axis=0)
    acc["count_values"] += samples.shape[0] * samples.shape[2]
    acc["count_samples"] += samples.shape[0]


def finalize_accumulator(acc: dict, channel_names: list[str], condition: str) -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame(
        {
            "channel": channel_names,
            "mean_abs": acc["sum_abs"] / acc["count_values"],
            "rms": np.sqrt(acc["sum_sq"] / acc["count_values"]),
            "mean_p2p": acc["sum_p2p"] / acc["count_samples"],
        }
    )
    for metric in ("mean_abs", "rms", "mean_p2p"):
        df[f"{metric}_rel"] = df[metric] / df[metric].mean()

    summary = {
        "condition": condition,
        "mean_abs_cv": float(df["mean_abs"].std(ddof=0) / df["mean_abs"].mean()),
        "mean_abs_max_min_ratio": float(df["mean_abs"].max() / df["mean_abs"].min()),
        "rms_cv": float(df["rms"].std(ddof=0) / df["rms"].mean()),
        "rms_max_min_ratio": float(df["rms"].max() / df["rms"].min()),
        "mean_p2p_cv": float(df["mean_p2p"].std(ddof=0) / df["mean_p2p"].mean()),
        "mean_p2p_max_min_ratio": float(df["mean_p2p"].max() / df["mean_p2p"].min()),
    }
    return df.sort_values("mean_abs", ascending=False).reset_index(drop=True), summary


def reshape_subject_mat(mat_path: Path) -> np.ndarray:
    samples = scio.loadmat(mat_path)["data"].transpose(2, 3, 0, 1)
    return np.asarray(samples.reshape(-1, samples.shape[2], samples.shape[3]), dtype=np.float64)


def subject_ea(samples: np.ndarray) -> np.ndarray:
    r_inv_sqrt = EA_R_inv_sqrt(samples)
    return np.einsum("ij,njk->nik", r_inv_sqrt, samples)


def analyze_raw_mat(root_path: Path, channel_names: list[str]) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    raw_acc = init_accumulator(len(channel_names))
    ea_acc = init_accumulator(len(channel_names))

    mat_files = sorted(root_path.glob("S*.mat"), key=lambda path: int(path.stem[1:]))
    for mat_file in mat_files:
        samples = reshape_subject_mat(mat_file)
        update_accumulator(raw_acc, samples)
        update_accumulator(ea_acc, subject_ea(samples))

    raw_df, raw_summary = finalize_accumulator(raw_acc, channel_names, "raw_mat")
    ea_df, ea_summary = finalize_accumulator(ea_acc, channel_names, "raw_mat_ea")
    return {
        "raw_mat": raw_df,
        "raw_mat_ea": ea_df,
    }, [raw_summary, ea_summary]


def load_subject_cached_samples(dataset, subject_indices: np.ndarray) -> np.ndarray:
    samples = []
    for index in subject_indices:
        info = dataset.read_info(int(index))
        sample = dataset.read_eeg(str(info["_record_id"]), str(info["clip_id"]))
        samples.append(np.asarray(sample, dtype=np.float64))
    return np.stack(samples, axis=0)


def analyze_cached(root_path: Path, channel_names: list[str]) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    dataset, _ = load_thubenchmark()
    raw_acc = init_accumulator(len(channel_names))
    ea_acc = init_accumulator(len(channel_names))

    grouped = dataset.info.groupby("subject_id").indices
    for subject_id in sorted(grouped):
        samples = load_subject_cached_samples(dataset, np.asarray(grouped[subject_id], dtype=int))
        update_accumulator(raw_acc, samples)
        update_accumulator(ea_acc, subject_ea(samples))

    raw_df, raw_summary = finalize_accumulator(raw_acc, channel_names, "cached_no_ea")
    ea_df, ea_summary = finalize_accumulator(ea_acc, channel_names, "cached_subject_ea")
    return {
        "cached_no_ea": raw_df,
        "cached_subject_ea": ea_df,
    }, [raw_summary, ea_summary]


def top_bottom_text(df: pd.DataFrame, metric: str, top_k: int = 8) -> str:
    top_rows = df.nlargest(top_k, metric)[["channel", metric, f"{metric}_rel"]]
    bottom_rows = df.nsmallest(top_k, metric)[["channel", metric, f"{metric}_rel"]]

    top_text = ", ".join(
        f"{row.channel}={row[metric]:.4f} ({row[f'{metric}_rel']:.3f}x)"
        for _, row in top_rows.iterrows()
    )
    bottom_text = ", ".join(
        f"{row.channel}={row[metric]:.4f} ({row[f'{metric}_rel']:.3f}x)"
        for _, row in bottom_rows.iterrows()
    )
    return f"top {top_k}: {top_text}\nbottom {top_k}: {bottom_text}"


def write_report(output_dir: Path, result_frames: dict[str, pd.DataFrame], summaries: list[dict]) -> None:
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    lines = []
    for condition, df in result_frames.items():
        lines.append(f"[{condition}]")
        lines.append(top_bottom_text(df, "mean_abs"))
        lines.append(top_bottom_text(df, "mean_p2p"))
        lines.append("")

    lines.append("[spread_summary]")
    lines.append(summary_df.to_string(index=False))
    (output_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze channel amplitude differences for THUBenchmark.")
    parser.add_argument(
        "--root-path",
        type=Path,
        default=Path("/home/yhj/pythonProject/data/THUBenchmark"),
        help="Path to the original THUBenchmark .mat files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualization/channel_amplitude/thubenchmark"),
        help="Directory to save csv and text outputs.",
    )
    args = parser.parse_args()

    channel_names = list(TSUBENCHMARK_CHANNEL_LIST)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_frames: dict[str, pd.DataFrame] = {}
    summaries: list[dict] = []

    raw_frames, raw_summaries = analyze_raw_mat(args.root_path, channel_names)
    cached_frames, cached_summaries = analyze_cached(args.root_path, channel_names)
    result_frames.update(raw_frames)
    result_frames.update(cached_frames)
    summaries.extend(raw_summaries)
    summaries.extend(cached_summaries)

    for condition, df in result_frames.items():
        df.to_csv(args.output_dir / f"{condition}.csv", index=False)

    write_report(args.output_dir, result_frames, summaries)

    print(f"Saved results to: {args.output_dir}")
    print(pd.DataFrame(summaries).to_string(index=False))
    print()
    for condition, df in result_frames.items():
        print(f"[{condition}]")
        print(top_bottom_text(df, "mean_abs"))
        print(top_bottom_text(df, "mean_p2p"))
        print()


if __name__ == "__main__":
    main()
