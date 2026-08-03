#!/usr/bin/env python3
"""对同一批 trial 的多-rank净化结果执行 EXP-028 HOSVD 低秩分析。"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from trial_lowrank_analysis import analyze_trial_hosvd_lowrank as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "加载 EXP-028 bundle 和逐-rank EEG_TNP payload，分析净化后的 "
            "clean/adv/perturbation 低秩谱。"
        )
    )
    parser.add_argument("--source_bundle", required=True)
    parser.add_argument("--source_summary", default=None)
    parser.add_argument("--purification_paths", nargs="+", required=True)
    parser.add_argument("--expected_ranks", default="15,20,25,30,35,40")
    parser.add_argument("--dataset", default="thubenchmark")
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--min_trials", type=int, default=3)
    parser.add_argument(
        "--views",
        nargs="+",
        default=list(base.DEFAULT_VIEWS),
        choices=base.VIEW_ORDER,
    )
    parser.add_argument(
        "--frequency_representation",
        default="complex",
        choices=("complex", "magnitude", "power", "real_imag"),
    )
    parser.add_argument(
        "--channel_view_space",
        default="interpolated_grid",
        choices=("raw", "grid", "interpolated_grid"),
    )
    parser.add_argument("--tf_n_fft", type=int, default=128)
    parser.add_argument("--tf_hop_length", type=int, default=64)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def parse_expected_ranks(value: str) -> list[int]:
    ranks = [int(token) for token in re.split(r"[\s,]+", value.strip()) if token]
    if not ranks or len(set(ranks)) != len(ranks):
        raise ValueError("expected_ranks must contain unique integer ranks.")
    return ranks


def load_source_bundle(path: str, args: argparse.Namespace) -> dict:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise ValueError("Source bundle must be a dict.")
    for key in ("clean", "adv", "labels", "metadata", "args"):
        if key not in bundle:
            raise ValueError(f"Source bundle missing {key}.")

    clean = torch.as_tensor(bundle["clean"]).float().cpu().contiguous()
    adv = torch.as_tensor(bundle["adv"]).float().cpu().contiguous()
    labels = torch.as_tensor(bundle["labels"]).long().view(-1).cpu()
    metadata = pd.DataFrame(bundle["metadata"])
    if clean.shape != adv.shape or clean.size(0) != labels.numel():
        raise ValueError("Source bundle tensor shapes are inconsistent.")
    if len(metadata) != clean.size(0):
        raise ValueError("Source bundle metadata length mismatch.")
    if "original_split_index" not in metadata.columns:
        raise KeyError("Source metadata must contain original_split_index.")

    source_indices = metadata["original_split_index"].astype(int).tolist()
    if len(set(source_indices)) != len(source_indices):
        raise ValueError("Source original_split_index values must be unique.")

    source_args = dict(bundle["args"])
    for key, expected in {
        "dataset": args.dataset,
        "model": args.model,
        "fold": args.fold,
        "seed": args.seed,
        "eps": args.eps,
    }.items():
        actual = source_args.get(key)
        if key == "eps":
            matches = actual is not None and abs(
                float(actual) - float(expected)
            ) <= 1e-12
        else:
            matches = str(actual) == str(expected)
        if not matches:
            raise ValueError(
                f"Source bundle mismatch: {key}={actual}, expected {expected}."
            )

    return {
        "clean": clean,
        "adv": adv,
        "labels": labels,
        "metadata": metadata.reset_index(drop=True),
        "source_indices": source_indices,
        "args": source_args,
    }


def validate_and_align_payload(
    payload: dict,
    source: dict,
    args: argparse.Namespace,
    path: str,
) -> list[dict]:
    """按 EXP-028 原始顺序重排 payload，并返回其中每个 rank 的净化张量。"""
    if not isinstance(payload, dict):
        raise ValueError(f"Purification payload must be a dict: {path}")
    required = (
        "clean",
        "adversarial",
        "clean_pur_by_rank",
        "adv_pur_by_rank",
        "labels",
        "source_indices",
        "ranks",
        "meta",
    )
    for key in required:
        if key not in payload:
            raise ValueError(f"Purification payload {path} missing {key}.")

    meta = payload["meta"]
    for key, expected in {
        "dataset": args.dataset,
        "model": args.model,
        "fold": args.fold,
        "seed": args.seed,
        "eps": args.eps,
    }.items():
        actual = meta.get(key)
        if key == "eps":
            matches = actual is not None and abs(
                float(actual) - float(expected)
            ) <= 1e-12
        else:
            matches = str(actual) == str(expected)
        if not matches:
            raise ValueError(
                f"Purification payload {path} mismatch: "
                f"meta.{key}={actual}, expected {expected}."
            )

    payload_sources = [int(value) for value in payload["source_indices"]]
    if len(set(payload_sources)) != len(payload_sources):
        raise ValueError(f"Purification payload {path} has duplicate source indices.")
    source_position = {
        source_index: position
        for position, source_index in enumerate(source["source_indices"])
    }
    missing = sorted(set(payload_sources) - set(source_position))
    if missing:
        raise ValueError(
            f"Purification payload {path} contains unknown source indices: "
            f"{missing[:10]}"
        )

    # payload 可能因为稳定随机子集逻辑而重排；统一恢复成 EXP-028 bundle 顺序。
    canonical_sources = [
        source_index
        for source_index in source["source_indices"]
        if source_index in set(payload_sources)
    ]
    payload_position = {
        source_index: position
        for position, source_index in enumerate(payload_sources)
    }
    reorder = torch.as_tensor(
        [payload_position[source_index] for source_index in canonical_sources],
        dtype=torch.long,
    )
    source_reorder = torch.as_tensor(
        [source_position[source_index] for source_index in canonical_sources],
        dtype=torch.long,
    )

    labels = torch.as_tensor(payload["labels"]).long().view(-1).index_select(0, reorder)
    expected_labels = source["labels"].index_select(0, source_reorder)
    if not torch.equal(labels, expected_labels):
        raise ValueError(f"Purification payload {path} labels mismatch.")

    clean = torch.as_tensor(payload["clean"]).float().index_select(0, reorder)
    adv = torch.as_tensor(payload["adversarial"]).float().index_select(0, reorder)
    expected_clean = source["clean"].index_select(0, source_reorder)
    expected_adv = source["adv"].index_select(0, source_reorder)
    if not torch.allclose(clean, expected_clean, atol=1e-6, rtol=1e-5):
        raise ValueError(f"Purification payload {path} clean tensor mismatch.")
    if not torch.allclose(adv, expected_adv, atol=1e-6, rtol=1e-5):
        raise ValueError(f"Purification payload {path} adversarial tensor mismatch.")

    clean_pur = torch.as_tensor(payload["clean_pur_by_rank"]).float()
    adv_pur = torch.as_tensor(payload["adv_pur_by_rank"]).float()
    ranks = [int(rank) for rank in payload["ranks"]]
    if clean_pur.shape != adv_pur.shape or clean_pur.size(1) != len(ranks):
        raise ValueError(f"Purification payload {path} rank tensor mismatch.")
    clean_pur = clean_pur.index_select(0, reorder).cpu().contiguous()
    adv_pur = adv_pur.index_select(0, reorder).cpu().contiguous()

    metadata = source["metadata"].iloc[source_reorder.numpy()].copy().reset_index(drop=True)
    records = []
    for rank_index, rank in enumerate(ranks):
        records.append(
            {
                "rank": rank,
                "clean_pur": clean_pur[:, rank_index].contiguous(),
                "adv_pur": adv_pur[:, rank_index].contiguous(),
                "labels": labels,
                "metadata": metadata,
                "source_indices": canonical_sources,
                "config": str(meta.get("configs", [""])[rank_index])
                if len(meta.get("configs", [])) > rank_index
                else "",
                "path": str(path),
            }
        )
    return records


def purified_spectra_to_npz(spectra_records: list[dict]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    manifest = []
    for record in spectra_records:
        prefix = "__".join(
            [
                f"rank_{record['rank']}",
                base.safe_key(record["center_mode"]),
                base.safe_key(record["data_type"]),
                f"subject_{base.safe_key(record['subject_id'])}",
                base.safe_key(record["hosvd_mode"]),
            ]
        )
        arrays[f"{prefix}__singular_values"] = record["singular_values"]
        arrays[f"{prefix}__energy"] = record["energy"]
        arrays[f"{prefix}__cumulative_energy"] = record["cumulative_energy"]
        manifest.append(
            {
                "prefix": prefix,
                "rank": int(record["rank"]),
                "center_mode": record["center_mode"],
                "data_type": record["data_type"],
                "subject_id": str(record["subject_id"]),
                "hosvd_mode": record["hosvd_mode"],
                "matrix_shape": list(record.get("matrix_shape", ())),
                "view_info": record.get("view_info", {}),
            }
        )
    arrays["manifest_json"] = np.asarray(json.dumps(manifest, ensure_ascii=False))
    return arrays


def plot_rank_trend(
    summary: pd.DataFrame,
    *,
    metric: str,
    center_mode: str,
    views: tuple[str, ...],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(views),
        figsize=(5 * len(views), 4),
        sharey=False,
    )
    axes = np.atleast_1d(axes)
    fig.suptitle(f"{metric} across purification ranks ({center_mode})")
    for axis, view in zip(axes, views):
        filtered = summary[
            (summary["center_mode"] == center_mode)
            & (summary["hosvd_mode"] == view)
        ]
        for data_type in base.DATA_TYPE_ORDER:
            grouped = (
                filtered[filtered["data_type"] == data_type]
                .groupby("rank")[metric]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("rank")
            )
            if grouped.empty:
                continue
            std = grouped["std"].fillna(0.0).to_numpy()
            mean = grouped["mean"].to_numpy()
            ranks = grouped["rank"].to_numpy()
            color = base.PLOT_COLORS[data_type]
            axis.plot(ranks, mean, marker="o", label=data_type, color=color)
            axis.fill_between(
                ranks,
                mean - std,
                mean + std,
                alpha=0.15,
                color=color,
                linewidth=0,
            )
        axis.set_title(f"{view} view")
        axis.set_xlabel("purification rank")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel(metric)
    axes[-1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    run_start = time.perf_counter()
    args = parse_args()
    base.seed_everything(args.seed)
    views = tuple(dict.fromkeys(args.views))
    expected_ranks = parse_expected_ranks(args.expected_ranks)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base.REPO_ROOT / output_dir
    base.setup_logging(output_dir)

    logging.info("Args: %s", args)
    source = load_source_bundle(args.source_bundle, args)
    spatial_transform = None
    if base.views_need_spatial_projection(views):
        spatial_transform = base.build_channel_spatial_transform(
            dataset=args.dataset,
            channel_view_space=args.channel_view_space,
        )

    rank_records: dict[int, dict] = {}
    for path in args.purification_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for record in validate_and_align_payload(payload, source, args, path):
            rank = int(record["rank"])
            if rank in rank_records:
                raise ValueError(f"Duplicate purification rank {rank}.")
            rank_records[rank] = record
        del payload

    if sorted(rank_records) != sorted(expected_ranks):
        raise ValueError(
            f"Purification ranks={sorted(rank_records)}, "
            f"expected={sorted(expected_ranks)}."
        )

    all_rows = []
    spectra_records = []
    for rank in expected_ranks:
        record = rank_records[rank]
        tensors = {
            "clean": record["clean_pur"],
            "adv": record["adv_pur"],
            # 与 EXP-028 perturb 定义保持一致：比较配对 clean/adv 在相同 rank
            # 净化后的差，而不是输入到净化输出的 reconstruction residual。
            "perturb": record["adv_pur"] - record["clean_pur"],
        }
        for center_mode in base.CENTER_MODES:
            for data_type in base.DATA_TYPE_ORDER:
                rows, records = base.analyze_tensor_set(
                    data_type=data_type,
                    tensor=tensors[data_type],
                    metadata=record["metadata"],
                    center_mode=center_mode,
                    min_trials=args.min_trials,
                    views=views,
                    frequency_representation=args.frequency_representation,
                    spatial_transform=spatial_transform,
                    channel_view_space=args.channel_view_space,
                    tf_n_fft=args.tf_n_fft,
                    tf_hop_length=args.tf_hop_length,
                )
                for row in rows:
                    row.update(
                        {
                            "rank": rank,
                            "purified": True,
                            "config": record["config"],
                            "source_payload": record["path"],
                        }
                    )
                for spectrum_record in records:
                    spectrum_record["rank"] = rank
                all_rows.extend(rows)
                spectra_records.extend(records)
        del tensors
        logging.info("Finished purified HOSVD analysis for rank=%d", rank)

    summary = pd.DataFrame(all_rows)
    if summary.empty:
        raise RuntimeError("No subject had enough trials for purified analysis.")
    summary["hosvd_mode"] = pd.Categorical(
        summary["hosvd_mode"],
        categories=list(base.VIEW_ORDER),
        ordered=True,
    )
    summary = (
        summary.sort_values(
            ["rank", "center_mode", "data_type", "subject_id", "hosvd_mode"]
        )
        .reset_index(drop=True)
    )
    summary["hosvd_mode"] = summary["hosvd_mode"].astype(str)
    summary["view"] = summary["hosvd_mode"]

    metadata = rank_records[expected_ranks[0]]["metadata"]
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    summary.to_csv(output_dir / "hosvd_summary.csv", index=False)
    np.savez_compressed(
        output_dir / "spectra.npz",
        **purified_spectra_to_npz(spectra_records),
    )

    if args.source_summary:
        source_summary = pd.read_csv(args.source_summary)
        source_summary["rank"] = np.nan
        source_summary["purified"] = False
        source_summary["config"] = ""
        source_summary["source_payload"] = str(args.source_bundle)
        pd.concat([source_summary, summary], ignore_index=True, sort=False).to_csv(
            output_dir / "comparison_summary.csv",
            index=False,
        )

    torch.save(
        {
            "labels": rank_records[expected_ranks[0]]["labels"],
            "source_indices": rank_records[expected_ranks[0]]["source_indices"],
            "ranks": expected_ranks,
            "source_bundle": str(args.source_bundle),
            "source_summary": str(args.source_summary or ""),
            "purification_paths": list(args.purification_paths),
            "args": vars(args),
        },
        output_dir / "analysis_manifest.pt",
    )
    run_config = vars(args).copy()
    run_config["expected_ranks"] = expected_ranks
    run_config["views"] = list(views)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, ensure_ascii=False)

    for rank in expected_ranks:
        rank_spectra = [
            record for record in spectra_records if int(record["rank"]) == rank
        ]
        rank_summary = summary[summary["rank"] == rank]
        for center_mode in base.CENTER_MODES:
            base.plot_cumulative_energy(
                rank_spectra,
                center_mode=center_mode,
                views=views,
                output_path=output_dir
                / f"cum_energy_rank{rank}_{center_mode}.png",
            )
            base.plot_rank95(
                rank_summary,
                center_mode=center_mode,
                views=views,
                output_path=output_dir / f"rank95_rank{rank}_{center_mode}.png",
            )
    for center_mode in base.CENTER_MODES:
        for metric in ("rank95", "effective_rank"):
            plot_rank_trend(
                summary,
                metric=metric,
                center_mode=center_mode,
                views=views,
                output_path=output_dir / f"{metric}_trend_{center_mode}.png",
            )

    logging.info(
        "Purified trial low-rank analysis finished: ranks=%s, rows=%d, elapsed=%s",
        expected_ranks,
        len(summary),
        base.format_elapsed(time.perf_counter() - run_start),
    )


if __name__ == "__main__":
    main()
