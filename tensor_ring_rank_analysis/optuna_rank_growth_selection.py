import argparse
import csv
import datetime
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from runtime_env import configure_runtime_env

configure_runtime_env()

import numpy as np
import optuna
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/eegap_matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/eegap_xdg_cache")
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SOURCE_TYPES = ("clean", "adv")
SELECTION_MODES = {
    "threshold",
    "score",
    "js_only",
    "mse_only",
    "margin_only",
    "js_mse",
    "mse_margin",
    "js_margin",
    "entropy_only",
    "js_entropy",
    "mse_entropy",
    "entropy_margin",
    "js_mse_entropy",
    "score_entropy",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline Optuna tuning for rank-growth rank-selection rules."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--study_name", type=str, default="rank_growth_selection")
    parser.add_argument("--n_trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune_ratio", type=float, default=0.5)
    parser.add_argument("--selection_modes", type=str, default="threshold,score")
    parser.add_argument("--include_exp007_full_score", type=str, default=None,
                        help="optional EXP-007 best_config.json used as a fixed full-score baseline.")
    parser.add_argument("--fixed_rank_baselines", type=str, default="15,20,25,30,35,40",
                        help="fixed ranks included in the selector comparison plot.")
    parser.add_argument("--comparison_selectors", type=str, default="threshold,js_mse,mse_only",
                        help="selector names included in the fixed-rank comparison plot.")
    parser.add_argument("--objective", type=str, default="robust_priority",
                        choices=["robust_priority"])
    parser.add_argument("--clean_weight", type=float, default=0.20)
    parser.add_argument("--mse_weight", type=float, default=0.10)
    parser.add_argument("--rank_weight", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="limit sample count for smoke tests; keeps the first N selected samples.")
    parser.add_argument("--plot_format", type=str, default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--plot_dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def prepare_output_dir(output_dir, overwrite):
    output_dir = Path(output_dir)
    target_files = [
        "trials.csv",
        "best_config.json",
        "selected_rows.csv",
        "summary.csv",
        "mode_best_summary.csv",
        "fixed_rank_selector_comparison.csv",
        "meta.json",
    ]
    existing = [output_dir / name for name in target_files if (output_dir / name).exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist: {joined}. Use --overwrite to replace them.")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_selection_modes(value):
    modes = []
    for item in str(value).replace(";", ",").split(","):
        item = item.strip()
        if item:
            modes.append(item)
    invalid = [mode for mode in modes if mode not in SELECTION_MODES]
    if invalid:
        raise ValueError(f"Unknown selection mode(s): {invalid}. Allowed: {sorted(SELECTION_MODES)}")
    if not modes:
        raise ValueError("--selection_modes must contain at least one mode.")
    return modes


def parse_csv_items(value):
    items = []
    for item in str(value).replace(";", ",").split(","):
        item = item.strip()
        if item:
            items.append(item)
    return items


def parse_rank_list(value):
    ranks = []
    for item in parse_csv_items(value):
        ranks.append(int(item))
    return ranks


def safe_min_max(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        high = low + 1.0
    return low, high


def normalize(values, low, high):
    return np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)


def js_divergence_np(probs_a, probs_b):
    probs_a = np.clip(probs_a, 1e-12, None)
    probs_b = np.clip(probs_b, 1e-12, None)
    probs_a = probs_a / probs_a.sum(axis=-1, keepdims=True)
    probs_b = probs_b / probs_b.sum(axis=-1, keepdims=True)
    midpoint = 0.5 * (probs_a + probs_b)
    kl_a = (probs_a * np.log(probs_a / midpoint)).sum(axis=-1)
    kl_b = (probs_b * np.log(probs_b / midpoint)).sum(axis=-1)
    return 0.5 * (kl_a + kl_b)


def softmax_np(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp_values = np.exp(logits)
    return exp_values / exp_values.sum(axis=-1, keepdims=True)


def extract_rank_tensors(predictions, max_samples=None):
    labels = torch.as_tensor(predictions["labels"]).detach().cpu().long()
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max_samples must be positive when provided.")
        labels = labels[:max_samples]

    sample_count = int(labels.numel())
    traces = predictions["traces"]
    ranks = None
    data = {}
    for source_type in SOURCE_TYPES:
        source_traces = sorted(
            traces[source_type],
            key=lambda trace: int(trace["sample_id"]),
        )[:sample_count]
        if len(source_traces) != sample_count:
            raise ValueError(
                f"Trace count mismatch for {source_type}: expected {sample_count}, got {len(source_traces)}."
            )

        source_ranks = [
            int(record["rank"])
            for record in sorted(source_traces[0]["eval_records"], key=lambda record: int(record["rank"]))
        ]
        if ranks is None:
            ranks = source_ranks
        elif ranks != source_ranks:
            raise ValueError(f"Rank sequence mismatch for {source_type}: {source_ranks} vs {ranks}.")

        logits = []
        mse = []
        source_indices = []
        sample_ids = []
        for expected_sample_id, trace in enumerate(source_traces):
            sample_id = int(trace["sample_id"])
            if sample_id != expected_sample_id:
                raise ValueError(
                    f"Expected contiguous sample_id={expected_sample_id}, got {sample_id} for {source_type}."
                )
            records = sorted(trace["eval_records"], key=lambda record: int(record["rank"]))
            record_ranks = [int(record["rank"]) for record in records]
            if record_ranks != ranks:
                raise ValueError(f"Rank sequence mismatch at {source_type} sample_id={sample_id}.")
            logits.append(torch.stack([
                torch.as_tensor(record["logits"]).detach().cpu().float().view(-1)
                for record in records
            ]))
            mse.append([float(record["mse_to_input"]) for record in records])
            source_indices.append(int(trace["source_index"]))
            sample_ids.append(sample_id)

        logits_tensor = torch.stack(logits, dim=0)
        probs = torch.softmax(logits_tensor, dim=-1)
        probs_clamped = probs.clamp_min(1e-12)
        entropy = -(probs_clamped * probs_clamped.log()).sum(dim=-1)
        top1 = probs.argmax(dim=-1)
        correct = top1.eq(labels.view(-1, 1))
        data[source_type] = {
            "sample_ids": np.asarray(sample_ids, dtype=np.int64),
            "source_indices": np.asarray(source_indices, dtype=np.int64),
            "logits": logits_tensor.numpy(),
            "probs": probs.numpy(),
            "entropy": entropy.numpy().astype(np.float64),
            "top1": top1.numpy(),
            "correct": correct.numpy().astype(np.int64),
            "mse": np.asarray(mse, dtype=np.float64),
        }

    raw_positions = predictions.get("selected_positions", [])
    if torch.is_tensor(raw_positions):
        selected_positions = raw_positions.detach().cpu().long().view(-1).tolist()
    else:
        selected_positions = [int(position) for position in raw_positions]

    return {
        "labels": labels.numpy().astype(np.int64),
        "ranks": np.asarray(ranks, dtype=np.int64),
        "data": data,
        "meta": predictions.get("meta", {}),
        "selected_positions": selected_positions[:sample_count],
    }


def load_raw_logits_from_predictions(predictions, source_type, sample_count):
    raw_logits = predictions.get("raw_logits")
    if isinstance(raw_logits, dict) and source_type in raw_logits:
        return torch.as_tensor(raw_logits[source_type]).detach().cpu().float()[:sample_count]

    source_key = f"{source_type}_raw_logits"
    if source_key in predictions:
        return torch.as_tensor(predictions[source_key]).detach().cpu().float()[:sample_count]
    return None


def recompute_raw_logits(predictions, selection_data, args):
    from tensor_ring_rank_analysis.analyze_tr_rank_predictions import (
        classify_tensor_batch,
        load_classifier,
        load_test_ad_payload,
    )
    from data.load import load_thubenchmark

    meta = selection_data["meta"]
    required = ["ad_data_path", "checkpoint_path", "dataset", "fold", "use_ea"]
    missing = [key for key in required if key not in meta or meta[key] in (None, "")]
    if missing:
        raise ValueError(f"Cannot recompute raw logits because meta is missing: {missing}")
    if meta.get("dataset") != "thubenchmark":
        raise ValueError("Raw-logit recomputation currently supports thubenchmark only.")

    dataset, info = load_thubenchmark()
    load_args = SimpleNamespace(
        dataset=meta.get("dataset", "thubenchmark"),
        fold=int(meta.get("fold", 0)),
        seed=int(meta.get("seed", args.seed)),
        use_ea=bool(meta.get("use_ea", False)),
        batch_size=int(args.batch_size),
    )
    data_payload = load_test_ad_payload(
        load_args,
        dataset=dataset,
        info=info,
        ad_data_path=meta["ad_data_path"],
    )

    positions = selection_data["selected_positions"]
    if not positions:
        positions = list(range(len(selection_data["labels"])))
    positions_tensor = torch.as_tensor(positions, dtype=torch.long)
    clean_samples = data_payload["x"][positions_tensor]
    adv_samples = data_payload["x_adv"][positions_tensor]
    labels = data_payload["labels"][positions_tensor].long()
    expected_labels = torch.as_tensor(selection_data["labels"]).long()
    if not torch.equal(labels, expected_labels):
        mismatch = labels.ne(expected_labels).nonzero(as_tuple=False).view(-1)[:10].tolist()
        raise ValueError(f"Raw data labels differ from predictions labels. First mismatch: {mismatch}")

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = load_classifier(meta["checkpoint_path"], info, device)
    return {
        "clean": classify_tensor_batch(model, clean_samples, args.batch_size, device).detach().cpu().float(),
        "adv": classify_tensor_batch(model, adv_samples, args.batch_size, device).detach().cpu().float(),
    }


def attach_raw_margin_features(predictions, selection_data, args):
    raw_logits = {}
    sample_count = len(selection_data["labels"])
    for source_type in SOURCE_TYPES:
        loaded = load_raw_logits_from_predictions(predictions, source_type, sample_count)
        if loaded is not None:
            raw_logits[source_type] = loaded

    recomputed = False
    if any(source_type not in raw_logits for source_type in SOURCE_TYPES):
        raw_logits = recompute_raw_logits(predictions, selection_data, args)
        recomputed = True

    for source_type in SOURCE_TYPES:
        source_raw_logits = torch.as_tensor(raw_logits[source_type]).detach().cpu().float()
        if source_raw_logits.size(0) != sample_count:
            raise ValueError(
                f"Raw-logit count mismatch for {source_type}: "
                f"expected {sample_count}, got {source_raw_logits.size(0)}."
            )
        raw_probs = torch.softmax(source_raw_logits, dim=-1).numpy()
        raw_top1 = raw_probs.argmax(axis=-1)
        raw_confidence = raw_probs.max(axis=-1)

        probs = selection_data["data"][source_type]["probs"]
        margins = []
        for sample_id in range(sample_count):
            target_class = int(raw_top1[sample_id])
            target_probs = probs[sample_id, :, target_class]
            masked = probs[sample_id].copy()
            masked[:, target_class] = -np.inf
            other_max = masked.max(axis=-1)
            margins.append(target_probs - other_max)
        selection_data["data"][source_type]["raw_logits"] = source_raw_logits.numpy()
        selection_data["data"][source_type]["raw_top1"] = raw_top1.astype(np.int64)
        selection_data["data"][source_type]["raw_confidence"] = raw_confidence.astype(np.float64)
        selection_data["data"][source_type]["margin"] = np.asarray(margins, dtype=np.float64)

    selection_data["raw_logits_recomputed"] = recomputed
    return selection_data


def attach_js_features(selection_data):
    for source_type in SOURCE_TYPES:
        probs = selection_data["data"][source_type]["probs"]
        js_adjacent = js_divergence_np(probs[:, :-1, :], probs[:, 1:, :])
        rank_count = probs.shape[1]
        js_score = np.zeros((probs.shape[0], rank_count), dtype=np.float64)
        if rank_count > 1:
            # 选中 rank r 时，优先使用 r 与下一档 rank 的稳定性；末档用上一档稳定性兜底。
            js_score[:, :-1] = js_adjacent
            js_score[:, -1] = js_adjacent[:, -1]
        selection_data["data"][source_type]["js_adjacent"] = js_adjacent
        selection_data["data"][source_type]["js_score"] = js_score
    return selection_data


def build_split_indices(sample_count, tune_ratio, seed):
    if sample_count < 2:
        raise ValueError("At least two samples are required to build tune/holdout split.")
    if not 0.0 < tune_ratio < 1.0:
        raise ValueError("--tune_ratio must be in (0, 1).")
    rng = np.random.default_rng(seed)
    indices = np.arange(sample_count, dtype=np.int64)
    rng.shuffle(indices)
    tune_count = max(1, min(sample_count - 1, int(round(sample_count * tune_ratio))))
    tune_indices = np.sort(indices[:tune_count])
    holdout_indices = np.sort(indices[tune_count:])
    return tune_indices, holdout_indices


def build_feature_stats(selection_data, tune_indices):
    stats = {}
    for key in ("js_score", "mse", "margin", "entropy"):
        values = []
        for source_type in SOURCE_TYPES:
            values.append(selection_data["data"][source_type][key][tune_indices].reshape(-1))
        stats[key] = safe_min_max(np.concatenate(values))
    stats["rank_norm"] = (0.0, 1.0)
    return stats


def select_threshold_for_source(source_data, params):
    js_adjacent = source_data["js_adjacent"]
    mse = source_data["mse"]
    margin = source_data["margin"]
    sample_count, rank_count = mse.shape
    selected = np.full(sample_count, rank_count - 1, dtype=np.int64)
    if rank_count <= 1:
        return selected

    for sample_id in range(sample_count):
        for rank_id in range(rank_count - 1):
            if (
                js_adjacent[sample_id, rank_id] <= params["js_threshold"]
                and mse[sample_id, rank_id] <= params["mse_threshold"]
                and margin[sample_id, rank_id] >= params["margin_threshold"]
            ):
                selected[sample_id] = rank_id
                break
    return selected


def select_js_only_for_source(source_data, params):
    js_adjacent = source_data["js_adjacent"]
    sample_count = js_adjacent.shape[0]
    rank_count = source_data["mse"].shape[1]
    selected = np.full(sample_count, rank_count - 1, dtype=np.int64)
    if rank_count <= 1:
        return selected

    for sample_id in range(sample_count):
        for rank_id in range(rank_count - 1):
            if js_adjacent[sample_id, rank_id] <= params["js_threshold"]:
                selected[sample_id] = rank_id
                break
    return selected


def select_mse_only_for_source(source_data, params):
    mse = source_data["mse"]
    sample_count, rank_count = mse.shape
    selected = np.full(sample_count, rank_count - 1, dtype=np.int64)
    for sample_id in range(sample_count):
        for rank_id in range(rank_count):
            if mse[sample_id, rank_id] <= params["mse_threshold"]:
                selected[sample_id] = rank_id
                break
    return selected


def select_margin_only_for_source(source_data, params):
    margin = source_data["margin"]
    sample_count, rank_count = margin.shape
    selected = np.full(sample_count, rank_count - 1, dtype=np.int64)
    for sample_id in range(sample_count):
        for rank_id in range(rank_count):
            if margin[sample_id, rank_id] >= params["margin_threshold"]:
                selected[sample_id] = rank_id
                break
    return selected


def select_js_mse_for_source(source_data, params):
    js_adjacent = source_data["js_adjacent"]
    mse = source_data["mse"]
    sample_count, rank_count = mse.shape
    selected = np.full(sample_count, rank_count - 1, dtype=np.int64)
    if rank_count <= 1:
        return selected

    for sample_id in range(sample_count):
        for rank_id in range(rank_count - 1):
            if (
                js_adjacent[sample_id, rank_id] <= params["js_threshold"]
                and mse[sample_id, rank_id] <= params["mse_threshold"]
            ):
                selected[sample_id] = rank_id
                break
    return selected


def select_js_margin_for_source(source_data, params):
    js_adjacent = source_data["js_adjacent"]
    margin = source_data["margin"]
    sample_count, rank_count = margin.shape
    selected = np.full(sample_count, rank_count - 1, dtype=np.int64)
    if rank_count <= 1:
        return selected

    for sample_id in range(sample_count):
        for rank_id in range(rank_count - 1):
            if (
                js_adjacent[sample_id, rank_id] <= params["js_threshold"]
                and margin[sample_id, rank_id] >= params["margin_threshold"]
            ):
                selected[sample_id] = rank_id
                break
    return selected


def select_mse_margin_for_source(source_data, params):
    mse = source_data["mse"]
    margin = source_data["margin"]
    sample_count, rank_count = mse.shape
    selected = np.full(sample_count, rank_count - 1, dtype=np.int64)
    for sample_id in range(sample_count):
        for rank_id in range(rank_count):
            if (
                mse[sample_id, rank_id] <= params["mse_threshold"]
                and margin[sample_id, rank_id] >= params["margin_threshold"]
            ):
                selected[sample_id] = rank_id
                break
    return selected


def select_entropy_only_for_source(source_data, params):
    entropy = source_data["entropy"]
    sample_count, rank_count = entropy.shape
    selected = np.full(sample_count, rank_count - 1, dtype=np.int64)
    for sample_id in range(sample_count):
        for rank_id in range(rank_count):
            if entropy[sample_id, rank_id] <= params["entropy_threshold"]:
                selected[sample_id] = rank_id
                break
    return selected


def select_js_entropy_for_source(source_data, params):
    js_adjacent = source_data["js_adjacent"]
    entropy = source_data["entropy"]
    sample_count = js_adjacent.shape[0]
    rank_count = entropy.shape[1]
    selected = np.full(sample_count, rank_count - 1, dtype=np.int64)
    if rank_count <= 1:
        return selected

    for sample_id in range(sample_count):
        for rank_id in range(rank_count - 1):
            if (
                js_adjacent[sample_id, rank_id] <= params["js_threshold"]
                and entropy[sample_id, rank_id] <= params["entropy_threshold"]
            ):
                selected[sample_id] = rank_id
                break
    return selected


def select_mse_entropy_for_source(source_data, params):
    mse = source_data["mse"]
    entropy = source_data["entropy"]
    sample_count, rank_count = mse.shape
    selected = np.full(sample_count, rank_count - 1, dtype=np.int64)
    for sample_id in range(sample_count):
        for rank_id in range(rank_count):
            if (
                mse[sample_id, rank_id] <= params["mse_threshold"]
                and entropy[sample_id, rank_id] <= params["entropy_threshold"]
            ):
                selected[sample_id] = rank_id
                break
    return selected


def select_entropy_margin_for_source(source_data, params):
    entropy = source_data["entropy"]
    margin = source_data["margin"]
    sample_count, rank_count = entropy.shape
    selected = np.full(sample_count, rank_count - 1, dtype=np.int64)
    for sample_id in range(sample_count):
        for rank_id in range(rank_count):
            if (
                entropy[sample_id, rank_id] <= params["entropy_threshold"]
                and margin[sample_id, rank_id] >= params["margin_threshold"]
            ):
                selected[sample_id] = rank_id
                break
    return selected


def select_js_mse_entropy_for_source(source_data, params):
    js_adjacent = source_data["js_adjacent"]
    mse = source_data["mse"]
    entropy = source_data["entropy"]
    sample_count, rank_count = mse.shape
    selected = np.full(sample_count, rank_count - 1, dtype=np.int64)
    if rank_count <= 1:
        return selected

    for sample_id in range(sample_count):
        for rank_id in range(rank_count - 1):
            if (
                js_adjacent[sample_id, rank_id] <= params["js_threshold"]
                and mse[sample_id, rank_id] <= params["mse_threshold"]
                and entropy[sample_id, rank_id] <= params["entropy_threshold"]
            ):
                selected[sample_id] = rank_id
                break
    return selected


def select_score_for_source(source_data, ranks, params, feature_stats):
    rank_min = float(np.min(ranks))
    rank_max = float(np.max(ranks))
    rank_norm = (ranks.astype(np.float64) - rank_min) / max(rank_max - rank_min, 1e-12)

    js_norm = normalize(source_data["js_score"], *feature_stats["js_score"])
    mse_norm = normalize(source_data["mse"], *feature_stats["mse"])
    margin_norm = normalize(source_data["margin"], *feature_stats["margin"])
    score = (
        params["alpha"] * js_norm
        + params["beta"] * mse_norm
        - params["gamma"] * margin_norm
        + params["delta"] * rank_norm.reshape(1, -1)
    )
    return np.argmin(score, axis=1).astype(np.int64)


def select_score_entropy_for_source(source_data, ranks, params, feature_stats):
    rank_min = float(np.min(ranks))
    rank_max = float(np.max(ranks))
    rank_norm = (ranks.astype(np.float64) - rank_min) / max(rank_max - rank_min, 1e-12)

    js_norm = normalize(source_data["js_score"], *feature_stats["js_score"])
    mse_norm = normalize(source_data["mse"], *feature_stats["mse"])
    margin_norm = normalize(source_data["margin"], *feature_stats["margin"])
    entropy_norm = normalize(source_data["entropy"], *feature_stats["entropy"])
    score = (
        params["alpha"] * js_norm
        + params["beta"] * mse_norm
        - params["gamma"] * margin_norm
        + params["delta"] * rank_norm.reshape(1, -1)
        + params["eta"] * entropy_norm
    )
    return np.argmin(score, axis=1).astype(np.int64)


def select_ranks(selection_data, params, feature_stats):
    ranks = selection_data["ranks"]
    selected = {}
    for source_type in SOURCE_TYPES:
        source_data = selection_data["data"][source_type]
        if params["selection_mode"] == "threshold":
            selected[source_type] = select_threshold_for_source(source_data, params)
        elif params["selection_mode"] == "score":
            selected[source_type] = select_score_for_source(source_data, ranks, params, feature_stats)
        elif params["selection_mode"] == "js_only":
            selected[source_type] = select_js_only_for_source(source_data, params)
        elif params["selection_mode"] == "mse_only":
            selected[source_type] = select_mse_only_for_source(source_data, params)
        elif params["selection_mode"] == "margin_only":
            selected[source_type] = select_margin_only_for_source(source_data, params)
        elif params["selection_mode"] == "js_mse":
            selected[source_type] = select_js_mse_for_source(source_data, params)
        elif params["selection_mode"] == "mse_margin":
            selected[source_type] = select_mse_margin_for_source(source_data, params)
        elif params["selection_mode"] == "js_margin":
            selected[source_type] = select_js_margin_for_source(source_data, params)
        elif params["selection_mode"] == "entropy_only":
            selected[source_type] = select_entropy_only_for_source(source_data, params)
        elif params["selection_mode"] == "js_entropy":
            selected[source_type] = select_js_entropy_for_source(source_data, params)
        elif params["selection_mode"] == "mse_entropy":
            selected[source_type] = select_mse_entropy_for_source(source_data, params)
        elif params["selection_mode"] == "entropy_margin":
            selected[source_type] = select_entropy_margin_for_source(source_data, params)
        elif params["selection_mode"] == "js_mse_entropy":
            selected[source_type] = select_js_mse_entropy_for_source(source_data, params)
        elif params["selection_mode"] == "score_entropy":
            selected[source_type] = select_score_entropy_for_source(source_data, ranks, params, feature_stats)
        else:
            raise ValueError(f"Unknown selection_mode={params['selection_mode']}")
    return selected


def gather_selected_values(array, selected_indices):
    return array[np.arange(selected_indices.size), selected_indices]


def summarize_selection(selection_data, selected, sample_indices, args):
    ranks = selection_data["ranks"]
    labels = selection_data["labels"]
    rank_min = float(np.min(ranks))
    rank_max = float(np.max(ranks))
    source_metrics = {}
    all_correct = []
    all_mse = []
    all_ranks = []
    all_entropy = []

    for source_type in SOURCE_TYPES:
        source_data = selection_data["data"][source_type]
        selected_indices = selected[source_type][sample_indices]
        correct = gather_selected_values(source_data["correct"][sample_indices], selected_indices)
        mse = gather_selected_values(source_data["mse"][sample_indices], selected_indices)
        entropy = gather_selected_values(source_data["entropy"][sample_indices], selected_indices)
        rank_values = ranks[selected_indices]
        source_metrics[source_type] = {
            "count": int(sample_indices.size),
            "accuracy": float(np.mean(correct)) if correct.size else float("nan"),
            "mean_mse": float(np.mean(mse)) if mse.size else float("nan"),
            "mean_entropy": float(np.mean(entropy)) if entropy.size else float("nan"),
            "mean_rank": float(np.mean(rank_values)) if rank_values.size else float("nan"),
            "mean_rank_norm": float(np.mean((rank_values - rank_min) / max(rank_max - rank_min, 1e-12)))
            if rank_values.size else float("nan"),
        }
        all_correct.append(correct)
        all_mse.append(mse)
        all_ranks.append(rank_values)
        all_entropy.append(entropy)

    correct_all = np.concatenate(all_correct)
    mse_all = np.concatenate(all_mse)
    ranks_all = np.concatenate(all_ranks)
    entropy_all = np.concatenate(all_entropy)
    aggregate = {
        "count": int(correct_all.size),
        "accuracy": float(np.mean(correct_all)),
        "mean_mse": float(np.mean(mse_all)),
        "mean_entropy": float(np.mean(entropy_all)),
        "mean_rank": float(np.mean(ranks_all)),
        "mean_rank_norm": float(np.mean((ranks_all - rank_min) / max(rank_max - rank_min, 1e-12))),
    }
    objective_value = (
        source_metrics["adv"]["accuracy"]
        + args.clean_weight * source_metrics["clean"]["accuracy"]
        - args.mse_weight * aggregate["mean_mse"]
        - args.rank_weight * aggregate["mean_rank_norm"]
    )
    return source_metrics, aggregate, float(objective_value)


def suggest_params(trial, selection_modes):
    if len(selection_modes) == 1:
        selection_mode = selection_modes[0]
    else:
        selection_mode = trial.suggest_categorical("selection_mode", selection_modes)

    params = {"selection_mode": selection_mode}
    if selection_mode == "threshold":
        params.update({
            "js_threshold": trial.suggest_float("js_threshold", 1e-5, 0.25, log=True),
            "mse_threshold": trial.suggest_float("mse_threshold", 0.01, 0.50),
            "margin_threshold": trial.suggest_float("margin_threshold", -1.0, 1.0),
        })
    elif selection_mode == "score":
        params.update({
            "alpha": trial.suggest_float("alpha", 0.0, 2.0),
            "beta": trial.suggest_float("beta", 0.0, 2.0),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "delta": trial.suggest_float("delta", 0.0, 1.0),
        })
    elif selection_mode == "js_only":
        params["js_threshold"] = trial.suggest_float("js_threshold", 1e-5, 0.25, log=True)
    elif selection_mode == "mse_only":
        params["mse_threshold"] = trial.suggest_float("mse_threshold", 0.01, 0.50)
    elif selection_mode == "margin_only":
        params["margin_threshold"] = trial.suggest_float("margin_threshold", -1.0, 1.0)
    elif selection_mode == "js_mse":
        params.update({
            "js_threshold": trial.suggest_float("js_threshold", 1e-5, 0.25, log=True),
            "mse_threshold": trial.suggest_float("mse_threshold", 0.01, 0.50),
        })
    elif selection_mode == "mse_margin":
        params.update({
            "mse_threshold": trial.suggest_float("mse_threshold", 0.01, 0.50),
            "margin_threshold": trial.suggest_float("margin_threshold", -1.0, 1.0),
        })
    elif selection_mode == "js_margin":
        params.update({
            "js_threshold": trial.suggest_float("js_threshold", 1e-5, 0.25, log=True),
            "margin_threshold": trial.suggest_float("margin_threshold", -1.0, 1.0),
        })
    elif selection_mode == "entropy_only":
        params["entropy_threshold"] = trial.suggest_float("entropy_threshold", 0.0, 4.0)
    elif selection_mode == "js_entropy":
        params.update({
            "js_threshold": trial.suggest_float("js_threshold", 1e-5, 0.25, log=True),
            "entropy_threshold": trial.suggest_float("entropy_threshold", 0.0, 4.0),
        })
    elif selection_mode == "mse_entropy":
        params.update({
            "mse_threshold": trial.suggest_float("mse_threshold", 0.01, 0.50),
            "entropy_threshold": trial.suggest_float("entropy_threshold", 0.0, 4.0),
        })
    elif selection_mode == "entropy_margin":
        params.update({
            "entropy_threshold": trial.suggest_float("entropy_threshold", 0.0, 4.0),
            "margin_threshold": trial.suggest_float("margin_threshold", -1.0, 1.0),
        })
    elif selection_mode == "js_mse_entropy":
        params.update({
            "js_threshold": trial.suggest_float("js_threshold", 1e-5, 0.25, log=True),
            "mse_threshold": trial.suggest_float("mse_threshold", 0.01, 0.50),
            "entropy_threshold": trial.suggest_float("entropy_threshold", 0.0, 4.0),
        })
    elif selection_mode == "score_entropy":
        params.update({
            "alpha": trial.suggest_float("alpha", 0.0, 2.0),
            "beta": trial.suggest_float("beta", 0.0, 2.0),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "delta": trial.suggest_float("delta", 0.0, 1.0),
            "eta": trial.suggest_float("eta", 0.0, 2.0),
        })
    else:
        raise ValueError(f"Unknown selection_mode={selection_mode}")
    return params


def build_selected_rows(selection_data, selected, split_names, params, selector_name):
    rows = []
    ranks = selection_data["ranks"]
    labels = selection_data["labels"]
    for source_type in SOURCE_TYPES:
        source_data = selection_data["data"][source_type]
        selected_indices = selected[source_type]
        selected_ranks = ranks[selected_indices]
        selected_top1 = gather_selected_values(source_data["top1"], selected_indices)
        selected_correct = gather_selected_values(source_data["correct"], selected_indices)
        selected_mse = gather_selected_values(source_data["mse"], selected_indices)
        selected_margin = gather_selected_values(source_data["margin"], selected_indices)
        selected_js = gather_selected_values(source_data["js_score"], selected_indices)
        selected_entropy = gather_selected_values(source_data["entropy"], selected_indices)
        selected_confidence = gather_selected_values(source_data["probs"].max(axis=-1), selected_indices)
        for sample_id in range(labels.size):
            rows.append({
                "selector": selector_name,
                "sample_id": int(sample_id),
                "source_index": int(source_data["source_indices"][sample_id]),
                "split": split_names[sample_id],
                "source_type": source_type,
                "label": int(labels[sample_id]),
                "selection_mode": params["selection_mode"],
                "selected_rank": int(selected_ranks[sample_id]),
                "selected_rank_index": int(selected_indices[sample_id]),
                "selected_top1": int(selected_top1[sample_id]),
                "selected_confidence": float(selected_confidence[sample_id]),
                "selected_correct": int(selected_correct[sample_id]),
                "selected_mse": float(selected_mse[sample_id]),
                "selected_margin": float(selected_margin[sample_id]),
                "selected_js_feature": float(selected_js[sample_id]),
                "selected_entropy": float(selected_entropy[sample_id]),
                "raw_top1": int(source_data["raw_top1"][sample_id]),
                "raw_confidence": float(source_data["raw_confidence"][sample_id]),
            })
    return rows


def build_summary_rows(selection_data, selected, split_map, args, selector_name):
    rows = []
    for split_name, indices in split_map.items():
        source_metrics, aggregate, objective_value = summarize_selection(
            selection_data,
            selected,
            indices,
            args,
        )
        for source_type, metrics in source_metrics.items():
            rows.append({
                "selector": selector_name,
                "split": split_name,
                "source_type": source_type,
                "count": metrics["count"],
                "accuracy": metrics["accuracy"],
                "mean_mse": metrics["mean_mse"],
                "mean_entropy": metrics["mean_entropy"],
                "mean_rank": metrics["mean_rank"],
                "mean_rank_norm": metrics["mean_rank_norm"],
                "objective": "",
            })
        rows.append({
            "selector": selector_name,
            "split": split_name,
            "source_type": "all",
            "count": aggregate["count"],
            "accuracy": aggregate["accuracy"],
            "mean_mse": aggregate["mean_mse"],
            "mean_entropy": aggregate["mean_entropy"],
            "mean_rank": aggregate["mean_rank"],
            "mean_rank_norm": aggregate["mean_rank_norm"],
            "objective": objective_value,
        })
    return rows


def summarize_for_splits(selection_data, selected, split_map, args):
    evaluation = {}
    for split_name, indices in split_map.items():
        source_metrics, aggregate, objective_value = summarize_selection(
            selection_data,
            selected,
            indices,
            args,
        )
        evaluation[split_name] = {
            "source_metrics": source_metrics,
            "aggregate": aggregate,
            "objective": objective_value,
        }
    return evaluation


def build_mode_best_row(selector_name, best_trial_number, best_value, params, evaluation):
    row = {
        "selector": selector_name,
        "best_trial_number": best_trial_number,
        "best_value": best_value,
        "selection_mode": params.get("selection_mode", ""),
        "js_threshold": params.get("js_threshold", ""),
        "mse_threshold": params.get("mse_threshold", ""),
        "entropy_threshold": params.get("entropy_threshold", ""),
        "margin_threshold": params.get("margin_threshold", ""),
        "alpha": params.get("alpha", ""),
        "beta": params.get("beta", ""),
        "gamma": params.get("gamma", ""),
        "delta": params.get("delta", ""),
        "eta": params.get("eta", ""),
    }
    for split_name in ("tune", "holdout", "all"):
        split_eval = evaluation[split_name]
        row[f"{split_name}_objective"] = split_eval["objective"]
        row[f"{split_name}_all_acc"] = split_eval["aggregate"]["accuracy"]
        row[f"{split_name}_all_mse"] = split_eval["aggregate"]["mean_mse"]
        row[f"{split_name}_all_entropy"] = split_eval["aggregate"]["mean_entropy"]
        row[f"{split_name}_all_rank"] = split_eval["aggregate"]["mean_rank"]
        row[f"{split_name}_clean_acc"] = split_eval["source_metrics"]["clean"]["accuracy"]
        row[f"{split_name}_clean_entropy"] = split_eval["source_metrics"]["clean"]["mean_entropy"]
        row[f"{split_name}_adv_acc"] = split_eval["source_metrics"]["adv"]["accuracy"]
        row[f"{split_name}_adv_entropy"] = split_eval["source_metrics"]["adv"]["mean_entropy"]
    return row


def trial_to_row(trial, study_name="", optimization_mode=""):
    row = {
        "study_name": study_name,
        "optimization_mode": optimization_mode,
        "number": trial.number,
        "state": str(trial.state),
        "value": trial.value if trial.value is not None else "",
    }
    row.update({f"param_{key}": value for key, value in trial.params.items()})
    row.update({f"metric_{key}": value for key, value in trial.user_attrs.items()})
    return row


def write_trials_csv(path, study_records):
    rows = []
    for record in study_records:
        study = record["study"]
        for trial in study.trials:
            rows.append(trial_to_row(
                trial,
                study_name=study.study_name,
                optimization_mode=record["mode"],
            ))
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    write_csv(path, rows, fieldnames)


def plot_objective_history(study, output_path, dpi):
    values = [trial.value for trial in study.trials if trial.value is not None]
    if not values:
        return
    best_values = np.maximum.accumulate(np.asarray(values, dtype=np.float64))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(np.arange(len(values)), values, marker="o", linewidth=1.2, label="trial")
    ax.plot(np.arange(len(best_values)), best_values, linewidth=2.0, label="best")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective")
    ax.set_title("Optuna objective history")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_selected_rank_distribution(selection_data, selected, output_path, dpi):
    ranks = selection_data["ranks"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.38
    x = np.arange(ranks.size)
    for offset, source_type in ((-width / 2, "clean"), (width / 2, "adv")):
        counts = np.bincount(selected[source_type], minlength=ranks.size)
        ax.bar(x + offset, counts, width=width, label=source_type)
    ax.set_xlabel("Selected rank")
    ax.set_ylabel("Count")
    ax.set_title("Selected rank distribution")
    ax.set_xticks(x)
    ax.set_xticklabels([str(rank) for rank in ranks])
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_fixed_rank_selection(selection_data, fixed_rank):
    ranks = selection_data["ranks"]
    matches = np.where(ranks == int(fixed_rank))[0]
    if matches.size != 1:
        raise ValueError(f"Fixed rank {fixed_rank} is not available in ranks={ranks.tolist()}.")
    rank_index = int(matches[0])
    sample_count = int(selection_data["labels"].size)
    return {
        source_type: np.full(sample_count, rank_index, dtype=np.int64)
        for source_type in SOURCE_TYPES
    }


def build_comparison_rows(selection_data, selector_results, split_map, args, fixed_ranks, selector_names):
    rows = []
    selected_by_name = {result["selector"]: result["selected"] for result in selector_results}
    entries = []
    for fixed_rank in fixed_ranks:
        entries.append((
            f"rank{fixed_rank}",
            "fixed_rank",
            build_fixed_rank_selection(selection_data, fixed_rank),
        ))
    for selector_name in selector_names:
        if selector_name in selected_by_name:
            entries.append((selector_name, "selector", selected_by_name[selector_name]))

    for method, method_type, selected in entries:
        evaluation = summarize_for_splits(selection_data, selected, split_map, args)
        for split_name in ("tune", "holdout", "all"):
            split_eval = evaluation[split_name]
            for source_type in SOURCE_TYPES:
                metrics = split_eval["source_metrics"][source_type]
                rows.append({
                    "method": method,
                    "method_type": method_type,
                    "split": split_name,
                    "source_type": source_type,
                    "count": metrics["count"],
                    "accuracy": metrics["accuracy"],
                    "mean_mse": metrics["mean_mse"],
                    "mean_entropy": metrics["mean_entropy"],
                    "mean_rank": metrics["mean_rank"],
                    "objective": "",
                })
            aggregate = split_eval["aggregate"]
            rows.append({
                "method": method,
                "method_type": method_type,
                "split": split_name,
                "source_type": "all",
                "count": aggregate["count"],
                "accuracy": aggregate["accuracy"],
                "mean_mse": aggregate["mean_mse"],
                "mean_entropy": aggregate["mean_entropy"],
                "mean_rank": aggregate["mean_rank"],
                "objective": split_eval["objective"],
            })
    return rows


def plot_fixed_rank_selector_comparison(comparison_rows, output_path, dpi):
    all_rows = [
        row for row in comparison_rows
        if row["split"] == "all" and row["source_type"] in {"clean", "adv", "all"}
    ]
    if not all_rows:
        return
    methods = []
    method_types = {}
    metrics = {}
    for row in all_rows:
        method = row["method"]
        if method not in methods:
            methods.append(method)
        method_types[method] = row["method_type"]
        metrics[(method, row["source_type"])] = row

    x = np.arange(len(methods))
    labels = [method.replace("_", "\n") for method in methods]
    colors = ["#4C78A8" if method_types[method] == "fixed_rank" else "#F58518" for method in methods]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    width = 0.36
    clean_acc = [float(metrics[(method, "clean")]["accuracy"]) for method in methods]
    adv_acc = [float(metrics[(method, "adv")]["accuracy"]) for method in methods]
    axes[0].bar(x - width / 2, adv_acc, width=width, label="adv acc", color="#D55E00", alpha=0.86)
    axes[0].bar(x + width / 2, clean_acc, width=width, label="clean acc", color="#0072B2", alpha=0.82)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Fixed ranks vs tuned selectors")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)

    all_mse = [float(metrics[(method, "all")]["mean_mse"]) for method in methods]
    all_rank = [float(metrics[(method, "all")]["mean_rank"]) for method in methods]
    axes[1].bar(x, all_rank, width=0.58, color=colors, alpha=0.40, label="mean rank")
    axes_mse = axes[1].twinx()
    axes_mse.plot(x, all_mse, marker="o", color="#009E73", linewidth=2.0, label="mean MSE")
    axes[1].set_ylabel("Mean rank")
    axes_mse.set_ylabel("Mean MSE")
    axes[1].set_xlabel("Method")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)

    fixed_patch = plt.Rectangle((0, 0), 1, 1, color="#4C78A8", alpha=0.40)
    selector_patch = plt.Rectangle((0, 0), 1, 1, color="#F58518", alpha=0.40)
    handles_left, labels_left = axes[1].get_legend_handles_labels()
    handles_right, labels_right = axes_mse.get_legend_handles_labels()
    axes[1].legend(
        [fixed_patch, selector_patch] + handles_left + handles_right,
        ["fixed rank", "selector"] + labels_left + labels_right,
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_fixed_full_score_params(path):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    params = dict(payload.get("params", {}))
    if not params:
        raise ValueError(f"No params found in full-score config: {path}")
    params["selection_mode"] = "score"
    for key in ("alpha", "beta", "gamma", "delta"):
        if key not in params:
            raise ValueError(f"Full-score config is missing {key}: {path}")
        params[key] = float(params[key])
    return params


def main():
    args = parse_args()
    if args.n_trials <= 0:
        raise ValueError("--n_trials must be positive.")
    if args.plot_dpi <= 0:
        raise ValueError("--plot_dpi must be positive.")
    selection_modes = parse_selection_modes(args.selection_modes)
    fixed_rank_baselines = parse_rank_list(args.fixed_rank_baselines)
    comparison_selectors = parse_csv_items(args.comparison_selectors)
    input_dir = Path(args.input_dir)
    predictions_path = input_dir / "rank_growth_predictions.pt"
    if not predictions_path.exists():
        raise FileNotFoundError(f"rank_growth_predictions.pt not found under {input_dir}.")

    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    predictions = torch_load_cpu(predictions_path)
    selection_data = extract_rank_tensors(predictions, max_samples=args.max_samples)
    selection_data = attach_raw_margin_features(predictions, selection_data, args)
    selection_data = attach_js_features(selection_data)

    sample_count = int(len(selection_data["labels"]))
    tune_indices, holdout_indices = build_split_indices(sample_count, args.tune_ratio, args.seed)
    split_names = np.full(sample_count, "holdout", dtype=object)
    split_names[tune_indices] = "tune"
    split_map = {
        "tune": tune_indices,
        "holdout": holdout_indices,
        "all": np.arange(sample_count, dtype=np.int64),
    }
    feature_stats = build_feature_stats(selection_data, tune_indices)

    study_records = []
    selector_results = []
    for mode_index, selection_mode in enumerate(selection_modes):
        sampler = optuna.samplers.TPESampler(seed=args.seed + mode_index)
        study = optuna.create_study(
            study_name=f"{args.study_name}_{selection_mode}",
            direction="maximize",
            sampler=sampler,
        )

        def objective(trial, mode=selection_mode):
            params = suggest_params(trial, [mode])
            selected = select_ranks(selection_data, params, feature_stats)
            source_metrics, aggregate, objective_value = summarize_selection(
                selection_data,
                selected,
                tune_indices,
                args,
            )
            trial.set_user_attr("selection_mode", params["selection_mode"])
            trial.set_user_attr("tune_adv_acc", source_metrics["adv"]["accuracy"])
            trial.set_user_attr("tune_clean_acc", source_metrics["clean"]["accuracy"])
            trial.set_user_attr("tune_mean_mse", aggregate["mean_mse"])
            trial.set_user_attr("tune_mean_entropy", aggregate["mean_entropy"])
            trial.set_user_attr("tune_mean_rank", aggregate["mean_rank"])
            trial.set_user_attr("tune_mean_rank_norm", aggregate["mean_rank_norm"])
            return objective_value

        study.optimize(objective, n_trials=args.n_trials)
        best_params = dict(study.best_trial.params)
        best_params["selection_mode"] = selection_mode
        best_selected = select_ranks(selection_data, best_params, feature_stats)
        best_evaluation = summarize_for_splits(selection_data, best_selected, split_map, args)
        selector_results.append({
            "selector": selection_mode,
            "best_trial_number": study.best_trial.number,
            "best_value": study.best_value,
            "params": best_params,
            "selected": best_selected,
            "evaluation": best_evaluation,
        })
        study_records.append({"mode": selection_mode, "study": study})

    if args.include_exp007_full_score:
        full_score_params = load_fixed_full_score_params(args.include_exp007_full_score)
        full_score_selected = select_ranks(selection_data, full_score_params, feature_stats)
        full_score_evaluation = summarize_for_splits(
            selection_data,
            full_score_selected,
            split_map,
            args,
        )
        selector_results.append({
            "selector": "exp007_full_score",
            "best_trial_number": "",
            "best_value": full_score_evaluation["tune"]["objective"],
            "params": full_score_params,
            "selected": full_score_selected,
            "evaluation": full_score_evaluation,
            "source_config": args.include_exp007_full_score,
        })

    global_best = max(
        selector_results,
        key=lambda item: float(item["evaluation"]["tune"]["objective"]),
    )
    summary_rows = []
    selected_rows = []
    mode_best_rows = []
    for result in selector_results:
        summary_rows.extend(build_summary_rows(
            selection_data,
            result["selected"],
            split_map,
            args,
            selector_name=result["selector"],
        ))
        selected_rows.extend(build_selected_rows(
            selection_data,
            result["selected"],
            split_names,
            result["params"],
            selector_name=result["selector"],
        ))
        mode_best_rows.append(build_mode_best_row(
            selector_name=result["selector"],
            best_trial_number=result["best_trial_number"],
            best_value=result["best_value"],
            params=result["params"],
            evaluation=result["evaluation"],
        ))
    comparison_rows = build_comparison_rows(
        selection_data,
        selector_results,
        split_map,
        args,
        fixed_rank_baselines,
        comparison_selectors,
    )

    write_trials_csv(output_dir / "trials.csv", study_records)
    write_csv(
        output_dir / "selected_rows.csv",
        selected_rows,
        [
            "selector",
            "sample_id",
            "source_index",
            "split",
            "source_type",
            "label",
            "selection_mode",
            "selected_rank",
            "selected_rank_index",
            "selected_top1",
            "selected_confidence",
            "selected_correct",
            "selected_mse",
            "selected_margin",
            "selected_js_feature",
            "selected_entropy",
            "raw_top1",
            "raw_confidence",
        ],
    )
    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        [
            "selector",
            "split",
            "source_type",
            "count",
            "accuracy",
            "mean_mse",
            "mean_entropy",
            "mean_rank",
            "mean_rank_norm",
            "objective",
        ],
    )
    write_csv(
        output_dir / "mode_best_summary.csv",
        mode_best_rows,
        [
            "selector",
            "best_trial_number",
            "best_value",
            "selection_mode",
            "js_threshold",
            "mse_threshold",
            "entropy_threshold",
            "margin_threshold",
            "alpha",
            "beta",
            "gamma",
            "delta",
            "eta",
            "tune_objective",
            "tune_all_acc",
            "tune_all_mse",
            "tune_all_entropy",
            "tune_all_rank",
            "tune_clean_acc",
            "tune_clean_entropy",
            "tune_adv_acc",
            "tune_adv_entropy",
            "holdout_objective",
            "holdout_all_acc",
            "holdout_all_mse",
            "holdout_all_entropy",
            "holdout_all_rank",
            "holdout_clean_acc",
            "holdout_clean_entropy",
            "holdout_adv_acc",
            "holdout_adv_entropy",
            "all_objective",
            "all_all_acc",
            "all_all_mse",
            "all_all_entropy",
            "all_all_rank",
            "all_clean_acc",
            "all_clean_entropy",
            "all_adv_acc",
            "all_adv_entropy",
        ],
    )
    write_csv(
        output_dir / "fixed_rank_selector_comparison.csv",
        comparison_rows,
        [
            "method",
            "method_type",
            "split",
            "source_type",
            "count",
            "accuracy",
            "mean_mse",
            "mean_entropy",
            "mean_rank",
            "objective",
        ],
    )

    best_config = {
        "global_best_selector": global_best["selector"],
        "global_best_trial_number": global_best["best_trial_number"],
        "global_best_value": global_best["evaluation"]["tune"]["objective"],
        "global_best_params": global_best["params"],
        "feature_stats": feature_stats,
        "mode_results": {
            result["selector"]: {
                "best_trial_number": result["best_trial_number"],
                "best_value": result["best_value"],
                "params": result["params"],
                "evaluation": result["evaluation"],
                **({"source_config": result["source_config"]} if "source_config" in result else {}),
            }
            for result in selector_results
        },
    }
    with open(output_dir / "best_config.json", "w", encoding="utf-8") as file:
        json.dump(to_jsonable(best_config), file, ensure_ascii=False, indent=2)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for record in study_records:
        plot_objective_history(
            record["study"],
            plots_dir / f"objective_history_{record['mode']}.{args.plot_format}",
            args.plot_dpi,
        )
    plot_selected_rank_distribution(
        selection_data,
        global_best["selected"],
        plots_dir / f"selected_rank_distribution_global_best.{args.plot_format}",
        args.plot_dpi,
    )
    comparison_plot_name = f"fixed_rank_selector_comparison.{args.plot_format}"
    plot_fixed_rank_selector_comparison(
        comparison_rows,
        plots_dir / comparison_plot_name,
        args.plot_dpi,
    )

    meta = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "predictions_path": str(predictions_path),
        "study_name": args.study_name,
        "n_trials": args.n_trials,
        "seed": args.seed,
        "tune_ratio": args.tune_ratio,
        "tune_indices": tune_indices.tolist(),
        "holdout_indices": holdout_indices.tolist(),
        "selection_modes": selection_modes,
        "include_exp007_full_score": args.include_exp007_full_score,
        "fixed_rank_baselines": fixed_rank_baselines,
        "comparison_selectors": comparison_selectors,
        "objective": args.objective,
        "clean_weight": args.clean_weight,
        "mse_weight": args.mse_weight,
        "rank_weight": args.rank_weight,
        "max_samples": args.max_samples,
        "sample_count": sample_count,
        "ranks": selection_data["ranks"].tolist(),
        "raw_logits_recomputed": bool(selection_data.get("raw_logits_recomputed", False)),
        "input_meta": selection_data["meta"],
        "outputs": {
            "trials": "trials.csv",
            "best_config": "best_config.json",
            "selected_rows": "selected_rows.csv",
            "summary": "summary.csv",
            "mode_best_summary": "mode_best_summary.csv",
            "fixed_rank_selector_comparison": "fixed_rank_selector_comparison.csv",
            "meta": "meta.json",
            "plots": "plots/",
            "fixed_rank_selector_comparison_plot": f"plots/{comparison_plot_name}",
        },
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(to_jsonable(meta), file, ensure_ascii=False, indent=2)

    print(f"Saved Optuna rank-growth selection results to: {output_dir}")


if __name__ == "__main__":
    main()
