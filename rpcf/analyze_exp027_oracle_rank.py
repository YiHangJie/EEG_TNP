import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

from runtime_env import configure_runtime_env

configure_runtime_env()

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/eegap_matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rpcf.core import DATASET_LOADERS, load_model_checkpoint, parse_int_csv


DEFAULT_RANKS = (15, 20, 25, 30, 35, 40)
PROTOCOL_KEYS = ("dataset", "model", "fold", "seed", "eps", "sample_num")


def parse_args():
    parser = argparse.ArgumentParser(
        description="EXP-027：计算 EEG_TNP 样本级动态 rank 的 oracle headroom。"
    )
    parser.add_argument(
        "--method_payload",
        action="append",
        nargs="+",
        required=True,
        metavar="METHOD_OR_PATH",
        help=(
            "可重复传入；每组第一个值是方法名，后续值是该方法的单 rank 或多 rank "
            "rpcf_purification_eval payload。"
        ),
    )
    parser.add_argument(
        "--expected_ranks",
        default=",".join(str(rank) for rank in DEFAULT_RANKS),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _float_matches(actual, expected, atol=1e-12):
    return actual is not None and abs(float(actual) - float(expected)) <= atol


def _validate_meta(meta, path):
    if not isinstance(meta, dict) or meta.get("kind") != "rpcf_purification_eval":
        raise ValueError(f"{path}: meta.kind must be rpcf_purification_eval.")
    for key in PROTOCOL_KEYS:
        if key not in meta:
            raise ValueError(f"{path}: meta missing {key}.")
    for key in ("checkpoint_path", "attack_path"):
        if not meta.get(key):
            raise ValueError(f"{path}: meta missing {key}.")


def _reorder_payload(payload, order):
    index = torch.as_tensor(order, dtype=torch.long)
    reordered = dict(payload)
    for key in (
        "clean",
        "adversarial",
        "clean_pur_by_rank",
        "adv_pur_by_rank",
        "labels",
    ):
        reordered[key] = torch.as_tensor(payload[key]).index_select(0, index)
    reordered["source_indices"] = [
        int(payload["source_indices"][position]) for position in order
    ]
    return reordered


def _align_payload(payload, canonical_indices, path):
    source_indices = [int(index) for index in payload["source_indices"]]
    if len(source_indices) != len(set(source_indices)):
        raise ValueError(f"{path}: source_indices must be unique.")
    if set(source_indices) != set(canonical_indices):
        raise ValueError(f"{path}: source_indices set mismatch.")
    position_by_source = {source: position for position, source in enumerate(source_indices)}
    order = [position_by_source[source] for source in canonical_indices]
    return _reorder_payload(payload, order)


def _normalize_purification_payload(path):
    payload = torch_load_cpu(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: purification payload must be a dict.")
    required = {
        "clean",
        "adversarial",
        "clean_pur_by_rank",
        "adv_pur_by_rank",
        "labels",
        "source_indices",
        "ranks",
        "meta",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{path}: missing keys {missing}.")
    _validate_meta(payload["meta"], path)

    normalized = dict(payload)
    for key in ("clean", "adversarial", "clean_pur_by_rank", "adv_pur_by_rank"):
        normalized[key] = torch.as_tensor(payload[key]).detach().cpu().float()
    normalized["labels"] = torch.as_tensor(payload["labels"]).detach().cpu().long().view(-1)
    normalized["source_indices"] = [int(index) for index in payload["source_indices"]]
    normalized["ranks"] = [int(rank) for rank in payload["ranks"]]

    sample_count = normalized["labels"].numel()
    if normalized["clean"].size(0) != sample_count:
        raise ValueError(f"{path}: clean/labels sample count mismatch.")
    if normalized["adversarial"].shape != normalized["clean"].shape:
        raise ValueError(f"{path}: adversarial shape must match clean shape.")
    if len(normalized["source_indices"]) != sample_count:
        raise ValueError(f"{path}: source_indices/labels count mismatch.")
    rank_count = len(normalized["ranks"])
    expected_shape = (sample_count, rank_count, *normalized["clean"].shape[1:])
    for key in ("clean_pur_by_rank", "adv_pur_by_rank"):
        if tuple(normalized[key].shape) != expected_shape:
            raise ValueError(
                f"{path}: {key} shape={tuple(normalized[key].shape)}, "
                f"expected={expected_shape}."
            )
    return normalized


def _protocol_value_matches(key, actual, expected):
    if key == "eps":
        return _float_matches(actual, expected)
    return str(actual) == str(expected)


def load_method_payloads(method, paths, expected_ranks):
    """合并一个方法的多份净化产物，并按 source index 严格对齐。"""
    if not paths:
        raise ValueError(f"{method}: at least one purification payload is required.")
    payloads = [_normalize_purification_payload(path) for path in paths]
    canonical_indices = payloads[0]["source_indices"]
    payloads = [
        _align_payload(payload, canonical_indices, path)
        for payload, path in zip(payloads, paths)
    ]
    base = payloads[0]
    base_meta = base["meta"]
    rank_tensors = {}
    rank_paths = {}

    for payload, path in zip(payloads, paths):
        meta = payload["meta"]
        for key in PROTOCOL_KEYS:
            if not _protocol_value_matches(key, meta.get(key), base_meta.get(key)):
                raise ValueError(
                    f"{method}: {path} meta.{key}={meta.get(key)} does not match "
                    f"{base_meta.get(key)}."
                )
        for key in ("checkpoint_path", "attack_path"):
            if str(meta.get(key)) != str(base_meta.get(key)):
                raise ValueError(f"{method}: {path} meta.{key} mismatch.")
        if not torch.equal(payload["labels"], base["labels"]):
            raise ValueError(f"{method}: {path} labels mismatch.")
        if not torch.equal(payload["clean"], base["clean"]):
            raise ValueError(f"{method}: {path} clean tensor mismatch.")
        if not torch.equal(payload["adversarial"], base["adversarial"]):
            raise ValueError(f"{method}: {path} adversarial tensor mismatch.")
        for rank_index, rank in enumerate(payload["ranks"]):
            if rank in rank_tensors:
                raise ValueError(
                    f"{method}: duplicate rank {rank} in {path} and {rank_paths[rank]}."
                )
            rank_tensors[rank] = (
                payload["clean_pur_by_rank"][:, rank_index],
                payload["adv_pur_by_rank"][:, rank_index],
            )
            rank_paths[rank] = str(path)

    actual_ranks = sorted(rank_tensors)
    if actual_ranks != sorted(expected_ranks):
        raise ValueError(
            f"{method}: ranks={actual_ranks}, expected={sorted(expected_ranks)}."
        )
    ranks = sorted(expected_ranks)
    return {
        "method": method,
        "paths": [str(path) for path in paths],
        "rank_paths": {str(rank): rank_paths[rank] for rank in ranks},
        "ranks": ranks,
        "clean": base["clean"],
        "adversarial": base["adversarial"],
        "clean_pur_by_rank": torch.stack(
            [rank_tensors[rank][0] for rank in ranks], dim=1
        ),
        "adv_pur_by_rank": torch.stack(
            [rank_tensors[rank][1] for rank in ranks], dim=1
        ),
        "labels": base["labels"],
        "source_indices": list(canonical_indices),
        "meta": dict(base_meta),
    }


def align_methods(method_data):
    """跨方法只要求同一原始 trial；各方法 adversarial tensor 必须保持独立。"""
    methods = list(method_data)
    if not methods:
        raise ValueError("At least one method is required.")
    base = method_data[methods[0]]
    canonical_indices = base["source_indices"]
    canonical_set = set(canonical_indices)
    for method in methods[1:]:
        current = method_data[method]
        if set(current["source_indices"]) != canonical_set:
            raise ValueError(f"{method}: cross-method source_indices set mismatch.")
        position_by_source = {
            source: position for position, source in enumerate(current["source_indices"])
        }
        order = [position_by_source[source] for source in canonical_indices]
        index = torch.as_tensor(order, dtype=torch.long)
        for key in (
            "clean",
            "adversarial",
            "clean_pur_by_rank",
            "adv_pur_by_rank",
            "labels",
        ):
            current[key] = current[key].index_select(0, index)
        current["source_indices"] = list(canonical_indices)
        for key in ("dataset", "model", "fold", "seed", "eps", "sample_num"):
            if not _protocol_value_matches(
                key, current["meta"].get(key), base["meta"].get(key)
            ):
                raise ValueError(f"{method}: cross-method meta.{key} mismatch.")
        if current["ranks"] != base["ranks"]:
            raise ValueError(f"{method}: cross-method ranks mismatch.")
        if not torch.equal(current["labels"], base["labels"]):
            raise ValueError(f"{method}: cross-method labels mismatch.")
        if not torch.equal(current["clean"], base["clean"]):
            raise ValueError(f"{method}: cross-method clean tensor mismatch.")
    return method_data


def evaluate_tensor_grid(model, tensor_grid, batch_size, device):
    sample_count, rank_count = tensor_grid.shape[:2]
    flat = tensor_grid.reshape(sample_count * rank_count, *tensor_grid.shape[2:])
    logits = []
    model.eval()
    with torch.no_grad():
        for start in range(0, flat.size(0), batch_size):
            batch = flat[start : start + batch_size].to(device)
            logits.append(model(batch).detach().cpu().float())
    return torch.cat(logits, dim=0).reshape(sample_count, rank_count, -1)


def classification_grid(logits, labels):
    probs = torch.softmax(logits, dim=-1)
    labels_grid = labels.view(-1, 1, 1).expand(-1, logits.size(1), 1)
    target_probs = probs.gather(dim=-1, index=labels_grid).squeeze(-1)
    other_probs = probs.clone()
    other_probs.scatter_(dim=-1, index=labels_grid, value=-1.0)
    margin = target_probs - other_probs.max(dim=-1).values
    prediction = probs.argmax(dim=-1)
    correct = prediction.eq(labels.view(-1, 1))
    return {
        "logits": logits,
        "prediction": prediction,
        "correct": correct,
        "margin": margin,
    }


def per_sample_mse(purified, reference):
    delta = purified - reference.unsqueeze(1)
    return delta.square().flatten(start_dim=2).mean(dim=2)


def select_best_fixed(clean_correct, adv_correct, ranks):
    """按 adv acc、clean acc、低 rank 的顺序选择最佳固定 rank。"""
    best_index = None
    best_key = None
    for rank_index, rank in enumerate(ranks):
        key = (
            float(adv_correct[:, rank_index].float().mean().item()),
            float(clean_correct[:, rank_index].float().mean().item()),
            -int(rank),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_index = rank_index
    return int(best_index)


def select_robust_first_oracle(
    clean_correct,
    adv_correct,
    clean_margin,
    adv_margin,
    ranks,
):
    """逐 trial 选择共同 rank；真实标签只用于测量理论上限。"""
    selected = []
    for sample_index in range(clean_correct.size(0)):
        best_index = None
        best_key = None
        for rank_index, rank in enumerate(ranks):
            key = (
                int(adv_correct[sample_index, rank_index].item()),
                int(clean_correct[sample_index, rank_index].item()),
                float(adv_margin[sample_index, rank_index].item()),
                float(clean_margin[sample_index, rank_index].item()),
                -int(rank),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = rank_index
        selected.append(best_index)
    return torch.as_tensor(selected, dtype=torch.long)


def gather_by_rank(values, selected_indices):
    return values.gather(1, selected_indices.view(-1, 1)).squeeze(1)


def paired_bootstrap_difference(oracle_correct, fixed_correct, samples, seed):
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive.")
    oracle = np.asarray(oracle_correct, dtype=np.float64)
    fixed = np.asarray(fixed_correct, dtype=np.float64)
    if oracle.shape != fixed.shape or oracle.ndim != 1 or oracle.size == 0:
        raise ValueError("oracle/fixed correctness must be non-empty aligned vectors.")
    delta = oracle - fixed
    rng = np.random.RandomState(int(seed))
    indices = rng.randint(0, delta.size, size=(int(samples), delta.size))
    bootstrap = delta[indices].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "samples": int(samples),
        "seed": int(seed),
    }


def _decision(headroom_pp, ci_low_pp, clean_delta_pp):
    if headroom_pp >= 2.0 and ci_low_pp > 0.0 and clean_delta_pp >= -1.0:
        return "strong_support"
    if headroom_pp >= 0.5:
        return "weak_signal"
    return "not_supported"


def analyze_method(method_data, model, device, batch_size, bootstrap_samples, seed):
    ranks = method_data["ranks"]
    labels = method_data["labels"]
    clean_logits = evaluate_tensor_grid(
        model, method_data["clean_pur_by_rank"], batch_size, device
    )
    adv_logits = evaluate_tensor_grid(
        model, method_data["adv_pur_by_rank"], batch_size, device
    )
    clean = classification_grid(clean_logits, labels)
    adv = classification_grid(adv_logits, labels)
    clean_mse = per_sample_mse(method_data["clean_pur_by_rank"], method_data["clean"])
    adv_mse = per_sample_mse(method_data["adv_pur_by_rank"], method_data["adversarial"])

    fixed_rows = []
    for rank_index, rank in enumerate(ranks):
        clean_correct = clean["correct"][:, rank_index]
        adv_correct = adv["correct"][:, rank_index]
        fixed_rows.append(
            {
                "method": method_data["method"],
                "rank": int(rank),
                "sample_num": int(labels.numel()),
                "clean_accuracy": float(clean_correct.float().mean().item()),
                "adv_accuracy": float(adv_correct.float().mean().item()),
                "both_correct_rate": float((clean_correct & adv_correct).float().mean().item()),
                "mean_clean_margin": float(clean["margin"][:, rank_index].mean().item()),
                "mean_adv_margin": float(adv["margin"][:, rank_index].mean().item()),
                "mean_clean_mse": float(clean_mse[:, rank_index].mean().item()),
                "mean_adv_mse": float(adv_mse[:, rank_index].mean().item()),
            }
        )

    fixed_index = select_best_fixed(clean["correct"], adv["correct"], ranks)
    selected = select_robust_first_oracle(
        clean["correct"],
        adv["correct"],
        clean["margin"],
        adv["margin"],
        ranks,
    )
    fixed_clean = clean["correct"][:, fixed_index]
    fixed_adv = adv["correct"][:, fixed_index]
    fixed_both = fixed_clean & fixed_adv
    oracle_clean = gather_by_rank(clean["correct"], selected)
    oracle_adv = gather_by_rank(adv["correct"], selected)
    oracle_both = oracle_clean & oracle_adv
    selected_ranks = torch.as_tensor(ranks, dtype=torch.long)[selected]
    bootstrap = paired_bootstrap_difference(
        oracle_adv.numpy(), fixed_adv.numpy(), bootstrap_samples, seed
    )

    fixed_clean_acc = float(fixed_clean.float().mean().item())
    fixed_adv_acc = float(fixed_adv.float().mean().item())
    oracle_clean_acc = float(oracle_clean.float().mean().item())
    oracle_adv_acc = float(oracle_adv.float().mean().item())
    headroom_pp = 100.0 * (oracle_adv_acc - fixed_adv_acc)
    clean_delta_pp = 100.0 * (oracle_clean_acc - fixed_clean_acc)
    ci_low_pp = 100.0 * bootstrap["ci_low"]
    ci_high_pp = 100.0 * bootstrap["ci_high"]
    if int((fixed_adv & ~oracle_adv).sum().item()) != 0:
        raise AssertionError("Robust-first oracle unexpectedly loses a fixed-rank adv success.")

    summary = {
        "method": method_data["method"],
        "sample_num": int(labels.numel()),
        "best_fixed_rank": int(ranks[fixed_index]),
        "fixed_clean_accuracy": fixed_clean_acc,
        "fixed_adv_accuracy": fixed_adv_acc,
        "fixed_both_correct_rate": float(fixed_both.float().mean().item()),
        "oracle_clean_accuracy": oracle_clean_acc,
        "oracle_adv_accuracy": oracle_adv_acc,
        "oracle_both_correct_rate": float(oracle_both.float().mean().item()),
        "oracle_mean_rank": float(selected_ranks.float().mean().item()),
        "robust_headroom_pp": headroom_pp,
        "clean_delta_pp": clean_delta_pp,
        "robust_headroom_ci_low_pp": ci_low_pp,
        "robust_headroom_ci_high_pp": ci_high_pp,
        "adv_rescued_count": int((~fixed_adv & oracle_adv).sum().item()),
        "adv_lost_count": int((fixed_adv & ~oracle_adv).sum().item()),
        "clean_rescued_count": int((~fixed_clean & oracle_clean).sum().item()),
        "clean_lost_count": int((fixed_clean & ~oracle_clean).sum().item()),
        "decision": _decision(headroom_pp, ci_low_pp, clean_delta_pp),
        "bootstrap": bootstrap,
    }

    selected_rows = []
    for sample_index, source_index in enumerate(method_data["source_indices"]):
        rank_index = int(selected[sample_index].item())
        selected_rows.append(
            {
                "method": method_data["method"],
                "sample_id": sample_index,
                "source_index": int(source_index),
                "label": int(labels[sample_index].item()),
                "best_fixed_rank": int(ranks[fixed_index]),
                "selected_rank": int(ranks[rank_index]),
                "fixed_clean_correct": int(fixed_clean[sample_index].item()),
                "fixed_adv_correct": int(fixed_adv[sample_index].item()),
                "oracle_clean_prediction": int(clean["prediction"][sample_index, rank_index].item()),
                "oracle_adv_prediction": int(adv["prediction"][sample_index, rank_index].item()),
                "oracle_clean_correct": int(oracle_clean[sample_index].item()),
                "oracle_adv_correct": int(oracle_adv[sample_index].item()),
                "oracle_clean_margin": float(clean["margin"][sample_index, rank_index].item()),
                "oracle_adv_margin": float(adv["margin"][sample_index, rank_index].item()),
                "oracle_clean_mse": float(clean_mse[sample_index, rank_index].item()),
                "oracle_adv_mse": float(adv_mse[sample_index, rank_index].item()),
                "adv_rescued": int((~fixed_adv[sample_index] & oracle_adv[sample_index]).item()),
                "clean_rescued": int((~fixed_clean[sample_index] & oracle_clean[sample_index]).item()),
                "clean_lost": int((fixed_clean[sample_index] & ~oracle_clean[sample_index]).item()),
            }
        )
    distribution = Counter(int(rank) for rank in selected_ranks.tolist())
    distribution_rows = [
        {
            "method": method_data["method"],
            "rank": int(rank),
            "count": int(distribution.get(rank, 0)),
            "fraction": float(distribution.get(rank, 0) / labels.numel()),
        }
        for rank in ranks
    ]
    return fixed_rows, summary, selected_rows, distribution_rows


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def render_markdown(summaries, overall_decision):
    lines = [
        "# EXP-027 Oracle Rank Headroom",
        "",
        "| Method | Best fixed rank | Fixed clean | Fixed adv | Oracle clean | Oracle adv | Robust headroom | 95% CI | Mean oracle rank | Decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summaries:
        lines.append(
            "| {method} | {best_fixed_rank} | {fixed_clean_accuracy:.2%} | "
            "{fixed_adv_accuracy:.2%} | {oracle_clean_accuracy:.2%} | "
            "{oracle_adv_accuracy:.2%} | {robust_headroom_pp:+.2f} pp | "
            "[{robust_headroom_ci_low_pp:+.2f}, {robust_headroom_ci_high_pp:+.2f}] pp | "
            "{oracle_mean_rank:.2f} | {decision} |".format(**row)
        )
    lines.extend(
        [
            "",
            f"Overall decision: `{overall_decision}`.",
            "",
            "> Oracle 使用真实标签，只表示逐样本动态 rank 的理论上限，不是可部署防御结果。",
            "> Madry AT 与 RPCF_AT 分别使用自身 white-box AutoAttack；跨方法仅对齐原始 trial source indices。",
        ]
    )
    return "\n".join(lines) + "\n"


def plot_fixed_vs_oracle(summaries, output_path):
    methods = [row["method"] for row in summaries]
    x = np.arange(len(methods), dtype=np.float64)
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(7, 2.5 * len(methods)), 4.8))
    series = (
        ("Fixed clean", [row["fixed_clean_accuracy"] for row in summaries]),
        ("Fixed adv", [row["fixed_adv_accuracy"] for row in summaries]),
        ("Oracle clean", [row["oracle_clean_accuracy"] for row in summaries]),
        ("Oracle adv", [row["oracle_adv_accuracy"] for row in summaries]),
    )
    for index, (label, values) in enumerate(series):
        ax.bar(x + (index - 1.5) * width, values, width, label=label)
    ax.set_xticks(x, methods)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Best fixed rank vs robust-first oracle")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_rank_distribution(rows, output_path):
    methods = list(dict.fromkeys(row["method"] for row in rows))
    ranks = sorted(set(int(row["rank"]) for row in rows))
    x = np.arange(len(ranks), dtype=np.float64)
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method_index, method in enumerate(methods):
        by_rank = {
            int(row["rank"]): float(row["fraction"])
            for row in rows
            if row["method"] == method
        }
        values = [by_rank.get(rank, 0.0) for rank in ranks]
        ax.bar(
            x + (method_index - (len(methods) - 1) / 2) * width,
            values,
            width,
            label=method,
        )
    ax.set_xticks(x, [str(rank) for rank in ranks])
    ax.set_xlabel("Oracle selected rank")
    ax.set_ylabel("Fraction")
    ax.set_title("Oracle rank distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def determine_overall_decision(summaries):
    by_method = {row["method"]: row["decision"] for row in summaries}
    rpcf = by_method.get("rpcf_at")
    madry = by_method.get("madry_at")
    if rpcf == "strong_support" and madry == "strong_support":
        return "general_dynamic_rank_headroom"
    if rpcf == "strong_support":
        return "rpcf_at_conditional_synergy"
    if rpcf == "weak_signal":
        return "weak_signal_do_not_train_selector"
    return "no_support_do_not_train_selector"


def _prepare_output_dir(path, overwrite):
    output_dir = Path(path)
    target = output_dir / "summary.json"
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; use --overwrite to replace it.")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    args = parse_args()
    expected_ranks = parse_int_csv(args.expected_ranks)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    method_specs = {}
    for values in args.method_payload:
        method, *paths = values
        if method in method_specs:
            raise ValueError(f"Duplicate --method_payload method: {method}")
        method_specs[method] = paths
    method_data = {
        method: load_method_payloads(method, paths, expected_ranks)
        for method, paths in method_specs.items()
    }
    align_methods(method_data)

    output_dir = _prepare_output_dir(args.output_dir, args.overwrite)
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    dataset_name = next(iter(method_data.values()))["meta"]["dataset"]
    dataset, info = DATASET_LOADERS[dataset_name]()
    del dataset

    fixed_rows = []
    summaries = []
    selected_rows = []
    distribution_rows = []
    for method, data in method_data.items():
        model = load_model_checkpoint(
            data["meta"]["model"], dataset_name, info, data["meta"]["checkpoint_path"], device
        )
        method_result = analyze_method(
            data,
            model,
            device,
            args.batch_size,
            args.bootstrap_samples,
            args.seed,
        )
        method_fixed, method_summary, method_selected, method_distribution = method_result
        fixed_rows.extend(method_fixed)
        summaries.append(method_summary)
        selected_rows.extend(method_selected)
        distribution_rows.extend(method_distribution)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    overall_decision = determine_overall_decision(summaries)
    _write_csv(output_dir / "fixed_rank_metrics.csv", fixed_rows)
    _write_csv(output_dir / "oracle_summary.csv", summaries)
    _write_csv(output_dir / "oracle_selected_rows.csv", selected_rows)
    _write_csv(output_dir / "rank_distribution.csv", distribution_rows)
    summary_payload = {
        "kind": "exp027_oracle_rank_headroom",
        "experiment_id": "EXP-027",
        "expected_ranks": expected_ranks,
        "oracle_rule": [
            "adv_correct",
            "clean_correct",
            "adv_true_probability_margin",
            "clean_true_probability_margin",
            "lower_rank",
        ],
        "best_fixed_rule": ["adv_accuracy", "clean_accuracy", "lower_rank"],
        "methods": summaries,
        "overall_decision": overall_decision,
        "protocol": {
            key: next(iter(method_data.values()))["meta"].get(key)
            for key in PROTOCOL_KEYS
        },
        "artifacts": {
            method: {
                "checkpoint_path": data["meta"]["checkpoint_path"],
                "attack_path": data["meta"]["attack_path"],
                "purification_paths": data["paths"],
                "rank_paths": data["rank_paths"],
            }
            for method, data in method_data.items()
        },
        "limitations": [
            "Oracle uses true labels and is not deployable.",
            "Each method uses its own white-box adversarial examples.",
            "This experiment selects one terminal rank per sample and does not test stage-wise trajectories.",
        ],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(_jsonable(summary_payload), file, ensure_ascii=False, indent=2)
    with open(output_dir / "comparison.md", "w", encoding="utf-8") as file:
        file.write(render_markdown(summaries, overall_decision))
    plot_fixed_vs_oracle(summaries, output_dir / "fixed_vs_oracle.png")
    plot_rank_distribution(distribution_rows, output_dir / "rank_distribution.png")
    print(output_dir / "comparison.md")


if __name__ == "__main__":
    main()
