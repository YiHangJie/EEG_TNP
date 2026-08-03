"""从已有六秩 EEG_TNP 输出构造无标签的样本级 knee 自动选秩 payload。"""

import argparse
import os

from runtime_env import configure_runtime_env

configure_runtime_env()

import torch

from rpcf.analyze_exp027_oracle_rank import (
    DEFAULT_RANKS,
    load_method_payloads,
)
from rpcf.core import parse_int_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="EXP-030：按样本重构 MSE–rank 曲线的 knee 自动选秩。"
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--payload_paths", nargs="+", required=True)
    parser.add_argument(
        "--expected_ranks",
        default=",".join(str(rank) for rank in DEFAULT_RANKS),
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def reconstruction_mse_curves(inputs, purified_by_rank):
    """返回每个样本、每个 rank 相对其输入的重构 MSE。"""
    if purified_by_rank.ndim != inputs.ndim + 1:
        raise ValueError("purified_by_rank must add exactly one rank dimension.")
    if purified_by_rank.shape[0] != inputs.shape[0]:
        raise ValueError("inputs and purified_by_rank sample count mismatch.")
    if purified_by_rank.shape[2:] != inputs.shape[1:]:
        raise ValueError("inputs and purified_by_rank trial shape mismatch.")
    return (purified_by_rank - inputs[:, None]).square().flatten(2).mean(2)


def select_log_mse_knee(mse_curves, ranks):
    """
    用 log-MSE 曲线相对端点连线的最大垂距寻找 diminishing-return knee。

    先取 cumulative-min envelope，避免独立 TN 优化的微小非单调抖动改变拐点；
    该函数只接收重构误差和 rank，不接收标签或分类器输出。
    """
    mse_curves = torch.as_tensor(mse_curves).detach().float()
    ranks = [int(rank) for rank in ranks]
    if mse_curves.ndim != 2 or mse_curves.size(1) != len(ranks):
        raise ValueError("mse_curves must have shape [sample, rank].")
    if ranks != sorted(set(ranks)):
        raise ValueError("ranks must be strictly increasing and unique.")
    if len(ranks) < 3:
        raise ValueError("At least three ranks are required for knee selection.")

    monotone_mse = torch.cummin(mse_curves.clamp_min(1e-12), dim=1).values
    log_mse = monotone_mse.log()
    denominator = (log_mse[:, :1] - log_mse[:, -1:]).clamp_min(1e-12)
    normalized_mse = (log_mse - log_mse[:, -1:]) / denominator
    rank_tensor = torch.tensor(ranks, dtype=log_mse.dtype, device=log_mse.device)
    normalized_rank = (rank_tensor - rank_tensor[0]) / (
        rank_tensor[-1] - rank_tensor[0]
    )
    knee_scores = (1.0 - normalized_rank.unsqueeze(0)) - normalized_mse
    # torch.argmax 的首个最大值自然实现低 rank tie-break。
    selected_indices = knee_scores.argmax(dim=1)
    return {
        "selected_indices": selected_indices,
        "monotone_mse": monotone_mse,
        "normalized_log_mse": normalized_mse,
        "knee_scores": knee_scores,
    }


def gather_selected(purified_by_rank, selected_indices):
    selected_indices = torch.as_tensor(selected_indices).long().view(-1)
    if selected_indices.numel() != purified_by_rank.size(0):
        raise ValueError("selected_indices sample count mismatch.")
    return purified_by_rank[
        torch.arange(purified_by_rank.size(0)), selected_indices
    ]


def _diagnostics(inputs, mse_curves, selection, ranks):
    diagnostics = []
    selected_indices = selection["selected_indices"]
    for sample_index, rank_index in enumerate(selected_indices.tolist()):
        selected_rank = int(ranks[rank_index])
        selected_mse = float(mse_curves[sample_index, rank_index])
        variance = float(inputs[sample_index].var(unbiased=False))
        curve = {
            "ranks": list(ranks),
            "mse": [float(value) for value in mse_curves[sample_index]],
            "monotone_mse": [
                float(value) for value in selection["monotone_mse"][sample_index]
            ],
            "normalized_log_mse": [
                float(value)
                for value in selection["normalized_log_mse"][sample_index]
            ],
            "knee_scores": [
                float(value) for value in selection["knee_scores"][sample_index]
            ],
            "selected_rank": selected_rank,
        }
        diagnostics.append(
            {
                "method": "rank_log_mse_knee",
                "final_rank_vector": [selected_rank],
                "mean_rank": float(selected_rank),
                "min_rank": selected_rank,
                "max_rank": selected_rank,
                "effective_parameters": -1,
                "stage_trajectory": [
                    {
                        "stage_index": 0,
                        "resolution": int(inputs.shape[-1]),
                        "rank_vector_before": [int(ranks[-1])],
                        "rank_vector_after": [selected_rank],
                        "pruned_by_bond": {},
                        "regrown_by_bond": {},
                        "pruned_count": int(ranks[-1] - selected_rank),
                        "regrown_count": 0,
                        "mse": selected_mse,
                        "normalized_residual": selected_mse / max(variance, 1e-12),
                        "heldout_mse": None,
                        "rank_knee_curve": curve,
                    }
                ],
                "rank_knee_curve": curve,
                "rank_inference_signals": [
                    "input_reconstruction_mse",
                    "log_mse_rank_curve",
                    "maximum_chord_distance_knee",
                ],
                "uses_labels": False,
                "uses_classifier_logits": False,
                "purification_time_sec": 0.0,
                "mse_to_input": selected_mse,
            }
        )
    return diagnostics


def build_payload(method_data):
    ranks = list(method_data["ranks"])
    selections = {}
    outputs = {}
    diagnostics = {}
    mses = {}
    for input_kind, input_key, purified_key in (
        ("clean", "clean", "clean_pur_by_rank"),
        ("adv", "adversarial", "adv_pur_by_rank"),
    ):
        inputs = method_data[input_key]
        purified = method_data[purified_key]
        mse_curves = reconstruction_mse_curves(inputs, purified)
        selection = select_log_mse_knee(mse_curves, ranks)
        outputs[input_kind] = gather_selected(
            purified, selection["selected_indices"]
        )
        diagnostics[input_kind] = _diagnostics(
            inputs, mse_curves, selection, ranks
        )
        mses[input_kind] = [
            float(mse_curves[index, rank_index])
            for index, rank_index in enumerate(
                selection["selected_indices"].tolist()
            )
        ]
        selections[input_kind] = selection

    meta = dict(method_data["meta"])
    meta.update(
        {
            "kind": "exp030_auto_rank_eval",
            "experiment_id": "EXP-030",
            "method": method_data["method"],
            "variant": "auto_rank_log_mse_knee",
            "config": None,
            "candidate_ranks": ranks,
            "candidate_rank_paths": method_data["rank_paths"],
            "rank_inference_uses_labels": False,
            "rank_inference_uses_classifier_logits": False,
            "classifier_loaded_after_rank_inference": True,
            "selection_strategy": "maximum_chord_distance_on_log_mse_curve",
        }
    )
    return {
        "clean": method_data["clean"],
        "adversarial": method_data["adversarial"],
        "clean_pur": outputs["clean"],
        "adv_pur": outputs["adv"],
        "labels": method_data["labels"],
        "source_indices": method_data["source_indices"],
        "clean_mses": mses["clean"],
        "adv_mses": mses["adv"],
        "clean_rank_diagnostics": diagnostics["clean"],
        "adv_rank_diagnostics": diagnostics["adv"],
        "meta": meta,
    }


def main():
    args = parse_args()
    if os.path.exists(args.output_path) and not args.overwrite:
        print(args.output_path)
        return
    ranks = parse_int_csv(args.expected_ranks)
    method_data = load_method_payloads(
        args.method, args.payload_paths, expected_ranks=ranks
    )
    payload = build_payload(method_data)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(payload, args.output_path)
    print(args.output_path)


if __name__ == "__main__":
    main()
