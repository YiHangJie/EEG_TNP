#!/usr/bin/env python3
"""将 EXP-028 bundle 转换为 EEG_TNP 净化入口可读取的 attack payload。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 trial_lowrank_analysis bundle 导出严格配对的净化输入。"
    )
    parser.add_argument("--bundle_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--dataset", default="thubenchmark")
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_attack_payload(
    bundle: dict,
    *,
    bundle_path: str,
    dataset: str,
    model: str,
    fold: int,
    seed: int,
    eps: float,
) -> dict:
    """校验 EXP-028 bundle，并保留原样本顺序导出 attack payload。"""
    if not isinstance(bundle, dict):
        raise ValueError("Source bundle must be a dict.")
    for key in ("clean", "adv", "labels", "metadata", "args"):
        if key not in bundle:
            raise ValueError(f"Source bundle missing {key}.")

    clean = torch.as_tensor(bundle["clean"]).detach().cpu().float().contiguous()
    adversarial = (
        torch.as_tensor(bundle["adv"]).detach().cpu().float().contiguous()
    )
    labels = torch.as_tensor(bundle["labels"]).detach().cpu().long().view(-1)
    if clean.shape != adversarial.shape:
        raise ValueError(
            "Source clean/adv shape mismatch: "
            f"{tuple(clean.shape)} vs {tuple(adversarial.shape)}."
        )
    if clean.size(0) != labels.numel():
        raise ValueError("Source labels length must match sample count.")

    metadata = pd.DataFrame(bundle["metadata"])
    if len(metadata) != clean.size(0):
        raise ValueError("Source metadata length must match sample count.")
    if "original_split_index" not in metadata.columns:
        raise KeyError("Source metadata must contain original_split_index.")
    source_indices = metadata["original_split_index"].astype(int).tolist()
    if len(set(source_indices)) != len(source_indices):
        raise ValueError("original_split_index values must be unique.")

    source_args = dict(bundle.get("args", {}))
    expected = {
        "dataset": dataset,
        "model": model,
        "fold": int(fold),
        "seed": int(seed),
        "eps": float(eps),
    }
    for key, expected_value in expected.items():
        actual = source_args.get(key)
        if key == "eps":
            matches = actual is not None and abs(
                float(actual) - float(expected_value)
            ) <= 1e-12
        else:
            matches = str(actual) == str(expected_value)
        if not matches:
            raise ValueError(
                f"Source bundle args mismatch: {key}={actual}, "
                f"expected {expected_value}."
            )

    return {
        "clean": clean,
        "adversarial": adversarial,
        "labels": labels,
        "source_indices": source_indices,
        "meta": {
            "kind": "exp028_trial_lowrank_attack_payload",
            "dataset": dataset,
            "model": model,
            "fold": int(fold),
            "seed": int(seed),
            "eps": float(eps),
            "attack": "pgd",
            "pgd_steps": int(source_args.get("pgd_steps", 0)),
            "pgd_alpha": float(source_args.get("pgd_alpha", 0.0)),
            "sample_num": int(clean.size(0)),
            "source_bundle": str(Path(bundle_path).resolve()),
            "source_indices": source_indices,
        },
    }


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    if output_path.exists() and not args.overwrite:
        print(output_path)
        return

    bundle = torch.load(args.bundle_path, map_location="cpu", weights_only=False)
    payload = build_attack_payload(
        bundle,
        bundle_path=args.bundle_path,
        dataset=args.dataset,
        model=args.model,
        fold=args.fold,
        seed=args.seed,
        eps=args.eps,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
