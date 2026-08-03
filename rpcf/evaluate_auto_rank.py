"""EXP-030：在同一 attack payload 上执行无标签的自动定秩 EEG_TNP。"""

import argparse
import logging
import os
import shutil

from runtime_env import configure_runtime_env

configure_runtime_env()

import numpy as np
import torch

from purify import purify
from rpcf.core import (
    DATASET_LOADERS,
    MODEL_CHOICES,
    evaluate_classifier,
    load_model_checkpoint,
    seed_everything,
    stable_subset_indices,
)
from rpcf.evaluate_purification import validate_attack_payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="EXP-030：无标签、无 classifier logits 的自动定秩净化评估。"
    )
    parser.add_argument("--attack_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--dataset", required=True, choices=DATASET_LOADERS)
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--method_tag", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--sample_num", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--checkpoint_every", type=int, default=4)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep_work_dir", action="store_true")
    return parser.parse_args()


def _torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _save_partial(
    path,
    clean_parts,
    adv_parts,
    clean_mses,
    adv_mses,
    clean_diagnostics,
    adv_diagnostics,
):
    torch.save(
        {
            "clean_pur": torch.stack(clean_parts) if clean_parts else None,
            "adv_pur": torch.stack(adv_parts) if adv_parts else None,
            "clean_mses": list(clean_mses),
            "adv_mses": list(adv_mses),
            "clean_rank_diagnostics": list(clean_diagnostics),
            "adv_rank_diagnostics": list(adv_diagnostics),
            "completed": len(clean_parts),
        },
        path,
    )


def _restore_partial(path):
    if not os.path.exists(path):
        return [], [], [], [], [], []
    payload = _torch_load_cpu(path)
    completed = int(payload.get("completed", 0))
    fields = []
    for key in (
        "clean_pur",
        "adv_pur",
        "clean_mses",
        "adv_mses",
        "clean_rank_diagnostics",
        "adv_rank_diagnostics",
    ):
        value = payload.get(key)
        if key.endswith("_pur"):
            value = [] if value is None else list(value)
        else:
            value = list(value or [])
        if len(value) != completed:
            raise ValueError(
                f"Partial checkpoint {key} has {len(value)} rows, "
                f"expected {completed}."
            )
        fields.append(value)
    return tuple(fields)


def _build_loader(data, labels, batch_size):
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, labels),
        batch_size=batch_size,
        shuffle=False,
    )


def _mean_rank(diagnostics):
    values = [float(row["mean_rank"]) for row in diagnostics]
    return float(np.mean(values)) if values else None


def main():
    args = parse_args()
    if os.path.exists(args.output_path) and not args.overwrite:
        print(args.output_path)
        return

    seed_everything(args.seed)
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    logging.basicConfig(
        filename=f"{os.path.splitext(args.output_path)[0]}.log",
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    attack_payload = _torch_load_cpu(args.attack_path)
    clean, adversarial, labels, attack_source_indices, attack_meta = (
        validate_attack_payload(attack_payload, args)
    )
    sample_num = min(int(args.sample_num), clean.size(0))
    selected_positions, selection_seed = stable_subset_indices(
        clean.size(0), sample_num, args.seed, args.fold
    )
    index = torch.as_tensor(selected_positions, dtype=torch.long)
    clean = clean.index_select(0, index)
    adversarial = adversarial.index_select(0, index)
    labels = labels.index_select(0, index)
    source_indices = [
        int(attack_source_indices[position]) for position in selected_positions
    ]

    # 这里只读取 sampling_rate；分类器刻意延迟到全部 rank inference 完成后加载。
    dataset, info = DATASET_LOADERS[args.dataset]()
    del dataset

    work_dir = f"{args.output_path}.work"
    partial_path = os.path.join(work_dir, "auto_rank.partial.pth")
    if args.overwrite and os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    (
        clean_parts,
        adv_parts,
        clean_mses,
        adv_mses,
        clean_diagnostics,
        adv_diagnostics,
    ) = _restore_partial(partial_path)
    completed = len(clean_parts)

    args.visualize = False
    for sample_index in range(completed, sample_num):
        # purify 收到的只有单个 EEG tensor；不传 labels，也不传 classifier。
        clean_pur, clean_mse, clean_diag = purify(
            args,
            sample_index,
            clean[sample_index],
            info["sampling_rate"],
            device,
            logging,
            classifier=None,
            return_diagnostics=True,
        )
        adv_pur, adv_mse, adv_diag = purify(
            args,
            sample_index + sample_num,
            adversarial[sample_index],
            info["sampling_rate"],
            device,
            logging,
            classifier=None,
            return_diagnostics=True,
        )
        for diagnostics in (clean_diag, adv_diag):
            if diagnostics.get("uses_labels") is not False:
                raise RuntimeError("Automatic rank diagnostics must declare uses_labels=False.")
            if diagnostics.get("uses_classifier_logits") is not False:
                raise RuntimeError(
                    "Automatic rank diagnostics must declare "
                    "uses_classifier_logits=False."
                )
        clean_parts.append(clean_pur.detach().cpu().float())
        adv_parts.append(adv_pur.detach().cpu().float())
        clean_mses.append(float(clean_mse))
        adv_mses.append(float(adv_mse))
        clean_diagnostics.append(clean_diag)
        adv_diagnostics.append(adv_diag)
        if (
            len(clean_parts) % int(args.checkpoint_every) == 0
            or len(clean_parts) == sample_num
        ):
            _save_partial(
                partial_path,
                clean_parts,
                adv_parts,
                clean_mses,
                adv_mses,
                clean_diagnostics,
                adv_diagnostics,
            )
        logging.info(
            "EXP-030 auto-rank: method=%s sample=%d/%d source_index=%d",
            args.method_tag,
            sample_index + 1,
            sample_num,
            source_indices[sample_index],
        )

    clean_pur = torch.stack(clean_parts)
    adv_pur = torch.stack(adv_parts)

    # 分类器只在 rank 已冻结、净化张量已生成后用于结果评估。
    model = load_model_checkpoint(
        args.model, args.dataset, info, args.checkpoint_path, device
    )
    model.eval()
    clean_metric = evaluate_classifier(
        model, _build_loader(clean_pur, labels, args.batch_size), device
    )
    adv_metric = evaluate_classifier(
        model, _build_loader(adv_pur, labels, args.batch_size), device
    )
    payload = {
        "clean": clean,
        "adversarial": adversarial,
        "clean_pur": clean_pur,
        "adv_pur": adv_pur,
        "labels": labels,
        "source_indices": source_indices,
        "clean_mses": clean_mses,
        "adv_mses": adv_mses,
        "clean_rank_diagnostics": clean_diagnostics,
        "adv_rank_diagnostics": adv_diagnostics,
        "metrics": {
            "purified_clean_accuracy": float(clean_metric["accuracy"]),
            "purified_clean_loss": float(clean_metric["loss"]),
            "purified_adv_accuracy": float(adv_metric["accuracy"]),
            "purified_adv_loss": float(adv_metric["loss"]),
            "mean_clean_mse": float(np.mean(clean_mses)),
            "mean_adv_mse": float(np.mean(adv_mses)),
            "mean_clean_rank": _mean_rank(clean_diagnostics),
            "mean_adv_rank": _mean_rank(adv_diagnostics),
        },
        "meta": {
            "kind": "exp030_auto_rank_eval",
            "experiment_id": "EXP-030",
            "dataset": args.dataset,
            "model": args.model,
            "method": args.method_tag,
            "variant": args.variant,
            "fold": int(args.fold),
            "seed": int(args.seed),
            "eps": float(args.eps),
            "sample_num": sample_num,
            "checkpoint_path": args.checkpoint_path,
            "attack_path": args.attack_path,
            "attack_meta": attack_meta,
            "config": args.config,
            "selection_strategy": "random_without_replacement",
            "selection_seed_rule": "seed + fold * 1000",
            "selection_seed": selection_seed,
            "selected_positions": selected_positions,
            "source_indices": source_indices,
            "rank_inference_uses_labels": False,
            "rank_inference_uses_classifier_logits": False,
            "classifier_loaded_after_rank_inference": True,
        },
    }
    torch.save(payload, args.output_path)
    logging.info("Saved EXP-030 auto-rank payload: %s", args.output_path)
    if not args.keep_work_dir:
        shutil.rmtree(work_dir)
    print(args.output_path)


if __name__ == "__main__":
    main()
