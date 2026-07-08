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
    parse_int_csv,
    parse_path_csv,
    rank_config_map,
    seed_everything,
    stable_subset_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="对 RPCF attack artifact 的相同子集执行多 rank EEG_TNP 净化评估。"
    )
    parser.add_argument("--attack_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--dataset", required=True, choices=DATASET_LOADERS)
    parser.add_argument(
        "--model", required=True, choices=MODEL_CHOICES
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--sample_num", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ranks", default="15,20,25,30,35,40")
    parser.add_argument(
        "--configs",
        default=",".join(
            f"PTR3d_8_2048_rank{rank}_3d_interpolate.yaml"
            for rank in (15, 20, 25, 30, 35, 40)
        ),
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--checkpoint_every", type=int, default=8)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep_work_dir", action="store_true")
    return parser.parse_args()


def validate_attack_payload(payload, args):
    if not isinstance(payload, dict):
        raise ValueError("Attack artifact must be a dict.")
    for key in ("clean", "adversarial", "labels", "source_indices", "meta"):
        if key not in payload:
            raise ValueError(f"Attack artifact missing {key}.")
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
            matches = actual is not None and abs(float(actual) - float(expected)) <= 1e-12
        else:
            matches = str(actual) == str(expected)
        if not matches:
            raise ValueError(
                f"Attack artifact mismatch: {key}={actual}, expected {expected}."
            )
    clean = torch.as_tensor(payload["clean"]).float()
    adversarial = torch.as_tensor(payload["adversarial"]).float()
    labels = torch.as_tensor(payload["labels"]).long().view(-1)
    if clean.shape != adversarial.shape or clean.size(0) != labels.numel():
        raise ValueError("Attack artifact tensor shapes are inconsistent.")
    return clean, adversarial, labels, [int(i) for i in payload["source_indices"]], meta


def save_partial(path, clean_parts, adv_parts, clean_mses, adv_mses, completed):
    torch.save(
        {
            "clean_pur": torch.stack(clean_parts) if clean_parts else None,
            "adv_pur": torch.stack(adv_parts) if adv_parts else None,
            "clean_mses": clean_mses,
            "adv_mses": adv_mses,
            "completed": completed,
        },
        path,
    )


def main():
    args = parse_args()
    if os.path.exists(args.output_path) and not args.overwrite:
        print(args.output_path)
        return
    seed_everything(args.seed)
    ranks = parse_int_csv(args.ranks)
    configs = parse_path_csv(args.configs)
    rank_configs = rank_config_map(ranks, configs)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    logging.basicConfig(
        filename=f"{os.path.splitext(args.output_path)[0]}.log",
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    attack_payload = torch.load(args.attack_path, map_location="cpu")
    clean, adversarial, labels, attack_source_indices, attack_meta = (
        validate_attack_payload(attack_payload, args)
    )
    sample_num = min(args.sample_num, clean.size(0))
    selected_positions, selection_seed = stable_subset_indices(
        clean.size(0), sample_num, args.seed, args.fold
    )
    index_tensor = torch.as_tensor(selected_positions, dtype=torch.long)
    clean = clean.index_select(0, index_tensor)
    adversarial = adversarial.index_select(0, index_tensor)
    labels = labels.index_select(0, index_tensor)
    source_indices = [attack_source_indices[position] for position in selected_positions]

    dataset, info = DATASET_LOADERS[args.dataset]()
    del dataset
    model = load_model_checkpoint(
        args.model, args.dataset, info, args.checkpoint_path, device
    )
    model.eval()
    work_dir = f"{args.output_path}.work"
    if args.overwrite and os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    clean_rank_parts = []
    adv_rank_parts = []
    metrics = []
    for rank, config in rank_configs.items():
        rank_path = os.path.join(work_dir, f"rank{rank}.pth")
        partial_path = os.path.join(work_dir, f"rank{rank}.partial.pth")
        if os.path.exists(rank_path) and not args.overwrite:
            rank_payload = torch.load(rank_path, map_location="cpu")
        else:
            clean_parts = []
            adv_parts = []
            clean_mses = []
            adv_mses = []
            completed = 0
            if os.path.exists(partial_path) and not args.overwrite:
                partial = torch.load(partial_path, map_location="cpu")
                completed = int(partial.get("completed", 0))
                if completed:
                    clean_parts = list(partial["clean_pur"])
                    adv_parts = list(partial["adv_pur"])
                    clean_mses = list(partial["clean_mses"])
                    adv_mses = list(partial["adv_mses"])
            args.config = config
            args.visualize = False
            for sample_index in range(completed, sample_num):
                clean_pur, clean_mse = purify(
                    args,
                    sample_index,
                    clean[sample_index],
                    info["sampling_rate"],
                    device,
                    logging,
                    classifier=model,
                )
                adv_pur, adv_mse = purify(
                    args,
                    sample_index + sample_num,
                    adversarial[sample_index],
                    info["sampling_rate"],
                    device,
                    logging,
                    classifier=model,
                )
                clean_parts.append(clean_pur.detach().cpu().float())
                adv_parts.append(adv_pur.detach().cpu().float())
                clean_mses.append(float(clean_mse))
                adv_mses.append(float(adv_mse))
                completed = sample_index + 1
                if completed % args.checkpoint_every == 0 or completed == sample_num:
                    save_partial(
                        partial_path,
                        clean_parts,
                        adv_parts,
                        clean_mses,
                        adv_mses,
                        completed,
                    )
                logging.info(
                    "Purification eval: rank=%d sample=%d/%d",
                    rank,
                    completed,
                    sample_num,
                )
            rank_payload = {
                "rank": rank,
                "config": config,
                "clean_pur": torch.stack(clean_parts),
                "adv_pur": torch.stack(adv_parts),
                "clean_mses": clean_mses,
                "adv_mses": adv_mses,
            }
            torch.save(rank_payload, rank_path)
            if os.path.exists(partial_path):
                os.remove(partial_path)

        clean_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(rank_payload["clean_pur"], labels),
            batch_size=args.batch_size,
            shuffle=False,
        )
        adv_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(rank_payload["adv_pur"], labels),
            batch_size=args.batch_size,
            shuffle=False,
        )
        clean_metric = evaluate_classifier(model, clean_loader, device)
        adv_metric = evaluate_classifier(model, adv_loader, device)
        metrics.append(
            {
                "rank": rank,
                "config": config,
                "purified_clean_accuracy": clean_metric["accuracy"],
                "purified_clean_loss": clean_metric["loss"],
                "purified_adv_accuracy": adv_metric["accuracy"],
                "purified_adv_loss": adv_metric["loss"],
                "mean_clean_mse": float(np.mean(rank_payload["clean_mses"])),
                "mean_adv_mse": float(np.mean(rank_payload["adv_mses"])),
            }
        )
        clean_rank_parts.append(rank_payload["clean_pur"].float())
        adv_rank_parts.append(rank_payload["adv_pur"].float())

    payload = {
        "clean": clean,
        "adversarial": adversarial,
        "clean_pur_by_rank": torch.stack(clean_rank_parts, dim=1),
        "adv_pur_by_rank": torch.stack(adv_rank_parts, dim=1),
        "labels": labels,
        "source_indices": source_indices,
        "ranks": ranks,
        "metrics": metrics,
        "meta": {
            "kind": "rpcf_purification_eval",
            "dataset": args.dataset,
            "model": args.model,
            "fold": args.fold,
            "seed": args.seed,
            "eps": args.eps,
            "checkpoint_path": args.checkpoint_path,
            "attack_path": args.attack_path,
            "attack_meta": attack_meta,
            "sample_num": sample_num,
            "selection_strategy": "random_without_replacement",
            "selection_seed_rule": "seed + fold * 1000",
            "selection_seed": selection_seed,
            "selected_positions": selected_positions,
            "source_indices": source_indices,
            "ranks": ranks,
            "configs": configs,
        },
    }
    torch.save(payload, args.output_path)
    logging.info("Saved purification evaluation: %s", args.output_path)
    if not args.keep_work_dir:
        shutil.rmtree(work_dir)
    print(args.output_path)


if __name__ == "__main__":
    main()
