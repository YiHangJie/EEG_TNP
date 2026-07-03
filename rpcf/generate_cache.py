import argparse
import datetime
import logging
import os
import shutil

from runtime_env import configure_runtime_env

configure_runtime_env()

import numpy as np
import torch

from data.subject_ea import get_protocol_tag, prepare_subject_fold
from purify import purify
from utils.experiment_artifacts import eeg_classification_collate, safe_token

from rpcf.core import (
    DATASET_LOADERS,
    build_attack,
    build_cache_path,
    load_model_checkpoint,
    parse_int_csv,
    parse_path_csv,
    rank_config_map,
    seed_everything,
    stable_subset_indices,
    validate_rpcf_cache,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成攻击一次、跨 rank 严格对齐的 RPCF 训练缓存。"
    )
    parser.add_argument("--dataset", default="thubenchmark", choices=DATASET_LOADERS)
    parser.add_argument(
        "--model", default="eegnet", choices=["eegnet", "tsception", "atcnet", "conformer"]
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attack", default="autoattack", choices=["fgsm", "pgd", "cw", "autoattack"])
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--sample_num", type=int, default=512)
    parser.add_argument("--attack_batch_size", type=int, default=32)
    parser.add_argument("--ranks", default="15,20,25,30,35,40")
    parser.add_argument(
        "--configs",
        default=",".join(
            f"PTR3d_8_2048_rank{rank}_3d_interpolate.yaml"
            for rank in (15, 20, 25, 30, 35, 40)
        ),
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--tag", default="rpcf")
    parser.add_argument("--output_dir", default="./purified_data/rpcf_train")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--checkpoint_every", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep_work_dir", action="store_true")
    parser.add_argument("--base_only", action="store_true", help="只生成/校验 base shard 后退出。")
    parser.add_argument("--rank_shard_only", action="store_true", help="只生成指定 rank shard，不写最终统一 cache。")
    parser.add_argument("--finalize_only", action="store_true", help="只汇总已有 rank shard 并写最终统一 cache。")
    return parser.parse_args()


def save_partial(path, clean_parts, adv_parts, clean_mses, adv_mses, completed):
    torch.save(
        {
            "x_pur": torch.stack(clean_parts, dim=0) if clean_parts else None,
            "x_adv_pur": torch.stack(adv_parts, dim=0) if adv_parts else None,
            "clean_mses": clean_mses,
            "adv_mses": adv_mses,
            "completed": completed,
        },
        path,
    )


def main():
    args = parse_args()
    mode_count = int(args.base_only) + int(args.rank_shard_only) + int(args.finalize_only)
    if mode_count > 1:
        raise ValueError("base_only, rank_shard_only and finalize_only are mutually exclusive.")
    ranks = parse_int_csv(args.ranks)
    configs = parse_path_csv(args.configs)
    rank_configs = rank_config_map(ranks, configs)
    if args.attack_batch_size <= 0 or args.checkpoint_every <= 0:
        raise ValueError("Batch size and checkpoint interval must be positive.")
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = args.output_path or build_cache_path(
        args.output_dir,
        args.dataset,
        args.model,
        args.fold,
        args.seed,
        args.attack,
        args.eps,
        args.sample_num,
        args.tag,
    )
    if os.path.exists(output_path) and not args.overwrite:
        validate_rpcf_cache(torch.load(output_path, map_location="cpu"))
        print(output_path)
        return

    work_dir = f"{output_path}.work"
    if args.overwrite and os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs("./log_purify", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        "./log_purify",
        f"rpcf_cache_{args.dataset}_{args.model}_fold{args.fold}_seed{args.seed}_"
        f"eps{safe_token(args.eps)}_{timestamp}.log",
    )
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.info("RPCF cache generation start: %s", vars(args))
    logging.info("Ranks/configs: %s", rank_configs)

    dataset, info = DATASET_LOADERS[args.dataset]()
    train_dataset, _, _, split_path = prepare_subject_fold(
        dataset_name=args.dataset,
        dataset=dataset,
        info=info,
        fold_id=args.fold,
        seed=args.seed,
        use_ea=False,
    )
    protocol = get_protocol_tag(use_ea=False)
    selected_indices, selection_seed = stable_subset_indices(
        len(train_dataset), args.sample_num, args.seed, args.fold
    )
    base_path = os.path.join(work_dir, "base.pth")

    if os.path.exists(base_path) and not args.overwrite:
        base = torch.load(base_path, map_location="cpu")
        if base.get("source_indices") != selected_indices:
            raise ValueError("Existing RPCF base shard uses different source indices.")
        x = base["x"].float()
        x_adv = base["x_adv"].float()
        labels = base["labels"].long()
        logging.info("Resume from base shard: %s", base_path)
    else:
        model = load_model_checkpoint(
            args.model, args.dataset, info, args.checkpoint_path, device
        )
        model.eval()
        attack = build_attack(
            args.attack, model, args.eps, info, device, seed=args.seed
        )
        x_parts = []
        x_adv_parts = []
        label_parts = []
        for start in range(0, len(selected_indices), args.attack_batch_size):
            batch_indices = selected_indices[start:start + args.attack_batch_size]
            clean, target = eeg_classification_collate(
                [train_dataset[index] for index in batch_indices]
            )
            clean = clean.to(device)
            target = target.to(device)
            model.zero_grad(set_to_none=True)
            adversarial = attack(clean, target)
            model.zero_grad(set_to_none=True)
            x_parts.append(clean.detach().cpu().float())
            x_adv_parts.append(adversarial.detach().cpu().float())
            label_parts.append(target.detach().cpu().long())
            logging.info(
                "Generated cached adversarial batch: %d/%d",
                min(start + len(batch_indices), len(selected_indices)),
                len(selected_indices),
            )
        x = torch.cat(x_parts, dim=0)
        x_adv = torch.cat(x_adv_parts, dim=0)
        labels = torch.cat(label_parts, dim=0)
        torch.save(
            {
                "x": x,
                "x_adv": x_adv,
                "labels": labels,
                "source_indices": selected_indices,
            },
            base_path,
        )
        del attack
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.base_only:
        logging.info("Base-only RPCF cache step completed: %s", base_path)
        print(base_path)
        return

    clean_rank_parts = []
    adv_rank_parts = []
    rank_metrics = []
    for rank, config in rank_configs.items():
        rank_path = os.path.join(work_dir, f"rank{rank}.pth")
        partial_path = os.path.join(work_dir, f"rank{rank}.partial.pth")
        if args.finalize_only:
            if not os.path.exists(rank_path):
                raise FileNotFoundError(f"Missing RPCF rank shard for finalize: {rank_path}")
            rank_payload = torch.load(rank_path, map_location="cpu")
            logging.info("Load completed rank shard for finalize: rank=%d", rank)
        elif os.path.exists(rank_path) and not args.overwrite:
            rank_payload = torch.load(rank_path, map_location="cpu")
            logging.info("Resume completed rank shard: rank=%d", rank)
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
                    clean_parts = list(partial["x_pur"])
                    adv_parts = list(partial["x_adv_pur"])
                    clean_mses = list(partial["clean_mses"])
                    adv_mses = list(partial["adv_mses"])
                    logging.info(
                        "Resume partial rank shard: rank=%d, completed=%d", rank, completed
                    )

            args.config = config
            args.visualize = False
            for sample_index in range(completed, x.size(0)):
                clean_pur, clean_mse = purify(
                    args,
                    sample_index,
                    x[sample_index],
                    info["sampling_rate"],
                    device,
                    logging,
                )
                adv_pur, adv_mse = purify(
                    args,
                    sample_index + x.size(0),
                    x_adv[sample_index],
                    info["sampling_rate"],
                    device,
                    logging,
                )
                clean_parts.append(clean_pur.detach().cpu().float())
                adv_parts.append(adv_pur.detach().cpu().float())
                clean_mses.append(float(clean_mse))
                adv_mses.append(float(adv_mse))
                completed = sample_index + 1
                if completed % args.checkpoint_every == 0 or completed == x.size(0):
                    save_partial(
                        partial_path,
                        clean_parts,
                        adv_parts,
                        clean_mses,
                        adv_mses,
                        completed,
                    )
                logging.info(
                    "RPCF rank purification: rank=%d, sample=%d/%d",
                    rank,
                    completed,
                    x.size(0),
                )
            rank_payload = {
                "rank": rank,
                "config": config,
                "x_pur": torch.stack(clean_parts, dim=0),
                "x_adv_pur": torch.stack(adv_parts, dim=0),
                "clean_mses": clean_mses,
                "adv_mses": adv_mses,
            }
            torch.save(rank_payload, rank_path)
            if os.path.exists(partial_path):
                os.remove(partial_path)

        if int(rank_payload.get("rank", rank)) != rank:
            raise ValueError(f"Rank shard identity mismatch: expected {rank}.")
        if rank_payload.get("config") != config:
            raise ValueError(f"Rank {rank} config mismatch.")
        if rank_payload["x_pur"].shape != x.shape:
            raise ValueError(f"Rank {rank} clean purified shard shape mismatch.")
        if rank_payload["x_adv_pur"].shape != x.shape:
            raise ValueError(f"Rank {rank} adversarial purified shard shape mismatch.")
        clean_rank_parts.append(rank_payload["x_pur"].float())
        adv_rank_parts.append(rank_payload["x_adv_pur"].float())
        rank_metrics.append(
            {
                "rank": rank,
                "config": config,
                "mean_clean_mse": float(np.mean(rank_payload["clean_mses"])),
                "mean_adv_mse": float(np.mean(rank_payload["adv_mses"])),
            }
        )

    if args.rank_shard_only:
        logging.info("Rank-shard-only RPCF cache step completed: %s", work_dir)
        print(work_dir)
        return

    payload = {
        "x": x,
        "x_adv": x_adv,
        "x_pur_by_rank": torch.stack(clean_rank_parts, dim=1),
        "x_adv_pur_by_rank": torch.stack(adv_rank_parts, dim=1),
        "labels": labels,
        "source_indices": selected_indices,
        "ranks": ranks,
        "meta": {
            "kind": "rpcf_train_cache",
            "dataset": args.dataset,
            "model": args.model,
            "fold": args.fold,
            "seed": args.seed,
            "protocol": protocol,
            "use_ea": False,
            "source_split": "train",
            "attack": args.attack,
            "eps": args.eps,
            "checkpoint_path": args.checkpoint_path,
            "sample_num": args.sample_num,
            "selection_strategy": "random_without_replacement",
            "selection_seed_rule": "seed + fold * 1000",
            "selection_seed": selection_seed,
            "source_indices": selected_indices,
            "split_path": split_path,
            "ranks": ranks,
            "configs": configs,
            "rank_metrics": rank_metrics,
            "tag": args.tag,
        },
    }
    payload = validate_rpcf_cache(payload)
    torch.save(payload, output_path)
    logging.info("Saved unified RPCF cache: %s", output_path)
    if not args.keep_work_dir:
        shutil.rmtree(work_dir)
    print(output_path)


if __name__ == "__main__":
    main()
