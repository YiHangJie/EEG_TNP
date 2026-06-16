import argparse
import logging
import os

from runtime_env import configure_runtime_env

configure_runtime_env()

import torch

from data.subject_ea import get_protocol_tag, prepare_subject_fold
from utils.experiment_artifacts import eeg_classification_collate

from rpcf.core import (
    DATASET_LOADERS,
    build_attack,
    evaluate_classifier,
    load_model_checkpoint,
    seed_everything,
    stable_subset_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="对指定 AT/RPCF checkpoint 运行 white-box attack 并保存统一评估数据。"
    )
    parser.add_argument("--dataset", required=True, choices=DATASET_LOADERS)
    parser.add_argument(
        "--model", required=True, choices=["eegnet", "tsception", "atcnet", "conformer"]
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--method_tag", required=True)
    parser.add_argument("--attack", default="autoattack", choices=["fgsm", "pgd", "cw", "autoattack"])
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sample_num", type=int, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.exists(args.output_path) and not args.overwrite:
        print(args.output_path)
        return
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    logging.basicConfig(
        filename=f"{os.path.splitext(args.output_path)[0]}.log",
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    dataset, info = DATASET_LOADERS[args.dataset]()
    _, _, test_dataset, split_path = prepare_subject_fold(
        dataset_name=args.dataset,
        dataset=dataset,
        info=info,
        fold_id=args.fold,
        seed=args.seed,
        use_ea=False,
    )
    if args.sample_num is None:
        source_indices = list(range(len(test_dataset)))
        selection_seed = None
    else:
        sample_num = min(args.sample_num, len(test_dataset))
        source_indices, selection_seed = stable_subset_indices(
            len(test_dataset), sample_num, args.seed, args.fold
        )
        test_dataset = torch.utils.data.Subset(test_dataset, source_indices)

    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=eeg_classification_collate,
    )
    model = load_model_checkpoint(
        args.model, args.dataset, info, args.checkpoint_path, device
    )
    model.eval()
    clean_metrics = evaluate_classifier(model, loader, device)
    attack = build_attack(
        args.attack, model, args.eps, info, device, seed=args.seed
    )

    clean_parts = []
    adv_parts = []
    label_parts = []
    for batch_index, (clean, labels) in enumerate(loader):
        clean = clean.to(device)
        labels = labels.to(device)
        model.zero_grad(set_to_none=True)
        adversarial = attack(clean, labels)
        model.zero_grad(set_to_none=True)
        clean_parts.append(clean.detach().cpu().float())
        adv_parts.append(adversarial.detach().cpu().float())
        label_parts.append(labels.detach().cpu().long())
        logging.info("Attack batch %d/%d", batch_index + 1, len(loader))
    clean = torch.cat(clean_parts, dim=0)
    adversarial = torch.cat(adv_parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    adversarial_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(adversarial, labels),
        batch_size=args.batch_size,
        shuffle=False,
    )
    adv_metrics = evaluate_classifier(model, adversarial_loader, device)
    mse = torch.nn.functional.mse_loss(adversarial, clean).item()
    payload = {
        "clean": clean,
        "adversarial": adversarial,
        "labels": labels,
        "source_indices": source_indices,
        "meta": {
            "kind": "rpcf_attack_eval",
            "dataset": args.dataset,
            "model": args.model,
            "fold": args.fold,
            "seed": args.seed,
            "protocol": get_protocol_tag(use_ea=False),
            "source_split": "test",
            "split_path": split_path,
            "checkpoint_path": args.checkpoint_path,
            "method_tag": args.method_tag,
            "attack": args.attack,
            "attack_seed": args.seed,
            "eps": args.eps,
            "sample_num": len(source_indices),
            "selection_strategy": (
                "full_test_split"
                if args.sample_num is None
                else "random_without_replacement"
            ),
            "selection_seed_rule": (
                None if args.sample_num is None else "seed + fold * 1000"
            ),
            "selection_seed": selection_seed,
            "clean_accuracy": clean_metrics["accuracy"],
            "clean_loss": clean_metrics["loss"],
            "adv_accuracy": adv_metrics["accuracy"],
            "adv_loss": adv_metrics["loss"],
            "attack_mse": mse,
        },
    }
    torch.save(payload, args.output_path)
    logging.info("Saved attack evaluation: %s", args.output_path)
    print(args.output_path)


if __name__ == "__main__":
    main()
