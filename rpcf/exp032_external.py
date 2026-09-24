"""EXP-032 MagNet/DCAE 训练与冻结分类器下的净化评估。"""

import argparse
from pathlib import Path

from rpcf.exp032_common import (
    SOURCE_RUN, align_payload, load_classifier, load_condition, load_payload, metric_row,
    norm_audit, predict, sha256, write_json,
)
import torch
from torch import nn
import torch.nn.functional as F
from rpcf.core import DATASET_LOADERS, build_attack, seed_everything
from rpcf.exp032_attacks import pgd_l2
from rpcf.exp032_purifiers import EEGReformer, sparsity_penalty
from rpcf.exp031_artifacts import atomic_torch_save
from utils.experiment_artifacts import eeg_classification_collate
from data.subject_ea import prepare_subject_fold


def train(args):
    seed_everything(args.seed)
    data, info = DATASET_LOADERS[args.dataset]()
    train_data, val_data, _, split = prepare_subject_fold(
        args.dataset, data, info, fold_id=0, seed=args.seed, use_ea=False
    )
    if args.train_sample_num:
        from utils.reproducibility import stable_subset_indices
        ids, _ = stable_subset_indices(len(train_data), min(args.train_sample_num, len(train_data)), args.seed)
        train_data = torch.utils.data.Subset(train_data, ids)
        val_data = torch.utils.data.Subset(val_data, list(range(min(args.train_sample_num, len(val_data)))))
    loader_args = dict(batch_size=args.batch_size, num_workers=0, collate_fn=eeg_classification_collate)
    bounds_loader = torch.utils.data.DataLoader(train_data, shuffle=False, **loader_args)
    low, high = float("inf"), float("-inf")
    for x, _ in bounds_loader:
        low, high = min(low, x.min().item()), max(high, x.max().item())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    reformer = EEGReformer(args.purifier, low, high).to(device)
    optimizer = torch.optim.Adam(reformer.parameters(), lr=0.001)
    loader = torch.utils.data.DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, **loader_args)
    noise_std = 0.5 if args.purifier == "dcae" else 0.1
    history = []
    for epoch in range(args.epochs):
        reformer.train()
        total_loss, count = 0.0, 0
        for x, _ in loader:
            target = reformer.encode_range(x.to(device))
            noisy = (target + noise_std * torch.randn_like(target)).clamp(0, 1)
            output, activity = reformer.network(noisy, return_activity=True)
            loss = F.mse_loss(output, target)
            if activity is not None:
                loss = loss + 7.5e-5 * sparsity_penalty(activity, rho=0.02)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
            count += len(x)
        reformer.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                val_loss += F.mse_loss(reformer(x), x).item() * len(x)
                val_count += len(x)
        entry = {"epoch": epoch + 1, "train_loss_encoded": total_loss / count,
                 "val_mse_native": val_loss / val_count}
        history.append(entry)
        print(f"TRAIN {args.purifier} {entry}", flush=True)
        write_json(str(args.output_path) + ".history.json", history)
    atomic_torch_save({"state_dict": reformer.cpu().state_dict(), "meta": {
        "experiment_id": "EXP-032", "dataset": args.dataset, "seed": args.seed, "fold": 0,
        "method": args.purifier, "low": low, "high": high, "train_split": split,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": 0.001,
        "optimizer": "Adam", "noise_std_encoded": noise_std, "noise_std_native": noise_std * (high - low),
        "range_source": "train_only_scalar_minmax", "checkpoint_selection": "final_epoch",
        "sparsity_rho": 0.02 if args.purifier == "dcae" else None,
        "sparsity_weight": 7.5e-5 if args.purifier == "dcae" else 0,
        "magnet_activity_l2": 0, "magnet_activity_l2_deviation": "author_1e-9_omitted",
        "train_samples": len(train_data), "val_samples": len(val_data),
        "assumptions_document": "docs/EXP032_BASELINES.md", "history": history,
    }}, args.output_path)


def evaluate(args):
    data, info, _, _, clean, labels, indices, split = load_condition(args)
    classifier, _, classifier_path = load_classifier(args, info, "clean", data)
    device = next(classifier.parameters()).device
    artifact = load_payload(args.purifier_path)
    meta = artifact["meta"]
    for key, expected in {"dataset": args.dataset, "seed": args.seed, "method": args.purifier}.items():
        if meta[key] != expected:
            raise ValueError(f"Purifier {key} mismatch")
    purifier = EEGReformer(args.purifier, meta["low"], meta["high"]).to(device)
    purifier.load_state_dict(artifact["state_dict"])
    model = nn.Sequential(purifier, classifier).eval()
    # 模型参数冻结，仍保留对输入求梯度的计算图。
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cp = predict(model, clean, args.batch_size)
    source = load_payload(args.attack_path)
    aligned = align_payload(source, indices, labels, clean)
    ap = predict(model, aligned["adversarial"], args.batch_size)
    protocol = source["meta"]["attack_protocol"]
    rows = [metric_row(args, f"clean_{args.purifier}", args.attack, cp, ap, labels,
                       attack_protocol=protocol, evaluation="nonadaptive_purification")]
    seed_everything(args.seed)
    attacker = None if args.attack == "pgd_l2" else build_attack(args.attack, model, 0.03, info, device, args.seed)
    adversarial_parts = []
    for start in range(0, len(clean), args.batch_size):
        x, y = clean[start:start + args.batch_size].to(device), labels[start:start + args.batch_size].to(device)
        adv = pgd_l2(model, x, y, steps=args.l2_steps, restarts=args.l2_restarts) if attacker is None else attacker(x, y)
        adversarial_parts.append(adv.cpu().detach())
        print(f"ADAPTIVE {args.purifier}/{args.attack} {start + len(x)}/{len(clean)}", flush=True)
    adaptive = torch.cat(adversarial_parts)
    norms = norm_audit(clean, adaptive, protocol["norm"], protocol.get("eps") if args.attack != "cw" else None)
    ap_adaptive = predict(model, adaptive, args.batch_size)
    rows.append(metric_row(args, f"clean_{args.purifier}", args.attack, cp, ap_adaptive, labels,
                           attack_protocol=protocol, evaluation="adaptive_exact_gradient", norm_audit=norms))
    write_json(args.output_path, {"experiment_id": "EXP-032", "source_indices": indices,
                                "labels": labels.tolist(), "split": split, "rows": rows,
                                "purifier_meta": meta, "purifier_sha256": sha256(args.purifier_path),
                                "classifier_sha256": sha256(classifier_path)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "evaluate"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run", default=SOURCE_RUN)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--purifier", choices=["magnet", "dcae"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-sample-num", type=int)
    parser.add_argument("--sample-num", type=int, default=512)
    parser.add_argument("--attack", default="pgd")
    parser.add_argument("--attack-path")
    parser.add_argument("--purifier-path")
    parser.add_argument("--l2-steps", type=int, default=200)
    parser.add_argument("--l2-restarts", type=int, default=5)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    if Path(args.output_path).exists():
        raise FileExistsError(args.output_path)
    (train if args.action == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
