import argparse
import logging
import os
from types import SimpleNamespace

from runtime_env import configure_runtime_env

configure_runtime_env()

import numpy as np
import torch

from data.subject_ea import get_protocol_tag, prepare_subject_fold
from purify import purify
from utils.experiment_artifacts import eeg_classification_collate

from rpcf.core import (
    DATASET_LOADERS,
    evaluate_classifier,
    load_model_checkpoint,
    seed_everything,
    stable_subset_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "对 EEG_TNP + RPCF_AT 执行 BPDA+PGD adaptive attack；"
            "forward 使用真实 EEG_TNP，backward 将 EEG_TNP 近似为恒等变换。"
        )
    )
    parser.add_argument("--dataset", default="thubenchmark", choices=DATASET_LOADERS)
    parser.add_argument(
        "--model", default="eegnet", choices=["eegnet", "tsception", "atcnet", "conformer"]
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--pgd_steps", type=int, default=10)
    parser.add_argument("--pgd_alpha", type=float, default=0.006)
    parser.add_argument("--sample_num", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_purify_args(args):
    """构造 purify() 所需的最小参数对象，避免污染 purify.py 默认入口。"""
    return SimpleNamespace(
        dataset=args.dataset,
        config=args.config,
        visualize=False,
        gpu_id=args.gpu_id,
        checkpoint_path=args.checkpoint_path,
        checkpoint_tag=None,
        model_tag="rpcf_at_bpda_pgd",
        output_tag=None,
        adv_output_tag=None,
        seed=args.seed,
        fold=args.fold,
        model=args.model,
    )


class BPDAIdentityPurifier(torch.autograd.Function):
    """forward 调用真实 EEG_TNP；backward 直接透传梯度。"""

    @staticmethod
    def forward(ctx, x, purifier):
        return purifier.forward_batch(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class EEGTNPPurifier:
    def __init__(self, purify_args, sampling_rate, device, logger):
        self.purify_args = purify_args
        self.sampling_rate = sampling_rate
        self.device = device
        self.logger = logger
        self.call_index = 0

    def forward_batch(self, batch):
        purified_parts = []
        for sample in batch:
            index = self.call_index
            self.call_index += 1
            # autograd.Function.forward 默认处在 no-grad 上下文；但 EEG_TNP
            # 自身需要通过梯度优化张量网络参数。这里仅恢复净化器内部训练所需梯度，
            # 外层 BPDA 的 backward 仍由 BPDAIdentityPurifier.backward 恒等透传。
            with torch.enable_grad():
                purified, _ = purify(
                    self.purify_args,
                    index,
                    sample.detach().cpu(),
                    self.sampling_rate,
                    self.device,
                    self.logger,
                    classifier=None,
                )
            purified_parts.append(purified.detach().float())
        return torch.stack(purified_parts, dim=0).to(
            device=batch.device, dtype=batch.dtype
        )

    def __call__(self, batch):
        return BPDAIdentityPurifier.apply(batch, self)


def tensor_metrics(model, data, labels, device, batch_size):
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, labels),
        batch_size=batch_size,
        shuffle=False,
    )
    return evaluate_classifier(model, loader, device)


def purify_tensor_batches(purifier, tensor, batch_size):
    parts = []
    for start in range(0, tensor.size(0), batch_size):
        batch = tensor[start : start + batch_size]
        parts.append(purifier.forward_batch(batch).detach().cpu().float())
    return torch.cat(parts, dim=0)


def bpda_pgd_attack(model, purifier, clean, labels, args, device):
    model.eval()
    adversarial_parts = []
    total = clean.size(0)
    for start in range(0, total, args.batch_size):
        x = clean[start : start + args.batch_size].to(device)
        y = labels[start : start + args.batch_size].to(device)
        x_anchor = x.detach()
        x_adv = x_anchor.clone()
        for step in range(args.pgd_steps):
            x_adv.requires_grad_(True)
            logits = model(purifier(x_adv))
            loss = torch.nn.functional.cross_entropy(logits, y)
            gradient = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + args.pgd_alpha * gradient.sign()
            delta = torch.clamp(x_adv - x_anchor, -args.eps, args.eps)
            x_adv = (x_anchor + delta).detach()
            logging.info(
                "BPDA PGD batch=%d/%d step=%d/%d loss=%.6f",
                start // args.batch_size + 1,
                int(np.ceil(total / args.batch_size)),
                step + 1,
                args.pgd_steps,
                float(loss.detach().cpu().item()),
            )
        adversarial_parts.append(x_adv.detach().cpu().float())
    return torch.cat(adversarial_parts, dim=0)


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
    sample_num = min(args.sample_num, len(test_dataset))
    source_indices, selection_seed = stable_subset_indices(
        len(test_dataset), sample_num, args.seed, args.fold
    )
    subset = torch.utils.data.Subset(test_dataset, source_indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=eeg_classification_collate,
    )

    clean_parts = []
    label_parts = []
    for clean_batch, labels in loader:
        clean_parts.append(clean_batch.detach().cpu().float())
        label_parts.append(labels.detach().cpu().long())
    clean = torch.cat(clean_parts, dim=0)
    labels = torch.cat(label_parts, dim=0).view(-1)

    model = load_model_checkpoint(
        args.model, args.dataset, info, args.checkpoint_path, device
    )
    model.eval()
    purify_args = make_purify_args(args)
    purifier = EEGTNPPurifier(
        purify_args,
        sampling_rate=info["sampling_rate"],
        device=device,
        logger=logging,
    )

    clean_pur = purify_tensor_batches(purifier, clean, args.batch_size)
    adversarial = bpda_pgd_attack(model, purifier, clean, labels, args, device)
    adv_pur = purify_tensor_batches(purifier, adversarial, args.batch_size)

    clean_pur_metrics = tensor_metrics(
        model, clean_pur, labels, device, args.eval_batch_size
    )
    adv_pur_metrics = tensor_metrics(
        model, adv_pur, labels, device, args.eval_batch_size
    )
    mse = torch.nn.functional.mse_loss(adversarial, clean).item()
    clean_pur_mse = torch.nn.functional.mse_loss(clean_pur, clean).item()
    adv_pur_mse = torch.nn.functional.mse_loss(adv_pur, adversarial).item()

    payload = {
        "clean": clean,
        "adversarial": adversarial,
        "labels": labels,
        "source_indices": source_indices,
        "clean_pur": clean_pur,
        "adv_pur": adv_pur,
        "rank": args.rank,
        "config": args.config,
        "metrics": {
            "purified_clean_accuracy": clean_pur_metrics["accuracy"],
            "purified_clean_loss": clean_pur_metrics["loss"],
            "bpda_purified_adv_accuracy": adv_pur_metrics["accuracy"],
            "bpda_purified_adv_loss": adv_pur_metrics["loss"],
            "attack_mse": mse,
            "mean_clean_mse": clean_pur_mse,
            "mean_adv_mse": adv_pur_mse,
        },
        "meta": {
            "kind": "rpcf_bpda_pgd_eval",
            "experiment_id": "EXP-023",
            "dataset": args.dataset,
            "model": args.model,
            "fold": args.fold,
            "seed": args.seed,
            "protocol": get_protocol_tag(use_ea=False),
            "source_split": "test",
            "split_path": split_path,
            "checkpoint_path": args.checkpoint_path,
            "method_tag": "rpcf_at",
            "attack": "bpda_pgd",
            "attack_seed": args.seed,
            "eps": args.eps,
            "pgd_steps": args.pgd_steps,
            "pgd_alpha": args.pgd_alpha,
            "bpda_identity": True,
            "rank": args.rank,
            "config": args.config,
            "sample_num": sample_num,
            "selection_strategy": "random_without_replacement",
            "selection_seed_rule": "seed + fold * 1000",
            "selection_seed": selection_seed,
        },
    }
    torch.save(payload, args.output_path)
    logging.info("Saved EXP-023 BPDA+PGD evaluation: %s", args.output_path)
    print(args.output_path)


if __name__ == "__main__":
    main()
