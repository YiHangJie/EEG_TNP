"""EXP-032 协议修订：外部净化仅使用与历史一致的 PGD-10 自适应优化设置。"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from rpcf.core import seed_everything
from rpcf.exp032_common import (SOURCE_RUN, align_payload, load_classifier, load_condition,
                               load_payload, metric_row, norm_audit, predict, sha256, write_json)
from rpcf.exp032_purifiers import EEGReformer


REVISION = "adaptive_pgd10_v1"
ADAPTIVE_PROTOCOL = {
    "name": "PGD-10", "norm": "Linf", "eps": 0.03, "alpha": 0.006,
    "steps": 10, "random_start": False, "restarts": 1, "eot_samples": 1,
    "eot_enabled": False, "attack_batch_size": 1, "extra_input_clamp": False,
    "iterate_selection": "last", "gradient": "exact_through_purifier_and_classifier",
    "reference": "EXP-031 BPDA PGD-10 optimization settings; differentiable purifier uses exact gradient",
}


def pgd10(model, clean, labels, device, progress=None):
    """与历史 BPDA PGD 的起点、更新、投影和末步输出一致；可微净化保留真实梯度。"""
    model.eval()
    parts = []
    for start in range(len(clean)):
        anchor = clean[start:start + 1].to(device).detach()
        target = labels[start:start + 1].to(device)
        adversarial = anchor.clone()
        for _ in range(10):
            adversarial.requires_grad_(True)
            loss = F.cross_entropy(model(adversarial), target)
            gradient = torch.autograd.grad(loss, adversarial)[0]
            adversarial = adversarial.detach() + 0.006 * gradient.sign()
            delta = torch.clamp(adversarial - anchor, -0.03, 0.03)
            adversarial = (anchor + delta).detach()
        parts.append(adversarial.cpu().float())
        if progress:
            progress(start + 1, len(clean))
    return torch.cat(parts)


def evaluate(args):
    data, info, _, _, clean, labels, indices, split = load_condition(args)
    classifier, _, classifier_path = load_classifier(args, info, "clean", data)
    device = next(classifier.parameters()).device
    artifact = load_payload(args.purifier_path)
    meta = artifact["meta"]
    for key, expected in {"dataset": args.dataset, "seed": args.seed,
                          "method": args.purifier, "fold": 0}.items():
        if meta[key] != expected:
            raise ValueError(f"Purifier {key} mismatch")
    purifier = EEGReformer(args.purifier, meta["low"], meta["high"]).to(device)
    purifier.load_state_dict(artifact["state_dict"])
    model = nn.Sequential(purifier, classifier).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cp = predict(model, clean, args.batch_size)
    source = load_payload(args.attack_path)
    aligned = align_payload(source, indices, labels, clean)
    ap = predict(model, aligned["adversarial"], args.batch_size)
    rows = [metric_row(args, f"clean_{args.purifier}", args.attack, cp, ap, labels,
                       attack_protocol=source["meta"]["attack_protocol"],
                       evaluation="nonadaptive_purification")]
    # 每条件只在原 pgd 槽位计算一次 PGD-10；其他四攻击仅保留非自适应净化。
    if args.attack == "pgd":
        seed_everything(args.seed)
        print(f"ADAPTIVE_PROTOCOL {ADAPTIVE_PROTOCOL}", flush=True)
        adaptive = pgd10(model, clean, labels, device,
                        lambda done, total: print(
                            f"ADAPTIVE {args.purifier}/pgd10 {done}/{total}", flush=True))
        norms = norm_audit(clean, adaptive, "Linf", 0.03)
        ap_adaptive = predict(model, adaptive, args.batch_size)
        rows.append(metric_row(args, f"clean_{args.purifier}", "pgd10", cp, ap_adaptive, labels,
                               attack_protocol=dict(ADAPTIVE_PROTOCOL),
                               evaluation="adaptive_exact_gradient", norm_audit=norms))
    write_json(args.output_path, {
        "experiment_id": "EXP-032", "protocol_revision": REVISION,
        "source_indices": indices, "labels": labels.tolist(), "split": split, "rows": rows,
        "purifier_meta": meta, "purifier_sha256": sha256(args.purifier_path),
        "classifier_sha256": sha256(classifier_path), "evaluation_batch_size": args.batch_size,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["evaluate"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run", default=SOURCE_RUN)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--purifier", choices=["magnet", "dcae"], required=True)
    parser.add_argument("--batch-size", type=int, default=8, help="仅推理批量；攻击批量固定为 1")
    parser.add_argument("--sample-num", type=int, default=512)
    parser.add_argument("--attack", choices=["fgsm", "pgd", "autoattack", "cw", "pgd_l2"], required=True)
    parser.add_argument("--attack-path", required=True)
    parser.add_argument("--purifier-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    if Path(args.output_path).exists():
        raise FileExistsError(args.output_path)
    evaluate(args)


if __name__ == "__main__":
    main()
