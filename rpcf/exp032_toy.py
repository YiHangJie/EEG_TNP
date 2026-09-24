"""逐 trial 在真实 TNP 插值张量上诊断低秩结构，包含等范数随机对照。"""

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

from rpcf.exp032_common import load_payload, write_json
import numpy as np
import torch
from purify import interpolate
from utils.reproducibility import seed_everything


def multiply_mode(tensor, matrix, mode):
    output = torch.tensordot(matrix, tensor, dims=([1], [mode]))
    return output.movedim(0, mode)


def statistics(tensor):
    """HOSVD 截断误差由正交 core 保留能量精确计算，不用不成立的谱和近似。"""
    tensor = tensor.double()
    bases, ranks = [], []
    for mode in range(tensor.ndim):
        unfolded = tensor.movedim(mode, 0).reshape(tensor.shape[mode], -1)
        u, singular, _ = torch.linalg.svd(unfolded, full_matrices=False)
        bases.append(u)
        total, energy = singular.sum(), singular.square().sum()
        if total <= 1e-20:
            ranks.append({"mode": mode, "effective_rank": 0.0, "stable_rank": 0.0, "rank95": 0,
                          "singular_values": singular.tolist()})
        else:
            p = singular / total
            ranks.append({"mode": mode, "effective_rank": float(torch.exp(-(p * p.clamp_min(1e-30).log()).sum())),
                          "stable_rank": float(energy / singular[0].square()),
                          "rank95": int(torch.searchsorted(singular.square().cumsum(0), 0.95 * energy)) + 1,
                          "singular_values": singular.tolist()})
    core = tensor
    for mode, basis in enumerate(bases):
        core = multiply_mode(core, basis.T, mode)
    energy = tensor.square().sum().item()
    errors = []
    for fraction in (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
        size = [max(1, int(np.ceil(fraction * s))) for s in core.shape]
        retained = core[tuple(slice(0, s) for s in size)].square().sum().item()
        error = float(np.sqrt(max(0, 1 - retained / energy))) if energy > 1e-20 else 0.0
        errors.append({"rank_fraction": fraction, "mode_ranks": size, "relative_frobenius_error": error})
    return ranks, errors, energy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--sample-num", type=int, default=512)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    output = Path(args.output_path)
    if output.exists():
        raise FileExistsError(output)
    seed_everything(args.seed)
    torch.set_num_threads(2)
    payload = load_payload(args.attack_path)
    n = min(args.sample_num, len(payload["labels"]))
    # 输入 artifact 已使用规范 n512 抽样；只允许 smoke 取前缀，正式不能重新选好样本。
    config = SimpleNamespace(dataset=args.dataset, config="PTR3d_8_2048_rank25_3d_interpolate.yaml")
    samples, spectrum_rows, reconstruction_rows = [], [], []
    output.parent.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        clean = payload["clean"][i].float()
        delta = payload["adversarial"][i].float() - clean
        gaussian = torch.randn_like(delta)
        noise_l2 = gaussian * (delta.norm() / gaussian.norm().clamp_min(1e-20))
        noise_linf = gaussian * (delta.abs().max() / gaussian.abs().max().clamp_min(1e-20))
        if not torch.allclose(noise_l2.norm(), delta.norm(), atol=1e-5, rtol=1e-5):
            raise ValueError("L2 noise matching failed")
        if not torch.allclose(noise_linf.abs().max(), delta.abs().max(), atol=1e-6, rtol=1e-5):
            raise ValueError("Linf noise matching failed")
        variants = {"clean": clean, "delta": delta, "clean_plus_delta": clean + delta,
                    "random_l2": noise_l2, "clean_plus_random_l2": clean + noise_l2,
                    "random_linf": noise_linf, "clean_plus_random_linf": clean + noise_linf}
        base = {"dataset": args.dataset, "seed": args.seed, "source_model": payload["meta"]["model"],
                "source_method": args.method, "source_index": int(payload["source_indices"][i])}
        for name, signal in variants.items():
            tensor = interpolate(config, signal, sampling_rate=250)
            if not torch.isfinite(tensor).all():
                raise ValueError("Non-finite TNP representation")
            ranks, errors, energy = statistics(tensor)
            samples.append({**base, "variant": name, "tensor_shape": list(tensor.shape),
                            "native_l2": signal.norm().item(), "native_linf": signal.abs().max().item(),
                            "tensor_energy": energy, "zero_energy": energy <= 1e-20})
            spectrum_rows.extend({**base, "variant": name, **row} for row in ranks)
            reconstruction_rows.extend({**base, "variant": name, **row} for row in errors)
        print(f"TOY {args.dataset}/{args.method} {i + 1}/{n}", flush=True)
    for suffix, rows in (("spectra", spectrum_rows), ("reconstruction", reconstruction_rows), ("samples", samples)):
        with output.with_suffix(f".{suffix}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    names = list(variants)
    for name in names:
        values = sorted(row["effective_rank"] for row in spectrum_rows if row["variant"] == name and row["mode"] == 2)
        axes[0].plot(values, np.arange(1, len(values) + 1) / len(values), label=name)
        fractions = sorted(set(row["rank_fraction"] for row in reconstruction_rows))
        ys = [np.median([row["relative_frobenius_error"] for row in reconstruction_rows
                         if row["variant"] == name and row["rank_fraction"] == fraction]) for fraction in fractions]
        axes[1].plot(fractions, ys, label=name)
    axes[0].set(xlabel="Temporal unfolding effective rank", ylabel="ECDF")
    axes[1].set(xlabel="HOSVD mode-rank fraction", ylabel="Median relative Frobenius error")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=200)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    write_json(output, {"experiment_id": "EXP-032", "sample_num": n, "source_path": args.attack_path,
                        "source_indices": [int(i) for i in payload["source_indices"][:n]],
                        "noise_seed": args.seed, "noise_rng": "torch.randn_like",
                        "representation": "purify.interpolate / 3d_interpolate", "complete": True})


if __name__ == "__main__":
    main()
