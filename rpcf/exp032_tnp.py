"""EXP-032 固定 rank 净化：保留逐样本审计信息，避免重复持久化大型张量。"""

import argparse
import hashlib
import logging
import random
from pathlib import Path
from types import SimpleNamespace

from rpcf.exp032_common import (SOURCE_RUN, align_payload, load_payload, load_classifier,
                                metric_row, predict, write_json)
import numpy as np
import torch
from rpcf.core import DATASET_LOADERS, seed_everything
from rpcf.exp031 import tnp_path
from rpcf.exp031_artifacts import atomic_torch_save
from purify import purify


def rng_state():
    """续跑恢复同一随机流，不改变净化器已有的随机种子规则。"""
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run", default=SOURCE_RUN)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--attack-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    output = Path(args.output_path)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(output) + ".log", filemode="a", level=logging.INFO)
    seed_everything(args.seed)
    attack = load_payload(args.attack_path)
    clean, adv, labels = attack["clean"], attack["adversarial"], attack["labels"]
    indices = [int(i) for i in attack["source_indices"]]
    canonical_path = tnp_path(args.source_run, args.dataset, "eegnet", args.seed, "madry", "autoattack")
    canonical = load_payload(canonical_path)
    for key, expected in {"dataset": args.dataset, "seed": args.seed, "fold": 0}.items():
        if canonical["meta"][key] != expected:
            raise ValueError(f"Canonical clean {key} mismatch")
    if list(canonical["ranks"]) != [25, 30]:
        raise ValueError("Expected frozen rank25/30")
    shared = align_payload(canonical, indices, labels, clean)["clean_pur_by_rank"]
    data, info = DATASET_LOADERS[args.dataset]()
    model, _, _ = load_classifier(args, info, args.method, data)
    device = next(model.parameters()).device
    rows, diagnostics, previews = [], {}, {}
    for ri, rank in enumerate((25, 30)):
        config = SimpleNamespace(dataset=args.dataset, model=args.model, seed=args.seed, visualize=False,
                                 config=f"PTR3d_8_2048_rank{rank}_3d_interpolate.yaml")
        partial_path = Path(str(output) + f".rank{rank}.partial.pth")
        predictions, sample_checks, examples = [], [], []
        if partial_path.exists():
            partial = load_payload(partial_path)
            if partial["source_indices"] != indices or partial["attack_path"] != args.attack_path:
                raise ValueError("Partial provenance mismatch")
            predictions, sample_checks, examples = partial["predictions"], partial["sample_checks"], partial["examples"]
            restore_rng(partial["rng_state"])
        for start in range(len(predictions), len(clean), args.batch_size):
            block = []
            for i in range(start, min(start + args.batch_size, len(clean))):
                purified, mse = purify(config, i, adv[i], info["sampling_rate"], device, logging, classifier=model)
                tensor = purified.detach().cpu().float().contiguous()
                block.append(tensor)
                sample_checks.append({"source_index": indices[i], "mse_to_attack": float(mse),
                                      "mse_to_clean": (tensor - clean[i]).square().mean().item(),
                                      "purified_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()})
                if i < 2:
                    examples.append(tensor)
            predictions.extend(predict(model, torch.stack(block), args.batch_size))
            atomic_torch_save({"source_indices": indices, "attack_path": args.attack_path,
                               "predictions": predictions, "sample_checks": sample_checks,
                               "examples": examples, "rng_state": rng_state()}, partial_path)
            print(f"TNP rank{rank} {len(predictions)}/{len(clean)}", flush=True)
        cp = predict(model, shared[:, ri], args.batch_size)
        rows.append(metric_row(args, f"{args.method}_tnp_r{rank}", attack["meta"]["attack"], cp, predictions, labels,
                               evaluation="nonadaptive_purification", rank=rank,
                               attack_protocol=attack["meta"]["attack_protocol"]))
        diagnostics[rank] = sample_checks
        previews[rank] = torch.stack(examples)
    atomic_torch_save({"labels": labels, "source_indices": indices, "ranks": [25, 30],
                       "attack_path": args.attack_path, "shared_clean_source": canonical_path,
                       "preview_source_indices": indices[:2], "previews_by_rank": previews,
                       "sample_diagnostics": diagnostics, "rows": rows,
                       "meta": {**attack["meta"], "kind": "exp032_tnp_summary",
                                "storage_policy": "predictions_mse_sha256_two_previews_per_rank"}}, output)
    write_json(str(output) + ".metrics.json", {"source_indices": indices, "labels": labels.tolist(), "rows": rows})


if __name__ == "__main__":
    main()
