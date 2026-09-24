"""仅供 EXP-033 smoke：原训练流程，验证前两例、缓存训练首批。"""

import itertools
import sys
from pathlib import Path

from rpcf.exp033_common import read_json, write_json


def smoke_options(argv):
    """拒绝正式清单或放宽过的训练参数，避免该包装器被误用于正式实验。"""
    def value(name):
        matches = [i for i, token in enumerate(argv) if token == name or token.startswith(name + "=")]
        if len(matches) != 1:
            raise ValueError(f"Require exactly one smoke option: {name}")
        index = matches[0]
        if argv[index].startswith(name + "="):
            result = argv[index].split("=", 1)[1]
        elif index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            result = argv[index + 1]
        else:
            raise ValueError(f"Missing value for smoke option: {name}")
        if not result:
            raise ValueError(f"Empty value for smoke option: {name}")
        return result
    for name, expected in (("--epochs", 1), ("--online_train_sample_num", 2), ("--max_cache_batches", 1)):
        if int(value(name)) != expected:
            raise ValueError(f"Smoke requires {name}={expected}")
    output = Path(value("--output_checkpoint")).parent
    manifests = [p / "manifest.json" for p in (output, *output.parents) if (p / "manifest.json").exists()]
    if not manifests or not read_json(manifests[0]).get("smoke"):
        raise ValueError("Smoke validation wrapper requires a smoke manifest")
    return output


def main():
    output = smoke_options(sys.argv[1:])
    import torch
    from rpcf import finetune
    original = finetune.prepare_subject_fold
    original_train_epoch = finetune.train_epoch
    records = []

    def limited_validation(*args, **kwargs):
        train, val, test, split = original(*args, **kwargs)
        indices = list(range(min(2, len(val))))
        if not indices:
            raise ValueError("Empty validation split")
        records.append(dict(original_sample_num=len(val), sample_num=len(indices),
                            validation_indices=indices, split_path=split))
        write_json(output / "smoke_validation.json", dict(smoke=True, selections=records,
                   note="validation prefix only; formal training and test attacks are unchanged"))
        return train, torch.utils.data.Subset(val, indices), test, split

    def limited_epoch(model, loader, *args, **kwargs):
        # 原 max_cache_batches 只作用于 balanced sampler；smoke 显式截断真实首批。
        result = original_train_epoch(model, itertools.islice(loader, 1), *args, **kwargs)
        audit = read_json(output / "smoke_validation.json")
        audit.update(cache_batch_limit=1, cache_batch_limit_enforced=True)
        write_json(output / "smoke_validation.json", audit)
        return result

    finetune.prepare_subject_fold = limited_validation
    finetune.train_epoch = limited_epoch
    try:
        finetune.main()
    finally:
        finetune.prepare_subject_fold = original
        finetune.train_epoch = original_train_epoch


if __name__ == "__main__":
    main()
