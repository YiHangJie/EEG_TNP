"""EXP-031 长跑产物的原子落盘、显存批量和存储审计工具。"""

import json
import os
import tempfile
from pathlib import Path

import torch

from utils.reproducibility import stable_subset_indices


EXP031_ARTIFACT_SAMPLE_NUM = 512


def atomic_torch_save(payload, path):
    """先写同目录临时文件再原子替换，避免中断留下伪完整 artifact。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass


def exp031_run_id(output_path):
    """从独立 EXP-031 artifact 路径中提取 run id；其他实验返回 None。"""
    if not output_path:
        return None
    parts = Path(output_path).parts
    try:
        index = parts.index("exp031")
    except ValueError:
        return None
    return parts[index + 1] if index + 1 < len(parts) else None


def resolve_exp031_attack_batch(output_path, dataset, model, requested_batch):
    """正式攻击复用 cache 阶段已经验证过的 AutoAttack 安全 batch。"""
    run_id = exp031_run_id(output_path)
    if run_id is None:
        return int(requested_batch)
    path = Path("logs/exp031") / run_id / "actual_cache_attack_batch_sizes.json"
    if not path.exists():
        return int(requested_batch)
    values = json.loads(path.read_text(encoding="utf-8"))
    backbone = str(model).removesuffix("_ea_forward")
    validated = int(values.get(f"{dataset}_{backbone}", requested_batch))
    return min(int(requested_batch), validated)


def compact_exp031_attack_artifact(
    clean, adversarial, labels, source_indices, output_path, seed, fold, *extra_tensors
):
    """完整评估后仅保留 EEG_TNP 需要的确定性 n512 张量，指标仍来自 full test。"""
    total = int(clean.size(0))
    if exp031_run_id(output_path) is None or total <= EXP031_ARTIFACT_SAMPLE_NUM:
        positions = list(range(total))
        selection_seed = None
        strategy = "full_test_split"
    else:
        positions, selection_seed = stable_subset_indices(
            total, EXP031_ARTIFACT_SAMPLE_NUM, seed, fold
        )
        strategy = "random_without_replacement"
    index = torch.as_tensor(positions, dtype=torch.long)
    compact = (
        clean.index_select(0, index),
        adversarial.index_select(0, index),
        labels.index_select(0, index),
        [source_indices[position] for position in positions],
        *(tensor.index_select(0, index) for tensor in extra_tensors),
    )
    audit = {
        "evaluation_sample_num": total,
        "evaluated_source_indices": list(source_indices),
        "artifact_sample_num": len(positions),
        "artifact_selection_strategy": strategy,
        "artifact_selection_seed_rule": (
            "seed + fold * 1000" if selection_seed is not None else None
        ),
        "artifact_selection_seed": selection_seed,
    }
    return compact, audit
