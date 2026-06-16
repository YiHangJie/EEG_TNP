import os
import random

import numpy as np
import torch


def seed_everything(seed=42):
    """统一固定 Python、NumPy、PyTorch 和 CUDA 随机源。"""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def stable_subset_indices(dataset_size, sample_num, seed, fold=0, offset=0):
    """按统一 seed 规则执行无放回抽样，并返回实际 selection seed。"""
    dataset_size = int(dataset_size)
    sample_num = int(sample_num)
    if sample_num <= 0:
        raise ValueError("sample_num must be positive.")
    if sample_num > dataset_size:
        raise ValueError(
            f"Requested {sample_num} samples but dataset has {dataset_size}."
        )
    selection_seed = int(seed) + int(fold) * 1000 + int(offset)
    rng = np.random.RandomState(selection_seed)
    indices = rng.choice(dataset_size, size=sample_num, replace=False).tolist()
    return indices, selection_seed
