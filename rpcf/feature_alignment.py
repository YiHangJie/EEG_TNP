"""EXP-026 的分类前特征抽取、分布对齐损失和均衡采样工具。"""

import math
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler


class PenultimateFeatureAdapter:
    """在不改变模型 forward 接口的前提下抽取 dropout 前分类表征。"""

    def __init__(self, model_name, model):
        self.model_name = str(model_name)
        self.model = model
        self.feature_layer_names = self._resolve_feature_layers()
        self.feature_dim = None

    def _resolve_feature_layers(self):
        if self.model_name == "eegnet":
            names = ["lin"]
        elif self.model_name == "tsception":
            # fc.2 是最终 Linear 前的 Dropout；取其输入可避开随机 mask。
            names = ["fc.2"]
        elif self.model_name == "atcnet":
            names = sorted(
                (
                    name
                    for name, _ in self.model.named_modules()
                    if name.startswith("dense") and name[5:].isdigit()
                ),
                key=lambda name: int(name[5:]),
            )
        elif self.model_name == "conformer":
            # cls.fc.5 是最终 Linear 前的 Dropout。
            names = ["cls.fc.5"]
        elif self.model_name == "tcnet":
            names = ["classifier"]
        elif self.model_name == "deepconvnet":
            names = ["final_layer.conv_classifier"]
        else:
            raise ValueError(
                f"Unsupported model for penultimate feature extraction: {self.model_name}"
            )
        if not names:
            raise ValueError(f"No classifier feature layers found for {self.model_name}.")
        for name in names:
            self.model.get_submodule(name)
        return names

    def forward_with_features(self, x):
        """返回原 logits 和按样本展平、L2 归一化后的分类前表征。"""
        captured = {}
        handles = []

        def capture(name):
            def hook(_module, inputs):
                if not inputs or not torch.is_tensor(inputs[0]):
                    raise TypeError(f"Unsupported pre-hook input for feature layer {name}.")
                captured[name] = inputs[0]

            return hook

        try:
            for name in self.feature_layer_names:
                handles.append(
                    self.model.get_submodule(name).register_forward_pre_hook(
                        capture(name)
                    )
                )
            logits = self.model(x)
        finally:
            for handle in handles:
                handle.remove()

        missing = [name for name in self.feature_layer_names if name not in captured]
        if missing:
            raise RuntimeError(
                f"Feature hooks were not triggered for {self.model_name}: {missing}"
            )
        batch_size = x.size(0)
        features = []
        for name in self.feature_layer_names:
            value = captured[name]
            if value.size(0) != batch_size:
                raise ValueError(
                    f"Feature batch mismatch at {name}: {value.size(0)} != {batch_size}."
                )
            features.append(value.reshape(batch_size, -1))
        feature = torch.cat(features, dim=1)
        feature = F.normalize(feature, p=2, dim=1, eps=1e-12)
        if self.feature_dim is None:
            self.feature_dim = int(feature.size(1))
        elif self.feature_dim != int(feature.size(1)):
            raise ValueError(
                f"Feature dimension changed: {self.feature_dim} -> {feature.size(1)}."
            )
        return logits, feature


def _pairwise_squared_distance(x, y):
    return torch.cdist(x.float(), y.float(), p=2).pow(2)


def biased_rbf_mmd(x, y, kernel_scales=(0.5, 1.0, 2.0), eps=1e-12):
    """使用 median bandwidth 的 multi-kernel biased MMD²。"""
    if x.dim() != 2 or y.dim() != 2 or x.size(1) != y.size(1):
        raise ValueError(f"MMD expects [N,D]/[M,D], got {x.shape}/{y.shape}.")
    if x.size(0) == 0 or y.size(0) == 0:
        raise ValueError("MMD inputs cannot be empty.")
    scales = tuple(float(scale) for scale in kernel_scales)
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("MMD kernel scales must be positive.")

    combined = torch.cat([x.detach(), y.detach()], dim=0).float()
    distances = torch.pdist(combined, p=2).pow(2)
    positive = distances[distances > eps]
    bandwidth = positive.median() if positive.numel() else combined.new_tensor(1.0)
    bandwidth = bandwidth.clamp_min(eps)

    d_xx = _pairwise_squared_distance(x, x)
    d_yy = _pairwise_squared_distance(y, y)
    d_xy = _pairwise_squared_distance(x, y)
    value = x.new_tensor(0.0, dtype=torch.float32)
    for scale in scales:
        denominator = 2.0 * bandwidth * scale
        value = value + (
            torch.exp(-d_xx / denominator).mean()
            + torch.exp(-d_yy / denominator).mean()
            - 2.0 * torch.exp(-d_xy / denominator).mean()
        )
    return (value / len(scales)).clamp_min(0.0).to(dtype=x.dtype)


def class_conditional_mmd_loss(
    clean_features,
    clean_pur_features,
    adv_pur_features,
    labels,
    rank_weights,
    kernel_scales=(0.5, 1.0, 2.0),
):
    """按 rank/类别对齐 clean 与 clean-pur、adv-pur 的特征分布。"""
    batch_size, rank_count, feature_dim = clean_pur_features.shape
    expected = (batch_size, feature_dim)
    if tuple(clean_features.shape) != expected:
        raise ValueError(
            f"Clean feature shape must be {expected}, got {tuple(clean_features.shape)}."
        )
    if adv_pur_features.shape != clean_pur_features.shape:
        raise ValueError("clean-pur and adv-pur feature shapes must match.")
    if labels.numel() != batch_size or rank_weights.numel() != rank_count:
        raise ValueError("CMMD labels/rank weights do not match feature shapes.")

    total = clean_features.new_tensor(0.0)
    classes = torch.unique(labels, sorted=True)
    for rank_index in range(rank_count):
        class_losses = []
        for class_id in classes:
            mask = labels.eq(class_id)
            clean_class = clean_features[mask]
            clean_pur_class = clean_pur_features[mask, rank_index]
            adv_pur_class = adv_pur_features[mask, rank_index]
            class_losses.append(
                0.5
                * (
                    biased_rbf_mmd(
                        clean_class, clean_pur_class, kernel_scales=kernel_scales
                    )
                    + biased_rbf_mmd(
                        clean_class, adv_pur_class, kernel_scales=kernel_scales
                    )
                )
            )
        rank_loss = torch.stack(class_losses).mean()
        total = total + rank_weights[rank_index] * rank_loss
    return total


def prototype_alignment_losses(
    clean_features,
    adv_pur_features,
    labels,
    rank_weights,
    margin=1.0,
):
    """返回 clean/adv-pur prototype 对齐损失和 adv-pur 类间 margin 损失。"""
    batch_size, rank_count, feature_dim = adv_pur_features.shape
    if tuple(clean_features.shape) != (batch_size, feature_dim):
        raise ValueError("Clean/prototype feature shapes do not match.")
    if labels.numel() != batch_size or rank_weights.numel() != rank_count:
        raise ValueError("Prototype labels/rank weights do not match feature shapes.")
    if margin <= 0:
        raise ValueError("Prototype margin must be positive.")

    classes = torch.unique(labels, sorted=True)
    clean_prototypes = torch.stack(
        [clean_features[labels.eq(class_id)].mean(dim=0) for class_id in classes]
    )
    alignment_total = clean_features.new_tensor(0.0)
    margin_total = clean_features.new_tensor(0.0)
    for rank_index in range(rank_count):
        pur_prototypes = torch.stack(
            [
                adv_pur_features[labels.eq(class_id), rank_index].mean(dim=0)
                for class_id in classes
            ]
        )
        alignment = (clean_prototypes - pur_prototypes).pow(2).sum(dim=1).mean()
        if pur_prototypes.size(0) >= 2:
            distances = torch.pdist(pur_prototypes, p=2)
            margin_loss = F.relu(float(margin) - distances).mean()
        else:
            margin_loss = pur_prototypes.sum() * 0.0
        alignment_total = alignment_total + rank_weights[rank_index] * alignment
        margin_total = margin_total + rank_weights[rank_index] * margin_loss
    return alignment_total, margin_total


def trial_contrastive_loss(
    clean_anchors,
    adv_features,
    adv_pur_features,
    temperature=0.1,
):
    """以 clean 为 anchor、同 trial 派生 view 为正样本的 InfoNCE。"""
    if temperature <= 0:
        raise ValueError("Contrastive temperature must be positive.")
    batch_size, rank_count, feature_dim = adv_pur_features.shape
    if tuple(clean_anchors.shape) != (batch_size, feature_dim):
        raise ValueError("Clean anchor shape does not match adv-pur features.")
    if tuple(adv_features.shape) != (batch_size, feature_dim):
        raise ValueError("Adversarial feature shape does not match clean anchors.")
    if batch_size < 2:
        return (adv_features.sum() + adv_pur_features.sum()) * 0.0, True

    candidates = torch.cat([adv_features.unsqueeze(1), adv_pur_features], dim=1)
    logits = torch.einsum("bd,jvd->bjv", clean_anchors, candidates)
    logits = logits / float(temperature)
    log_denominator = torch.logsumexp(logits.reshape(batch_size, -1), dim=1)
    positive_logits = logits[
        torch.arange(batch_size, device=logits.device),
        torch.arange(batch_size, device=logits.device),
    ]
    loss = -(positive_logits.mean(dim=1) - log_denominator).mean()
    return loss, False


def stratified_cache_split_indices(labels, holdout_fraction, seed):
    """按类别确定性划分 cache；singleton 类别保留在训练侧。"""
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be in (0, 1).")
    labels = torch.as_tensor(labels).detach().cpu().long().reshape(-1)
    by_class = defaultdict(list)
    for index, class_id in enumerate(labels.tolist()):
        by_class[int(class_id)].append(index)
    rng = random.Random(int(seed))
    train_indices = []
    holdout_indices = []
    for class_id in sorted(by_class):
        indices = list(by_class[class_id])
        rng.shuffle(indices)
        if len(indices) <= 1:
            holdout_count = 0
        else:
            holdout_count = max(1, int(round(len(indices) * holdout_fraction)))
            holdout_count = min(holdout_count, len(indices) - 1)
        holdout_indices.extend(indices[:holdout_count])
        train_indices.extend(indices[holdout_count:])
    rng.shuffle(train_indices)
    rng.shuffle(holdout_indices)
    if not train_indices or not holdout_indices:
        raise ValueError("Stratified cache split produced an empty partition.")
    return train_indices, holdout_indices


class ClassBalancedBatchSampler(Sampler):
    """确定性构造 P 类 × K trial 的 batch，并在类别池耗尽时循环采样。"""

    def __init__(
        self,
        labels,
        classes_per_batch=4,
        samples_per_class=2,
        seed=42,
        num_batches=None,
    ):
        labels = torch.as_tensor(labels).detach().cpu().long().reshape(-1)
        self.by_class = defaultdict(list)
        for index, class_id in enumerate(labels.tolist()):
            self.by_class[int(class_id)].append(index)
        self.classes = sorted(self.by_class)
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.epoch = 0
        if self.classes_per_batch <= 0 or self.samples_per_class <= 0:
            raise ValueError("Balanced sampler dimensions must be positive.")
        if self.classes_per_batch > len(self.classes):
            raise ValueError(
                f"Need {self.classes_per_batch} classes, cache only has {len(self.classes)}."
            )
        batch_size = self.classes_per_batch * self.samples_per_class
        self.num_batches = int(num_batches or math.ceil(len(labels) / batch_size))
        if self.num_batches <= 0:
            raise ValueError("Balanced sampler must produce at least one batch.")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pools = {}
        offsets = {}
        for class_id in self.classes:
            pools[class_id] = list(self.by_class[class_id])
            rng.shuffle(pools[class_id])
            offsets[class_id] = 0
        class_use_count = {class_id: 0 for class_id in self.classes}

        def draw(class_id):
            values = []
            for _ in range(self.samples_per_class):
                if offsets[class_id] >= len(pools[class_id]):
                    rng.shuffle(pools[class_id])
                    offsets[class_id] = 0
                values.append(pools[class_id][offsets[class_id]])
                offsets[class_id] += 1
            return values

        for _ in range(self.num_batches):
            tie_break = {class_id: rng.random() for class_id in self.classes}
            selected = sorted(
                self.classes,
                key=lambda class_id: (class_use_count[class_id], tie_break[class_id]),
            )[: self.classes_per_batch]
            batch = []
            for class_id in selected:
                class_use_count[class_id] += 1
                batch.extend(draw(class_id))
            rng.shuffle(batch)
            yield batch
