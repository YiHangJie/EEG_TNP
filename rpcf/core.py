import csv
import json
import math
import os
from collections import OrderedDict

from runtime_env import configure_runtime_env

configure_runtime_env()

import torch
from torch import nn
from models.registry import MODEL_CLASSES, MODEL_CHOICES

from attack.autoattack import AutoAttack
from attack.cw import CW
from attack.fgsm import FGSM
from attack.pgd import PGD
from data.load import load_bciciv2a, load_m3cv, load_seediv, load_thubenchmark
from models.model_args import get_model_args
from utils.experiment_artifacts import safe_token, torch_load_cpu
from utils.reproducibility import seed_everything, stable_subset_indices


DEFAULT_RANKS = (15, 20, 25, 30, 35, 40)
DATASET_LOADERS = {
    "seediv": load_seediv,
    "m3cv": load_m3cv,
    "bciciv2a": load_bciciv2a,
    "thubenchmark": load_thubenchmark,
}
# MODEL_CLASSES is imported from models.registry.
ATTACK_CLASSES = {
    "fgsm": FGSM,
    "pgd": PGD,
    "cw": CW,
    "autoattack": AutoAttack,
}


def parse_int_csv(value):
    values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    if len(set(values)) != len(values):
        raise ValueError(f"Values must be unique, got {values}.")
    return values


def parse_path_csv(value):
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one path.")
    return values


def rank_config_map(ranks, configs):
    if len(ranks) != len(configs):
        raise ValueError(
            f"Rank/config count mismatch: ranks={len(ranks)}, configs={len(configs)}."
        )
    return OrderedDict(zip(ranks, configs))


def build_model(model_name, dataset_name, info, device=None):
    model = MODEL_CLASSES[model_name](
        **get_model_args(model_name, dataset_name, info)
    )
    if device is not None:
        model.to(device)
    return model


def load_state_dict(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload: {path}")
    return payload


def load_model_checkpoint(model_name, dataset_name, info, checkpoint_path, device):
    model = build_model(model_name, dataset_name, info, device)
    model.load_state_dict(load_state_dict(checkpoint_path, device))
    return model


def build_attack(name, model, eps, info, device, seed):
    kwargs = {
        "eps": eps,
        "device": device,
        "n_classes": info["num_classes"],
    }
    if name == "autoattack":
        kwargs["seed"] = int(seed)
    return ATTACK_CLASSES[name](model, **kwargs)


def build_cache_path(output_dir, dataset, model, fold, seed, attack, eps, sample_num, tag):
    file_name = (
        f"{dataset}_{model}_no_ea_fold{fold}_seed{seed}_rpcf_train_"
        f"{attack}_eps{safe_token(eps)}_n{sample_num}_{safe_token(tag)}.pth"
    )
    return os.path.join(output_dir, file_name)


def validate_rpcf_cache(payload, expected=None):
    """校验统一 RPCF cache 的结构、shape 和关键实验元数据。"""
    if not isinstance(payload, dict):
        raise ValueError("RPCF cache must be a dict payload.")
    required = {
        "x",
        "x_adv",
        "x_pur_by_rank",
        "x_adv_pur_by_rank",
        "labels",
        "source_indices",
        "ranks",
        "meta",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"RPCF cache missing keys: {missing}")

    tensors = {}
    for key in ("x", "x_adv", "x_pur_by_rank", "x_adv_pur_by_rank"):
        value = payload[key]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        tensors[key] = value.detach().cpu().float()
    labels = torch.as_tensor(payload["labels"], dtype=torch.long).view(-1)
    ranks = [int(rank) for rank in payload["ranks"]]
    source_indices = [int(index) for index in payload["source_indices"]]
    meta = payload["meta"]

    if not isinstance(meta, dict) or meta.get("kind") != "rpcf_train_cache":
        raise ValueError("RPCF cache meta.kind must be rpcf_train_cache.")
    sample_count = tensors["x"].size(0)
    rank_count = len(ranks)
    if rank_count == 0 or len(set(ranks)) != rank_count:
        raise ValueError(f"RPCF ranks must be non-empty and unique, got {ranks}.")
    if tensors["x_adv"].shape != tensors["x"].shape:
        raise ValueError("x_adv shape must match x.")
    expected_rank_shape = (sample_count, rank_count, *tensors["x"].shape[1:])
    if tuple(tensors["x_pur_by_rank"].shape) != expected_rank_shape:
        raise ValueError(
            f"x_pur_by_rank shape is {tuple(tensors['x_pur_by_rank'].shape)}, "
            f"expected {expected_rank_shape}."
        )
    if tuple(tensors["x_adv_pur_by_rank"].shape) != expected_rank_shape:
        raise ValueError(
            f"x_adv_pur_by_rank shape is {tuple(tensors['x_adv_pur_by_rank'].shape)}, "
            f"expected {expected_rank_shape}."
        )
    if labels.numel() != sample_count or len(source_indices) != sample_count:
        raise ValueError("labels/source_indices length must match x.")
    if len(set(source_indices)) != len(source_indices):
        raise ValueError("source_indices must be unique.")
    if [int(rank) for rank in meta.get("ranks", [])] != ranks:
        raise ValueError("meta.ranks must match payload ranks.")

    if expected:
        for key, expected_value in expected.items():
            if expected_value is None:
                continue
            actual = meta.get(key)
            if key == "eps":
                matches = actual is not None and abs(
                    float(actual) - float(expected_value)
                ) <= 1e-12
            else:
                matches = str(actual) == str(expected_value)
            if not matches:
                raise ValueError(
                    f"RPCF cache metadata mismatch: {key}={actual}, "
                    f"expected {expected_value}."
                )

    normalized = dict(payload)
    normalized.update(tensors)
    normalized["labels"] = labels
    normalized["source_indices"] = source_indices
    normalized["ranks"] = ranks
    return normalized


def load_rpcf_cache(path, expected=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"RPCF cache not found: {path}")
    return validate_rpcf_cache(torch_load_cpu(path), expected=expected)


def logical_layer_names(model_name, model):
    """返回可稳定 hook 和独立冻结的模型逻辑层。"""
    if model_name == "eegnet":
        names = ["block1", "block2", "lin"]
    elif model_name == "tsception":
        names = [
            "Tception1",
            "Tception2",
            "Tception3",
            "Sception1",
            "Sception2",
            "fusion_layer",
            "fc",
        ]
    elif model_name == "conformer":
        names = ["embd"]
        names.extend(f"encoder.{index}" for index in range(len(model.encoder)))
        names.append("cls")
    elif model_name == "atcnet":
        top_level = dict(model.named_children())
        names = ["conv_block"]
        for prefix in ("msa", "tcn", "re", "dense"):
            names.extend(
                sorted(
                    (
                        name
                        for name, module in top_level.items()
                        if name.startswith(prefix)
                        and name[len(prefix):].isdigit()
                        and any(True for _ in module.parameters())
                    ),
                    key=lambda name: (
                        int(name[len(prefix):]) if name[len(prefix):].isdigit() else 0,
                        name,
                    ),
                )
            )
    elif model_name == "tcnet":
        names = ["eegnet"]
        names.extend(f"tcn_blocks.{index}" for index in range(len(model.tcn_blocks)))
        names.append("classifier")
    elif model_name == "deepconvnet":
        names = ["conv_time_spat", "conv_2", "conv_3", "conv_4", "final_layer"]
    else:
        raise ValueError(f"Unsupported model for logical layer registry: {model_name}")

    for name in names:
        model.get_submodule(name)
    return names


def feature_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported hooked feature type: {type(output)!r}")


def normalized_feature_shift(anchor, view, eps=1e-12):
    """逐样本计算归一化 L2 feature shift。"""
    if anchor.shape != view.shape:
        raise ValueError(
            f"Feature shape mismatch: anchor={tuple(anchor.shape)}, view={tuple(view.shape)}."
        )
    anchor_flat = anchor.reshape(anchor.size(0), -1).float()
    view_flat = view.reshape(view.size(0), -1).float()
    numerator = torch.linalg.vector_norm(view_flat - anchor_flat, ord=2, dim=1)
    denominator = torch.linalg.vector_norm(anchor_flat, ord=2, dim=1) + eps
    return numerator / denominator


def select_sensitive_layers(scores, ratio):
    if not 0 < ratio <= 1:
        raise ValueError("sensitive layer ratio must be in (0, 1].")
    if not scores:
        raise ValueError("Layer scores cannot be empty.")
    count = max(1, math.ceil(len(scores) * ratio))
    ordered = sorted(scores, key=lambda name: (-float(scores[name]), name))
    return ordered[:count]


def compute_interlayer_sensitivity(raw_by_layer, layer_names, input_values, eps=1e-12):
    """将绝对 feature shift 转为 input/相邻逻辑层之间的相对放大率。"""
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if not layer_names:
        raise ValueError("layer_names cannot be empty.")
    previous = torch.as_tensor(input_values, dtype=torch.float64)
    relative = {}
    for layer_name in layer_names:
        if layer_name not in raw_by_layer:
            raise ValueError(f"Missing raw sensitivity for layer: {layer_name}")
        current = torch.as_tensor(raw_by_layer[layer_name], dtype=torch.float64)
        if current.shape != previous.shape:
            raise ValueError(
                f"Sensitivity shape mismatch for {layer_name}: "
                f"{tuple(current.shape)} vs {tuple(previous.shape)}"
            )
        relative[layer_name] = current / (previous + eps)
        previous = current
    return relative


def compute_rank_weights(ranks, epoch, epochs, temperature=0.5, static=False):
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    rank_tensor = torch.as_tensor(ranks, dtype=torch.float32)
    if rank_tensor.numel() == 0:
        raise ValueError("ranks cannot be empty.")
    if static or rank_tensor.numel() == 1:
        return torch.full_like(rank_tensor, 1.0 / rank_tensor.numel())
    rank_min = rank_tensor.min()
    rank_max = rank_tensor.max()
    highness = (rank_tensor - rank_min) / (rank_max - rank_min)
    progress = 1.0 if epochs <= 1 else float(epoch) / float(epochs - 1)
    logits = ((2.0 * progress - 1.0) * (2.0 * highness - 1.0)) / temperature
    return torch.softmax(logits, dim=0)


def configure_trainable_layers(model, selected_layers, all_layers=False):
    """冻结非敏感层，并返回可训练参数统计。"""
    selected_layers = list(selected_layers)
    if all_layers:
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for layer_name in selected_layers:
            module = model.get_submodule(layer_name)
            for parameter in module.parameters():
                parameter.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0:
        raise ValueError("No trainable parameters after sensitive-layer selection.")
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": trainable / total,
    }


def set_frozen_batchnorm_eval(model, selected_layers=None, all_layers=False):
    """model.train() 后调用，避免冻结层的 BatchNorm 继续更新统计量。"""
    selected_layers = tuple(selected_layers or ())
    for module_name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            belongs_to_selected_layer = all_layers or any(
                module_name == layer_name
                or module_name.startswith(f"{layer_name}.")
                for layer_name in selected_layers
            )
            if not belongs_to_selected_layer:
                module.eval()


def checkpoint_is_better(candidate, best, tolerance=1e-12):
    """按 robust acc、clean acc、validation loss 的优先级比较。"""
    if best is None:
        return True
    if candidate["robust_acc"] > best["robust_acc"] + tolerance:
        return True
    if abs(candidate["robust_acc"] - best["robust_acc"]) <= tolerance:
        if candidate["clean_acc"] > best["clean_acc"] + tolerance:
            return True
        if abs(candidate["clean_acc"] - best["clean_acc"]) <= tolerance:
            return candidate["val_loss"] < best["val_loss"] - tolerance
    return False


def evaluate_classifier(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            labels = labels.to(device)
            logits = model(data)
            loss_sum += torch.nn.functional.cross_entropy(
                logits, labels, reduction="sum"
            ).item()
            correct += logits.argmax(dim=1).eq(labels).sum().item()
            total += labels.numel()
    if total == 0:
        raise ValueError("Cannot evaluate an empty loader.")
    return {"accuracy": correct / total, "loss": loss_sum / total}


def pgd_adversarial_examples(
    model,
    x,
    y,
    epsilon,
    step_size,
    steps,
    random_start=True,
):
    x_anchor = x.detach()
    if random_start:
        x_adv = x_anchor + torch.empty_like(x_anchor).uniform_(-epsilon, epsilon)
    else:
        x_adv = x_anchor.clone()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(model(x_adv), y)
        gradient = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * gradient.sign()
        delta = torch.clamp(x_adv - x_anchor, -epsilon, epsilon)
        x_adv = (x_anchor + delta).detach()
    return x_adv


def evaluate_pgd(model, loader, device, epsilon, steps=10, step_size=None):
    model.eval()
    step_size = epsilon / 5 if step_size is None else step_size
    correct = 0
    total = 0
    for data, labels in loader:
        data = data.to(device)
        labels = labels.to(device)
        adversarial = pgd_adversarial_examples(
            model,
            data,
            labels,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
        )
        with torch.no_grad():
            correct += model(adversarial).argmax(dim=1).eq(labels).sum().item()
        total += labels.numel()
    return correct / total


def write_json(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
