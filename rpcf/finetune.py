import argparse
import logging
import os

from runtime_env import configure_runtime_env

configure_runtime_env()

import torch

from rpcf.exp031_artifacts import atomic_torch_save
import torch.nn.functional as F

from data.subject_ea import prepare_subject_fold
from utils.experiment_artifacts import eeg_classification_collate

from rpcf.core import (
    DATASET_LOADERS,
    MODEL_CHOICES,
    compute_rank_weights,
    configure_trainable_layers,
    evaluate_classifier,
    evaluate_pgd,
    load_model_checkpoint,
    load_rpcf_cache,
    logical_layer_names,
    seed_everything,
    set_frozen_batchnorm_eval,
    write_csv,
    write_json,
)
from rpcf.feature_alignment import (
    ClassBalancedBatchSampler,
    PenultimateFeatureAdapter,
    class_conditional_mmd_loss,
    prototype_alignment_losses,
    stratified_cache_split_indices,
    trial_contrastive_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="从 AT checkpoint 初始化并执行 RPCF sensitive-layer fine-tuning。"
    )
    parser.add_argument("--cache_path", required=True)
    parser.add_argument(
        "--sensitivity_path",
        default=None,
        help="选择性微调时必填；--all_layers 模式不需要 sensitivity artifact。",
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=DATASET_LOADERS)
    parser.add_argument(
        "--model", required=True, choices=MODEL_CHOICES
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="兼容旧命令保留；当前固定训练满 epochs，不启用 early stopping。",
    )
    parser.add_argument("--rank_temperature", type=float, default=0.5)
    parser.add_argument("--consistancy_temperature", type=float, default=2.0)
    parser.add_argument("--clean_ce_weight", type=float, default=1.0)
    parser.add_argument("--adv_ce_weight", type=float, default=1.0)
    parser.add_argument("--pur_ce_weight", type=float, default=0.5)
    parser.add_argument("--adv_pur_ce_weight", type=float, default=1.0)
    parser.add_argument("--lambda_adv", type=float, default=0.5)
    parser.add_argument("--lambda_pur", type=float, default=0.2)
    parser.add_argument("--lambda_adv_pur", type=float, default=0.5)
    parser.add_argument("--pgd_steps", type=int, default=10)
    parser.add_argument(
        "--online_madry_at",
        action="store_true",
        help="每个 epoch 先在完整训练集上执行在线 Madry PGD 对抗训练。",
    )
    parser.add_argument("--online_at_batch_size", type=int, default=128)
    parser.add_argument("--online_at_pgd_steps", type=int, default=10)
    parser.add_argument(
        "--online_at_step_size",
        type=float,
        default=None,
        help="在线 PGD 步长；默认使用 epsilon / 5。",
    )
    parser.add_argument(
        "--online_train_sample_num",
        type=int,
        default=None,
        help="仅用于 smoke；默认使用完整训练 split。",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--all_layers", action="store_true")
    parser.add_argument(
        "--selected_layers_override",
        default=None,
        help="逗号分隔的逻辑层名；用于覆盖 sensitivity.json 中的 selected_layers。",
    )
    parser.add_argument(
        "--layer_selection_rule",
        default=None,
        help="记录层选择来源，例如 score_prefix；不影响训练逻辑。",
    )
    parser.add_argument(
        "--prefix_k",
        type=int,
        default=None,
        help="记录敏感性降序累加前缀长度；不影响训练逻辑。",
    )
    parser.add_argument("--static_rank_weights", action="store_true")
    parser.add_argument(
        "--feature_objective",
        choices=("none", "cmmd", "prototype", "contrastive"),
        default="none",
        help="EXP-026 分类前特征适配目标；none 保留原 RPCF/RPCF_AT 行为。",
    )
    parser.add_argument("--feature_loss_weight", type=float, default=0.0)
    parser.add_argument("--prototype_margin", type=float, default=1.0)
    parser.add_argument("--prototype_margin_weight", type=float, default=0.0)
    parser.add_argument("--contrastive_temperature", type=float, default=0.1)
    parser.add_argument(
        "--balanced_classes_per_batch",
        type=int,
        default=None,
        help="启用 class-balanced sampler 时每个 batch 的类别数。",
    )
    parser.add_argument("--balanced_samples_per_class", type=int, default=2)
    parser.add_argument(
        "--max_cache_batches",
        type=int,
        default=None,
        help="仅用于 smoke，限制每个 epoch 的 cache batch 数。",
    )
    parser.add_argument(
        "--cache_holdout_fraction",
        type=float,
        default=0.0,
        help="仅用于 EXP-026 预筛的分层 cache holdout 比例；正式训练保持 0。",
    )
    parser.add_argument("--history_prefix", required=True)
    return parser.parse_args()


def load_sensitivity(path, args, cache):
    if not path:
        raise ValueError("--sensitivity_path is required unless --all_layers is set.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sensitivity artifact not found: {path}")
    import json

    with open(path, "r", encoding="utf-8") as file:
        artifact = json.load(file)
    for key, expected in {
        "dataset": args.dataset,
        "model": args.model,
        "fold": args.fold,
        "seed": args.seed,
        "eps": args.epsilon,
    }.items():
        actual = artifact.get(key)
        if key == "eps":
            matches = actual is not None and abs(float(actual) - float(expected)) <= 1e-12
        else:
            matches = str(actual) == str(expected)
        if not matches:
            raise ValueError(
                f"Sensitivity metadata mismatch: {key}={actual}, expected {expected}."
            )
    if [int(rank) for rank in artifact.get("ranks", [])] != cache["ranks"]:
        raise ValueError("Sensitivity ranks do not match the RPCF cache.")
    selected_layers = artifact.get("selected_layers")
    if not selected_layers:
        raise ValueError("Sensitivity artifact has no selected layers.")
    return artifact, selected_layers


def resolve_layer_configuration(args, cache, model):
    """解析微调层；全层模式不依赖或读取 sensitivity artifact。"""
    if args.all_layers:
        if args.selected_layers_override:
            raise ValueError(
                "--selected_layers_override cannot be combined with --all_layers."
            )
        selected_layers = logical_layer_names(args.model, model)
        return {
            "kind": "rpcf_all_layers",
            "dataset": args.dataset,
            "model": args.model,
            "fold": args.fold,
            "seed": args.seed,
            "eps": args.epsilon,
            "ranks": list(cache["ranks"]),
            "logical_layers": selected_layers,
            "selected_layers": selected_layers,
            "selected_param_ratio": 1.0,
        }, selected_layers

    sensitivity, selected_layers = load_sensitivity(
        args.sensitivity_path, args, cache
    )
    override_layers = parse_selected_layers_override(
        args.selected_layers_override, sensitivity
    )
    if override_layers is not None:
        selected_layers = override_layers
    return sensitivity, selected_layers


def parse_selected_layers_override(value, sensitivity):
    if value is None:
        return None
    selected_layers = [item.strip() for item in value.split(",") if item.strip()]
    if not selected_layers:
        raise ValueError("--selected_layers_override cannot be empty when provided.")
    known_layers = set(sensitivity.get("logical_layers") or [])
    known_layers.update((sensitivity.get("layers") or {}).keys())
    unknown_layers = [layer for layer in selected_layers if layer not in known_layers]
    if unknown_layers:
        raise ValueError(
            "selected layer override contains unknown layers: "
            + ", ".join(unknown_layers)
        )
    if len(set(selected_layers)) != len(selected_layers):
        raise ValueError("--selected_layers_override contains duplicate layers.")
    return selected_layers


def rank_weighted_ce(logits_flat, labels, rank_count, rank_weights):
    batch_size = labels.size(0)
    targets = labels.unsqueeze(1).expand(-1, rank_count).reshape(-1)
    losses = torch.nn.functional.cross_entropy(
        logits_flat, targets, reduction="none"
    ).reshape(batch_size, rank_count)
    return (losses * rank_weights.view(1, rank_count)).sum(dim=1).mean()


def kl_to_clean_teacher(student_logits, teacher_probs, temperature):
    """复用 consistancy 的温度 KL，让 student logits 对齐 clean teacher。"""
    log_probs = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(log_probs, teacher_probs, reduction="none").sum(dim=1) * (
        temperature ** 2
    )


def rank_weighted_kl(
    logits_flat,
    teacher_probs,
    rank_count,
    rank_weights,
    temperature,
):
    batch_size = teacher_probs.size(0)
    teacher_by_rank = (
        teacher_probs.unsqueeze(1)
        .expand(-1, rank_count, -1)
        .reshape(batch_size * rank_count, -1)
    )
    losses = kl_to_clean_teacher(
        logits_flat, teacher_by_rank, temperature
    ).reshape(batch_size, rank_count)
    return (losses * rank_weights.view(1, rank_count)).sum(dim=1).mean()


def pgd_adversarial_examples(
    model,
    x,
    labels,
    epsilon,
    step_size,
    steps,
    random_start=True,
):
    """基于当前模型在线生成 L∞ PGD 对抗样本。"""
    x_adv = x.detach()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = F.cross_entropy(model(x_adv), labels)
        gradient = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * gradient.sign()
        perturbation = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = (x + perturbation).detach()
    return x_adv


def train_epoch_online_madry(
    model,
    loader,
    optimizer,
    device,
    selected_layers,
    all_layers,
    args,
):
    """在原始训练 split 上执行在线 Madry AT，只优化 adversarial CE。"""
    model.train()
    set_frozen_batchnorm_eval(
        model,
        selected_layers=selected_layers,
        all_layers=all_layers,
    )
    total_loss = 0.0
    total_samples = 0
    step_size = (
        args.online_at_step_size
        if args.online_at_step_size is not None
        else args.epsilon / 5.0
    )
    for x, labels in loader:
        x = x.to(device)
        labels = labels.to(device)
        x_adv = pgd_adversarial_examples(
            model,
            x,
            labels,
            epsilon=args.epsilon,
            step_size=step_size,
            steps=args.online_at_pgd_steps,
            random_start=True,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x_adv), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            max_norm=0.01,
        )
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / total_samples


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    rank_weights,
    selected_layers,
    all_layers,
    args,
    use_cached_adv=True,
    feature_adapter=None,
):
    model.train()
    set_frozen_batchnorm_eval(
        model,
        selected_layers=selected_layers,
        all_layers=all_layers,
    )
    totals = {
        "loss": 0.0,
        "clean_ce": 0.0,
        "adv_ce": 0.0,
        "clean_pur_ce": 0.0,
        "adv_pur_ce": 0.0,
        "adv_kl": 0.0,
        "clean_pur_kl": 0.0,
        "adv_pur_kl": 0.0,
        "feature_loss": 0.0,
        "prototype_alignment": 0.0,
        "prototype_margin": 0.0,
    }
    feature_objective = getattr(args, "feature_objective", "none")
    feature_loss_weight = float(getattr(args, "feature_loss_weight", 0.0))
    skipped_feature_batches = 0
    total_samples = 0
    for x, x_adv, x_pur_by_rank, x_adv_pur_by_rank, labels in loader:
        x = x.to(device)
        x_adv = x_adv.to(device)
        x_pur_by_rank = x_pur_by_rank.to(device)
        x_adv_pur_by_rank = x_adv_pur_by_rank.to(device)
        labels = labels.to(device)
        batch_size, rank_count = x_pur_by_rank.shape[:2]
        sample_shape = x_pur_by_rank.shape[2:]
        x_pur_flat = x_pur_by_rank.reshape(
            batch_size * rank_count, *sample_shape
        )
        x_adv_pur_flat = x_adv_pur_by_rank.reshape(
            batch_size * rank_count, *sample_shape
        )

        optimizer.zero_grad(set_to_none=True)
        if feature_adapter is None:
            clean_logits = model(x)
            clean_features = None
        else:
            clean_logits, clean_features = feature_adapter.forward_with_features(x)
            clean_features = clean_features.detach()
        clean_ce = F.cross_entropy(clean_logits, labels)
        with torch.no_grad():
            teacher_probs = F.softmax(
                clean_logits.detach() / args.consistancy_temperature, dim=1
            )

        if feature_objective == "cmmd":
            clean_pur_logits, clean_pur_features = (
                feature_adapter.forward_with_features(x_pur_flat)
            )
            clean_pur_features = clean_pur_features.reshape(
                batch_size, rank_count, -1
            )
        else:
            clean_pur_logits = model(x_pur_flat)
            clean_pur_features = None

        if feature_adapter is None:
            adv_pur_logits = model(x_adv_pur_flat)
            adv_pur_features = None
        else:
            adv_pur_logits, adv_pur_features = (
                feature_adapter.forward_with_features(x_adv_pur_flat)
            )
            adv_pur_features = adv_pur_features.reshape(
                batch_size, rank_count, -1
            )

        clean_pur_ce = rank_weighted_ce(
            clean_pur_logits, labels, rank_count, rank_weights
        )
        adv_pur_ce = rank_weighted_ce(
            adv_pur_logits, labels, rank_count, rank_weights
        )
        if use_cached_adv:
            adv_logits = model(x_adv)
            adv_ce = F.cross_entropy(adv_logits, labels)
            adv_kl = kl_to_clean_teacher(
                adv_logits, teacher_probs, args.consistancy_temperature
            ).mean()
        else:
            adv_ce = clean_ce.new_tensor(0.0)
            adv_kl = clean_ce.new_tensor(0.0)
        clean_pur_kl = rank_weighted_kl(
            clean_pur_logits,
            teacher_probs,
            rank_count,
            rank_weights,
            args.consistancy_temperature,
        )
        adv_pur_kl = rank_weighted_kl(
            adv_pur_logits,
            teacher_probs,
            rank_count,
            rank_weights,
            args.consistancy_temperature,
        )
        feature_loss = clean_ce.new_tensor(0.0)
        prototype_alignment = clean_ce.new_tensor(0.0)
        prototype_margin = clean_ce.new_tensor(0.0)
        if feature_objective == "cmmd":
            feature_loss = class_conditional_mmd_loss(
                clean_features,
                clean_pur_features,
                adv_pur_features,
                labels,
                rank_weights,
            )
        elif feature_objective == "prototype":
            prototype_alignment, prototype_margin = prototype_alignment_losses(
                clean_features,
                adv_pur_features,
                labels,
                rank_weights,
                margin=getattr(args, "prototype_margin", 1.0),
            )
            feature_loss = (
                prototype_alignment
                + getattr(args, "prototype_margin_weight", 0.0) * prototype_margin
            )
        elif feature_objective == "contrastive":
            _, adv_features = feature_adapter.forward_with_features(x_adv)
            feature_loss, skipped = trial_contrastive_loss(
                clean_features,
                adv_features,
                adv_pur_features,
                temperature=getattr(args, "contrastive_temperature", 0.1),
            )
            skipped_feature_batches += int(skipped)

        loss = (
            args.clean_ce_weight * clean_ce
            + args.adv_ce_weight * adv_ce
            + args.pur_ce_weight * clean_pur_ce
            + args.adv_pur_ce_weight * adv_pur_ce
            + args.lambda_adv * adv_kl
            + args.lambda_pur * clean_pur_kl
            + args.lambda_adv_pur * adv_pur_kl
            + feature_loss_weight * feature_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            max_norm=0.01,
        )
        optimizer.step()

        total_samples += batch_size
        totals["loss"] += loss.item() * batch_size
        totals["clean_ce"] += clean_ce.item() * batch_size
        totals["adv_ce"] += adv_ce.item() * batch_size
        totals["clean_pur_ce"] += clean_pur_ce.item() * batch_size
        totals["adv_pur_ce"] += adv_pur_ce.item() * batch_size
        totals["adv_kl"] += adv_kl.item() * batch_size
        totals["clean_pur_kl"] += clean_pur_kl.item() * batch_size
        totals["adv_pur_kl"] += adv_pur_kl.item() * batch_size
        totals["feature_loss"] += feature_loss.item() * batch_size
        totals["prototype_alignment"] += prototype_alignment.item() * batch_size
        totals["prototype_margin"] += prototype_margin.item() * batch_size
    metrics = {key: value / total_samples for key, value in totals.items()}
    metrics["feature_skipped_batches"] = skipped_feature_batches
    return metrics


def evaluate_cache_holdout(model, loader, device, ranks):
    """评估预筛 holdout 中 clean、clean-pur 和 adv-pur 的逐 rank accuracy。"""
    model.eval()
    total = 0
    clean_correct = 0
    clean_pur_correct = torch.zeros(len(ranks), dtype=torch.long)
    adv_pur_correct = torch.zeros(len(ranks), dtype=torch.long)
    with torch.no_grad():
        for x, _x_adv, x_pur_by_rank, x_adv_pur_by_rank, labels in loader:
            x = x.to(device)
            x_pur_by_rank = x_pur_by_rank.to(device)
            x_adv_pur_by_rank = x_adv_pur_by_rank.to(device)
            labels = labels.to(device)
            batch_size, rank_count = x_pur_by_rank.shape[:2]
            sample_shape = x_pur_by_rank.shape[2:]
            clean_correct += model(x).argmax(dim=1).eq(labels).sum().item()
            clean_pur_predictions = model(
                x_pur_by_rank.reshape(batch_size * rank_count, *sample_shape)
            ).argmax(dim=1).reshape(batch_size, rank_count)
            adv_pur_predictions = model(
                x_adv_pur_by_rank.reshape(batch_size * rank_count, *sample_shape)
            ).argmax(dim=1).reshape(batch_size, rank_count)
            expanded_labels = labels.unsqueeze(1)
            clean_pur_correct += clean_pur_predictions.eq(
                expanded_labels
            ).sum(dim=0).cpu()
            adv_pur_correct += adv_pur_predictions.eq(
                expanded_labels
            ).sum(dim=0).cpu()
            total += batch_size
    if total == 0:
        raise ValueError("Cache holdout loader is empty.")
    metrics = {"holdout_clean_acc": clean_correct / total}
    for index, rank in enumerate(ranks):
        metrics[f"holdout_clean_pur_rank{rank}_acc"] = (
            clean_pur_correct[index].item() / total
        )
        metrics[f"holdout_adv_pur_rank{rank}_acc"] = (
            adv_pur_correct[index].item() / total
        )
    target_indices = [index for index, rank in enumerate(ranks) if rank in (25, 30)]
    if target_indices:
        metrics["holdout_rank25_30_clean_pur_acc"] = sum(
            clean_pur_correct[index].item() / total for index in target_indices
        ) / len(target_indices)
        metrics["holdout_rank25_30_adv_pur_acc"] = sum(
            adv_pur_correct[index].item() / total for index in target_indices
        ) / len(target_indices)
    return metrics


def main():
    args = parse_args()
    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.online_at_batch_size <= 0
        or args.online_at_pgd_steps <= 0
    ):
        raise ValueError("epochs and batch_size must be positive.")
    if args.consistancy_temperature <= 0:
        raise ValueError("--consistancy_temperature must be positive.")
    if args.online_at_step_size is not None and args.online_at_step_size <= 0:
        raise ValueError("--online_at_step_size must be positive when provided.")
    if (
        args.online_train_sample_num is not None
        and args.online_train_sample_num <= 0
    ):
        raise ValueError("--online_train_sample_num must be positive when provided.")
    if args.feature_objective == "none":
        if args.feature_loss_weight != 0 or args.prototype_margin_weight != 0:
            raise ValueError(
                "Feature/margin weights must be zero when --feature_objective=none."
            )
    elif args.feature_loss_weight <= 0:
        raise ValueError("--feature_loss_weight must be positive for feature objectives.")
    if args.prototype_margin <= 0 or args.prototype_margin_weight < 0:
        raise ValueError("Prototype margin must be positive and its weight non-negative.")
    if args.contrastive_temperature <= 0:
        raise ValueError("--contrastive_temperature must be positive.")
    if not 0 <= args.cache_holdout_fraction < 1:
        raise ValueError("--cache_holdout_fraction must be in [0, 1).")
    if args.balanced_samples_per_class <= 0:
        raise ValueError("--balanced_samples_per_class must be positive.")
    if args.max_cache_batches is not None and args.max_cache_batches <= 0:
        raise ValueError("--max_cache_batches must be positive when provided.")
    if args.balanced_classes_per_batch is not None:
        if args.balanced_classes_per_batch <= 0:
            raise ValueError("--balanced_classes_per_batch must be positive.")
        balanced_batch_size = (
            args.balanced_classes_per_batch * args.balanced_samples_per_class
        )
        if args.batch_size != balanced_batch_size:
            raise ValueError(
                "--batch_size must equal balanced_classes_per_batch × "
                f"balanced_samples_per_class ({balanced_batch_size})."
            )
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output_checkpoint) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.history_prefix) or ".", exist_ok=True)
    logging.basicConfig(
        filename=f"{args.history_prefix}.log",
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.info("RPCF fine-tuning start: %s", vars(args))

    cache = load_rpcf_cache(
        args.cache_path,
        expected={
            "dataset": args.dataset,
            "model": args.model,
            "fold": args.fold,
            "seed": args.seed,
            "eps": args.epsilon,
        },
    )
    dataset, info = DATASET_LOADERS[args.dataset]()
    raw_train_dataset, val_dataset, _, split_path = prepare_subject_fold(
        dataset_name=args.dataset,
        dataset=dataset,
        info=info,
        fold_id=args.fold,
        seed=args.seed,
        use_ea=False,
    )
    model = load_model_checkpoint(
        args.model, args.dataset, info, args.checkpoint_path, device
    )
    sensitivity, selected_layers = resolve_layer_configuration(
        args, cache, model
    )
    trainable_stats = configure_trainable_layers(
        model, selected_layers, all_layers=args.all_layers
    )
    logging.info("Selected layers: %s", selected_layers)
    logging.info("Trainable stats: %s", trainable_stats)
    feature_adapter = (
        None
        if args.feature_objective == "none"
        else PenultimateFeatureAdapter(args.model, model)
    )

    full_cache_dataset = torch.utils.data.TensorDataset(
        cache["x"],
        cache["x_adv"],
        cache["x_pur_by_rank"],
        cache["x_adv_pur_by_rank"],
        cache["labels"],
    )
    train_indices = list(range(len(full_cache_dataset)))
    holdout_indices = []
    holdout_loader = None
    if args.cache_holdout_fraction > 0:
        train_indices, holdout_indices = stratified_cache_split_indices(
            cache["labels"],
            args.cache_holdout_fraction,
            seed=args.seed + args.fold * 1000,
        )
        train_dataset = torch.utils.data.Subset(
            full_cache_dataset, train_indices
        )
        holdout_dataset = torch.utils.data.Subset(
            full_cache_dataset, holdout_indices
        )
        holdout_loader = torch.utils.data.DataLoader(
            holdout_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        train_dataset = full_cache_dataset

    balanced_sampler = None
    if args.balanced_classes_per_batch is not None:
        train_labels = cache["labels"][torch.as_tensor(train_indices)]
        balanced_sampler = ClassBalancedBatchSampler(
            train_labels,
            classes_per_batch=args.balanced_classes_per_batch,
            samples_per_class=args.balanced_samples_per_class,
            seed=args.seed + args.fold * 1000,
            num_batches=args.max_cache_batches,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=balanced_sampler,
            num_workers=0,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
    online_train_loader = None
    online_train_sample_num = len(raw_train_dataset)
    if args.online_madry_at:
        online_train_dataset = raw_train_dataset
        if args.online_train_sample_num is not None:
            online_train_sample_num = min(
                args.online_train_sample_num, len(raw_train_dataset)
            )
            generator = torch.Generator().manual_seed(
                args.seed + args.fold * 1000 + 17
            )
            indices = torch.randperm(
                len(raw_train_dataset), generator=generator
            )[:online_train_sample_num].tolist()
            online_train_dataset = torch.utils.data.Subset(
                raw_train_dataset, indices
            )
        online_train_loader = torch.utils.data.DataLoader(
            online_train_dataset,
            batch_size=args.online_at_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=eeg_classification_collate,
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=eeg_classification_collate,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    initial_clean = evaluate_classifier(model, val_loader, device)
    initial_robust = evaluate_pgd(
        model,
        val_loader,
        device,
        epsilon=args.epsilon,
        steps=args.pgd_steps,
    )
    initial_metric = {
        "robust_acc": initial_robust,
        "clean_acc": initial_clean["accuracy"],
        "val_loss": initial_clean["loss"],
    }
    if holdout_loader is not None:
        initial_metric.update(
            evaluate_cache_holdout(model, holdout_loader, device, cache["ranks"])
        )
    history = []
    logging.info("Initial validation metric: %s", initial_metric)

    for epoch in range(args.epochs):
        if balanced_sampler is not None:
            balanced_sampler.set_epoch(epoch)
        rank_weights = compute_rank_weights(
            cache["ranks"],
            epoch,
            args.epochs,
            temperature=args.rank_temperature,
            static=args.static_rank_weights,
        ).to(device)
        online_at_loss = None
        if online_train_loader is not None:
            online_at_loss = train_epoch_online_madry(
                model,
                online_train_loader,
                optimizer,
                device,
                selected_layers,
                args.all_layers,
                args,
            )
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            rank_weights,
            selected_layers,
            args.all_layers,
            args,
            use_cached_adv=not args.online_madry_at,
            feature_adapter=feature_adapter,
        )
        if online_at_loss is not None:
            train_metrics["online_at_loss"] = online_at_loss
        clean_metrics = evaluate_classifier(model, val_loader, device)
        robust_acc = evaluate_pgd(
            model,
            val_loader,
            device,
            epsilon=args.epsilon,
            steps=args.pgd_steps,
        )
        candidate = {
            "robust_acc": robust_acc,
            "clean_acc": clean_metrics["accuracy"],
            "val_loss": clean_metrics["loss"],
        }
        if holdout_loader is not None:
            candidate.update(
                evaluate_cache_holdout(
                    model, holdout_loader, device, cache["ranks"]
                )
            )
        row = {
            "epoch": epoch + 1,
            **train_metrics,
            **candidate,
            "lr": optimizer.param_groups[0]["lr"],
            "rank_weights": [float(value) for value in rank_weights.detach().cpu()],
        }
        history.append(row)
        logging.info("Epoch %d: %s", epoch + 1, row)

    final_metric = {
        "robust_acc": history[-1]["robust_acc"],
        "clean_acc": history[-1]["clean_acc"],
        "val_loss": history[-1]["val_loss"],
    }
    final_epoch = len(history)
    atomic_torch_save(model.state_dict(), args.output_checkpoint)
    history_payload = {
        "kind": "rpcf_finetune_history",
        "dataset": args.dataset,
        "model": args.model,
        "fold": args.fold,
        "seed": args.seed,
        "epsilon": args.epsilon,
        "cache_path": args.cache_path,
        "sensitivity_path": args.sensitivity_path,
        "at_checkpoint_path": args.checkpoint_path,
        "output_checkpoint": args.output_checkpoint,
        "split_path": split_path,
        "ranks": cache["ranks"],
        "selected_layers": selected_layers,
        "selected_layers_override": args.selected_layers_override,
        "layer_selection_rule": (
            args.layer_selection_rule
            or (
                "all_layers"
                if args.all_layers
                else (
                    "override"
                    if args.selected_layers_override
                    else "sensitivity_selected_layers"
                )
            )
        ),
        "prefix_k": args.prefix_k,
        "all_layers": args.all_layers,
        "static_rank_weights": args.static_rank_weights,
        "feature_objective": args.feature_objective,
        "feature_alignment": {
            "objective": args.feature_objective,
            "loss_weight": args.feature_loss_weight,
            "clean_anchor_detached": args.feature_objective != "none",
            "feature_layers": (
                feature_adapter.feature_layer_names if feature_adapter else []
            ),
            "feature_dim": feature_adapter.feature_dim if feature_adapter else None,
            "l2_normalized": args.feature_objective != "none",
            "mmd_kernel": {
                "kind": "multi_rbf_median_biased",
                "scales": [0.5, 1.0, 2.0],
            },
            "prototype_margin": args.prototype_margin,
            "prototype_margin_weight": args.prototype_margin_weight,
            "contrastive_temperature": args.contrastive_temperature,
            "contrastive_positive_views": ["cached_adv", "adv_pur_by_rank"],
            "contrastive_negative_rule": "all_views_from_other_trials",
        },
        "cache_split": {
            "holdout_fraction": args.cache_holdout_fraction,
            "train_size": len(train_indices),
            "holdout_size": len(holdout_indices),
            "train_indices": train_indices if holdout_indices else None,
            "holdout_indices": holdout_indices if holdout_indices else None,
        },
        "cache_sampler": {
            "kind": (
                "class_balanced" if balanced_sampler is not None else "legacy_shuffle"
            ),
            "classes_per_batch": args.balanced_classes_per_batch,
            "samples_per_class": args.balanced_samples_per_class,
            "batch_size": args.batch_size,
            "seed": args.seed + args.fold * 1000,
            "num_batches": (
                len(balanced_sampler) if balanced_sampler is not None else len(train_loader)
            ),
            "max_cache_batches": args.max_cache_batches,
        },
        "online_madry_at": args.online_madry_at,
        "online_at": {
            "train_sample_num": (
                online_train_sample_num if args.online_madry_at else 0
            ),
            "batch_size": args.online_at_batch_size,
            "pgd_steps": args.online_at_pgd_steps,
            "step_size": (
                args.online_at_step_size
                if args.online_at_step_size is not None
                else args.epsilon / 5.0
            ),
            "random_start": True,
            "loss": "adversarial_ce",
        },
        "rank_temperature": args.rank_temperature,
        "loss_rule": (
            "online Madry AT + purification-only clean-teacher consistancy CE + KL"
            if args.online_madry_at
            else "clean-teacher consistancy CE + KL"
        ),
        "consistancy_temperature": args.consistancy_temperature,
        "loss_weights": {
            "clean_ce": args.clean_ce_weight,
            "adv_ce": args.adv_ce_weight,
            "pur_ce": args.pur_ce_weight,
            "adv_pur_ce": args.adv_pur_ce_weight,
            "adv_kl": args.lambda_adv,
            "pur_kl": args.lambda_pur,
            "adv_pur_kl": args.lambda_adv_pur,
            "feature_loss": args.feature_loss_weight,
            "prototype_margin": args.prototype_margin_weight,
        },
        "cached_adv_loss_enabled": not args.online_madry_at,
        "trainable_stats": trainable_stats,
        "sensitivity_selected_param_ratio": sensitivity.get(
            "selected_param_ratio"
        ),
        "checkpoint_policy": "final_epoch_no_early_stopping",
        "configured_patience_ignored": args.patience,
        "initial_metric": initial_metric,
        # 保留旧字段，兼容 EXP-018 汇总脚本；此处表示最终保存的 epoch。
        "best_epoch": final_epoch,
        "best_metric": final_metric,
        "history": history,
    }
    write_json(f"{args.history_prefix}.json", history_payload)
    csv_rows = []
    for row in history:
        flat = {key: value for key, value in row.items() if key != "rank_weights"}
        for rank, weight in zip(cache["ranks"], row["rank_weights"]):
            flat[f"weight_rank{rank}"] = weight
        csv_rows.append(flat)
    fieldnames = list(csv_rows[0]) if csv_rows else [
        "epoch",
        "loss",
        "clean_ce",
        "adv_ce",
        "clean_pur_ce",
        "adv_pur_ce",
        "online_at_loss",
        "robust_acc",
        "clean_acc",
        "val_loss",
        "lr",
    ]
    write_csv(f"{args.history_prefix}.csv", fieldnames, csv_rows)
    logging.info(
        "Saved final RPCF checkpoint=%s, epoch=%d, final_metric=%s",
        args.output_checkpoint,
        final_epoch,
        final_metric,
    )
    print(args.output_checkpoint)


if __name__ == "__main__":
    main()
