import argparse
import logging
import os

from runtime_env import configure_runtime_env

configure_runtime_env()

import torch
import torch.nn.functional as F

from data.subject_ea import prepare_subject_fold
from utils.experiment_artifacts import eeg_classification_collate

from rpcf.core import (
    DATASET_LOADERS,
    compute_rank_weights,
    configure_trainable_layers,
    evaluate_classifier,
    evaluate_pgd,
    load_model_checkpoint,
    load_rpcf_cache,
    seed_everything,
    set_frozen_batchnorm_eval,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="从 AT checkpoint 初始化并执行 RPCF sensitive-layer fine-tuning。"
    )
    parser.add_argument("--cache_path", required=True)
    parser.add_argument("--sensitivity_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=DATASET_LOADERS)
    parser.add_argument(
        "--model", required=True, choices=["eegnet", "tsception", "atcnet", "conformer"]
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
    parser.add_argument("--static_rank_weights", action="store_true")
    parser.add_argument("--history_prefix", required=True)
    return parser.parse_args()


def load_sensitivity(path, args, cache):
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
    }
    total_samples = 0
    for x, x_adv, x_pur_by_rank, x_adv_pur_by_rank, labels in loader:
        x = x.to(device)
        x_adv = x_adv.to(device)
        x_pur_by_rank = x_pur_by_rank.to(device)
        x_adv_pur_by_rank = x_adv_pur_by_rank.to(device)
        labels = labels.to(device)
        batch_size, rank_count = x_pur_by_rank.shape[:2]
        sample_shape = x_pur_by_rank.shape[2:]

        optimizer.zero_grad(set_to_none=True)
        clean_logits = model(x)
        clean_ce = F.cross_entropy(clean_logits, labels)
        with torch.no_grad():
            teacher_probs = F.softmax(
                clean_logits.detach() / args.consistancy_temperature, dim=1
            )
        clean_pur_logits = model(
            x_pur_by_rank.reshape(batch_size * rank_count, *sample_shape)
        )
        adv_pur_logits = model(
            x_adv_pur_by_rank.reshape(batch_size * rank_count, *sample_shape)
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
        loss = (
            args.clean_ce_weight * clean_ce
            + args.adv_ce_weight * adv_ce
            + args.pur_ce_weight * clean_pur_ce
            + args.adv_pur_ce_weight * adv_pur_ce
            + args.lambda_adv * adv_kl
            + args.lambda_pur * clean_pur_kl
            + args.lambda_adv_pur * adv_pur_kl
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
    return {key: value / total_samples for key, value in totals.items()}


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
    sensitivity, selected_layers = load_sensitivity(
        args.sensitivity_path, args, cache
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
    trainable_stats = configure_trainable_layers(
        model, selected_layers, all_layers=args.all_layers
    )
    logging.info("Selected layers: %s", selected_layers)
    logging.info("Trainable stats: %s", trainable_stats)

    train_dataset = torch.utils.data.TensorDataset(
        cache["x"],
        cache["x_adv"],
        cache["x_pur_by_rank"],
        cache["x_adv_pur_by_rank"],
        cache["labels"],
    )
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
    history = []
    logging.info("Initial validation metric: %s", initial_metric)

    for epoch in range(args.epochs):
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
    torch.save(model.state_dict(), args.output_checkpoint)
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
        "all_layers": args.all_layers,
        "static_rank_weights": args.static_rank_weights,
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
