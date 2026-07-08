import argparse
import copy
import datetime
import os
import re

from runtime_env import configure_runtime_env

configure_runtime_env()

import numpy as np
import torch
import torch.nn.functional as F

from data.load import load_bciciv2a, load_m3cv, load_seediv, load_thubenchmark
from data.subject_ea import get_protocol_tag, prepare_subject_fold
from models.model_args import get_model_args
from models.registry import MODEL_CHOICES, MODEL_CLASSES
from train_AT_consistancy import evaluate, seed_everything, train_epoch_madry
from utils.experiment_artifacts import (
    as_label_tensor,
    build_checkpoint_path,
    eeg_classification_collate,
    normalize_path_args,
    safe_token,
    torch_load_cpu,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='seediv',
                        choices=['seediv', 'm3cv', 'bciciv2a', 'thubenchmark'], help='choose dataset')
    parser.add_argument('--model', type=str, default='eegnet',
                        choices=MODEL_CHOICES, help='choose model')
    parser.add_argument('--at_strategy', type=str, default='madry', choices=['madry'],
                        help='consistency2 version only supports Madry AT')
    parser.add_argument('--fold', type=int, default=0, help='which fold to train')
    parser.add_argument('--epsilon', type=float, default=0.1, help='max perturbation budget for adversarial training')
    parser.add_argument('--pgd_step_size', type=float, default=0.02, help='step size for PGD updates')
    parser.add_argument('--pgd_steps', type=int, default=10, help='number of PGD steps for adversarial examples')
    parser.add_argument('--clean_ratio', type=float, default=0.0, help='portion of clean loss mixed with Madry loss')
    parser.add_argument('--clip_min', type=float, default=None, help='minimum value to clamp adversarial examples')
    parser.add_argument('--clip_max', type=float, default=None, help='maximum value to clamp adversarial examples')
    parser.add_argument('--pgd_random_start', action='store_true', default=True,
                        help='enable random start for PGD attacks')
    parser.add_argument('--no_pgd_random_start', dest='pgd_random_start', action='store_false',
                        help='disable random start for PGD attacks')
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for original training data')
    parser.add_argument('--consistency_batch_size', '--consistancy_batch_size',
                        dest='consistency_batch_size', type=int, default=None,
                        help='batch size for paired consistency2 data; default equals --batch_size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu_id', type=int, default=0, help='which gpu to use')
    parser.add_argument('--use_ea', dest='use_ea', action='store_true', default=False,
                        help='use EA-aligned data after subject split')
    parser.add_argument('--no_ea', dest='use_ea', action='store_false',
                        help='use raw subject-split data without EA alignment')
    parser.add_argument('--use_consistency_aug', '--use_consistancy_aug',
                        dest='use_consistency_aug', action='store_true',
                        help='train with paired EEG_TNP consistency2 augmentation')
    parser.add_argument('--consistency_aug_paths', '--consistancy_aug_paths',
                        dest='consistency_aug_paths', nargs='*', default=None,
                        help='two or more paired consistency2 .pth files; comma-separated input is accepted')
    parser.add_argument('--consistency_aug_tag', '--consistancy_aug_tag',
                        dest='consistency_aug_tag', type=str, default='consistency2',
                        help='experiment tag added to logs and checkpoints')
    parser.add_argument('--consistency_temperature', '--consistancy_temperature',
                        dest='consistency_temperature', type=float, default=2.0,
                        help='temperature for KL distribution alignment')
    parser.add_argument('--consistency_lambda_pur', '--consistancy_lambda_pur',
                        dest='consistency_lambda_pur', type=float, default=0.2,
                        help='KL weight for aligning x_pur to x')
    parser.add_argument('--consistency_lambda_adv', '--consistancy_lambda_adv',
                        dest='consistency_lambda_adv', type=float, default=0.5,
                        help='KL weight for aligning x_adv_pur to x')
    parser.add_argument('--consistency_lambda_rank', '--consistancy_lambda_rank',
                        dest='consistency_lambda_rank', type=float, default=0.2,
                        help='one-way KL weight from lower-rank purified samples to higher-rank purified samples')
    parser.add_argument('--consistency_ce_adv_pur_weight', '--consistancy_ce_adv_pur_weight',
                        dest='consistency_ce_adv_pur_weight', type=float, default=1.0,
                        help='hard-label CE weight for x_adv_pur')
    parser.add_argument('--consistency_ce_pur_weight', '--consistancy_ce_pur_weight',
                        dest='consistency_ce_pur_weight', type=float, default=0.5,
                        help='hard-label CE weight for x_pur')
    parser.add_argument('--consistency_warmup_epochs', '--consistancy_warmup_epochs',
                        dest='consistency_warmup_epochs', type=int, default=0,
                        help='number of epochs to train CE-only before enabling KL alignment')
    return parser.parse_args()


def _validate_args(args):
    if args.consistency_temperature <= 0:
        raise ValueError('--consistency_temperature must be positive.')
    if args.consistency_warmup_epochs < 0:
        raise ValueError('--consistency_warmup_epochs must be non-negative.')
    if args.consistency_batch_size is not None and args.consistency_batch_size <= 0:
        raise ValueError('--consistency_batch_size must be positive when set.')
    weighted_args = {
        'consistency_lambda_pur': args.consistency_lambda_pur,
        'consistency_lambda_adv': args.consistency_lambda_adv,
        'consistency_lambda_rank': args.consistency_lambda_rank,
        'consistency_ce_adv_pur_weight': args.consistency_ce_adv_pur_weight,
        'consistency_ce_pur_weight': args.consistency_ce_pur_weight,
    }
    for name, value in weighted_args.items():
        if value < 0:
            raise ValueError(f'--{name} must be non-negative.')


def _validate_consistency2_meta(meta, path, args, protocol_tag):
    if not isinstance(meta, dict):
        raise ValueError(f'Consistency2 metadata missing or invalid for {path}.')
    checks = {
        'dataset': args.dataset,
        'model': args.model,
        'fold': args.fold,
        'seed': args.seed,
    }
    for key, expected in checks.items():
        if key not in meta:
            raise ValueError(f'Consistency2 metadata missing {key} for {path}.')
        if str(meta[key]) != str(expected):
            raise ValueError(
                f'Consistency2 metadata mismatch for {path}: {key}={meta[key]} but expected {expected}.'
            )

    if meta.get('kind') != 'consistency2_pair':
        raise ValueError(f'Consistency2 file must have kind=consistency2_pair: {path}')
    meta_protocol = meta.get('protocol', meta.get('protocol_tag'))
    if meta_protocol is None:
        raise ValueError(f'Consistency2 metadata missing protocol for {path}.')
    if str(meta_protocol) != str(protocol_tag):
        raise ValueError(
            f'Consistency2 metadata mismatch for {path}: protocol={meta_protocol} but expected {protocol_tag}.'
        )
    if meta.get('source_split') != 'train':
        raise ValueError(f'Consistency2 file must come from train split: {path}')
    if 'eps' in meta and abs(float(meta['eps']) - float(args.epsilon)) > 1e-12:
        raise ValueError(
            f'Consistency2 eps mismatch for {path}: eps={meta["eps"]} but expected {args.epsilon}.'
        )


def _as_float_tensor(payload, key, path):
    if key not in payload:
        raise ValueError(f'Consistency2 payload missing {key}: {path}')
    value = payload[key]
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.detach().cpu().float()


def _source_index_position_map(source_indices, path):
    positions = {}
    for position, source_index in enumerate(source_indices):
        source_index = int(source_index)
        if source_index in positions:
            raise ValueError(f'Duplicate source_index={source_index} in {path}.')
        positions[source_index] = position
    return positions


def _ordered_by_source(tensor, positions, source_order):
    index = torch.tensor([positions[int(source_index)] for source_index in source_order], dtype=torch.long)
    return tensor.index_select(0, index)


def _validate_pair_tensors(path, tensors, labels, source_indices, expected_sample_shape):
    pair_count = tensors['x'].size(0)
    for name, tensor in tensors.items():
        if tensor.size(0) != pair_count:
            raise ValueError(
                f'Consistency2 length mismatch for {path}: {name} has {tensor.size(0)} '
                f'but x has {pair_count}.'
            )
        if tuple(tensor.shape[1:]) != tuple(expected_sample_shape):
            raise ValueError(
                f'Consistency2 shape mismatch for {path}: {name} sample shape '
                f'{tuple(tensor.shape[1:])} but expected {tuple(expected_sample_shape)}.'
            )
    if labels.size(0) != pair_count:
        raise ValueError(
            f'Consistency2 label length mismatch for {path}: labels has {labels.size(0)} '
            f'but x has {pair_count}.'
        )
    if source_indices is None or len(source_indices) != pair_count:
        raise ValueError(f'Consistency2 source_indices missing or length mismatch for {path}.')


def _infer_rank_value(meta, path, fallback):
    """从配置名中解析 rank 数值；解析失败时退回到输入顺序，保证旧 payload 仍可运行。"""
    candidates = [
        meta.get('config'),
        os.path.basename(str(path)),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        match = re.search(r'rank(\d+)', str(candidate))
        if match:
            return int(match.group(1))
    return fallback


def load_consistency2_rank_dataset(paths, expected_sample_shape, args, protocol_tag, logger):
    """按 source_indices 对齐多个 rank 文件，构造 [N, R, ...] 的净化样本张量。"""
    if len(paths) < 2:
        raise ValueError('Consistency2 rank KL requires at least two --consistency_aug_paths files.')

    base_x = None
    base_labels = None
    base_source_order = None
    base_source_set = None
    rank_entries = []

    for rank_index, path in enumerate(paths):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Consistency2 augmentation file not found: {path}')
        payload = torch_load_cpu(path)
        if not isinstance(payload, dict):
            raise ValueError(f'Unsupported consistency2 payload in {path}; expected dict payload.')

        meta = payload.get('meta', {})
        _validate_consistency2_meta(meta, path, args, protocol_tag)
        x = _as_float_tensor(payload, 'x', path)
        x_pur = _as_float_tensor(payload, 'x_pur', path)
        x_adv = _as_float_tensor(payload, 'x_adv', path)
        x_adv_pur = _as_float_tensor(payload, 'x_adv_pur', path)
        if 'labels' not in payload:
            raise ValueError(f'Consistency2 payload missing labels: {path}')
        labels = as_label_tensor(payload['labels'])
        source_indices = payload.get('source_indices')

        tensors = {'x': x, 'x_pur': x_pur, 'x_adv': x_adv, 'x_adv_pur': x_adv_pur}
        _validate_pair_tensors(path, tensors, labels, source_indices, expected_sample_shape)

        positions = _source_index_position_map(source_indices, path)
        source_set = set(positions.keys())
        if rank_index == 0:
            base_source_order = [int(source_index) for source_index in source_indices]
            base_source_set = source_set
            base_x = x
            base_labels = labels
        elif source_set != base_source_set:
            missing = sorted(base_source_set - source_set)[:10]
            extra = sorted(source_set - base_source_set)[:10]
            raise ValueError(
                f'Consistency2 source_indices mismatch for {path}; '
                f'missing preview={missing}, extra preview={extra}.'
            )

        ordered_x = _ordered_by_source(x, positions, base_source_order)
        ordered_labels = _ordered_by_source(labels, positions, base_source_order)
        ordered_x_pur = _ordered_by_source(x_pur, positions, base_source_order)
        ordered_x_adv_pur = _ordered_by_source(x_adv_pur, positions, base_source_order)
        if rank_index > 0:
            if not torch.equal(ordered_labels, base_labels):
                raise ValueError(f'Consistency2 labels do not align with the first rank file: {path}')
            if not torch.allclose(ordered_x, base_x, atol=1e-6, rtol=1e-5):
                raise ValueError(f'Consistency2 clean x does not align with the first rank file: {path}')

        rank_value = _infer_rank_value(meta, path, rank_index)
        rank_tag = safe_token(meta.get('config', f'rank{rank_value}'))
        rank_entries.append((rank_value, rank_tag, ordered_x_pur, ordered_x_adv_pur))
        logger.info(
            f'Loaded consistency2 rank augmentation: {path}, pairs: {x.size(0)}, '
            f'config: {meta.get("config")}, rank: {rank_value}, '
            f'attack: {meta.get("attack")}, eps: {meta.get("eps")}'
        )

    rank_values = [entry[0] for entry in rank_entries]
    if len(set(rank_values)) != len(rank_values):
        raise ValueError(f'Consistency2 rank values must be unique, got: {rank_values}.')

    # 低 rank 到高 rank 的 KL 依赖这个顺序，不能沿用命令行输入顺序。
    rank_entries.sort(key=lambda item: item[0])
    rank_values = [entry[0] for entry in rank_entries]
    rank_tags = [entry[1] for entry in rank_entries]
    x_pur_rank_parts = [entry[2] for entry in rank_entries]
    x_adv_pur_rank_parts = [entry[3] for entry in rank_entries]
    x_pur_by_rank = torch.stack(x_pur_rank_parts, dim=1)
    x_adv_pur_by_rank = torch.stack(x_adv_pur_rank_parts, dim=1)
    dataset = torch.utils.data.TensorDataset(
        base_x,
        x_pur_by_rank,
        x_adv_pur_by_rank,
        base_labels,
    )
    logger.info(
        f'Built consistency2 rank dataset: samples={base_x.size(0)}, ranks={len(rank_tags)}, '
        f'x_pur_by_rank={tuple(x_pur_by_rank.shape)}, x_adv_pur_by_rank={tuple(x_adv_pur_by_rank.shape)}, '
        f'rank_values={rank_values}, rank_tags={rank_tags}'
    )
    return dataset, base_x.size(0), len(rank_tags), rank_tags, rank_values


def _kl_to_teacher(student_logits, teacher_probs, temperature):
    log_probs = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)


def _lower_to_higher_rank_kl(logits_by_rank, temperature):
    """让低 rank 的净化 logits 单向靠近高 rank logits，并对 rank pair 数取均值。"""
    rank_count = logits_by_rank.size(1)
    if rank_count < 2:
        return logits_by_rank.new_tensor(0.0)

    log_probs = F.log_softmax(logits_by_rank / temperature, dim=-1)
    probs = F.softmax(logits_by_rank / temperature, dim=-1)
    pair_losses = []
    for low_rank_index in range(rank_count):
        for high_rank_index in range(low_rank_index + 1, rank_count):
            # 高 rank 作为 teacher，只更新低 rank student 的对齐方向。
            pair_losses.append(F.kl_div(
                log_probs[:, low_rank_index, :],
                probs[:, high_rank_index, :].detach(),
                reduction='batchmean',
            ))
    return torch.stack(pair_losses).mean() * (temperature ** 2)


def train_epoch_consistency2_aug(model, loader, optimizer, criterion, device, args, enable_kl=True):
    """在多 rank paired 数据上训练 CE、原始 KL 和低 rank 到高 rank KL。"""
    model.train()
    totals = {
        'loss': 0.0,
        'ce_pur': 0.0,
        'ce_adv_pur': 0.0,
        'kl_pur': 0.0,
        'kl_adv': 0.0,
        'kl_rank_pur': 0.0,
        'kl_rank_adv': 0.0,
    }
    total_samples = 0
    temperature = args.consistency_temperature

    for x, x_pur_by_rank, x_adv_pur_by_rank, target in loader:
        x = x.to(device)
        x_pur_by_rank = x_pur_by_rank.to(device)
        x_adv_pur_by_rank = x_adv_pur_by_rank.to(device)
        target = target.to(device)

        batch_size, rank_count = x_pur_by_rank.shape[:2]
        sample_shape = x_pur_by_rank.shape[2:]
        target_by_rank = target.unsqueeze(1).expand(-1, rank_count).reshape(-1)
        x_pur_flat = x_pur_by_rank.reshape(batch_size * rank_count, *sample_shape)
        x_adv_pur_flat = x_adv_pur_by_rank.reshape(batch_size * rank_count, *sample_shape)

        optimizer.zero_grad()
        with torch.no_grad():
            teacher_probs = F.softmax(model(x).detach() / temperature, dim=1)
            teacher_by_rank = teacher_probs.unsqueeze(1).expand(-1, rank_count, -1).reshape(
                batch_size * rank_count,
                -1,
            )

        logits_pur_flat = model(x_pur_flat)
        logits_adv_pur_flat = model(x_adv_pur_flat)
        class_count = logits_pur_flat.size(1)
        logits_pur_by_rank = logits_pur_flat.reshape(batch_size, rank_count, class_count)
        logits_adv_pur_by_rank = logits_adv_pur_flat.reshape(batch_size, rank_count, class_count)

        ce_pur = criterion(logits_pur_flat, target_by_rank)
        ce_adv_pur = criterion(logits_adv_pur_flat, target_by_rank)

        if enable_kl:
            kl_pur = _kl_to_teacher(logits_pur_flat, teacher_by_rank, temperature)
            kl_adv = _kl_to_teacher(logits_adv_pur_flat, teacher_by_rank, temperature)
            kl_rank_pur = _lower_to_higher_rank_kl(logits_pur_by_rank, temperature)
            kl_rank_adv = _lower_to_higher_rank_kl(logits_adv_pur_by_rank, temperature)
        else:
            kl_pur = logits_pur_flat.new_tensor(0.0)
            kl_adv = logits_adv_pur_flat.new_tensor(0.0)
            kl_rank_pur = logits_pur_flat.new_tensor(0.0)
            kl_rank_adv = logits_adv_pur_flat.new_tensor(0.0)

        loss = (
            args.consistency_ce_adv_pur_weight * ce_adv_pur
            + args.consistency_ce_pur_weight * ce_pur
            + args.consistency_lambda_pur * kl_pur
            + args.consistency_lambda_adv * kl_adv
            + args.consistency_lambda_rank * (kl_rank_pur + kl_rank_adv)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
        optimizer.step()

        total_samples += batch_size
        totals['loss'] += loss.item() * batch_size
        totals['ce_pur'] += ce_pur.item() * batch_size
        totals['ce_adv_pur'] += ce_adv_pur.item() * batch_size
        totals['kl_pur'] += kl_pur.item() * batch_size
        totals['kl_adv'] += kl_adv.item() * batch_size
        totals['kl_rank_pur'] += kl_rank_pur.item() * batch_size
        totals['kl_rank_adv'] += kl_rank_adv.item() * batch_size

    return {key: value / total_samples for key, value in totals.items()}


def main():
    args = parse_args()
    _validate_args(args)
    args.pgd_step_size = args.epsilon / 5
    args.pgd_steps = 5 * 2
    seed_everything(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    import logging

    os.makedirs('./log_train_AT', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_tag = None
    if args.use_consistency_aug:
        checkpoint_tag = safe_token(args.consistency_aug_tag, default='consistency2')
    log_suffix = f'_{checkpoint_tag}' if checkpoint_tag else ''
    logfile_directory = (
        f'./log_train_AT/train_consistency2_{args.dataset}_{args.model}_{args.at_strategy}_'
        f'eps{args.epsilon}_{args.seed}_{args.lr}_{args.weight_decay}_{args.batch_size}'
        f'{log_suffix}_{timestamp}.log'
    )
    logging.basicConfig(
        filename=logfile_directory,
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.info(f'Training consistency2 {args.dataset} with {args.model}')
    logging.info(args)
    protocol_tag = get_protocol_tag(use_ea=args.use_ea)
    logging.info(f'Data protocol: {protocol_tag}, use_ea: {args.use_ea}')
    logging.info(
        f'Consistency2 augmentation enabled: {args.use_consistency_aug}, '
        f'tag: {args.consistency_aug_tag}, fold: {args.fold}'
    )
    logging.info(
        f"AT config | strategy: {args.at_strategy}, eps: {args.epsilon}, "
        f"step size: {args.pgd_step_size}, steps: {args.pgd_steps}, "
        f"clean_ratio: {args.clean_ratio}, random_start: {args.pgd_random_start}, "
        f"clip_min/max: ({args.clip_min}, {args.clip_max})"
    )
    logging.info(
        f"Consistency2 loss | T: {args.consistency_temperature}, "
        f"lambda_pur: {args.consistency_lambda_pur}, lambda_adv: {args.consistency_lambda_adv}, "
        f"lambda_rank: {args.consistency_lambda_rank}, "
        f"ce_adv_pur_weight: {args.consistency_ce_adv_pur_weight}, "
        f"ce_pur_weight: {args.consistency_ce_pur_weight}, "
        f"warmup_epochs: {args.consistency_warmup_epochs}"
    )

    dataset_dict = {
        'seediv': load_seediv,
        'm3cv': load_m3cv,
        'bciciv2a': load_bciciv2a,
        'thubenchmark': load_thubenchmark,
    }
    dataset, info = dataset_dict[args.dataset]()
    logging.info(f'Dataset: {args.dataset}, Sample_num: {len(dataset)}, num_classes: {info["num_classes"]}')

    train_dataset, val_dataset, test_dataset, split_path = prepare_subject_fold(
        dataset_name=args.dataset,
        dataset=dataset,
        info=info,
        fold_id=args.fold,
        seed=args.seed,
        use_ea=args.use_ea,
    )
    logging.info(f'Split path: {split_path}')
    logging.info(
        f'sample num in train set: {len(train_dataset)}, sample num in val set: {len(val_dataset)}, '
        f'sample num in test set: {len(test_dataset)}'
    )

    pair_loader = None
    if args.use_consistency_aug:
        consistency_aug_paths = normalize_path_args(args.consistency_aug_paths)
        if not consistency_aug_paths:
            raise ValueError('--use_consistency_aug requires at least two --consistency_aug_paths files.')
        sample_data, _ = train_dataset[0]
        pair_dataset, pair_count, rank_count, rank_tags, rank_values = load_consistency2_rank_dataset(
            paths=consistency_aug_paths,
            expected_sample_shape=tuple(sample_data.shape),
            args=args,
            protocol_tag=protocol_tag,
            logger=logging,
        )
        pair_batch_size = args.consistency_batch_size or args.batch_size
        pair_loader = torch.utils.data.DataLoader(
            pair_dataset,
            batch_size=pair_batch_size,
            shuffle=True,
            num_workers=0,
        )
        logging.info(
            f'Using consistency2 paired augmentation: {pair_count} aligned samples, '
            f'ranks: {rank_count}, rank_values: {rank_values}, rank_tags: {rank_tags}, '
            f'pair batch size: {pair_batch_size}'
        )

    model_dict = {
        'eegnet': EEGNet,
        'tsception': TSCeption,
        'atcnet': ATCNet,
        'conformer': Conformer,
    }
    model = model_dict[args.model](**get_model_args(args.model, args.dataset, info))
    model.to(device)
    logging.info(f'Model: {args.model}, Parameter Num: {sum(p.numel() for p in model.parameters())}')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=eeg_classification_collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=eeg_classification_collate,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=eeg_classification_collate,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience // 2
    )

    best_val_loss = float('inf')
    patience = getattr(args, 'patience', args.patience)
    min_delta = getattr(args, 'min_delta', 0.0)
    no_improve_epochs = 0
    criterion = torch.nn.CrossEntropyLoss()
    best_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        madry_loss = train_epoch_madry(model, train_loader, optimizer, criterion, device, args)
        consistency_metrics = None
        if pair_loader is not None:
            enable_kl = epoch >= args.consistency_warmup_epochs
            consistency_metrics = train_epoch_consistency2_aug(
                model, pair_loader, optimizer, criterion, device, args, enable_kl=enable_kl
            )
            train_loss = madry_loss + consistency_metrics['loss']
        else:
            train_loss = madry_loss

        val_acc, val_loss = evaluate(model, val_loader)
        test_acc, test_loss = evaluate(model, test_loader)
        scheduler.step(val_loss)
        if consistency_metrics is None:
            logging.info(
                f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, '
                f'Madry Loss: {madry_loss:.4f}, Val Acc: {val_acc:.4f}, '
                f'Val Loss: {val_loss:.4f}, Test Acc: {test_acc:.4f}, '
                f'Test Loss: {test_loss:.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}'
            )
        else:
            logging.info(
                f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, '
                f'Madry Loss: {madry_loss:.4f}, Consistency2 Loss: {consistency_metrics["loss"]:.4f}, '
                f'Consistency2 CE Pur: {consistency_metrics["ce_pur"]:.4f}, '
                f'Consistency2 CE AdvPur: {consistency_metrics["ce_adv_pur"]:.4f}, '
                f'Consistency2 KL Pur: {consistency_metrics["kl_pur"]:.4f}, '
                f'Consistency2 KL AdvPur: {consistency_metrics["kl_adv"]:.4f}, '
                f'Consistency2 KL RankPur: {consistency_metrics["kl_rank_pur"]:.4f}, '
                f'Consistency2 KL RankAdvPur: {consistency_metrics["kl_rank_adv"]:.4f}, '
                f'Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}, '
                f'Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}, '
                f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}'
            )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improve_epochs = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(
                best_state_dict,
                build_checkpoint_path(
                    args.dataset, args.model, protocol_tag, args.at_strategy,
                    args.epsilon, args.seed, args.fold, tag=checkpoint_tag,
                    lr=args.lr, weight_decay=args.weight_decay,
                ),
            )
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logging.info(
                    f'Early stopping at epoch {epoch + 1} '
                    f'(no improvement in {patience} epochs). Best Val Loss: {best_val_loss:.4f}'
                )
                break

    best_model = model_dict[args.model](**get_model_args(args.model, args.dataset, info))
    best_model.load_state_dict(best_state_dict)
    best_model.to(device)
    torch.save(
        best_state_dict,
        build_checkpoint_path(
            args.dataset, args.model, protocol_tag, args.at_strategy,
            args.epsilon, args.seed, args.fold, tag=checkpoint_tag,
        ),
    )

    test_acc, test_loss = evaluate(best_model, test_loader)
    logging.info(f'Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}')
    logging.info(f'Acc: {np.mean([test_acc]):.4f}±{np.std([test_acc]):.4f}, '
                 f'Loss: {np.mean([test_loss]):.4f}±{np.std([test_loss]):.4f}')


if __name__ == '__main__':
    main()
