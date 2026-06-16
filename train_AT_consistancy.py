import argparse
import copy
import datetime
import os

from runtime_env import configure_runtime_env

configure_runtime_env()

import numpy as np
import torch
import torch.nn.functional as F

from data.load import load_bciciv2a, load_m3cv, load_seediv, load_thubenchmark
from data.subject_ea import get_protocol_tag, iter_subject_folds
from models.model_args import get_model_args
from torcheeg.models import ATCNet, Conformer, EEGNet, TSCeption
from utils.experiment_artifacts import (
    as_label_tensor,
    build_checkpoint_path,
    eeg_classification_collate,
    normalize_path_args,
    safe_token,
    torch_load_cpu,
)
from utils.reproducibility import seed_everything, stable_subset_indices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='seediv',
                        choices=['seediv', 'm3cv', 'bciciv2a', 'thubenchmark'], help='choose dataset')
    parser.add_argument('--model', type=str, default='eegnet',
                        choices=['eegnet', 'tsception', 'atcnet', 'conformer'], help='choose model')
    parser.add_argument('--at_strategy', type=str, default='madry', choices=['madry'],
                        help='first consistancy version only supports Madry AT')
    parser.add_argument('--fold', type=int, default=0, help='which subject split fold to train')
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
    parser.add_argument('--train_sample_num', type=int, default=None,
                        help='optional random subset size for training smoke tests; default uses full train split')
    parser.add_argument('--consistancy_batch_size', type=int, default=None,
                        help='batch size for paired consistancy data; default equals --batch_size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu_id', type=int, default=0, help='which gpu to use')
    parser.add_argument('--use_ea', dest='use_ea', action='store_true', default=False,
                        help='use EA-aligned data after subject split')
    parser.add_argument('--no_ea', dest='use_ea', action='store_false',
                        help='use raw subject-split data without EA alignment')
    parser.add_argument('--use_consistancy_aug', action='store_true',
                        help='train with paired EEG_TNP consistancy augmentation')
    parser.add_argument('--consistancy_aug_paths', nargs='*', default=None,
                        help='one or more paired consistancy .pth files; comma-separated input is accepted')
    parser.add_argument('--consistancy_aug_tag', type=str, default='consistancy',
                        help='experiment tag added to logs and checkpoints')
    parser.add_argument('--consistancy_temperature', type=float, default=2.0,
                        help='temperature for KL distribution alignment')
    parser.add_argument('--consistancy_lambda_pur', type=float, default=0.2,
                        help='KL weight for aligning x_pur to x')
    parser.add_argument('--consistancy_lambda_adv', type=float, default=0.5,
                        help='KL weight for aligning x_adv_pur to x')
    parser.add_argument('--consistancy_ce_adv_pur_weight', type=float, default=1.0,
                        help='hard-label CE weight for x_adv_pur')
    parser.add_argument('--consistancy_ce_pur_weight', type=float, default=0.5,
                        help='hard-label CE weight for x_pur')
    parser.add_argument('--consistancy_warmup_epochs', type=int, default=0,
                        help='number of epochs to train CE-only before enabling KL alignment')
    return parser.parse_args()


def evaluate(model, loader):
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
            loss += F.cross_entropy(output, target).item() * data.size(0)
    return correct / total, loss / len(loader.dataset)


def clamp_tensor(x, clip_min=None, clip_max=None):
    if clip_min is None and clip_max is None:
        return x
    lower = -float('inf') if clip_min is None else clip_min
    upper = float('inf') if clip_max is None else clip_max
    return torch.clamp(x, min=lower, max=upper)


def maybe_subset_train_dataset(train_dataset, args, fold_index, logger):
    """仅用于 smoke：从训练 split 中抽一个稳定子集，full 默认不启用。"""
    if args.train_sample_num is None:
        return train_dataset
    if args.train_sample_num <= 0:
        raise ValueError('--train_sample_num must be positive when provided.')
    sample_num = min(args.train_sample_num, len(train_dataset))
    selected_indices, selection_seed = stable_subset_indices(
        len(train_dataset), sample_num, args.seed, fold_index, offset=17
    )
    logger.info(
        f'Using train_sample_num={sample_num}; selection_seed={selection_seed}; '
        f'source index preview: {selected_indices[:20]}'
    )
    return torch.utils.data.Subset(train_dataset, selected_indices)


def pgd_adversarial_examples(model, x, y, epsilon, step_size, steps, criterion,
                             random_start=True, clip_min=None, clip_max=None):
    """在线生成当前模型的 PGD 对抗样本，用于标准 Madry AT 阶段。"""
    x_adv = x.detach()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = clamp_tensor(x_adv, clip_min, clip_max)
    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = criterion(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad)
        perturbation = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = clamp_tensor(x + perturbation, clip_min, clip_max)
    return x_adv.detach()


def train_epoch_madry(model, loader, optimizer, criterion, device, args):
    """先在完整原始训练集上执行 Madry AT，保持基础鲁棒训练口径不变。"""
    model.train()
    total_loss = 0.0
    clean_ratio = max(0.0, min(1.0, args.clean_ratio))
    adv_ratio = 1.0 - clean_ratio
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        adv_data = pgd_adversarial_examples(
            model, data, target, epsilon=args.epsilon, step_size=args.pgd_step_size,
            steps=args.pgd_steps, criterion=criterion, random_start=args.pgd_random_start,
            clip_min=args.clip_min, clip_max=args.clip_max,
        )
        optimizer.zero_grad()
        adv_loss = criterion(model(adv_data), target)
        if clean_ratio > 0:
            clean_loss = criterion(model(data), target)
            loss = adv_ratio * adv_loss + clean_ratio * clean_loss
        else:
            loss = adv_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)


def _validate_consistancy_meta(meta, path, args, fold_index, protocol_tag):
    if not isinstance(meta, dict):
        raise ValueError(f'Consistancy metadata missing or invalid for {path}.')
    checks = {
        'dataset': args.dataset,
        'model': args.model,
        'fold': fold_index,
        'seed': args.seed,
    }
    for key, expected in checks.items():
        if key not in meta:
            raise ValueError(f'Consistancy metadata missing {key} for {path}.')
        if str(meta[key]) != str(expected):
            raise ValueError(
                f'Consistancy metadata mismatch for {path}: {key}={meta[key]} but expected {expected}.'
            )

    if meta.get('kind') != 'consistancy_pair':
        raise ValueError(f'Consistancy file must have kind=consistancy_pair: {path}')
    meta_protocol = meta.get('protocol', meta.get('protocol_tag'))
    if meta_protocol is None:
        raise ValueError(f'Consistancy metadata missing protocol for {path}.')
    if str(meta_protocol) != str(protocol_tag):
        raise ValueError(
            f'Consistancy metadata mismatch for {path}: protocol={meta_protocol} but expected {protocol_tag}.'
        )
    if meta.get('source_split') != 'train':
        raise ValueError(f'Consistancy file must come from train split: {path}')
    if 'eps' in meta and abs(float(meta['eps']) - float(args.epsilon)) > 1e-12:
        raise ValueError(
            f'Consistancy eps mismatch for {path}: eps={meta["eps"]} but expected {args.epsilon}.'
        )


def _as_float_tensor(payload, key, path):
    if key not in payload:
        raise ValueError(f'Consistancy payload missing {key}: {path}')
    value = payload[key]
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.detach().cpu().float()


def load_consistancy_pair_dataset(paths, expected_sample_shape, args, fold_index, protocol_tag, logger):
    """读取 paired consistancy 数据，并保留 x/x_pur/x_adv/x_adv_pur 的一一对应关系。"""
    x_parts = []
    x_pur_parts = []
    x_adv_parts = []
    x_adv_pur_parts = []
    label_parts = []
    total_pairs = 0

    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Consistancy augmentation file not found: {path}')
        payload = torch_load_cpu(path)
        if not isinstance(payload, dict):
            raise ValueError(f'Unsupported consistancy payload in {path}; expected dict payload.')

        meta = payload.get('meta', {})
        _validate_consistancy_meta(meta, path, args, fold_index, protocol_tag)
        x = _as_float_tensor(payload, 'x', path)
        x_pur = _as_float_tensor(payload, 'x_pur', path)
        x_adv = _as_float_tensor(payload, 'x_adv', path)
        x_adv_pur = _as_float_tensor(payload, 'x_adv_pur', path)
        if 'labels' not in payload:
            raise ValueError(f'Consistancy payload missing labels: {path}')
        labels = as_label_tensor(payload['labels'])
        source_indices = payload.get('source_indices')

        tensors = {'x': x, 'x_pur': x_pur, 'x_adv': x_adv, 'x_adv_pur': x_adv_pur}
        pair_count = x.size(0)
        for name, tensor in tensors.items():
            if tensor.size(0) != pair_count:
                raise ValueError(
                    f'Consistancy length mismatch for {path}: {name} has {tensor.size(0)} '
                    f'but x has {pair_count}.'
                )
            if tuple(tensor.shape[1:]) != tuple(expected_sample_shape):
                raise ValueError(
                    f'Consistancy shape mismatch for {path}: {name} sample shape '
                    f'{tuple(tensor.shape[1:])} but expected {tuple(expected_sample_shape)}.'
                )
        if labels.size(0) != pair_count:
            raise ValueError(
                f'Consistancy label length mismatch for {path}: labels has {labels.size(0)} '
                f'but x has {pair_count}.'
            )
        if source_indices is None or len(source_indices) != pair_count:
            raise ValueError(f'Consistancy source_indices missing or length mismatch for {path}.')

        x_parts.append(x)
        x_pur_parts.append(x_pur)
        x_adv_parts.append(x_adv)
        x_adv_pur_parts.append(x_adv_pur)
        label_parts.append(labels)
        total_pairs += pair_count
        logger.info(
            f'Loaded consistancy augmentation: {path}, pairs: {pair_count}, '
            f'attack: {meta.get("attack")}, eps: {meta.get("eps")}, data shape: {tuple(x.shape)}'
        )

    dataset = torch.utils.data.TensorDataset(
        torch.cat(x_parts, dim=0),
        torch.cat(x_pur_parts, dim=0),
        torch.cat(x_adv_parts, dim=0),
        torch.cat(x_adv_pur_parts, dim=0),
        torch.cat(label_parts, dim=0),
    )
    return dataset, total_pairs


def _kl_to_teacher(student_logits, teacher_probs, temperature):
    log_probs = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)


def train_epoch_consistancy_aug(model, loader, optimizer, criterion, device, args, enable_kl=True):
    """在 paired EEG_TNP 增强数据上训练 CE + KL 分布对齐损失。"""
    model.train()
    totals = {
        'loss': 0.0,
        'ce_pur': 0.0,
        'ce_adv_pur': 0.0,
        'kl_pur': 0.0,
        'kl_adv': 0.0,
    }
    total_samples = 0
    temperature = args.consistancy_temperature

    for x, x_pur, x_adv, x_adv_pur, target in loader:
        x = x.to(device)
        x_pur = x_pur.to(device)
        x_adv_pur = x_adv_pur.to(device)
        target = target.to(device)
        _ = x_adv  # x_adv 目前只作为成对数据完整性和后续排查依据，不参与第一版损失。

        optimizer.zero_grad()
        with torch.no_grad():
            teacher_probs = F.softmax(model(x).detach() / temperature, dim=1)

        logits_pur = model(x_pur)
        logits_adv_pur = model(x_adv_pur)
        ce_pur = criterion(logits_pur, target)
        ce_adv_pur = criterion(logits_adv_pur, target)

        if enable_kl:
            kl_pur = _kl_to_teacher(logits_pur, teacher_probs, temperature)
            kl_adv = _kl_to_teacher(logits_adv_pur, teacher_probs, temperature)
        else:
            kl_pur = logits_pur.new_tensor(0.0)
            kl_adv = logits_adv_pur.new_tensor(0.0)

        loss = (
            args.consistancy_ce_adv_pur_weight * ce_adv_pur
            + args.consistancy_ce_pur_weight * ce_pur
            + args.consistancy_lambda_pur * kl_pur
            + args.consistancy_lambda_adv * kl_adv
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
        optimizer.step()

        batch_size = target.size(0)
        total_samples += batch_size
        totals['loss'] += loss.item() * batch_size
        totals['ce_pur'] += ce_pur.item() * batch_size
        totals['ce_adv_pur'] += ce_adv_pur.item() * batch_size
        totals['kl_pur'] += kl_pur.item() * batch_size
        totals['kl_adv'] += kl_adv.item() * batch_size

    return {key: value / total_samples for key, value in totals.items()}


def main():
    args = parse_args()
    if args.consistancy_temperature <= 0:
        raise ValueError('--consistancy_temperature must be positive.')
    if args.consistancy_warmup_epochs < 0:
        raise ValueError('--consistancy_warmup_epochs must be non-negative.')
    args.pgd_step_size = args.epsilon / 5
    args.pgd_steps = 5 * 2
    seed_everything(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    import logging

    os.makedirs('./log_train_AT', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_tag = None
    if args.use_consistancy_aug:
        checkpoint_tag = safe_token(args.consistancy_aug_tag, default='consistancy')
    log_suffix = f'_{checkpoint_tag}' if checkpoint_tag else ''
    logfile_directory = (
        f'./log_train_AT/train_consistancy_{args.dataset}_{args.model}_{args.at_strategy}_'
        f'eps{args.epsilon}_{args.seed}_fold{args.fold}_{args.lr}_{args.weight_decay}_{args.batch_size}'
        f'{log_suffix}_{timestamp}.log'
    )
    logging.basicConfig(
        filename=logfile_directory,
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.info(f'Training consistancy {args.dataset} with {args.model}')
    logging.info(args)
    protocol_tag = get_protocol_tag(use_ea=args.use_ea)
    logging.info(f'Data protocol: {protocol_tag}, use_ea: {args.use_ea}')
    logging.info(
        f'Consistancy augmentation enabled: {args.use_consistancy_aug}, '
        f'tag: {args.consistancy_aug_tag}'
    )
    logging.info(
        f"AT config | strategy: {args.at_strategy}, eps: {args.epsilon}, "
        f"step size: {args.pgd_step_size}, steps: {args.pgd_steps}, "
        f"clean_ratio: {args.clean_ratio}, random_start: {args.pgd_random_start}, "
        f"clip_min/max: ({args.clip_min}, {args.clip_max})"
    )
    logging.info(
        f"Consistancy loss | T: {args.consistancy_temperature}, "
        f"lambda_pur: {args.consistancy_lambda_pur}, lambda_adv: {args.consistancy_lambda_adv}, "
        f"ce_adv_pur_weight: {args.consistancy_ce_adv_pur_weight}, "
        f"ce_pur_weight: {args.consistancy_ce_pur_weight}, "
        f"warmup_epochs: {args.consistancy_warmup_epochs}"
    )

    dataset_dict = {
        'seediv': load_seediv,
        'm3cv': load_m3cv,
        'bciciv2a': load_bciciv2a,
        'thubenchmark': load_thubenchmark,
    }
    dataset, info = dataset_dict[args.dataset]()
    logging.info(f'Dataset: {args.dataset}, Sample_num: {len(dataset)}, num_classes: {info["num_classes"]}')

    accs = []
    losses = []
    best_models = []
    ran_requested_fold = False
    for index, train_dataset, val_dataset, test_dataset, split_path in iter_subject_folds(
        dataset_name=args.dataset,
        dataset=dataset,
        info=info,
        seed=args.seed,
        use_ea=args.use_ea,
    ):
        if index != args.fold:
            continue
        ran_requested_fold = True
        logging.info(f'Split path: {split_path}')
        logging.info(
            f'sample num in train set: {len(train_dataset)}, sample num in val set: {len(val_dataset)}, '
            f'sample num in test set: {len(test_dataset)}'
        )
        train_dataset = maybe_subset_train_dataset(train_dataset, args, index, logging)

        pair_loader = None
        if args.use_consistancy_aug:
            consistancy_aug_paths = normalize_path_args(args.consistancy_aug_paths)
            if not consistancy_aug_paths:
                raise ValueError('--use_consistancy_aug requires at least one --consistancy_aug_paths file.')
            sample_data, _ = train_dataset[0]
            pair_dataset, pair_count = load_consistancy_pair_dataset(
                paths=consistancy_aug_paths,
                expected_sample_shape=tuple(sample_data.shape),
                args=args,
                fold_index=index,
                protocol_tag=protocol_tag,
                logger=logging,
            )
            pair_batch_size = args.consistancy_batch_size or args.batch_size
            pair_loader = torch.utils.data.DataLoader(
                pair_dataset,
                batch_size=pair_batch_size,
                shuffle=True,
                num_workers=0,
            )
            logging.info(
                f'Using consistancy paired augmentation: {pair_count} pairs, '
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
            consistancy_metrics = None
            if pair_loader is not None:
                enable_kl = epoch >= args.consistancy_warmup_epochs
                consistancy_metrics = train_epoch_consistancy_aug(
                    model, pair_loader, optimizer, criterion, device, args, enable_kl=enable_kl
                )
                train_loss = madry_loss + consistancy_metrics['loss']
            else:
                train_loss = madry_loss

            val_acc, val_loss = evaluate(model, val_loader)
            test_acc, test_loss = evaluate(model, test_loader)
            scheduler.step(val_loss)
            if consistancy_metrics is None:
                logging.info(
                    f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, '
                    f'Madry Loss: {madry_loss:.4f}, Val Acc: {val_acc:.4f}, '
                    f'Val Loss: {val_loss:.4f}, Test Acc: {test_acc:.4f}, '
                    f'Test Loss: {test_loss:.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}'
                )
            else:
                logging.info(
                    f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, '
                    f'Madry Loss: {madry_loss:.4f}, Consistancy Loss: {consistancy_metrics["loss"]:.4f}, '
                    f'Consistancy CE Pur: {consistancy_metrics["ce_pur"]:.4f}, '
                    f'Consistancy CE AdvPur: {consistancy_metrics["ce_adv_pur"]:.4f}, '
                    f'Consistancy KL Pur: {consistancy_metrics["kl_pur"]:.4f}, '
                    f'Consistancy KL AdvPur: {consistancy_metrics["kl_adv"]:.4f}, '
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
                        args.epsilon, args.seed, index, tag=checkpoint_tag,
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
        best_models.append(best_model)
        torch.save(
            best_state_dict,
            build_checkpoint_path(
                args.dataset, args.model, protocol_tag, args.at_strategy,
                args.epsilon, args.seed, index, tag=checkpoint_tag,
            ),
        )

        test_acc, test_loss = evaluate(best_model, test_loader)
        logging.info(f'Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}')
        accs.append(test_acc)
        losses.append(test_loss)
        break

    if not ran_requested_fold:
        raise ValueError(f'Requested fold {args.fold} was not produced by iter_subject_folds.')
    logging.info(f'Acc: {np.mean(accs):.4f}±{np.std(accs):.4f}, Loss: {np.mean(losses):.4f}±{np.std(losses):.4f}')


if __name__ == '__main__':
    main()
