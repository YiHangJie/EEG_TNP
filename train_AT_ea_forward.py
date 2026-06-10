import argparse
import copy
import datetime
import os
import random

from runtime_env import configure_runtime_env

configure_runtime_env()

import numpy as np
import torch

from data.load import load_bciciv2a, load_m3cv, load_seediv, load_thubenchmark
from data.subject_ea import RAW_PROTOCOL_TAG, prepare_subject_ea_forward_fold
from models.eegnet_ea_forward import SubjectEAConformer, SubjectEAEEGNet
from models.model_args import get_model_args
from utils.experiment_artifacts import build_checkpoint_path, eeg_subject_classification_collate, safe_token


def seed_everything(seed=42):
    """固定随机源，保证 EA-in-forward AT 实验可复现。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='thubenchmark',
                        choices=['seediv', 'm3cv', 'bciciv2a', 'thubenchmark'])
    parser.add_argument('--model', type=str, default='eegnet_ea_forward',
                        choices=['eegnet_ea_forward', 'conformer_ea_forward'])
    parser.add_argument('--at_strategy', type=str, default='madry', choices=['madry'])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epsilon', type=float, default=0.03)
    parser.add_argument('--pgd_step_size', type=float, default=None)
    parser.add_argument('--pgd_steps', type=int, default=10)
    parser.add_argument('--clean_ratio', type=float, default=0.0)
    parser.add_argument('--clip_min', type=float, default=None)
    parser.add_argument('--clip_max', type=float, default=None)
    parser.add_argument('--pgd_random_start', action='store_true', default=True)
    parser.add_argument('--no_pgd_random_start', dest='pgd_random_start', action='store_false')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_sample_num', type=int, default=None,
                        help='optional random subset size for training smoke tests; default uses full train split')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_ea', dest='use_ea', action='store_true', default=False,
                        help='not supported here; EA is applied inside the model')
    parser.add_argument('--no_ea', dest='use_ea', action='store_false',
                        help='use raw data and apply EA inside the model forward')
    parser.add_argument('--checkpoint_tag', type=str, default=None)
    return parser.parse_args()


def clamp_tensor(x, clip_min=None, clip_max=None):
    if clip_min is None and clip_max is None:
        return x
    lower = -float('inf') if clip_min is None else clip_min
    upper = float('inf') if clip_max is None else clip_max
    return torch.clamp(x, min=lower, max=upper)


def pgd_adversarial_examples(model, x, y, subject_ids, epsilon, step_size, steps,
                             criterion, random_start=True, clip_min=None, clip_max=None):
    """在 raw/no_ea 输入空间生成 PGD，对模型内 EA 的梯度一并反传。"""
    x_adv = x.detach()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = clamp_tensor(x_adv, clip_min, clip_max)
    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = criterion(model(x_adv, subject_ids), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad)
        perturbation = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = clamp_tensor(x + perturbation, clip_min, clip_max)
    return x_adv.detach()


def train_epoch_madry(model, loader, optimizer, criterion, device, args):
    model.train()
    total_loss = 0.0
    clean_ratio = max(0.0, min(1.0, args.clean_ratio))
    adv_ratio = 1.0 - clean_ratio
    for data, target, subject_ids in loader:
        data = data.to(device)
        target = target.to(device)
        subject_ids = subject_ids.to(device)
        adv_data = pgd_adversarial_examples(
            model, data, target, subject_ids,
            epsilon=args.epsilon,
            step_size=args.pgd_step_size,
            steps=args.pgd_steps,
            criterion=criterion,
            random_start=args.pgd_random_start,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
        )
        optimizer.zero_grad()
        adv_loss = criterion(model(adv_data, subject_ids), target)
        if clean_ratio > 0:
            clean_loss = criterion(model(data, subject_ids), target)
            loss = adv_ratio * adv_loss + clean_ratio * clean_loss
        else:
            loss = adv_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data, target, subject_ids in loader:
            data = data.to(device)
            target = target.to(device)
            subject_ids = subject_ids.to(device)
            output = model(data, subject_ids)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
            loss += torch.nn.functional.cross_entropy(output, target).item() * data.size(0)
    return correct / total, loss / len(loader.dataset)


def create_model(args, info, ea_matrices, device):
    model_map = {
        'eegnet_ea_forward': ('eegnet', SubjectEAEEGNet),
        'conformer_ea_forward': ('conformer', SubjectEAConformer),
    }
    base_model, model_cls = model_map[args.model]
    model_args = get_model_args(base_model, args.dataset, info)
    model = model_cls(ea_matrices=ea_matrices, **model_args)
    return model.to(device)


def maybe_subset_train_dataset(train_dataset, args, logger):
    """仅用于 smoke：从训练 split 中抽一个稳定子集，full 默认不启用。"""
    if args.train_sample_num is None:
        return train_dataset
    if args.train_sample_num <= 0:
        raise ValueError('--train_sample_num must be positive when provided.')
    sample_num = min(args.train_sample_num, len(train_dataset))
    selection_seed = args.seed + args.fold * 1000 + 17
    rng = np.random.RandomState(selection_seed)
    selected_indices = rng.choice(len(train_dataset), size=sample_num, replace=False).tolist()
    logger.info(
        f'Using train_sample_num={sample_num}; selection_seed={selection_seed}; '
        f'source index preview: {selected_indices[:20]}'
    )
    return torch.utils.data.Subset(train_dataset, selected_indices)


def checkpoint_path(args, fold):
    return build_checkpoint_path(
        args.dataset,
        args.model,
        RAW_PROTOCOL_TAG,
        args.at_strategy,
        args.epsilon,
        args.seed,
        fold,
        tag=args.checkpoint_tag,
    )


def main():
    args = parse_args()
    if args.use_ea:
        raise ValueError('train_AT_ea_forward.py expects --no_ea/raw input; EA is applied inside forward.')
    if args.pgd_step_size is None:
        args.pgd_step_size = args.epsilon / 5

    seed_everything(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    os.makedirs('./log_train_AT', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)

    import logging

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    tag_suffix = f'_{safe_token(args.checkpoint_tag)}' if args.checkpoint_tag else ''
    log_path = (
        f'./log_train_AT/train_ea_forward_{args.dataset}_{args.model}_{args.at_strategy}_'
        f'eps{args.epsilon}_{args.seed}_{args.lr}_{args.weight_decay}_{args.batch_size}'
        f'{tag_suffix}_{timestamp}.log'
    )
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.info(f'Training EA-forward {args.dataset} with {args.model}')
    logging.info(args)
    logging.info(
        f'AT config | strategy: {args.at_strategy}, eps: {args.epsilon}, '
        f'step size: {args.pgd_step_size}, steps: {args.pgd_steps}, '
        f'clean_ratio: {args.clean_ratio}, random_start: {args.pgd_random_start}, '
        f'clip_min/max: ({args.clip_min}, {args.clip_max})'
    )

    dataset_dict = {
        'seediv': load_seediv,
        'm3cv': load_m3cv,
        'bciciv2a': load_bciciv2a,
        'thubenchmark': load_thubenchmark,
    }
    dataset, info = dataset_dict[args.dataset]()
    train_dataset, val_dataset, test_dataset, split_path, ea_matrices, subject_to_index = (
        prepare_subject_ea_forward_fold(
            dataset_name=args.dataset,
            dataset=dataset,
            info=info,
            fold_id=args.fold,
            seed=args.seed,
        )
    )
    logging.info(f'Dataset: {args.dataset}, Sample_num: {len(dataset)}, num_classes: {info["num_classes"]}')
    logging.info(f'Split path: {split_path}')
    logging.info(
        f'sample num in train set: {len(train_dataset)}, sample num in val set: {len(val_dataset)}, '
        f'sample num in test set: {len(test_dataset)}'
    )
    train_dataset = maybe_subset_train_dataset(train_dataset, args, logging)
    logging.info(f'EA matrices shape: {tuple(ea_matrices.shape)}')
    logging.info(f'Subject to index: {subject_to_index}')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=eeg_subject_classification_collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=eeg_subject_classification_collate,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=eeg_subject_classification_collate,
    )

    model = create_model(args, info, ea_matrices, device)
    logging.info(f'Model: {args.model}, Parameter Num: {sum(p.numel() for p in model.parameters())}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience // 2
    )
    criterion = torch.nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    best_state_dict = copy.deepcopy(model.state_dict())
    no_improve_epochs = 0
    min_delta = 0.0

    for epoch in range(args.epochs):
        train_loss = train_epoch_madry(model, train_loader, optimizer, criterion, device, args)
        val_acc, val_loss = evaluate(model, val_loader, device)
        test_acc, test_loss = evaluate(model, test_loader, device)
        scheduler.step(val_loss)
        logging.info(
            f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, '
            f'Val Loss: {val_loss:.4f}, Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}, '
            f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}'
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improve_epochs = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(
                best_state_dict,
                build_checkpoint_path(
                    args.dataset, args.model, RAW_PROTOCOL_TAG, args.at_strategy,
                    args.epsilon, args.seed, args.fold, tag=args.checkpoint_tag,
                    lr=args.lr, weight_decay=args.weight_decay,
                ),
            )
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.patience:
                logging.info(
                    f'Early stopping at epoch {epoch + 1} '
                    f'(no improvement in {args.patience} epochs). Best Val Loss: {best_val_loss:.4f}'
                )
                break

    model.load_state_dict(best_state_dict)
    torch.save(best_state_dict, checkpoint_path(args, args.fold))
    test_acc, test_loss = evaluate(model, test_loader, device)
    logging.info(f'Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}')
    logging.info(f'Checkpoint: {checkpoint_path(args, args.fold)}')


if __name__ == '__main__':
    main()
