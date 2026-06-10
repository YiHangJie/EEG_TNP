import argparse
import logging
import os
import random

from runtime_env import configure_runtime_env

configure_runtime_env()

import numpy as np
import torch
from torch import nn

from attack.apgd import APGD
from attack.apgdt import APGDT
from attack.autoattack import AutoAttack  # noqa: F401  # 保留导入语义，实际 autoattack 使用下方 subject-aware 版本。
from attack.cw import CW
from attack.fab import FAB
from attack.fgsm import FGSM
from attack.pgd import PGD
from attack.square import Square
from data.load import load_bciciv2a, load_m3cv, load_seediv, load_thubenchmark
from data.subject_ea import RAW_PROTOCOL_TAG, prepare_subject_ea_forward_fold
from models.eegnet_ea_forward import SubjectEAConformer, SubjectEAEEGNet
from models.model_args import get_model_args
from utils.experiment_artifacts import (
    build_checkpoint_path,
    eeg_subject_classification_collate,
    safe_token,
    short_protocol_tag,
)


def seed_everything(seed=42):
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
    parser.add_argument('--attack', type=str, default='autoattack',
                        choices=['fgsm', 'pgd', 'cw', 'autoattack'])
    parser.add_argument('--eps', type=float, default=0.03)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--attack_sample_num', type=int, default=None,
                        help='optional random subset size for attack smoke tests; default attacks the full test split')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_ea', dest='use_ea', action='store_true', default=False,
                        help='not supported here; EA is applied inside the model')
    parser.add_argument('--no_ea', dest='use_ea', action='store_false',
                        help='use raw data and apply EA inside model forward')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--checkpoint_tag', type=str, default=None)
    parser.add_argument('--save_adv', action='store_true')
    parser.add_argument('--adv_output_tag', type=str, default=None)
    return parser.parse_args()


def select_random_indices(dataset_size, sample_num, seed, fold):
    """从测试 split 中无放回随机抽样，用于和现有 attack.py 的 512 样本日志对齐。"""
    if sample_num > dataset_size:
        raise ValueError(
            f'Requested {sample_num} random samples but dataset has only {dataset_size} samples.'
        )
    selection_seed = seed + fold * 1000
    rng = np.random.RandomState(selection_seed)
    return rng.choice(dataset_size, size=sample_num, replace=False).tolist(), selection_seed


def build_default_checkpoint_path(args):
    return build_checkpoint_path(
        args.dataset,
        args.model,
        RAW_PROTOCOL_TAG,
        args.at_strategy,
        args.eps,
        args.seed,
        args.fold,
        tag=args.checkpoint_tag,
    )


def resolve_checkpoint_path(args):
    if args.checkpoint_path:
        return args.checkpoint_path
    return build_default_checkpoint_path(args)


def infer_adv_model_tag(args):
    if args.adv_output_tag:
        return safe_token(args.adv_output_tag)
    if args.checkpoint_tag:
        return safe_token(args.checkpoint_tag)
    return 'ea_forward'


def build_adv_output_path(args):
    model_tag = infer_adv_model_tag(args)
    protocol_short = short_protocol_tag(False)
    file_name = (
        f'{args.dataset}_{args.model}_{protocol_short}_{model_tag}_{args.at_strategy}_'
        f'{args.attack}_eps{args.eps}_seed{args.seed}_fold{args.fold}.pth'
    )
    return os.path.join('./ad_data', file_name), model_tag


class SubjectBatchModelWrapper(nn.Module):
    """把 `(x, subject_ids)` 模型包装成只接收 `x` 的接口，供现有攻击类调用。"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.subject_ids = None

    def set_subject_ids(self, subject_ids):
        self.subject_ids = subject_ids.detach().long()

    def forward(self, x):
        if self.subject_ids is None:
            raise RuntimeError('Subject ids must be set before calling the wrapped model.')
        subject_ids = self.subject_ids
        if subject_ids.numel() != x.size(0):
            # AutoAttack 内部会继续筛选子集；按 subject 分组后这些子集都属于同一被试。
            unique_subjects = torch.unique(subject_ids)
            if unique_subjects.numel() != 1:
                raise ValueError(
                    f'Wrapped subject id count {subject_ids.numel()} does not match input batch {x.size(0)}.'
                )
            subject_ids = unique_subjects[0].repeat(x.size(0))
        return self.model(x, subject_ids.to(x.device))


class SubjectAwareAutoAttack:
    """AutoAttack 的 subject-aware 轻量版本，保证每个失败子集使用对应 subject_ids。"""
    def __init__(self, model_wrapper, device='cuda', norm='Linf', eps=8 / 255,
                 seed=42, n_classes=10, verbose=False):
        self.model_wrapper = model_wrapper
        self.device = device
        self.verbose = verbose
        self.attacks = [
            APGD(
                model_wrapper,
                eps=eps,
                norm=norm,
                seed=seed,
                verbose=verbose,
                loss='ce',
                n_restarts=1,
            ),
            APGDT(
                model_wrapper,
                eps=eps,
                norm=norm,
                seed=seed,
                verbose=verbose,
                n_classes=n_classes,
                n_restarts=1,
            ),
            FAB(
                model_wrapper,
                eps=eps,
                norm=norm,
                seed=seed,
                verbose=verbose,
                multi_targeted=True,
                n_classes=n_classes,
                n_restarts=1,
            ),
            Square(
                model_wrapper,
                eps=eps,
                norm=norm,
                seed=seed,
                verbose=verbose,
                n_queries=5000,
                n_restarts=1,
            ),
        ]

    def _attack_one_subject_group(self, images, labels, subject_ids):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        subject_ids = subject_ids.clone().detach().long().to(self.device)
        batch_size = images.size(0)
        fails = torch.arange(batch_size, device=self.device)
        final_images = images.clone().detach()

        for attack in self.attacks:
            if len(fails) == 0:
                break
            fail_subject_ids = subject_ids[fails]
            self.model_wrapper.set_subject_ids(fail_subject_ids)
            adv_images = attack(images[fails], labels[fails])
            self.model_wrapper.set_subject_ids(fail_subject_ids)
            outputs = self.model_wrapper(adv_images)
            preds = outputs.argmax(dim=1)
            corrects = preds == labels[fails]
            wrongs = ~corrects
            succeeds = torch.masked_select(fails, wrongs)
            succeeds_of_fails = torch.masked_select(torch.arange(fails.size(0), device=self.device), wrongs)
            final_images[succeeds] = adv_images[succeeds_of_fails]
            fails = torch.masked_select(fails, corrects)

        return final_images

    def __call__(self, images, labels, subject_ids):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        subject_ids = subject_ids.clone().detach().long().to(self.device)
        final_images = images.clone().detach()

        # 按 subject 分组后调用标准 AutoAttack 子攻击，避免内部子集筛选破坏 subject 对齐。
        for subject_id in torch.unique(subject_ids):
            group_indices = (subject_ids == subject_id).nonzero(as_tuple=False).view(-1)
            group_adv = self._attack_one_subject_group(
                images[group_indices],
                labels[group_indices],
                subject_ids[group_indices],
            )
            final_images[group_indices] = group_adv

        return final_images


def create_model(args, info, ea_matrices, device):
    model_map = {
        'eegnet_ea_forward': ('eegnet', SubjectEAEEGNet),
        'conformer_ea_forward': ('conformer', SubjectEAConformer),
    }
    base_model, model_cls = model_map[args.model]
    model_args = get_model_args(base_model, args.dataset, info)
    model = model_cls(ea_matrices=ea_matrices, **model_args)
    return model.to(device)


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


def build_attack(args, wrapper, device, num_classes):
    if args.attack == 'autoattack':
        return SubjectAwareAutoAttack(
            wrapper,
            eps=args.eps,
            device=device,
            seed=args.seed,
            n_classes=num_classes,
        )
    attack_dict = {
        'fgsm': FGSM,
        'pgd': PGD,
        'cw': CW,
    }
    return attack_dict[args.attack](wrapper, eps=args.eps, device=device, n_classes=num_classes)


def run_attack_batch(args, attack, wrapper, data, target, subject_ids):
    if args.attack == 'autoattack':
        return attack(data, target, subject_ids)
    wrapper.set_subject_ids(subject_ids)
    return attack(data, target)


def subject_to_meta(subject_to_index):
    return {str(subject_id): int(index) for subject_id, index in subject_to_index.items()}


def main():
    args = parse_args()
    if args.use_ea:
        raise ValueError('attack_ea_forward.py expects --no_ea/raw input; EA is applied inside forward.')

    seed_everything(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    os.makedirs('./log_attack', exist_ok=True)

    checkpoint_log_tag = os.path.splitext(os.path.basename(args.checkpoint_path))[0] if args.checkpoint_path else ''
    log_tag = safe_token(args.adv_output_tag or args.checkpoint_tag or checkpoint_log_tag or 'ea_forward')
    log_path = (
        f'./log_attack/attack_ea_forward_{args.dataset}_{args.model}_{args.at_strategy}_'
        f'{args.attack}_{args.eps}_{args.seed}_fold{args.fold}_{log_tag}.log'
    )
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.info(f'Attacking EA-forward {args.attack} on {args.dataset} with {args.model}')
    logging.info(args)

    dataset_dict = {
        'seediv': load_seediv,
        'm3cv': load_m3cv,
        'bciciv2a': load_bciciv2a,
        'thubenchmark': load_thubenchmark,
    }
    dataset, info = dataset_dict[args.dataset]()
    _, _, test_dataset, split_path, ea_matrices, subject_to_index = prepare_subject_ea_forward_fold(
        dataset_name=args.dataset,
        dataset=dataset,
        info=info,
        fold_id=args.fold,
        seed=args.seed,
    )
    logging.info(f'Dataset: {args.dataset}')
    logging.info(f'Split path: {split_path}')
    if args.attack_sample_num is not None:
        if args.attack_sample_num <= 0:
            raise ValueError('--attack_sample_num must be positive when provided.')
        attack_sample_num = min(args.attack_sample_num, len(test_dataset))
        selected_indices, selection_seed = select_random_indices(
            dataset_size=len(test_dataset),
            sample_num=attack_sample_num,
            seed=args.seed,
            fold=args.fold,
        )
        test_dataset = torch.utils.data.Subset(test_dataset, selected_indices)
        logging.info(
            f'Using attack_sample_num={attack_sample_num}; selection_seed={selection_seed}; '
            f'source index preview: {selected_indices[:20]}'
        )
    logging.info(f'EA matrices shape: {tuple(ea_matrices.shape)}')
    logging.info(f'Subject to index: {subject_to_index}')

    model = create_model(args, info, ea_matrices, device)
    model.eval()
    checkpoint_path = resolve_checkpoint_path(args)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint)
    logging.info(f'Model: {args.model}, fold: {args.fold}, checkpoint: {checkpoint_path}')

    wrapper = SubjectBatchModelWrapper(model).to(device)
    attack = build_attack(args, wrapper, device, info['num_classes'])

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=eeg_subject_classification_collate,
    )
    logging.info(f'test dataset size: {len(test_dataset)}')

    ad_data = []
    clean_data = []
    labels = []
    subject_indices = []
    for batch_index, (data, target, subject_ids) in enumerate(test_loader):
        data = data.to(device)
        target = target.to(device)
        subject_ids = subject_ids.to(device)
        adv_data = run_attack_batch(args, attack, wrapper, data, target, subject_ids)
        ad_data.append(adv_data.detach().cpu())
        clean_data.append(data.detach().cpu())
        labels.append(target.detach().cpu())
        subject_indices.append(subject_ids.detach().cpu())
        logging.info(
            f'Attacking {args.attack} on {args.dataset} with {args.model}, '
            f'fold: {args.fold}, batch: {batch_index + 1}/{len(test_loader)}'
        )

    ad_data = torch.cat(ad_data, dim=0).cpu()
    clean_data = torch.cat(clean_data, dim=0).cpu()
    labels = torch.cat(labels, dim=0).cpu()
    subject_indices = torch.cat(subject_indices, dim=0).cpu()

    evaluate_acc, evaluate_loss = evaluate(model, test_loader, device)
    logging.info(f'Before Attack - Test Accuracy: {evaluate_acc * 100:.2f}%, Test Loss: {evaluate_loss:.4f}')

    ad_dataset = torch.utils.data.TensorDataset(ad_data, labels, subject_indices)
    ad_loader = torch.utils.data.DataLoader(
        ad_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=eeg_subject_classification_collate,
    )
    ad_evaluate_acc, ad_evaluate_loss = evaluate(model, ad_loader, device)
    logging.info(f'After Attack - Test Accuracy: {ad_evaluate_acc * 100:.2f}%, Test Loss: {ad_evaluate_loss:.4f}')

    mse = torch.nn.functional.mse_loss(ad_data, clean_data)
    logging.info(f'MSE between clean data and adversarial data: {mse.item():.6f}')

    if args.save_adv:
        os.makedirs('./ad_data', exist_ok=True)
        adv_output_path, model_tag = build_adv_output_path(args)
        adv_meta = {
            'dataset': args.dataset,
            'model': args.model,
            'fold': args.fold,
            'seed': args.seed,
            'protocol': RAW_PROTOCOL_TAG,
            'protocol_short': short_protocol_tag(False),
            'use_ea': False,
            'model_tag': model_tag,
            'at_strategy': args.at_strategy,
            'attack': args.attack,
            'eps': args.eps,
            'checkpoint_path': checkpoint_path,
            'adv_output_tag': args.adv_output_tag,
            'attack_sample_num': args.attack_sample_num,
            'clean_accuracy': evaluate_acc,
            'adv_accuracy': ad_evaluate_acc,
            'mse': mse.item(),
            'subject_indices': subject_indices,
            'subject_to_index': subject_to_meta(subject_to_index),
        }
        torch.save((ad_data, labels, adv_meta), adv_output_path)
        logging.info(f'Saved adversarial data: {adv_output_path}')

    logging.info('Test model on 512 random samples for adversarial training')
    available_sample_num = min(clean_data.size(0), ad_data.size(0), labels.size(0), subject_indices.size(0))
    eval_sample_num = min(512, available_sample_num)
    selected_indices, selection_seed = select_random_indices(
        dataset_size=available_sample_num,
        sample_num=eval_sample_num,
        seed=args.seed,
        fold=args.fold,
    )
    selected_index_tensor = torch.as_tensor(selected_indices, dtype=torch.long)
    logging.info(
        f'Selected {len(selected_indices)} random test samples without replacement; '
        f'selection_seed: {selection_seed}'
    )
    logging.info(f'Source index preview: {selected_indices[:20]}')

    sampled_clean_dataset = torch.utils.data.TensorDataset(
        clean_data[selected_index_tensor],
        labels[selected_index_tensor],
        subject_indices[selected_index_tensor],
    )
    sampled_clean_loader = torch.utils.data.DataLoader(
        sampled_clean_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=eeg_subject_classification_collate,
    )
    sampled_ad_dataset = torch.utils.data.TensorDataset(
        ad_data[selected_index_tensor],
        labels[selected_index_tensor],
        subject_indices[selected_index_tensor],
    )
    sampled_ad_loader = torch.utils.data.DataLoader(
        sampled_ad_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=eeg_subject_classification_collate,
    )
    sampled_ad_acc, sampled_ad_loss = evaluate(model, sampled_ad_loader, device)
    sampled_clean_acc, sampled_clean_loss = evaluate(model, sampled_clean_loader, device)
    logging.info(
        f'After Attack - Test Accuracy on {eval_sample_num} random samples: '
        f'{sampled_ad_acc * 100:.2f}%, Test Loss: {sampled_ad_loss:.4f}'
    )
    logging.info(
        f'Before Attack - Test Accuracy on {eval_sample_num} random samples: '
        f'{sampled_clean_acc * 100:.2f}%, Test Loss: {sampled_clean_loss:.4f}'
    )


if __name__ == '__main__':
    main()
