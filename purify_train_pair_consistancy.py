import argparse
import datetime
import logging
import os

from runtime_env import configure_runtime_env

configure_runtime_env()

import numpy as np
import torch
from tqdm import tqdm

from attack.autoattack import AutoAttack
from attack.cw import CW
from attack.fgsm import FGSM
from attack.pgd import PGD
from data.load import load_bciciv2a, load_m3cv, load_seediv, load_thubenchmark
from data.subject_ea import get_protocol_tag, prepare_subject_fold
from models.model_args import get_model_args
from purify import purify
from torcheeg.models import ATCNet, Conformer, EEGNet, TSCeption
from utils.experiment_artifacts import eeg_classification_collate, safe_token, short_protocol_tag
from utils.reproducibility import seed_everything, stable_subset_indices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='thubenchmark',
                        choices=['seediv', 'm3cv', 'bciciv2a', 'thubenchmark'])
    parser.add_argument('--model', type=str, default='eegnet',
                        choices=['eegnet', 'tsception', 'atcnet', 'conformer'])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--config', type=str, required=True, help='purification config file under configs/<dataset>/')
    parser.add_argument('--sample_num', type=int, default=512, help='number of random train samples to purify')
    parser.add_argument('--attack', type=str, default='autoattack',
                        choices=['fgsm', 'pgd', 'cw', 'autoattack'])
    parser.add_argument('--eps', type=float, default=0.03)
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='explicit classifier checkpoint used to generate cached train adversarial samples')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_ea', dest='use_ea', action='store_true', default=False)
    parser.add_argument('--no_ea', dest='use_ea', action='store_false')
    parser.add_argument('--output_tag', type=str, default='consistancy')
    parser.add_argument('--output_dir', type=str, default='./purified_data/train_pair_consistancy')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    return parser.parse_args()


def build_output_path(args):
    """按 paired consistancy 实验关键信息生成稳定文件名。"""
    protocol_short = short_protocol_tag(args.use_ea)
    config_tag = safe_token(os.path.splitext(os.path.basename(args.config))[0])
    output_tag = safe_token(args.output_tag)
    eps_tag = safe_token(args.eps)
    file_name = (
        f'{args.dataset}_{args.model}_{protocol_short}_fold{args.fold}_seed{args.seed}_'
        f'train_pair_consistancy_{args.attack}_eps{eps_tag}_{config_tag}_'
        f'n{args.sample_num}_tag{output_tag}.pth'
    )
    return os.path.join(args.output_dir, file_name)


def load_classifier(args, info, device):
    """加载用于生成缓存对抗样本的分类器 checkpoint。"""
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint_path}')

    model_dict = {
        'eegnet': EEGNet,
        'tsception': TSCeption,
        'atcnet': ATCNet,
        'conformer': Conformer,
    }
    model = model_dict[args.model](**get_model_args(args.model, args.dataset, info))
    model.to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def build_attack(args, model, info, device):
    """构造与现有攻击入口一致的攻击对象，避免 paired 数据和测试攻击口径不一致。"""
    attack_dict = {
        'fgsm': FGSM,
        'pgd': PGD,
        'cw': CW,
        'autoattack': AutoAttack,
    }
    kwargs = {
        'eps': args.eps,
        'device': device,
        'n_classes': info['num_classes'],
    }
    if args.attack == 'autoattack':
        kwargs['seed'] = args.seed
    return attack_dict[args.attack](model, **kwargs)


def load_train_batch(train_dataset, indices):
    """按训练 split 局部索引取样，并复用统一分类 collate 处理标签形状。"""
    return eeg_classification_collate([train_dataset[index] for index in indices])


def main():
    args = parse_args()
    if args.sample_num <= 0:
        raise ValueError('--sample_num must be positive.')
    if args.batch_size <= 0:
        raise ValueError('--batch_size must be positive.')

    seed_everything(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('./log_purify', exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    config_tag = safe_token(os.path.splitext(os.path.basename(args.config))[0])
    log_path = (
        f'./log_purify/train_pair_consistancy_{args.dataset}_{args.model}_'
        f'{short_protocol_tag(args.use_ea)}_fold{args.fold}_seed{args.seed}_'
        f'{args.attack}_eps{safe_token(args.eps)}_{config_tag}_n{args.sample_num}_'
        f'tag{safe_token(args.output_tag)}_{timestamp}.log'
    )
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.info('Purifying paired consistancy training samples')
    logging.info(args)

    output_path = build_output_path(args)
    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(f'Output already exists: {output_path}. Use --overwrite to replace it.')

    dataset_dict = {
        'seediv': load_seediv,
        'm3cv': load_m3cv,
        'bciciv2a': load_bciciv2a,
        'thubenchmark': load_thubenchmark,
    }
    dataset, info = dataset_dict[args.dataset]()
    train_dataset, _, _, split_path = prepare_subject_fold(
        dataset_name=args.dataset,
        dataset=dataset,
        info=info,
        fold_id=args.fold,
        seed=args.seed,
        use_ea=args.use_ea,
    )
    protocol_tag = get_protocol_tag(use_ea=args.use_ea)
    logging.info(f'Data protocol: {protocol_tag}, use_ea: {args.use_ea}')
    logging.info(f'Split path: {split_path}')
    logging.info(f'Train dataset size: {len(train_dataset)}')

    selected_indices, selection_seed = stable_subset_indices(
        dataset_size=len(train_dataset),
        sample_num=args.sample_num,
        seed=args.seed,
        fold=args.fold,
    )
    logging.info(
        f'Selected {len(selected_indices)} random train samples without replacement; '
        f'selection_seed: {selection_seed}'
    )
    logging.info(f'Source index preview: {selected_indices[:20]}')

    model = load_classifier(args, info, device)
    attack = build_attack(args, model, info, device)
    logging.info(f'Model: {args.model}, fold: {args.fold}, checkpoint: {args.checkpoint_path}')
    logging.info(f'Attack: {args.attack}, eps: {args.eps}, batch_size: {args.batch_size}')

    x_samples = []
    x_pur_samples = []
    x_adv_samples = []
    x_adv_pur_samples = []
    labels = []
    source_indices = []
    clean_purify_mses = []
    adv_purify_mses = []
    attack_mses = []

    for batch_start in tqdm(range(0, len(selected_indices), args.batch_size), desc='purify train pair consistancy'):
        batch_indices = selected_indices[batch_start:batch_start + args.batch_size]
        clean_data, target = load_train_batch(train_dataset, batch_indices)
        clean_data = clean_data.to(device)
        target = target.to(device)
        model.zero_grad()
        adv_data = attack(clean_data, target).detach().cpu().float()
        model.zero_grad()
        clean_data_cpu = clean_data.detach().cpu().float()
        target_cpu = target.detach().cpu().long().view(-1)
        batch_attack_mse = torch.nn.functional.mse_loss(
            adv_data,
            clean_data_cpu,
            reduction='none',
        ).reshape(adv_data.size(0), -1).mean(dim=1)

        for offset, dataset_index in enumerate(batch_indices):
            local_index = len(x_samples)
            clean_sample = clean_data_cpu[offset]
            adv_sample = adv_data[offset]

            x_pur, clean_mse = purify(
                args,
                local_index,
                clean_sample,
                info['sampling_rate'],
                device,
                logging,
            )
            x_adv_pur, adv_mse = purify(
                args,
                local_index + args.sample_num,
                adv_sample,
                info['sampling_rate'],
                device,
                logging,
            )

            x_samples.append(clean_sample.detach().cpu().float())
            x_pur_samples.append(x_pur.detach().cpu().float())
            x_adv_samples.append(adv_sample.detach().cpu().float())
            x_adv_pur_samples.append(x_adv_pur.detach().cpu().float())
            labels.append(target_cpu[offset].view(1))
            source_indices.append(dataset_index)
            clean_purify_mses.append(float(clean_mse))
            adv_purify_mses.append(float(adv_mse))
            attack_mses.append(float(batch_attack_mse[offset].item()))
            logging.info(
                f'Purified consistancy pair {local_index}, source_index: {dataset_index}, '
                f'label: {int(target_cpu[offset].item())}'
            )

    x = torch.stack(x_samples, dim=0)
    x_pur = torch.stack(x_pur_samples, dim=0)
    x_adv = torch.stack(x_adv_samples, dim=0)
    x_adv_pur = torch.stack(x_adv_pur_samples, dim=0)
    labels = torch.cat(labels, dim=0).long()
    meta = {
        'dataset': args.dataset,
        'model': args.model,
        'fold': args.fold,
        'seed': args.seed,
        'protocol': protocol_tag,
        'protocol_short': short_protocol_tag(args.use_ea),
        'use_ea': args.use_ea,
        'config': args.config,
        'kind': 'consistancy_pair',
        'attack': args.attack,
        'eps': args.eps,
        'checkpoint_path': args.checkpoint_path,
        'sample_num': args.sample_num,
        'selection_strategy': 'random_without_replacement',
        'selection_seed': selection_seed,
        'source_split': 'train',
        'source_indices': source_indices,
        'output_tag': args.output_tag,
        'split_path': split_path,
        'data_shape': tuple(x.shape),
        'label_shape': tuple(labels.shape),
        'mean_clean_purify_mse': float(np.mean(clean_purify_mses)) if clean_purify_mses else None,
        'mean_adv_purify_mse': float(np.mean(adv_purify_mses)) if adv_purify_mses else None,
        'mean_attack_mse': float(np.mean(attack_mses)) if attack_mses else None,
    }
    payload = {
        'x': x,
        'x_pur': x_pur,
        'x_adv': x_adv,
        'x_adv_pur': x_adv_pur,
        'labels': labels,
        'source_indices': source_indices,
        'meta': meta,
    }
    torch.save(payload, output_path)
    logging.info(f'Saved paired consistancy data: {output_path}')
    logging.info(
        f'x shape: {tuple(x.shape)}, x_pur shape: {tuple(x_pur.shape)}, '
        f'x_adv shape: {tuple(x_adv.shape)}, x_adv_pur shape: {tuple(x_adv_pur.shape)}, '
        f'labels shape: {tuple(labels.shape)}'
    )
    logging.info(f'Mean clean purify mse: {meta["mean_clean_purify_mse"]}')
    logging.info(f'Mean adv purify mse: {meta["mean_adv_purify_mse"]}')
    logging.info(f'Mean attack mse: {meta["mean_attack_mse"]}')
    print(output_path)


if __name__ == '__main__':
    main()
