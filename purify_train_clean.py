import argparse
import datetime
import logging
import os
import random

from runtime_env import configure_runtime_env

configure_runtime_env()

import numpy as np
import torch

from data.load import load_bciciv2a, load_m3cv, load_seediv, load_thubenchmark
from data.subject_ea import get_protocol_tag, prepare_subject_fold
from purify import purify
from utils.experiment_artifacts import as_label_tensor, safe_token, short_protocol_tag


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
    parser.add_argument('--model', type=str, default='eegnet',
                        choices=['eegnet', 'tsception', 'atcnet', 'conformer'])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--config', type=str, required=True, help='purification config file under configs/<dataset>/')
    parser.add_argument('--sample_num', type=int, default=512, help='number of random train samples to purify')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_ea', dest='use_ea', action='store_true', default=False)
    parser.add_argument('--no_ea', dest='use_ea', action='store_false')
    parser.add_argument('--output_tag', type=str, default='default')
    parser.add_argument('--output_dir', type=str, default='./purified_data/train_clean')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    return parser.parse_args()


def build_output_path(args):
    protocol_short = short_protocol_tag(args.use_ea)
    config_tag = safe_token(os.path.splitext(os.path.basename(args.config))[0])
    output_tag = safe_token(args.output_tag)
    file_name = (
        f'{args.dataset}_{args.model}_{protocol_short}_fold{args.fold}_seed{args.seed}_'
        f'train_clean_{config_tag}_n{args.sample_num}_tag{output_tag}.pth'
    )
    return os.path.join(args.output_dir, file_name)


def select_random_indices(dataset_size, sample_num, seed, fold):
    """从训练 split 中无放回随机抽样，避免连续取样带来的 subject/session 偏置。"""
    if sample_num > dataset_size:
        raise ValueError(
            f'Requested {sample_num} random samples but train split has only {dataset_size} samples.'
        )
    selection_seed = seed + fold * 1000
    rng = np.random.RandomState(selection_seed)
    return rng.choice(dataset_size, size=sample_num, replace=False).tolist(), selection_seed


def main():
    args = parse_args()
    if args.sample_num <= 0:
        raise ValueError('--sample_num must be positive.')

    seed_everything(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('./log_purify', exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    config_tag = safe_token(os.path.splitext(os.path.basename(args.config))[0])
    log_path = (
        f'./log_purify/train_clean_{args.dataset}_{args.model}_{short_protocol_tag(args.use_ea)}_'
        f'fold{args.fold}_seed{args.seed}_{config_tag}_n{args.sample_num}_'
        f'tag{safe_token(args.output_tag)}_{timestamp}.log'
    )
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.info('Purifying clean training samples')
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

    selected_indices, selection_seed = select_random_indices(
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

    purified_data = []
    labels = []
    source_indices = []
    mses = []
    for local_index, dataset_index in enumerate(selected_indices):
        data, label = train_dataset[dataset_index]
        purified_sample, mse = purify(args, local_index, data, info['sampling_rate'], device, logging)
        purified_data.append(purified_sample.detach().cpu().float())
        labels.append(as_label_tensor(label))
        source_indices.append(dataset_index)
        mses.append(mse)

    purified_data = torch.stack(purified_data, dim=0)
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
        'kind': 'clean',
        'sample_num': args.sample_num,
        'selection_strategy': 'random_without_replacement',
        'selection_seed': selection_seed,
        'source_split': 'train',
        'source_indices': source_indices,
        'output_tag': args.output_tag,
        'split_path': split_path,
        'data_shape': tuple(purified_data.shape),
        'label_shape': tuple(labels.shape),
        'mean_mse': float(np.mean(mses)) if mses else None,
    }
    torch.save((purified_data, labels, meta), output_path)
    logging.info(f'Saved purified train-clean data: {output_path}')
    logging.info(f'Purified data shape: {tuple(purified_data.shape)}, labels shape: {tuple(labels.shape)}')
    logging.info(f'Mean mse: {meta["mean_mse"]}')
    print(output_path)


if __name__ == '__main__':
    main()
