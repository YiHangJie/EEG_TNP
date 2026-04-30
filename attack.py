import argparse
import logging
import os
from runtime_env import configure_runtime_env

configure_runtime_env()

import torch
import random
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager

from torcheeg.models import EEGNet, TSCeption, ATCNet, Conformer

from models.model_args import get_model_args
from data.load import load_seediv, load_m3cv, load_bciciv2a, load_thubenchmark
from data.subject_ea import get_protocol_tag, prepare_subject_fold
from attack.fgsm import FGSM
from attack.pgd import PGD
from attack.cw import CW
from attack.autoattack import AutoAttack
from utils.experiment_artifacts import (
    build_checkpoint_path,
    eeg_classification_collate,
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
    parser.add_argument('--dataset', type=str, default='seediv', choices=['seediv','m3cv', 'bciciv2a', 'thubenchmark'], help='choose dataset')
    parser.add_argument('--model', type=str, default='eegnet', choices=['eegnet', 'tsception', 'atcnet', 'conformer'], help='choose model')
    parser.add_argument('--at_strategy', type=str, default='clean', choices=['madry', 'fbf', 'trades', 'clean'], help='adversarial training strategy')
    parser.add_argument('--fold', type=int, default=0, help='which fold to use')
    parser.add_argument('--attack', type=str, default='fgsm', choices=['fgsm', 'pgd', 'cw', 'autoattack'], help='choose attack')
    parser.add_argument('--eps', type=float, default=0.1, help='attack budget, default is 8/255')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu_id', type=int, default=0, help='which gpu to use')
    parser.add_argument('--use_ea', dest='use_ea', action='store_true', default=False,
                        help='use EA-aligned data after subject split')
    parser.add_argument('--no_ea', dest='use_ea', action='store_false',
                        help='use raw subject-split data without EA alignment')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='explicit checkpoint path; takes priority over default checkpoint naming')
    parser.add_argument('--checkpoint_tag', type=str, default=None,
                        help='optional suffix inserted before _best.pth in the default checkpoint name')
    parser.add_argument('--save_adv', action='store_true',
                        help='save adversarial data for any attacked model, not only clean models')
    parser.add_argument('--adv_output_tag', type=str, default=None,
                        help='model/source tag used in saved adversarial-data file names')
    args = parser.parse_args()
    return args

def build_default_attack_checkpoint_path(args, protocol_tag):
    eps = args.eps if args.at_strategy != 'clean' else 0
    return build_checkpoint_path(
        args.dataset, args.model, protocol_tag, args.at_strategy,
        eps, args.seed, args.fold, tag=args.checkpoint_tag,
    )

def resolve_checkpoint_path(args, protocol_tag):
    if args.checkpoint_path:
        return args.checkpoint_path
    return build_default_attack_checkpoint_path(args, protocol_tag)

def select_random_indices(dataset_size, sample_num, seed, fold):
    """从测试 split 中无放回随机抽样，避免 512 样本评估集中在连续被试上。"""
    if sample_num > dataset_size:
        raise ValueError(
            f'Requested {sample_num} random samples but dataset has only {dataset_size} samples.'
        )
    selection_seed = seed + fold * 1000
    rng = np.random.RandomState(selection_seed)
    return rng.choice(dataset_size, size=sample_num, replace=False).tolist(), selection_seed

def infer_adv_model_tag(args):
    if args.adv_output_tag:
        return safe_token(args.adv_output_tag)
    if args.checkpoint_tag:
        return safe_token(args.checkpoint_tag)
    if args.checkpoint_path:
        return safe_token(os.path.splitext(os.path.basename(args.checkpoint_path))[0])
    return safe_token(args.at_strategy)

def build_adv_output_path(args):
    model_tag = infer_adv_model_tag(args)
    protocol_short = short_protocol_tag(args.use_ea)
    if model_tag == safe_token(args.at_strategy):
        source_tag = args.at_strategy
    else:
        source_tag = f'{model_tag}_{args.at_strategy}'
    file_name = (
        f'{args.dataset}_{args.model}_{protocol_short}_{source_tag}_{args.attack}_'
        f'eps{args.eps}_seed{args.seed}_fold{args.fold}.pth'
    )
    return os.path.join('./ad_data', file_name), model_tag

def _format_cuda_bytes(num_bytes):
    return f"{num_bytes / (1024 ** 2):.2f}MB"

def _cuda_memory_reserved(device):
    if hasattr(torch.cuda, "memory_reserved"):
        return torch.cuda.memory_reserved(device)
    return torch.cuda.memory_cached(device)

def _cuda_max_memory_reserved(device):
    if hasattr(torch.cuda, "max_memory_reserved"):
        return torch.cuda.max_memory_reserved(device)
    return torch.cuda.max_memory_cached(device)

def log_cuda_memory(tag, device, logger=None):
    if not torch.cuda.is_available():
        return
    if logger is None:
        logger = logging
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = _cuda_memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = _cuda_max_memory_reserved(device)
    logger.info(
        f"[CUDA] {tag} | alloc={_format_cuda_bytes(allocated)} "
        f"reserved={_format_cuda_bytes(reserved)} "
        f"peak_alloc={_format_cuda_bytes(peak_allocated)} "
        f"peak_reserved={_format_cuda_bytes(peak_reserved)}"
    )

@contextmanager
def cuda_mem_tracker(tag, device, logger=None):
    if not torch.cuda.is_available():
        yield
        return
    if logger is None:
        logger = logging
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start_alloc = torch.cuda.memory_allocated(device)
    start_reserved = _cuda_memory_reserved(device)
    yield
    torch.cuda.synchronize(device)
    end_alloc = torch.cuda.memory_allocated(device)
    end_reserved = _cuda_memory_reserved(device)
    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = _cuda_max_memory_reserved(device)
    logger.info(
        f"[CUDA] {tag} | start_alloc={_format_cuda_bytes(start_alloc)} "
        f"start_reserved={_format_cuda_bytes(start_reserved)} "
        f"end_alloc={_format_cuda_bytes(end_alloc)} "
        f"end_reserved={_format_cuda_bytes(end_reserved)} "
        f"peak_alloc={_format_cuda_bytes(peak_alloc)} "
        f"peak_reserved={_format_cuda_bytes(peak_reserved)}"
    )

def evaluate(model, loader):
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
            loss += torch.nn.functional.cross_entropy(output, target).item() * data.size(0)
        accuracy = correct / total
        loss /= len(loader.dataset)
    return accuracy, loss

if __name__ == '__main__':
    args = parse_args()
    if args.dataset == 'm3cv':
        args.batch_size = 16
    seed_everything(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # set log file
    checkpoint_log_tag = os.path.splitext(os.path.basename(args.checkpoint_path))[0] if args.checkpoint_path else ''
    log_tag = safe_token(args.adv_output_tag or args.checkpoint_tag or checkpoint_log_tag or '')
    log_suffix = f'_{log_tag}' if log_tag != 'none' else ''
    logfile_directory = f'./log_attack/attack_{args.dataset}_{args.model}_{args.at_strategy}_{args.attack}_{args.eps}_{args.seed}{log_suffix}.log'
    logging.basicConfig(filename=logfile_directory, level=logging.INFO, filemode='w', format='%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')  # 时间格式)
    logging.info(f'Attacking {args.attack} on {args.dataset} with {args.model}')
    logging.info(args)
    protocol_tag = get_protocol_tag(use_ea=args.use_ea)
    logging.info(f'Data protocol: {protocol_tag}, use_ea: {args.use_ea}')

    # load dataset
    dataset_dict = {
        'seediv': load_seediv,
        'm3cv': load_m3cv,
        'bciciv2a': load_bciciv2a,
        'thubenchmark': load_thubenchmark
    }
    dataset, info = dataset_dict[args.dataset]()
    logging.info(f'Dataset: {args.dataset}')
    _, _, test_dataset, split_path = prepare_subject_fold(
        dataset_name=args.dataset,
        dataset=dataset,
        info=info,
        fold_id=args.fold,
        seed=args.seed,
        use_ea=args.use_ea,
    )
    logging.info(f'Split path: {split_path}')

    # load model
    model_dict = {
        'eegnet': EEGNet,
        'tsception': TSCeption,
        'atcnet': ATCNet,
        'conformer': Conformer
    }
    model = model_dict[args.model](**get_model_args(args.model, args.dataset, info))
    model.to(device)
    model.eval()
    log_cuda_memory("after model init", device)
    checkpoint_path = resolve_checkpoint_path(args, protocol_tag)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    logging.info(f'Model: {args.model}, fold: {args.fold}, checkpoint: {checkpoint_path}')
    log_cuda_memory("after checkpoint load", device)

    # init attack
    attack_dict = {
        'fgsm': FGSM,
        'pgd': PGD,
        'cw': CW,
        'autoattack': AutoAttack
    }
    attack = attack_dict[args.attack](model, eps=args.eps, device=device, n_classes=info['num_classes'])
    log_cuda_memory("after attack init", device)

    logging.info(f"Sample num in test set: {len(test_dataset)}")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=eeg_classification_collate,
    )
    logging.info(f"test dataset size: {len(test_dataset)}")

    log_cuda_memory("before attack loop", device)
    ad_data = []
    clean_data = []
    labels = []
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        log_cuda_memory(f"after loading batch {i+1}/{len(test_loader)}", device)
        with cuda_mem_tracker(f'attack batch {i+1}/{len(test_loader)}', device):
            adv_data = attack(data, target)
        ad_data.append(adv_data.detach().cpu())
        clean_data.append(data.detach().cpu())
        labels.append(target.detach().cpu())
        logging.info(f'Attacking {args.attack} on {args.dataset} with {args.model}, fold: {args.fold}, batch: {i+1}/{len(test_loader)}')
    log_cuda_memory("after attack loop", device)
    ad_data = torch.cat(ad_data, dim=0).cpu()
    clean_data = torch.cat(clean_data, dim=0).cpu()
    labels = torch.cat(labels, dim=0).cpu()

    # evaluate
    with cuda_mem_tracker("evaluate clean", device):
        evaluate_acc, evaluate_loss = evaluate(model, test_loader)
    logging.info(f'Before Attack - Test Accuracy: {evaluate_acc*100:.2f}%, Test Loss: {evaluate_loss:.4f}')
    ad_dataset = torch.utils.data.TensorDataset(ad_data, labels)
    ad_loader = torch.utils.data.DataLoader(
        ad_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=eeg_classification_collate,
    )
    with cuda_mem_tracker("evaluate adversarial", device):
        ad_evaluate_acc, ad_evaluate_loss = evaluate(model, ad_loader)
    logging.info(f'After Attack - Test Accuracy: {ad_evaluate_acc*100:.2f}%, Test Loss: {ad_evaluate_loss:.4f}')

    # calculate mse
    MSELoss = torch.nn.MSELoss(reduction='mean')
    mse = MSELoss(ad_data, clean_data)
    logging.info(f'MSE between clean data and adversarial data: {mse.item():.6f}')

    should_save_adv = args.save_adv or args.at_strategy == 'clean'
    if should_save_adv:
        os.makedirs('./ad_data', exist_ok=True)
        if args.save_adv or args.adv_output_tag or args.checkpoint_tag or args.checkpoint_path:
            adv_output_path, model_tag = build_adv_output_path(args)
            adv_meta = {
                'dataset': args.dataset,
                'model': args.model,
                'fold': args.fold,
                'seed': args.seed,
                'protocol': protocol_tag,
                'protocol_short': short_protocol_tag(args.use_ea),
                'use_ea': args.use_ea,
                'at_strategy': args.at_strategy,
                'attack': args.attack,
                'eps': args.eps,
                'checkpoint_path': checkpoint_path,
                'model_tag': model_tag,
                'adv_output_tag': args.adv_output_tag,
                'clean_accuracy': evaluate_acc,
                'adv_accuracy': ad_evaluate_acc,
                'mse': mse.item(),
            }
            torch.save((ad_data, labels, adv_meta), adv_output_path)
        else:
            adv_output_path = f'./ad_data/{args.dataset}_{args.model}_{protocol_tag}_{args.at_strategy}_{args.attack}_eps{args.eps}_{args.seed}_fold{args.fold}.pth'
            torch.save((ad_data, labels), adv_output_path)
        logging.info(f'Saved adversarial data: {adv_output_path}')

    if args.at_strategy != 'clean':
        logging.info("Test model on 512 random samples for adversarial training")
        test_data = []
        test_label = []
        for data, target in test_loader:
            test_data.append(data.detach().cpu())
            test_label.append(target.detach().cpu())
        test_data = torch.cat(test_data, dim=0)
        test_label = torch.cat(test_label, dim=0)
        available_sample_num = min(test_data.size(0), ad_data.size(0), labels.size(0))
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

        test_dataset = torch.utils.data.TensorDataset(
            test_data[selected_index_tensor],
            test_label[selected_index_tensor],
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=eeg_classification_collate,
        )

        ad_dataset = torch.utils.data.TensorDataset(
            ad_data[selected_index_tensor],
            labels[selected_index_tensor],
        )
        ad_loader = torch.utils.data.DataLoader(
            ad_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=eeg_classification_collate,
        )
        ad_evaluate_acc, ad_evaluate_loss = evaluate(model, ad_loader)

        logging.info(f'After Attack - Test Accuracy on {eval_sample_num} random samples: {ad_evaluate_acc*100:.2f}%, Test Loss: {ad_evaluate_loss:.4f}')
        test_evaluate_acc, test_evaluate_loss = evaluate(model, test_loader)
        logging.info(f'Before Attack - Test Accuracy on {eval_sample_num} random samples: {test_evaluate_acc*100:.2f}%, Test Loss: {test_evaluate_loss:.4f}')

            

        
        

        
