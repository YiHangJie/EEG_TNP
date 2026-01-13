import argparse
import os
import torch
import random
import numpy as np
import yaml

from torcheeg.models import EEGNet, TSCeption, ATCNet, Conformer
from torcheeg.model_selection import KFold, train_test_split
from torcheeg.datasets.constants import SEED_IV_CHANNEL_LOCATION_DICT, M3CV_CHANNEL_LOCATION_DICT
from torcheeg.datasets.constants.motor_imagery import BCICIV2A_LOCATION_DICT
from torcheeg.datasets.constants.ssvep import TSUBENCHMARK_CHANNEL_LOCATION_DICT
from torcheeg.transforms import ToGrid
from utils.standalone_to_interpolated_grid import ToInterpolatedGrid

dataset_channel_location_dicts = {
   'seediv': SEED_IV_CHANNEL_LOCATION_DICT,
   'm3cv': M3CV_CHANNEL_LOCATION_DICT,
   'bciciv2a': BCICIV2A_LOCATION_DICT,
   'thubenchmark': TSUBENCHMARK_CHANNEL_LOCATION_DICT
}

from models.model_args import get_model_args
from data.load import load_seediv, load_m3cv, load_bciciv2a, load_thubenchmark
from TN.PTR import PTR
from TN.PTR_3d import PTR_3d
from TN.PTR_3d_fs import PTR_3d_fs
from TN.PTR_tfs import PTR_tfs
from TN.opt import *
from TN.utils import get_TN_args
from utils.visualize import plot_eeg

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
    parser.add_argument('--fold', type=int, default=0, help='which fold to use')
    parser.add_argument('--attack', type=str, default='fgsm', choices=['fgsm', 'pgd', 'cw', 'autoattack', 'clean'], help='choose attack')
    parser.add_argument('--eps', type=float, default=0.1, help='epsilon for attacks')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu_id', type=int, default=0, help='which gpu to use')

    parser.add_argument('--config', type=str, help='path to purify config file')
    parser.add_argument('--sample_num', type=int, default=512, help='number of samples to purify')

    parser.add_argument('--visualize', action='store_true', help='whether to visualize the purification process')

    args = parser.parse_args()
    return args

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

def fft_resample(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    使用 FFT 对一维信号进行重采样（最后一个维度视为时间维）

    :param x: (..., N) 形式的输入信号
    :param target_len: 目标重采样长度
    :return: (..., target_len) 形式的重采样结果
    """
    orig_len = x.size(-1)
    Xf = torch.fft.rfft(x, dim=-1)  # (..., N_freq)
    num_freqs_out = target_len // 2 + 1
    num_freqs_in = Xf.size(-1)

    if target_len > orig_len:
        # zero-pad FFT (upsampling)
        pad_size = num_freqs_out - num_freqs_in
        pad = torch.zeros(*Xf.shape[:-1], pad_size, dtype=Xf.dtype, device=Xf.device)
        Xf_resampled = torch.cat([Xf, pad], dim=-1)
    else:
        # truncate FFT (downsampling)
        Xf_resampled = Xf[..., :num_freqs_out]

    # IFFT to get time domain signal
    y = torch.fft.irfft(Xf_resampled, n=target_len, dim=-1)
    return y

def interpolate(args, data, sampling_rate):
    data = data.clone()
    config_path = f'./configs/{args.dataset}/{args.config}'
    # init TN
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config['device_type'] = "gpu" if torch.cuda.is_available() else "cpu"
    config_tmp = Config()
    for key, item in config.items():
        setattr(config_tmp, key, item)
    config = config_tmp

    data = data.permute(1, 2, 0)
    h, w, _ = data.shape
    if config.strategy == 'interpolate':
        new_h = round(2**np.ceil(np.log2(h)))
        # eeg_t_seg = round(w / sampling_rate * 10)   # 10 segments per second
        # k = np.ceil(w / (2 ** (config.stage - 1)) / eeg_t_seg)
        # new_w = int(k * eeg_t_seg * (2 ** (config.stage - 1)))
        new_w = round(2**np.ceil(np.log2(w)+config.c))
        pre_data = torch.nn.functional.interpolate(data.permute(2, 0, 1).unsqueeze(0).cpu(), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
    elif config.strategy == 'fft':
        new_h = round(2**np.ceil(np.log2(h)))
        pre_data = torch.nn.functional.interpolate(data.permute(2, 0, 1).unsqueeze(0).cpu(), size=(new_h, w), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
        # eeg_t_seg = round(w / sampling_rate * 10)   # 10 segments per second
        # k = np.ceil(w / (2 ** (config.stage - 1)) / eeg_t_seg)
        # new_w = int(k * eeg_t_seg * (2 ** (config.stage - 1)))
        new_w = round(2**np.ceil(np.log2(w)+config.c))
        pre_data = fft_resample(data.permute(2, 0, 1), target_len=new_w).permute(1, 2, 0)
    elif config.strategy == '3d':
        dataset_channel_location_dict = dataset_channel_location_dicts[args.dataset]
        t = ToGrid(dataset_channel_location_dict)
        pre_data = torch.from_numpy(t(eeg=data.squeeze().numpy())['eeg']).permute(1, 2, 0).float()
        new_w = round(2**np.ceil(np.log2(w)+config.c))
        pre_data = fft_resample(pre_data, target_len=new_w)
    elif config.strategy == '3d_interpolate':
        dataset_channel_location_dict = dataset_channel_location_dicts[args.dataset]
        t = ToInterpolatedGrid(dataset_channel_location_dict)
        pre_data = torch.from_numpy(t(eeg=data.squeeze().numpy())['eeg']).permute(1, 2, 0).float()
        new_w = round(2**np.ceil(np.log2(w)+config.c))
        pre_data = fft_resample(pre_data, target_len=new_w)
    elif config.strategy == '3d_fs':
        # 3d grid interpolation
        dataset_channel_location_dict = dataset_channel_location_dicts[args.dataset]
        t = ToInterpolatedGrid(dataset_channel_location_dict)
        pre_data = torch.from_numpy(t(eeg=data.squeeze().numpy())['eeg']).permute(1, 2, 0).float()
        # resample to 2^d
        new_w = round(2**np.ceil(np.log2(w)+config.c))
        pre_data = fft_resample(pre_data, target_len=new_w)
        # stft
        time_log = round(np.log2(new_w))
        n_fft_log = time_log // 2
        hop_length_log = time_log - n_fft_log
        n_fft = 2 ** n_fft_log
        hop_length = 2 ** hop_length_log
        new_w = max(new_w - hop_length, 1)
        pre_data = fft_resample(pre_data, target_len=new_w)
        pre_data = torch.stft(pre_data, n_fft=n_fft, hop_length=hop_length, return_complex=False, onesided=True, normalized=True).float().permute(4, 0, 1, 2, 3)

    elif config.strategy == 'tfs':
        new_h = round(2**np.ceil(np.log2(h)+config.c))
        data = torch.nn.functional.interpolate(data.permute(2, 0, 1).unsqueeze(0).cpu(), size=(new_h, w), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
        new_w = round(2**np.ceil(np.log2(w)+config.c))
        hop_length = new_w // new_h
        new_w -= hop_length
        data = fft_resample(data.squeeze(), target_len=new_w)
        pre_data = torch.stft(data, new_h*2-2, hop_length=hop_length, return_complex=False, onesided=True, normalized=True).float().permute(3, 0, 1, 2)

    return pre_data

def inv_interpolate(args, pre_data, original_shape, strategy):
    config_path = f'./configs/{args.dataset}/{args.config}'
    # init TN
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config['device_type'] = "gpu" if torch.cuda.is_available() else "cpu"
    config_tmp = Config()
    for key, item in config.items():
        setattr(config_tmp, key, item)
    config = config_tmp

    pre_data = pre_data.clone()
    h, w = original_shape
    if strategy == 'interpolate':
        data = torch.nn.functional.interpolate(pre_data.permute(2, 0, 1).unsqueeze(0), size=original_shape, mode='bilinear', align_corners=False).squeeze(0)
    elif strategy == 'fft':
        # data = torch.nn.functional.interpolate(pre_data.permute(2, 0, 1).unsqueeze(0), size=(original_shape[0], pre_data.shape[1]), mode='bilinear', align_corners=False).squeeze(0)
        data = fft_resample(pre_data.permute(2, 0, 1), target_len=original_shape[-1])
    elif strategy == '3d':
        pre_data = fft_resample(pre_data, target_len=original_shape[-1])
        dataset_channel_location_dict = dataset_channel_location_dicts[args.dataset]
        t = ToGrid(dataset_channel_location_dict)
        data = torch.from_numpy(t.reverse(eeg=pre_data.permute(2, 0, 1).cpu().numpy())['eeg']).unsqueeze(0).float()
    elif strategy == '3d_interpolate':
        pre_data = fft_resample(pre_data, target_len=original_shape[-1])
        dataset_channel_location_dict = dataset_channel_location_dicts[args.dataset]
        t = ToInterpolatedGrid(dataset_channel_location_dict)
        data = torch.from_numpy(t.reverse(eeg=pre_data.permute(2, 0, 1).cpu().numpy())['eeg']).unsqueeze(0).float()
    elif strategy == 'tfs':
        pre_data = pre_data.permute(1, 2, 3, 0)
        new_h = round(2**np.ceil(np.log2(h)+config.c))
        new_w = round(2**np.ceil(np.log2(w)+config.c))
        hop_length = new_w // new_h
        pre_data = torch.istft(torch.view_as_complex(pre_data.contiguous()), new_h*2-2, hop_length=hop_length, return_complex=False, onesided=True, normalized=True)
        data = fft_resample(pre_data, target_len=original_shape[-1]).unsqueeze(0).float()
        data = torch.nn.functional.interpolate(data.unsqueeze(0), size=original_shape, mode='bilinear', align_corners=False).squeeze(0)
    return data


def purify(args, index, data, sampling_rate, device, logging):
    config_path = f'./configs/{args.dataset}/{args.config}'
    # resize
    pre_data = interpolate(args, data, sampling_rate)
    logging.info(f"original data shape: {data.shape}, Resized data shape: {pre_data.shape}")
    # init TN
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config['device_type'] = "gpu" if torch.cuda.is_available() else "cpu"
    config_tmp = Config()
    for key, item in config.items():
        setattr(config_tmp, key, item)
    config = config_tmp
    TN_args = get_TN_args(config, pre_data.clone(), sampling_rate, None, config.device_type)
    TN_dict = {
        'PTR': PTR,
        'PTR_3d': PTR_3d,
        'PTR_3d_fs': PTR_3d_fs,
        'PTR_tfs': PTR_tfs,
    }
    TN = TN_dict[config.model]
    tn = TN(**TN_args)
    # purify
    purified_data_resized, t, mse_history = tn.train(pre_data.clone().to(device), config, index, logging=logging)
    # purified_data_resized = pre_data
    # inverse resize
    purified_data = inv_interpolate(args, purified_data_resized, data.shape[-2:], config.strategy)
    # logging
    mse = torch.nn.functional.mse_loss(purified_data.cpu(), data).item()
    logging.info(f"Purified data {index}, shape: {purified_data.shape}, mse:{mse}, compression rate: {tn.count_parameters()/data.numel()}, {data.numel()/tn.count_parameters()}x")
    logging.info('')

    # visualization
    if args.visualize:
        plot_eeg(data.cpu().squeeze().numpy(), title=f'Original data {index}', save_path=f'visualization/{args.dataset}_original_data_{index}.png')
        plot_eeg(purified_data.cpu().squeeze().numpy(), title=f'Purified data {index}', save_path=f'visualization/{args.dataset}_purified_data_{index}.png')
        # plot_eeg(pre_data.cpu().squeeze().numpy(), title=f'Resized data {index}', save_path=f'visualization/{args.dataset}_resized_data_{index}.png')
        # plot_eeg(purified_data_resized.cpu().squeeze().numpy(), title=f'Purified resized data {index}', save_path=f'visualization/{args.dataset}_purified_resized_data_{index}.png')
        # for i, t in enumerate(tn.targets):
        #     plot_eeg(t.cpu().squeeze().numpy(), title=f'Target {i} of data {index}', save_path=f'visualization/{args.dataset}_target_{i}_{index}.png')

    return purified_data, mse
    

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    # device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set log file
    import logging
    logfile_directory = f'./log_purify/purify_{args.dataset}_{args.model}_{args.attack}_eps{args.eps}_{args.seed}_{args.config}.log'
    logging.basicConfig(filename=logfile_directory, level=logging.INFO, filemode='w', format='%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')  # 时间格式)
    logging.info(f'Purifying {args.attack} on {args.dataset} with {args.model}')
    logging.info(args)

    # load dataset and adversarial dataset
    dataset_dict = {
        'seediv': load_seediv,
        'm3cv': load_m3cv,
        'bciciv2a': load_bciciv2a,
        'thubenchmark': load_thubenchmark
    }
    dataset, info = dataset_dict[args.dataset]()
    logging.info(f'Dataset: {args.dataset}')
    cv = KFold(n_splits=5, shuffle=True, random_state=args.seed, split_path=f'./cached_data/{args.dataset}_split/kfold_split')

    for index, (train_dataset, test_dataset) in enumerate(cv.split(dataset)):
        if index != args.fold:
            continue
        val_dataset, test_dataset = train_test_split(test_dataset, shuffle=True, test_size=0.5, random_state=args.seed, split_path=f'./cached_data/{args.dataset}_split/test_val_split_{index}')
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        ad_data, labels = torch.load(f'./ad_data/{args.dataset}_{args.model}_{"clean"}_{args.attack}_eps{args.eps}_{args.seed}_fold{args.fold}.pth')
        clean_data = []
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            clean_data.append(data.detach().cpu())
        clean_data = torch.cat(clean_data, dim=0)
    ad_data = ad_data[:args.sample_num]
    clean_data = clean_data[:args.sample_num]
    labels = labels[:args.sample_num]
    logging.info(f"Adversarial and clean data loaded, ad_data shape: {ad_data.shape}, clean_data shape: {clean_data.shape}")
    logging.info('')

    # load model
    model_dict = {
        'eegnet': EEGNet,
        'tsception': TSCeption,
        'atcnet': ATCNet,
        'conformer': Conformer
    }
    model = model_dict[args.model](**get_model_args(args.model, args.dataset, info))
    model.to(device)
    checkpoint = torch.load(f'./checkpoints/{args.dataset}_{args.model}_{"clean_eps0"}_{args.seed}_fold{args.fold}_best.pth', map_location=device)
    model.load_state_dict(checkpoint)
    logging.info(f'Model: {args.model}, fold: {args.fold}')

    # evaluate adversarial and clean data
    ad_data_dataset = torch.utils.data.TensorDataset(ad_data, labels)
    ad_data_dataloader = torch.utils.data.DataLoader(ad_data_dataset, batch_size=args.batch_size, shuffle=False)
    clean_data_dataset = torch.utils.data.TensorDataset(clean_data, labels)
    clean_data_dataloader = torch.utils.data.DataLoader(clean_data_dataset, batch_size=args.batch_size, shuffle=False)
    ad_accuracy, ad_loss = evaluate(model, ad_data_dataloader)
    logging.info(f'Adversarial data accuracy: {ad_accuracy}, loss: {ad_loss}')
    clean_accuracy, clean_loss = evaluate(model, clean_data_dataloader)
    logging.info(f'Clean data accuracy: {clean_accuracy}, loss: {clean_loss}')

    if args.attack != 'clean':
        # purify
        purified_ad_data = []
        mses_ad = []
        for i in range(args.sample_num):
            tmp_purified_ad_data, mse = purify(args, i, ad_data[i], info['sampling_rate'], device, logging)
            purified_ad_data.append(tmp_purified_ad_data)
            mses_ad.append(mse)
        logging.info('')
        logging.info('')
    
    purified_clean_data = []
    mses_clean = []
    for i in range(args.sample_num):
        tmp_purified_clean_data, mse = purify(args, i, clean_data[i], info['sampling_rate'], device, logging)
        purified_clean_data.append(tmp_purified_clean_data)
        mses_clean.append(mse)
    logging.info('')
    logging.info('')

    if args.attack != 'clean':
        purified_ad_data = torch.stack(purified_ad_data, dim=0)
        purified_ad_data_dataset = torch.utils.data.TensorDataset(purified_ad_data, labels)
        purified_ad_data_dataloader = torch.utils.data.DataLoader(purified_ad_data_dataset, batch_size=args.batch_size, shuffle=False)
        purified_ad_accuracy, purified_ad_loss = evaluate(model, purified_ad_data_dataloader)
        logging.info(f'Purified adversarial data accuracy: {purified_ad_accuracy}, loss: {purified_ad_loss}')
    purified_clean_data = torch.stack(purified_clean_data, dim=0)
    purified_clean_data_dataset = torch.utils.data.TensorDataset(purified_clean_data, labels)
    purified_clean_data_dataloader = torch.utils.data.DataLoader(purified_clean_data_dataset, batch_size=args.batch_size, shuffle=False)
    purified_clean_accuracy, purified_clean_loss = evaluate(model, purified_clean_data_dataloader)
    logging.info(f'Purified clean data accuracy: {purified_clean_accuracy}, loss: {purified_clean_loss}')
    logging.info(f'Mean mse of purified adversarial data: {np.mean(mses_ad)}')
    logging.info(f'Mean mse of purified clean data: {np.mean(mses_clean)}')

    # # save purified data
    # if args.attack != 'clean':
    #     torch.save((purified_ad_data, labels), f'./purified_data/{args.dataset}_{args.model}_{args.attack}_eps{args.eps}_{args.seed}_fold{args.fold}_{args.config}_ad.pth')
    # torch.save((purified_clean_data, labels), f'./purified_data/{args.dataset}_{args.model}_{args.attack}_eps{args.eps}_{args.seed}_fold{args.fold}_{args.config}_clean.pth')
