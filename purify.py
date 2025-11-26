import argparse
import torch
import random
import numpy as np
import yaml

from torcheeg.models import EEGNet, TSCeption, ATCNet, Conformer
from torcheeg.model_selection import KFold, train_test_split
from torcheeg.datasets.constants import SEED_IV_CHANNEL_LOCATION_DICT, M3CV_CHANNEL_LOCATION_DICT
from torcheeg.datasets.constants.motor_imagery import BCICIV2A_LOCATION_DICT
from torcheeg.datasets.constants.ssvep import TSUBENCHMARK_LOCATION_LIST
from torcheeg.transforms import ToGrid, ToInterpolatedGrid

dataset_channel_location_dicts = {
   'seediv': SEED_IV_CHANNEL_LOCATION_DICT,
   'm3cv': M3CV_CHANNEL_LOCATION_DICT,
   'bciciv2a': BCICIV2A_LOCATION_DICT,
   'thubenchmark': TSUBENCHMARK_LOCATION_LIST
}

from models.model_args import get_model_args
from data.load import load_seediv, load_m3cv, load_bciciv2a, load_thubenchmark
from TN.PTR import PTR
from TN.opt import *
from TN.utils import get_TN_args

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
    parser.add_argument('--attack', type=str, default='fgsm', choices=['fgsm', 'pgd', 'cw', 'autoattack'], help='choose attack')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu_id', type=int, default=0, help='which gpu to use')

    parser.add_argument('--config', type=str, help='path to purify config file')
    parser.add_argument('--sample_num', type=int, default=512, help='number of samples to purify')
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

def interpolate(args, data, sampling_rate, strategy):
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
    if strategy == '2d':
        new_h = round(2**np.ceil(np.log2(h)))
        eeg_t_seg = round(w / sampling_rate * 10)   # 10 segments per second
        k = np.ceil(w / (2 ** (config.stage - 1)) / eeg_t_seg)
        new_w = int(k * eeg_t_seg * (2 ** (config.stage - 1)))
        pre_data = torch.nn.functional.interpolate(data.permute(2, 0, 1).unsqueeze(0).cpu(), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
    elif strategy == '3d':
        dataset_channel_location_dict = dataset_channel_location_dicts[args.dataset]
        t = ToGrid(dataset_channel_location_dict)
        # t = ToInterpolatedGrid(dataset_channel_location_dict)
        pre_data = torch.from_numpy(t(eeg=data.squeeze().numpy())['eeg']).permute(1, 2, 0)
        
    return pre_data

def inv_interpolate(pre_data, original_shape, strategy):
    if strategy == '2d':
        data = torch.nn.functional.interpolate(pre_data.permute(2, 0, 1).unsqueeze(0), size=original_shape, mode='bilinear', align_corners=False).squeeze(0)
    elif strategy == '3d':
        dataset_channel_location_dict = dataset_channel_location_dicts[args.dataset]
        t = ToGrid(dataset_channel_location_dict)
        # t = ToInterpolatedGrid(dataset_channel_location_dict)
        data = torch.from_numpy(t.reverse(eeg=pre_data.permute(2, 0, 1).numpy())['eeg'])
    return data



def purify(args, index, data, sampling_rate, device, logging):
    config_path = f'./configs/{args.dataset}/{args.config}'
    dataset = args.dataset

    # init TN
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config['device_type'] = "gpu" if torch.cuda.is_available() else "cpu"
    config_tmp = Config()
    for key, item in config.items():
        setattr(config_tmp, key, item)
    config = config_tmp

    # preprocess data to fit TN input
    # resize data to power of 2
    data = data.permute(1, 2, 0)
    h, w, _ = data.shape
    new_h = round(2**np.ceil(np.log2(h)))
    # new_w = round(new_h * np.ceil(w // new_h))
    eeg_t_seg = round(w / sampling_rate * 10)   # 10 segments per second
    k = np.ceil(w / (2 ** (config.stage - 1)) / eeg_t_seg)
    new_w = int(k * eeg_t_seg * (2 ** (config.stage - 1)))
    pre_data = torch.nn.functional.interpolate(data.permute(2, 0, 1).unsqueeze(0).cpu(), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
    logging.info(f"Purifying data {index}, original shape: {data.shape}, new shape: {pre_data.shape}")

    # init TN
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config['device_type'] = "gpu" if torch.cuda.is_available() else "cpu"
    config_tmp = Config()
    for key, item in config.items():
        setattr(config_tmp, key, item)
    config = config_tmp
    TN_args = get_TN_args(config, pre_data, sampling_rate, None, config.device_type)
    ptr = PTR(**TN_args)
    purified_data, t, mse_history = ptr.train(pre_data.to(device), config, index, logging=logging)
    logging.info("")

    purified_data = torch.nn.functional.interpolate(purified_data.permute(2, 0, 1).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
    return purified_data
    

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # set log file
    import logging
    logfile_directory = f'./log/purify_{args.dataset}_{args.model}_{args.attack}_{args.seed}_{args.config}.log'
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
        ad_data, labels = torch.load(f'./ad_data/{args.dataset}_{args.model}_{args.attack}_{args.seed}_fold{args.fold}.pth')
        clean_data = []
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            clean_data.append(data.detach().cpu())
        clean_data = torch.cat(clean_data, dim=0)
    ad_data = ad_data[:args.sample_num]
    clean_data = clean_data[:args.sample_num]
    labels = labels[:args.sample_num]
    logging.info(f"Adversarial and clean data loaded, ad_data shape: {ad_data.shape}, clean_data shape: {clean_data.shape}")

    # load model
    model_dict = {
        'eegnet': EEGNet,
        'tsception': TSCeption,
        'atcnet': ATCNet,
        'conformer': Conformer
    }
    model = model_dict[args.model](**get_model_args(args.model, args.dataset, info))
    model.to(device)
    checkpoint = torch.load(f'./checkpoints/{args.dataset}_{args.model}_{args.seed}_fold{args.fold}_best.pth', map_location=device)
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

    # purify
    purified_ad_data = []
    for i in range(args.sample_num):
        tmp_purified_ad_data = purify(args, i, ad_data[i], info['sampling_rate'], device, logging)
        purified_ad_data.append(tmp_purified_ad_data)
    logging.info('')
    logging.info('')
    
    purified_clean_data = []
    for i in range(args.sample_num):
        tmp_purified_clean_data = purify(args, i, clean_data[i], info['sampling_rate'], device, logging)
        purified_clean_data.append(tmp_purified_clean_data)
    
    logging.info('')
    logging.info('')

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

    # save purified data
    torch.save((purified_ad_data, labels), f'./purified_data/{args.dataset}_{args.model}_{args.attack}_{args.seed}_fold{args.fold}_{args.config}_ad.pth')
    torch.save((purified_clean_data, labels), f'./purified_data/{args.dataset}_{args.model}_{args.attack}_{args.seed}_fold{args.fold}_{args.config}_clean.pth')