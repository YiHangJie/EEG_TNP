import argparse
import torch
import random
import numpy as np
from tqdm import tqdm

from torcheeg.models import EEGNet, TSCeption, ATCNet, Conformer
from torcheeg.model_selection import KFold, train_test_split

from models.model_args import get_model_args
from data.load import load_seediv, load_m3cv, load_bciciv2a, load_thubenchmark
from attack.fgsm import FGSM
from attack.pgd import PGD
from attack.cw import CW
from attack.autoattack import AutoAttack

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

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # set log file
    import logging
    logfile_directory = f'./log_attack/attack_{args.dataset}_{args.model}_{args.at_strategy}_{args.attack}_{args.eps}_{args.seed}.log'
    logging.basicConfig(filename=logfile_directory, level=logging.INFO, filemode='w', format='%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')  # 时间格式)
    logging.info(f'Attacking {args.attack} on {args.dataset} with {args.model}')
    logging.info(args)

    # load dataset
    dataset_dict = {
        'seediv': load_seediv,
        'm3cv': load_m3cv,
        'bciciv2a': load_bciciv2a,
        'thubenchmark': load_thubenchmark
    }
    dataset, info = dataset_dict[args.dataset]()
    logging.info(f'Dataset: {args.dataset}')
    cv = KFold(n_splits=5, shuffle=True, random_state=args.seed, split_path=f'./cached_data/{args.dataset}_split/kfold_split')

    # attack
    for index, (train_dataset, test_dataset) in enumerate(cv.split(dataset)):
        if index != args.fold:
            continue
        
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
        eps = args.eps if args.at_strategy != 'clean' else 0
        checkpoint = torch.load(f'./checkpoints/{args.dataset}_{args.model}_{args.at_strategy}_eps{eps}_{args.seed}_fold{args.fold}_best.pth', map_location=device)
        model.load_state_dict(checkpoint)
        logging.info(f'Model: {args.model}, fold: {args.fold}')

        # init attack
        attack_dict = {
            'fgsm': FGSM,
            'pgd': PGD,
            'cw': CW,
            'autoattack': AutoAttack
        }
        attack = attack_dict[args.attack](model, eps=args.eps, device=device, n_classes=info['num_classes'])
        
        val_dataset, test_dataset = train_test_split(test_dataset, shuffle=True, test_size=0.5, random_state=args.seed, split_path=f'./cached_data/{args.dataset}_split/test_val_split_{index}')
        logging.info(f"Sample num in test set: {len(test_dataset)}")
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        logging.info(f"test dataset size: {len(test_dataset)}")

        ad_data = []
        clean_data = []
        labels = []
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            adv_data = attack(data, target)
            ad_data.append(adv_data.detach())
            clean_data.append(data.detach())
            labels.append(target.detach())
            logging.info(f'Attacking {args.attack} on {args.dataset} with {args.model}, fold: {args.fold}, batch: {i+1}/{len(test_loader)}')
        ad_data = torch.cat(ad_data, dim=0).cpu()
        clean_data = torch.cat(clean_data, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        
        # evaluate
        evaluate_acc, evaluate_loss = evaluate(model, test_loader)
        logging.info(f'Before Attack - Test Accuracy: {evaluate_acc*100:.2f}%, Test Loss: {evaluate_loss:.4f}')
        ad_dataset = torch.utils.data.TensorDataset(ad_data, labels)
        ad_loader = torch.utils.data.DataLoader(ad_dataset, batch_size=args.batch_size, shuffle=False)
        ad_evaluate_acc, ad_evaluate_loss = evaluate(model, ad_loader)
        logging.info(f'After Attack - Test Accuracy: {ad_evaluate_acc*100:.2f}%, Test Loss: {ad_evaluate_loss:.4f}')

        # calculate mse
        MSELoss = torch.nn.MSELoss(reduction='mean')
        mse = MSELoss(ad_data, clean_data)
        logging.info(f'MSE between clean data and adversarial data: {mse.item():.6f}')

        # save adversarial data
        torch.save((ad_data, labels), f'./ad_data/{args.dataset}_{args.model}_{args.at_strategy}_{args.attack}_eps{args.eps}_{args.seed}_fold{args.fold}.pth')

        if args.at_strategy != 'clean':
            logging.info(f"Test model on 512 samples for adversarial training")
            test_data = []
            test_label = []
            for i, (data, target) in enumerate(test_loader):
                test_data.append(data.detach().cpu())
                test_label.append(target.detach().cpu())
            test_data = torch.cat(test_data, dim=0)
            test_label = torch.cat(test_label, dim=0)
            test_dataset = torch.utils.data.TensorDataset(test_data[:512], test_label[:512])
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            ad_dataset = torch.utils.data.TensorDataset(ad_data[:512], labels[:512])
            ad_loader = torch.utils.data.DataLoader(ad_dataset, batch_size=args.batch_size, shuffle=False)
            ad_evaluate_acc, ad_evaluate_loss = evaluate(model, ad_loader)
            
            logging.info(f'After Attack - Test Accuracy on 512 samples: {ad_evaluate_acc*100:.2f}%, Test Loss: {ad_evaluate_loss:.4f}')
            test_evaluate_acc, test_evaluate_loss = evaluate(model, test_loader)
            logging.info(f'Before Attack - Test Accuracy on 512 samples: {test_evaluate_acc*100:.2f}%, Test Loss: {test_evaluate_loss:.4f}')

            

        
        

        