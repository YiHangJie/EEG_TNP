import argparse
import copy
import datetime
import torch
import random
import numpy as np
from tqdm import tqdm

from torcheeg.models import EEGNet, TSCeption, ATCNet, Conformer
from torcheeg.model_selection import KFold, train_test_split

from models.model_args import get_model_args
from data.load import load_seediv, load_m3cv, load_bciciv2a, load_thubenchmark

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
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
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
    timestamp = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logfile_directory = f'./log/train_{args.dataset}_{args.model}_{args.seed}_{args.lr}_{args.weight_decay}_{args.batch_size}_{timestamp}.log'
    logging.basicConfig(filename=logfile_directory, level=logging.INFO, filemode='w', format='%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')  # 时间格式)
    logging.info(f'Training {args.dataset} with {args.model}')
    logging.info(args)

    # load dataset
    dataset_dict = {
        'seediv': load_seediv,
        'm3cv': load_m3cv,
        'bciciv2a': load_bciciv2a,
        'thubenchmark': load_thubenchmark
    }
    dataset, info = dataset_dict[args.dataset]()
    logging.info(f'Dataset: {args.dataset}, Sample_num: {len(dataset)}, num_classes: {info["num_classes"]}')
    cv = KFold(n_splits=5, shuffle=True, random_state=args.seed, split_path=f'./cached_data/{args.dataset}_split/kfold_split')

    # train model
    accs = []
    losses = []
    best_models = []
    for index, (train_dataset, test_dataset) in enumerate(cv.split(dataset)):
        if index >= 1:
            break
        logging.info(f"sample num in train set: {len(train_dataset)}, sample num in test set: {len(test_dataset)}")
        # initialize model
        model_dict = {
            'eegnet': EEGNet,
            'tsception': TSCeption,
            'atcnet': ATCNet,
            'conformer': Conformer
        }
        model = model_dict[args.model](**get_model_args(args.model, args.dataset, info))
        model.to(device)
        logging.info(f'Model: {args.model}, Parameter Num: {sum(p.numel() for p in model.parameters())}')

        # train_dataset, val_dataset = train_test_split(train_dataset, shuffle=True, test_size=0.25, random_state=args.seed, split_path=f'./cached_data/{args.dataset}_split/test_val_split_{index}')
        val_dataset, test_dataset = train_test_split(test_dataset, shuffle=True, test_size=0.5, random_state=args.seed, split_path=f'./cached_data/{args.dataset}_split/test_val_split_{index}')
        logging.info(f"sample num in train set: {len(train_dataset)}, sample num in val set: {len(val_dataset)}, sample num in test set: {len(test_dataset)}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # init optimizer and scheduler
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience//2, verbose=True)

        best_val_loss = float('inf')
        # 早停参数（可写进 args）
        patience = getattr(args, "patience", args.patience)   # 容忍无改进的 epoch 数
        min_delta = getattr(args, "min_delta", 0.) # 认为“有改进”的最小下降量
        no_improve_epochs = 0
        criterion = torch.nn.CrossEntropyLoss()
        # train model
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                # grad clip
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
                optimizer.step()
                train_loss += loss.item() * data.size(0)
            train_loss /= len(train_loader.dataset)
            val_acc, val_loss = evaluate(model, val_loader)
            test_acc, test_loss = evaluate(model, test_loader)
            scheduler.step(val_loss)
            logging.info(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}, Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            # logging.info(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}')

            # # save best model
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     best_model = model
            #     torch.save(model.state_dict(), f'./checkpoints/{args.dataset}_{args.model}_{args.seed}_fold{index}_best.pth')

            # --- Early Stopping 核心逻辑 ---
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                no_improve_epochs = 0
                best_state_dict = copy.deepcopy(model.state_dict())
                torch.save(best_state_dict, f'./checkpoints/{args.dataset}_{args.model}_{args.seed}_fold{index}_{args.lr}_{args.weight_decay}_best.pth')  # 仍然保存最佳模型
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    logging.info(
                        f'Early stopping at epoch {epoch+1} '
                        f'(no improvement in {patience} epochs). Best Val Loss: {best_val_loss:.4f}'
                    )
                    break
        best_model = model_dict[args.model](**get_model_args(args.model, args.dataset, info))
        # best_model.load_state_dict(torch.load(f'./checkpoints/{args.dataset}_{args.model}_{args.seed}_fold{index}_best.pth'))
        best_model.load_state_dict(best_state_dict)
        best_model.to(device)
        best_models.append(best_model)

        # evaluate model
        test_acc, test_loss = evaluate(best_model, test_loader)
        logging.info(f'Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}')
        accs.append(test_acc)
        losses.append(test_loss)
    logging.info(f'Acc: {np.mean(accs):.4f}±{np.std(accs):.4f}, Loss: {np.mean(losses):.4f}±{np.std(losses):.4f}')