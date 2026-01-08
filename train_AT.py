import argparse
import copy
import datetime
import torch
import torch.nn.functional as F
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
    parser.add_argument('--at_strategy', type=str, default='madry', choices=['madry', 'fbf', 'trades', 'clean'], help='adversarial training strategy')
    parser.add_argument('--epsilon', type=float, default=0.1, help='max perturbation budget for adversarial training')
    parser.add_argument('--pgd_step_size', type=float, default=0.02, help='step size for PGD/FGSM updates')
    parser.add_argument('--pgd_steps', type=int, default=10, help='number of PGD steps for adversarial example generation')
    parser.add_argument('--fbf_replays', type=int, default=3, help='number of repeats per batch for FBF training')
    parser.add_argument('--trades_beta', type=float, default=0.1, help='beta coefficient for TRADES loss') # 1, 6
    parser.add_argument('--clean_ratio', type=float, default=0.0, help='portion of clean loss mixed with adversarial loss (Madry)')
    parser.add_argument('--clip_min', type=float, default=None, help='minimum value to clamp adversarial examples')
    parser.add_argument('--clip_max', type=float, default=None, help='maximum value to clamp adversarial examples')
    parser.add_argument('--pgd_random_start', action='store_true', default=True, help='enable random start for PGD/FBF attacks')
    parser.add_argument('--no_pgd_random_start', dest='pgd_random_start', action='store_false', help='disable random start for PGD/FBF attacks')
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
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

def clamp_tensor(x, clip_min=None, clip_max=None):
    if clip_min is None and clip_max is None:
        return x
    lower = -float('inf') if clip_min is None else clip_min
    upper = float('inf') if clip_max is None else clip_max
    return torch.clamp(x, min=lower, max=upper)

def pgd_adversarial_examples(model, x, y, epsilon, step_size, steps, criterion, random_start=True, clip_min=None, clip_max=None):
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

def trades_adversarial_examples(model, x, natural_pred, step_size, epsilon, steps, clip_min=None, clip_max=None):
    x_adv = x.detach() + 0.001 * torch.randn_like(x)
    x_adv = clamp_tensor(x_adv, clip_min, clip_max)
    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = F.kl_div(F.log_softmax(model(x_adv), dim=1), natural_pred, reduction='batchmean')
        grad = torch.autograd.grad(loss_kl, x_adv)[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad)
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = clamp_tensor(x_adv, clip_min, clip_max)
    return x_adv.detach()

def train_epoch_madry(model, loader, optimizer, criterion, device, args):
    model.train()
    total_loss = 0.0
    clean_ratio = max(0.0, min(1.0, args.clean_ratio))
    adv_ratio = 1.0 - clean_ratio
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        adv_data = pgd_adversarial_examples(
            model, data, target, epsilon=args.epsilon, step_size=args.pgd_step_size,
            steps=args.pgd_steps, criterion=criterion, random_start=args.pgd_random_start,
            clip_min=args.clip_min, clip_max=args.clip_max
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

def train_epoch_fbf(model, loader, optimizer, criterion, device, args):
    model.train()
    total_loss = 0.0
    total_samples = len(loader.dataset) * args.fbf_replays
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        if args.pgd_random_start:
            delta = torch.empty_like(data).uniform_(-args.epsilon, args.epsilon)
        else:
            delta = torch.zeros_like(data)
        delta = clamp_tensor(delta, args.clip_min, args.clip_max)
        for _ in range(args.fbf_replays):
            delta.requires_grad_()
            optimizer.zero_grad()
            adv_data = clamp_tensor(data + delta, args.clip_min, args.clip_max)
            logits = model(adv_data)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
            optimizer.step()
            total_loss += loss.item() * data.size(0)
            grad = delta.grad.detach()
            delta = delta + args.pgd_step_size * torch.sign(grad)
            delta = torch.clamp(delta, -args.epsilon, args.epsilon)
            delta = clamp_tensor(delta, args.clip_min, args.clip_max)
            delta = delta.detach()
    return total_loss / total_samples

def train_epoch_trades(model, loader, optimizer, criterion, device, args):
    model.train()
    total_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits_natural = model(data)
        loss_natural = criterion(logits_natural, target)
        natural_pred = torch.softmax(logits_natural.detach(), dim=1)
        adv_data = trades_adversarial_examples(
            model, data, natural_pred, step_size=args.pgd_step_size,
            epsilon=args.epsilon, steps=args.pgd_steps, clip_min=args.clip_min, clip_max=args.clip_max
        )
        logits_adv = model(adv_data)
        loss_robust = F.kl_div(F.log_softmax(logits_adv, dim=1), natural_pred, reduction='batchmean')
        loss = loss_natural + args.trades_beta * loss_robust
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)

def train_epoch_clean(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)

if __name__ == '__main__':
    args = parse_args()
    args.pgd_step_size = args.epsilon / 5
    args.pgd_steps = 5*2
    if args.at_strategy == 'clean':
        args.epsilon = 0
    seed_everything(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # set log file
    import logging
    timestamp = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logfile_directory = f'./log_train_AT/train_{args.dataset}_{args.model}_{args.at_strategy}_eps{args.epsilon}_{args.seed}_{args.lr}_{args.weight_decay}_{args.batch_size}_{timestamp}.log'
    logging.basicConfig(filename=logfile_directory, level=logging.INFO, filemode='w', format='%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')  # 时间格式)
    logging.info(f'Training {args.dataset} with {args.model}')
    logging.info(args)
    logging.info(
        f"AT config | strategy: {args.at_strategy}, eps: {args.epsilon}, "
        f"step size: {args.pgd_step_size}, steps: {args.pgd_steps}, "
        f"fbf_replays: {args.fbf_replays}, trades_beta: {args.trades_beta}, "
        f"clean_ratio: {args.clean_ratio}, random_start: {args.pgd_random_start}, "
        f"clip_min/max: ({args.clip_min}, {args.clip_max})"
    )

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience//2)

        best_val_loss = float('inf')
        # 早停参数（可写进 args）
        patience = getattr(args, "patience", args.patience)   # 容忍无改进的 epoch 数
        min_delta = getattr(args, "min_delta", 0.) # 认为“有改进”的最小下降量
        no_improve_epochs = 0
        criterion = torch.nn.CrossEntropyLoss()
        best_state_dict = copy.deepcopy(model.state_dict())
        # train model
        for epoch in range(args.epochs):
            if args.at_strategy == 'madry':
                train_loss = train_epoch_madry(model, train_loader, optimizer, criterion, device, args)
            elif args.at_strategy == 'fbf':
                train_loss = train_epoch_fbf(model, train_loader, optimizer, criterion, device, args)
            elif args.at_strategy == 'trades':
                train_loss = train_epoch_trades(model, train_loader, optimizer, criterion, device, args)
            else:
                train_loss = train_epoch_clean(model, train_loader, optimizer, criterion, device)
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
                torch.save(best_state_dict, f'./checkpoints/{args.dataset}_{args.model}_{args.at_strategy}_eps{args.epsilon}_{args.seed}_fold{index}_{args.lr}_{args.weight_decay}_best.pth')  # 仍然保存最佳模型
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
        torch.save(best_state_dict, f'./checkpoints/{args.dataset}_{args.model}_{args.at_strategy}_eps{args.epsilon}_{args.seed}_fold{index}_best.pth')  # 保存最佳模型

        # evaluate model
        test_acc, test_loss = evaluate(best_model, test_loader)
        logging.info(f'Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}')
        accs.append(test_acc)
        losses.append(test_loss)
    logging.info(f'Acc: {np.mean(accs):.4f}±{np.std(accs):.4f}, Loss: {np.mean(losses):.4f}±{np.std(losses):.4f}')
