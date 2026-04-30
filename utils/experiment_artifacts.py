import os
import re

import torch


def safe_token(value, default='none'):
    value = str(value).strip()
    if not value:
        return default
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value)


def short_protocol_tag(use_ea):
    return 'ea' if use_ea else 'no_ea'


def normalize_path_args(paths):
    if not paths:
        return []
    normalized = []
    for path_group in paths:
        for path in str(path_group).split(','):
            path = path.strip()
            if path:
                normalized.append(path)
    return normalized


def as_label_tensor(labels):
    if torch.is_tensor(labels):
        return labels.detach().cpu().long().view(-1)
    return torch.as_tensor(labels, dtype=torch.long).view(-1)


def eeg_classification_collate(batch):
    data_parts = []
    label_parts = []
    for data, label in batch:
        data_parts.append(torch.as_tensor(data).float())
        label_tensor = torch.as_tensor(label, dtype=torch.long).view(-1)
        if label_tensor.numel() != 1:
            raise ValueError(f'Expected scalar class label, got shape {tuple(label_tensor.shape)}.')
        label_parts.append(label_tensor.item())
    return torch.stack(data_parts, dim=0), torch.tensor(label_parts, dtype=torch.long)


def torch_load_cpu(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def load_tensor_payload(path):
    payload = torch_load_cpu(path)
    if isinstance(payload, (tuple, list)) and len(payload) == 3:
        data, labels, meta = payload
    elif isinstance(payload, (tuple, list)) and len(payload) == 2:
        data, labels = payload
        meta = {}
    elif isinstance(payload, dict) and 'data' in payload and 'labels' in payload:
        data = payload['data']
        labels = payload['labels']
        meta = payload.get('meta', {})
    else:
        raise ValueError(
            f'Unsupported tensor payload in {path}; expected (data, labels), '
            f'(data, labels, meta), or a dict with data/labels.'
        )
    if not torch.is_tensor(data):
        data = torch.as_tensor(data)
    meta = meta if isinstance(meta, dict) else {}
    return data.detach().cpu().float(), as_label_tensor(labels), meta


def build_checkpoint_path(dataset, model, protocol_tag, at_strategy, eps, seed, fold,
                          tag=None, lr=None, weight_decay=None, checkpoint_dir='./checkpoints'):
    file_name = f'{dataset}_{model}_{protocol_tag}_{at_strategy}_eps{eps}_{seed}_fold{fold}'
    if lr is not None and weight_decay is not None:
        file_name += f'_{lr}_{weight_decay}'
    if tag:
        file_name += f'_{safe_token(tag)}'
    return os.path.join(checkpoint_dir, f'{file_name}_best.pth')
