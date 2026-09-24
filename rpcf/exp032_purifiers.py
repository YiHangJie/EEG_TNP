"""有论文来源的外部净化器；复现差异见 docs/EXP032_BASELINES.md。"""

import torch
from torch import nn
import torch.nn.functional as F


class MagNetReformer(nn.Module):
    """MagNet 作者 MNIST-I [3, average, 3] 对称结构，EEG 尺寸适配。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 3, 3, padding=1), nn.Sigmoid(), nn.AvgPool2d(2),
            nn.Conv2d(3, 3, 3, padding=1), nn.Sigmoid(),
            nn.Conv2d(3, 3, 3, padding=1), nn.Sigmoid(), nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(3, 3, 3, padding=1), nn.Sigmoid(),
            nn.Conv2d(3, 1, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x, return_activity=False):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % 2, 0, (-h) % 2), mode="replicate")
        out = self.net(x)[..., :h, :w]
        return (out, None) if return_activity else out


class DCAE(nn.Module):
    """Ding et al. 2024 Table 2；包含 sigmoid bottleneck，供式 (10)–(12) 使用。"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1), nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x, return_activity=False):
        h, w = x.shape[-2:]
        # 22/62 等通道数不能被 8 整除；仅右/下补齐，再裁剪恢复原形状。
        padded = F.pad(x, (0, (-w) % 8, 0, (-h) % 8), mode="replicate")
        if self.training and torch.is_grad_enabled():
            # 分块重算激活保留论文 batch128 与同一损失，避免长 EEG 反向显存溢出。
            from torch.utils.checkpoint import checkpoint
            activity = padded
            for block in (self.encoder[:3], self.encoder[3:6], self.encoder[6:]):
                activity = checkpoint(block, activity, use_reentrant=False)
            output = activity
            for block in (self.decoder[:3], self.decoder[3:6], self.decoder[6:9], self.decoder[9:]):
                output = checkpoint(block, output, use_reentrant=False)
            output = output[..., :h, :w]
        else:
            activity = self.encoder(padded)
            output = self.decoder(activity)[..., :h, :w]
        return (output, activity) if return_activity else output


def sparsity_penalty(activity, rho=0.02):
    """每个 bottleneck 单元在 batch 上平均，再对单元求 KL 总和。"""
    eta = activity.mean(dim=0).clamp(1e-6, 1 - 1e-6)
    return (rho * (rho / eta).log() + (1 - rho) * ((1 - rho) / (1 - eta)).log()).sum()


class EEGReformer(nn.Module):
    """用 train-only 标量范围桥接有符号 EEG 与论文 sigmoid 输出域。"""

    def __init__(self, method, low, high):
        super().__init__()
        if high <= low:
            raise ValueError("Invalid train-only signal range")
        self.register_buffer("low", torch.tensor(float(low)))
        self.register_buffer("scale", torch.tensor(float(high - low)))
        self.network = {"magnet": MagNetReformer, "dcae": DCAE}[method]()

    def encode_range(self, x):
        return ((x - self.low) / self.scale).clamp(0, 1)

    def forward(self, x):
        return self.network(self.encode_range(x)) * self.scale + self.low
