import torch
from torch import nn
from torcheeg.models import EEGNet


class SubjectEAEEGNet(nn.Module):
    """
    在 forward 中执行 subject-wise EA 的 EEGNet。

    输入保持 raw/no_ea EEG 张量 `[B, 1, C, T]`，`subject_ids` 是已经映射到
    `ea_matrices` 第一维的整数索引。EA 矩阵注册为 buffer，随 checkpoint 保存，
    但不作为可训练参数更新。
    """
    def __init__(self,
                 ea_matrices: torch.Tensor,
                 chunk_size: int = 151,
                 num_electrodes: int = 60,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
        super().__init__()
        ea_matrices = torch.as_tensor(ea_matrices, dtype=torch.float32)
        if ea_matrices.dim() != 3:
            raise ValueError(f'ea_matrices must be [num_subjects, C, C], got {tuple(ea_matrices.shape)}.')
        if ea_matrices.size(1) != num_electrodes or ea_matrices.size(2) != num_electrodes:
            raise ValueError(
                f'EA matrix shape {tuple(ea_matrices.shape[1:])} does not match '
                f'num_electrodes={num_electrodes}.'
            )

        self.register_buffer('ea_matrices', ea_matrices)
        self.backbone = EEGNet(
            chunk_size=chunk_size,
            num_electrodes=num_electrodes,
            F1=F1,
            F2=F2,
            D=D,
            num_classes=num_classes,
            kernel_1=kernel_1,
            kernel_2=kernel_2,
            dropout=dropout,
        )

    @staticmethod
    def _normalize_after_ea(x: torch.Tensor) -> torch.Tensor:
        """复现外部 EA 数据流中的逐通道时间维标准化，顺序固定为 EA -> normalize。"""
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        std = torch.sqrt(var)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return (x - mean) / std

    def apply_ea(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """按样本 subject_id 查表并执行 `R^{-1/2} @ X`，随后做单样本归一化。"""
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f'Expected x shape [B, 1, C, T], got {tuple(x.shape)}.')
        subject_ids = torch.as_tensor(subject_ids, dtype=torch.long, device=self.ea_matrices.device).view(-1)
        if subject_ids.numel() != x.size(0):
            raise ValueError(
                f'subject_ids length {subject_ids.numel()} does not match batch size {x.size(0)}.'
            )
        if torch.any(subject_ids < 0) or torch.any(subject_ids >= self.ea_matrices.size(0)):
            raise ValueError('subject_ids contains index outside ea_matrices range.')

        matrices = self.ea_matrices.index_select(0, subject_ids).to(device=x.device, dtype=x.dtype)
        signal = x.squeeze(1)
        aligned = torch.einsum('bij,bjt->bit', matrices, signal)
        aligned = self._normalize_after_ea(aligned)
        return aligned.unsqueeze(1)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        x = self.apply_ea(x, subject_ids)
        return self.backbone(x)
