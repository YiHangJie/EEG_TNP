"""EXP-032 独立 L2 攻击；不改变 EXP-031 的攻击默认值。"""

import torch
import torch.nn.functional as F


def pgd_l2(model, clean, labels, eps=1.0, alpha=0.1, steps=200, restarts=5):
    """在完整 EEG 样本 L2 球内优化；跨轨迹保留成功优先、CE 次优候选。

    EEG 有正负值，因此不使用图像的 [0,1] 裁剪。随机源沿用全局 torch seed。
    每步及每次重启均参与候选选择，避免最后一步把已成功样本改回正确。
    """
    if eps <= 0 or alpha <= 0 or steps < 1 or restarts < 1:
        raise ValueError("Positive attack budget/steps/restarts required")
    model.eval()
    clean = clean.detach()
    shape = (-1,) + (1,) * (clean.ndim - 1)
    with torch.no_grad():
        logits = model(clean)
        best_loss = F.cross_entropy(logits, labels, reduction="none")
        best_success = logits.argmax(1).ne(labels)
    best = clean.clone()

    def consider(candidate, logits):
        nonlocal best, best_loss, best_success
        loss = F.cross_entropy(logits, labels, reduction="none").detach()
        success = logits.argmax(1).ne(labels)
        take = (success & ~best_success) | ((success == best_success) & (loss > best_loss))
        best[take] = candidate.detach()[take]
        best_loss[take] = loss[take]
        best_success[take] = success[take]

    for _ in range(restarts):
        noise = torch.randn_like(clean)
        noise /= noise.flatten(1).norm(2, 1).clamp_min(1e-12).view(shape)
        radius = torch.rand(len(clean), device=clean.device).pow(1 / clean[0].numel())
        adv = clean + noise * (eps * radius).view(shape)
        for _ in range(steps):
            adv = adv.detach().requires_grad_(True)
            logits = model(adv)
            consider(adv, logits.detach())
            gradient, = torch.autograd.grad(F.cross_entropy(logits, labels), adv)
            gradient /= gradient.flatten(1).norm(2, 1).clamp_min(1e-12).view(shape)
            delta = (adv.detach() + alpha * gradient - clean)
            factor = (eps / delta.flatten(1).norm(2, 1).clamp_min(1e-12)).clamp_max(1)
            adv = clean + delta * factor.view(shape)
        with torch.no_grad():
            consider(adv, model(adv))
    return best.detach()
