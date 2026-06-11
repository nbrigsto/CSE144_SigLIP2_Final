"""Reproducibility helpers, EMA, mixup/cutmix, and metrics."""
import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int, deterministic: bool = True):
    """Seed all RNGs for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    """DataLoader worker seeding for reproducibility."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ModelEMA:
    """Exponential Moving Average of model weights (improves few-shot stability)."""

    def __init__(self, model, decay=0.9998):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for ema_p, p in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if ema_p.dtype.is_floating_point:
                ema_p.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                ema_p.copy_(p)


def rand_bbox(size, lam):
    """Random bounding box for CutMix."""
    H, W = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def mixup_cutmix(x, y, num_classes, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5,
                 label_smoothing=0.1):
    """Apply MixUp or CutMix and return soft (smoothed) targets.

    Returns x, soft_targets where soft_targets is (B, num_classes).
    """
    off = label_smoothing / num_classes
    on = 1.0 - label_smoothing + off
    y1 = torch.full((y.size(0), num_classes), off, device=x.device)
    y1.scatter_(1, y.unsqueeze(1), on)

    if np.random.rand() > prob:
        return x, y1

    perm = torch.randperm(x.size(0), device=x.device)
    y2 = y1[perm]

    if np.random.rand() < 0.5 and mixup_alpha > 0:  # MixUp
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        x = lam * x + (1 - lam) * x[perm]
    else:  # CutMix
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        x1c, y1c, x2c, y2c = rand_bbox(x.size(), lam)
        x[:, :, y1c:y2c, x1c:x2c] = x[perm, :, y1c:y2c, x1c:x2c]
        lam = 1 - ((x2c - x1c) * (y2c - y1c) / (x.size(-1) * x.size(-2)))

    targets = lam * y1 + (1 - lam) * y2
    return x, targets


class SoftTargetCrossEntropy(nn.Module):
    """Cross entropy against soft targets (for mixup/cutmix)."""

    def forward(self, logits, targets):
        loss = torch.sum(-targets * torch.log_softmax(logits, dim=-1), dim=-1)
        return loss.mean()


@torch.no_grad()
def accuracy(logits, targets):
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()
