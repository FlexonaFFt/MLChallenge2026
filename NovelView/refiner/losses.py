"""Losses tuned for the PSNR metric: L1 + (1-SSIM) + optional VGG perceptual.

PSNR = 20*log10(255/sqrt(MSE)), so we optimize pixel fidelity directly. SSIM keeps
structure/edges sharp; perceptual is a light extra (disabled by default to avoid a
weights download — enable with --perceptual once internet is confirmed on server).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean PSNR over the batch, inputs in [0,1]."""
    mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(1)
    return (20.0 * torch.log10(1.0 / torch.sqrt(mse.clamp_min(1e-10)))).mean()


def _gaussian_window(ch: int, size: int = 11, sigma: float = 1.5, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    win = (g.t() @ g).expand(ch, 1, size, size).contiguous()
    return win


def ssim(pred: torch.Tensor, target: torch.Tensor, win=None) -> torch.Tensor:
    ch = pred.shape[1]
    if win is None:
        win = _gaussian_window(ch, device=pred.device)
    pad = win.shape[-1] // 2
    mu1 = F.conv2d(pred, win, padding=pad, groups=ch)
    mu2 = F.conv2d(target, win, padding=pad, groups=ch)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(pred * pred, win, padding=pad, groups=ch) - mu1_sq
    s2 = F.conv2d(target * target, win, padding=pad, groups=ch) - mu2_sq
    s12 = F.conv2d(pred * target, win, padding=pad, groups=ch) - mu12
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    val = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return val.mean()


class RefinerLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_ssim=0.15, w_perc=0.0):
        super().__init__()
        self.w_l1, self.w_ssim, self.w_perc = w_l1, w_ssim, w_perc
        self.vgg = None
        if w_perc > 0:
            from torchvision.models import vgg16, VGG16_Weights

            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
            for p in vgg.parameters():
                p.requires_grad_(False)
            self.vgg = vgg
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        ssim_v = ssim(pred, target)
        loss = self.w_l1 * l1 + self.w_ssim * (1.0 - ssim_v)
        perc = torch.zeros((), device=pred.device)
        if self.vgg is not None:
            p = (pred - self.mean) / self.std
            t = (target - self.mean) / self.std
            perc = F.l1_loss(self.vgg(p), self.vgg(t))
            loss = loss + self.w_perc * perc
        return loss, {
            "l1": l1.detach(),
            "ssim": ssim_v.detach(),
            "perc": perc.detach(),
            "psnr": psnr(pred.detach(), target),
        }
