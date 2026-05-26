"""Residual U-Net refiner.

Takes the 10-channel geometric stack and predicts a residual over a geometry-based
base image (validity-weighted blend of the two warps). Predicting a residual makes
training converge fast because geometry already provides most of the answer.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(inplace=True),
    )


class UNetRefiner(nn.Module):
    def __init__(self, in_ch: int = 16, base: int = 48, depth: int = 4):
        super().__init__()
        self.depth = depth
        chs = [base * (2 ** i) for i in range(depth + 1)]

        self.inc = _conv(in_ch, chs[0])
        self.downs = nn.ModuleList([_conv(chs[i], chs[i + 1]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in range(depth)]
        )
        self.up_convs = nn.ModuleList([_conv(chs[i + 1], chs[i]) for i in range(depth)])
        self.outc = nn.Conv2d(chs[0], 3, 1)
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)  # start as identity (output == base)

    @staticmethod
    def _base_blend(x: torch.Tensor) -> torch.Tensor:
        # channel layout:
        #   raw_t0 0:3 | raw_t1 3:6 | warp_t0 6:9 | warp_t1 9:12
        #   flow_interp 12:15 | flowmag 15
        # Base = flow interpolation (~20.8 dB); the net learns a residual on top.
        return x[:, 12:15]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self._base_blend(x)
        h = self.inc(x)
        skips = [h]
        for d in range(self.depth):
            h = self.downs[d](self.pool(h))
            skips.append(h)
        for d in reversed(range(self.depth)):
            h = self.ups[d](h)
            skip = skips[d]
            # pad to match odd sizes
            dh = skip.shape[-2] - h.shape[-2]
            dw = skip.shape[-1] - h.shape[-1]
            if dh or dw:
                h = nn.functional.pad(h, [0, dw, 0, dh])
            h = self.up_convs[d](torch.cat([h, skip], dim=1))
        residual = self.outc(h)
        return (base + residual).clamp(0.0, 1.0)
