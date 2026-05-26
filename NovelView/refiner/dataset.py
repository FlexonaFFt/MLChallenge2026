"""Dataset that yields (geo_stack, target) tensors.

Geometric inputs are expensive (lidar projection + warps), so each sample's stack
is cached to NVMe on first use (--cache-dir). Training reads the cache afterwards,
which keeps epochs fast and CPU-light. Random crops are used for training; full
frames for validation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geo_warp import build_geo_inputs, stack_channels  # noqa: E402


class SampleDataset(Dataset):
    def __init__(self, root, split="train", crop=512, cache_dir=None, train=True):
        self.root = Path(root)
        self.dirs = sorted(p for p in (self.root / split).iterdir() if p.is_dir())
        self.crop = crop
        self.train = train
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.dirs:
            raise RuntimeError(f"No samples found under {self.root / split}")

    def __len__(self):
        return len(self.dirs)

    def _load_stack(self, sample_dir: Path):
        if self.cache_dir:
            cf = self.cache_dir / f"{sample_dir.name}.npz"
            if cf.exists():
                d = np.load(cf)
                return d["x"], (d["y"] if "y" in d else None)
        geo = build_geo_inputs(sample_dir)
        x = stack_channels(geo).astype(np.float32)
        y = (
            geo["target"].transpose(2, 0, 1).astype(np.float32)
            if geo["target"] is not None
            else None
        )
        if self.cache_dir:
            cf = self.cache_dir / f"{sample_dir.name}.npz"
            if y is not None:
                np.savez_compressed(cf, x=x, y=y)
            else:
                np.savez_compressed(cf, x=x)
        return x, y

    def __getitem__(self, i):
        sample_dir = self.dirs[i]
        x, y = self._load_stack(sample_dir)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y) if y is not None else torch.zeros(3, *x.shape[1:])

        if self.train and self.crop:
            _, H, W = x.shape
            ch = min(self.crop, H)
            cw = min(self.crop, W)
            top = np.random.randint(0, H - ch + 1)
            left = np.random.randint(0, W - cw + 1)
            x = x[:, top : top + ch, left : left + cw]
            y = y[:, top : top + ch, left : left + cw]
        return x, y, sample_dir.name
