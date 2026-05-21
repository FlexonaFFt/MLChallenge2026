import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.collector import DataCollector
from core.encoder import StateEncoder


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        collector: DataCollector,
        encoder: StateEncoder,
        batch_size: int = 256,
        lr: float = 1e-3,
    ):
        self.model = model
        self.collector = collector
        self.encoder = encoder
        self.batch_size = batch_size
        self.lr = lr

    def fit(
        self,
        deadline: float,
        num_pairs: int = 40000,
        max_walk: int = 60,
        seed: int = 239,
        max_epochs: int = 1000,
    ):
        print("collecting data...")
        t0 = time.time()
        tokens, labels = self.collector.collect(num_pairs, max_walk, seed)
        print(f"dataset: {tokens.shape[0]} pairs, {time.time() - t0:.1f}s")

        n = tokens.shape[0]
        if n == 0:
            return []

        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.SmoothL1Loss()
        history = []

        for epoch in range(max_epochs):
            if time.time() >= deadline:
                break
            idx = np.random.permutation(n)
            epoch_loss, steps = 0.0, 0

            for s in range(0, n, self.batch_size):
                if time.time() >= deadline:
                    break
                sel = idx[s:s + self.batch_size]
                if len(sel) < 2:
                    continue
                B, N = len(sel), tokens.shape[1]
                dense, cv, tv = self.encoder.to_tensors(tokens[sel], B, N)
                y = torch.from_numpy(labels[sel])

                pred = self.model(dense, cv, tv)
                loss = loss_fn(pred, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item())
                steps += 1

            avg = epoch_loss / max(1, steps)
            history.append(avg)
            print(f"  epoch {epoch}: loss={avg:.4f}")

        return history
