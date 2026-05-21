"""
Pre-train ValueNet on all three known puzzles with curriculum learning.
Run locally once; produces model.pt for submission.

Usage:
    python3 pretrain.py                     # auto-detect device
    python3 pretrain.py --device mps        # force Apple Silicon GPU
    python3 pretrain.py --epochs 30         # quick test run
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gym import Fifteen2DEnv, LightsOutEnv, RotateSlideEnv
from models.value_net import ValueNet
from core.encoder import StateEncoder
from core.collector import DataCollector


MODEL_PATH = "model.pt"

# Curriculum stages per puzzle: gradually increase walk length
PUZZLES = [
    {
        "name": "game_15_2d",
        "env_class": Fifteen2DEnv,
        "stages": [
            {"max_walk": 15,  "num_pairs": 8_000},
            {"max_walk": 50,  "num_pairs": 15_000},
            {"max_walk": 100, "num_pairs": 10_000},
        ],
    },
    {
        "name": "toggle_lights",
        "env_class": LightsOutEnv,
        "stages": [
            {"max_walk": 10, "num_pairs": 8_000},
            {"max_walk": 20, "num_pairs": 15_000},
            {"max_walk": 30, "num_pairs": 10_000},
        ],
    },
    {
        "name": "cylinder_game",
        "env_class": RotateSlideEnv,
        "stages": [
            {"max_walk": 20,  "num_pairs": 8_000},
            {"max_walk": 80,  "num_pairs": 15_000},
            {"max_walk": 150, "num_pairs": 10_000},
        ],
    },
]


def _get_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_stage(model, tokens, labels, optimizer, loss_fn, encoder, device, batch_size, epochs):
    n = tokens.shape[0]
    if n == 0:
        return []

    history = []
    for epoch in range(epochs):
        idx = np.random.permutation(n)
        epoch_loss, steps = 0.0, 0

        for s in range(0, n, batch_size):
            sel = idx[s:s + batch_size]
            if len(sel) < 2:
                continue

            B, N = len(sel), tokens.shape[1]
            dense, cv, tv = encoder.to_tensors(tokens[sel], B, N)
            dense, cv, tv = dense.to(device), cv.to(device), tv.to(device)
            y = torch.from_numpy(labels[sel]).to(device)

            pred = model(dense, cv, tv)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            steps += 1

        avg = epoch_loss / max(1, steps)
        history.append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"      epoch {epoch + 1:3d}/{epochs}: loss={avg:.4f}")

    return history


def train_puzzle(model, puzzle, encoder, device, batch_size, lr, epochs_per_stage, seed):
    env = puzzle["env_class"]()
    collector = DataCollector(env, encoder)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    print(f"\n{'='*50}")
    print(f"  Puzzle: {puzzle['name']}")
    print(f"{'='*50}")

    for i, stage in enumerate(puzzle["stages"]):
        print(f"\n  Stage {i + 1}/{len(puzzle['stages'])}: "
              f"max_walk={stage['max_walk']}, pairs={stage['num_pairs']}")

        t0 = time.time()
        tokens, labels = collector.collect(stage["num_pairs"], stage["max_walk"], seed + i)
        print(f"    data: {tokens.shape[0]} pairs in {time.time() - t0:.1f}s")

        history = train_stage(
            model, tokens, labels, optimizer, loss_fn,
            encoder, device, batch_size, epochs_per_stage,
        )
        if history:
            print(f"    loss: {history[0]:.4f} → {history[-1]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60,
                        help="Training epochs per curriculum stage (default: 60)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu / mps / cuda (auto-detected if not set)")
    parser.add_argument("--output", type=str, default=MODEL_PATH)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    device = _get_device(args.device)
    print(f"Device: {device}")

    model = ValueNet().to(device)
    if os.path.exists(args.output):
        ckpt = torch.load(args.output, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Resuming from {args.output}")

    encoder = StateEncoder()
    start = time.time()

    for puzzle in PUZZLES:
        train_puzzle(model, puzzle, encoder, device, args.batch_size, args.lr, args.epochs, args.seed)

    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(
        {"state_dict": model_cpu, "config": {"puzzles": [p["name"] for p in PUZZLES]}},
        args.output,
    )

    elapsed = time.time() - start
    print(f"\nDone. Saved → {args.output}  ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
