"""
Pre-train a separate ValueNet for each known puzzle with curriculum learning.
Run locally once; produces model_<env_id>.pt files for submission.

Usage:
    python3 pretrain.py                  # auto-detect device
    python3 pretrain.py --epochs 200     # longer training
    python3 pretrain.py --puzzle game_15_2d  # single puzzle only
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
from models.value_net import ValueNet, count_params
from core.encoder import StateEncoder
from core.collector import DataCollector


PUZZLES = [
    {
        "env_id": "game_15_2d",
        "env_class": Fifteen2DEnv,
        "stages": [
            {"max_walk": 15,  "num_pairs": 10_000},
            {"max_walk": 50,  "num_pairs": 20_000},
            {"max_walk": 100, "num_pairs": 15_000},
        ],
    },
    {
        "env_id": "toggle_lights",
        "env_class": LightsOutEnv,
        "stages": [
            {"max_walk": 10, "num_pairs": 10_000},
            {"max_walk": 20, "num_pairs": 20_000},
            {"max_walk": 30, "num_pairs": 15_000},
        ],
    },
    {
        "env_id": "cylinder_game",
        "env_class": RotateSlideEnv,
        "stages": [
            {"max_walk": 20,  "num_pairs": 10_000},
            {"max_walk": 80,  "num_pairs": 20_000},
            {"max_walk": 150, "num_pairs": 15_000},
        ],
    },
]


def model_path_for(env_id: str, output_dir: str) -> str:
    return os.path.join(output_dir, f"model_{env_id}.pt")


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
        if (epoch + 1) % 20 == 0:
            print(f"      epoch {epoch + 1:3d}/{epochs}: loss={avg:.4f}")
    return history


def train_puzzle(puzzle, encoder, device, batch_size, lr, epochs_per_stage, seed, output_dir):
    env_id = puzzle["env_id"]
    env = puzzle["env_class"]()
    collector = DataCollector(env, encoder)

    # Each puzzle gets a fresh model — no cross-puzzle contamination
    model = ValueNet().to(device)
    print(f"\n  params: {count_params(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    print(f"\n{'='*52}")
    print(f"  Puzzle: {env_id}")
    print(f"{'='*52}")

    for i, stage in enumerate(puzzle["stages"]):
        print(f"\n  Stage {i + 1}/{len(puzzle['stages'])}: "
              f"max_walk={stage['max_walk']}, pairs={stage['num_pairs']}")

        # Lower LR for later stages to avoid overshooting
        for g in optimizer.param_groups:
            g["lr"] = lr / (2 ** i)

        t0 = time.time()
        tokens, labels = collector.collect(stage["num_pairs"], stage["max_walk"], seed + i)
        print(f"    data: {tokens.shape[0]} pairs in {time.time() - t0:.1f}s")

        history = train_stage(
            model, tokens, labels, optimizer, loss_fn,
            encoder, device, batch_size, epochs_per_stage,
        )
        if history:
            print(f"    loss: {history[0]:.4f} → {history[-1]:.4f}")

    # Save puzzle-specific model (weights on CPU for portability)
    out_path = model_path_for(env_id, output_dir)
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": cpu_state, "config": {"env_id": env_id}}, out_path)
    print(f"\n  Saved → {out_path}")

    # Sanity check: V(s) at different depths
    print("  Sanity check V(s):")
    model.eval()
    model_cpu = ValueNet()
    model_cpu.load_state_dict(cpu_state)
    enc_cpu = StateEncoder()
    for depth in [0, 1, 5, 10, 20, 40]:
        env.reset()
        state, _ = env.scramble(depth, seed=1)
        tokens_d = enc_cpu.encode_tokens(env, state)[None]
        B, N, _ = tokens_d.shape
        dense, cv, tv = enc_cpu.to_tensors(tokens_d, B, N)
        with torch.no_grad():
            v = model_cpu(dense, cv, tv).item()
        print(f"    depth={depth:3d}  V={v:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save model_<env_id>.pt files")
    parser.add_argument("--puzzle", type=str, default=None,
                        help="Train only one puzzle: game_15_2d / toggle_lights / cylinder_game")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    device = _get_device(args.device)
    print(f"Device: {device}")

    encoder = StateEncoder()
    puzzles = [p for p in PUZZLES if args.puzzle is None or p["env_id"] == args.puzzle]
    if not puzzles:
        print(f"Unknown puzzle: {args.puzzle}")
        return

    start = time.time()
    for puzzle in puzzles:
        train_puzzle(
            puzzle, encoder, device,
            args.batch_size, args.lr, args.epochs, args.seed,
            args.output_dir,
        )

    print(f"\nDone. Total time: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
