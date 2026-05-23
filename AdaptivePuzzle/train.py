"""Hidden-game adaptation: fit V(s) on backward random walks; save model.pt."""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gym
import common
from model import ValueNet


TIME_LIMIT_DEFAULT = 1 * 60
SAFETY_MARGIN = 20
MODEL_PATH = "model.pt"
META_PATH = "meta.json"


def collect_dataset(env, num_pairs, max_walk, seed):
    pairs = common.backward_walks(
        env, num_walks=max(100, num_pairs // max(1, max_walk // 2)),
        min_len=1, max_len=max_walk, seed=seed,
    )
    random.Random(seed + 1).shuffle(pairs)
    pairs = pairs[:num_pairs]
    if not pairs:
        return (
            np.zeros((0, 1, common.TOKEN_FEAT_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    tokens = np.stack([common.encode_tokens(env, s) for s, _ in pairs])
    labels = np.array([d for _, d in pairs], dtype=np.float32)
    return tokens, labels


def to_tensors(tokens):
    B, N, _ = tokens.shape
    parts = common.split_token_features(tokens.reshape(B * N, -1))
    return (
        torch.from_numpy(parts["dense"].reshape(B, N, -1)),
        torch.from_numpy(parts["content_value"].reshape(B, N)),
        torch.from_numpy(parts["target_value"].reshape(B, N)),
    )


def finetune(model, tokens, labels, deadline, batch_size=256, lr=1e-3, max_epochs=1000):
    n = tokens.shape[0]
    if n == 0:
        return []
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    history = []

    for epoch in range(max_epochs):
        if time.time() >= deadline:
            break
        idx = np.random.permutation(n)
        epoch_loss, steps = 0.0, 0
        for s in range(0, n, batch_size):
            if time.time() >= deadline:
                break
            sel = idx[s:s + batch_size]
            if len(sel) < 2:
                continue
            dense, cv, tv = to_tensors(tokens[sel])
            y = torch.from_numpy(labels[sel])

            pred = model(dense, cv, tv)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_limit", type=int,
                        default=int(os.environ.get("TRAIN_TIME_LIMIT", TIME_LIMIT_DEFAULT)))
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--num_pairs", type=int, default=40000)
    parser.add_argument("--max_walk", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    start = time.time()
    deadline = start + args.time_limit - SAFETY_MARGIN

    env = gym.make_env()
    env_id = getattr(gym, "ENV_ID", "unknown")
    print(f"env_id={env_id}")

    print("collecting data...")
    t0 = time.time()
    tokens, labels = collect_dataset(env, args.num_pairs, args.max_walk, args.seed)
    print(f"dataset: {tokens.shape[0]} pairs, {time.time()-t0:.1f}s")

    model = ValueNet()

    print("fitting V...")
    history = finetune(
        model, tokens, labels, deadline,
        batch_size=args.batch_size, lr=args.lr,
    )

    torch.save(
        {"state_dict": model.state_dict(),
         "config": {"env_id": env_id, "history": history}},
        MODEL_PATH,
    )
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "env_id": env_id,
            "num_train_samples": int(tokens.shape[0]),
            "wall_time_sec": time.time() - start,
        }, f, indent=2)

    print(f"train.py done in {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
