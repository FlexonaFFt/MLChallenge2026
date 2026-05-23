"""Hidden-game adaptation: fit V(s) ~ cost-to-go via Deep Approximate Value
Iteration (DAVI, the core of DeepCubeA), then save model.pt.

Instead of regressing noisy random-walk depth, we bootstrap true shortest-path
distance with a Bellman backup toward the goal:

    V(s) <- min_a [ 1 + V_target(step(s, a)) ],   V(goal) = 0

V_target is a frozen copy of the network, refreshed every `target_update`
steps. The goal is anchored to 0, which keeps the recursion grounded.
"""

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


TIME_LIMIT_DEFAULT = 50 * 60
SAFETY_MARGIN = 60
MODEL_PATH = "model.pt"
META_PATH = "meta.json"


def to_tensors(tokens):
    B, N, _ = tokens.shape
    parts = common.split_token_features(tokens.reshape(B * N, -1))
    return (
        torch.from_numpy(parts["dense"].reshape(B, N, -1)),
        torch.from_numpy(parts["content_value"].reshape(B, N)),
        torch.from_numpy(parts["target_value"].reshape(B, N)),
    )


def net_values(net, tokens, chunk=8192):
    """Batched forward, no grad -> (B,) float32 numpy."""
    if tokens.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, tokens.shape[0], chunk):
            dense, cv, tv = to_tensors(tokens[s:s + chunk])
            out.append(net(dense, cv, tv).cpu().numpy())
    return np.concatenate(out)


def sample_states(env, n, max_scramble, rng):
    """Sample states by random scrambles of length 1..max_scramble from goal."""
    states = []
    for _ in range(n):
        L = rng.randint(1, max_scramble)
        env.reset(seed=rng.randint(0, 10**9))
        for _ in range(L):
            valid = env.valid_actions()
            if not valid:
                break
            env.step(rng.choice(valid))
        states.append(env.get_state())
    return states


def expand_children(env, states, solved_key):
    """For each parent state, enumerate children. Returns stacked child tokens,
    a parent-index array, and a boolean goal mask."""
    child_tok = []
    owner = []
    is_goal = []
    for i, s in enumerate(states):
        try:
            env.set_state(s)
            valid = env.valid_actions()
        except Exception:
            continue
        for a in valid:
            try:
                env.set_state(s)
                env.step(a)
                cs = env.get_state()
            except Exception:
                continue
            child_tok.append(common.encode_tokens(env, cs))
            owner.append(i)
            is_goal.append(common.state_key(common.to_jsonable(cs)) == solved_key)
    if not child_tok:
        return None, None, None
    return np.stack(child_tok), np.asarray(owner), np.asarray(is_goal, dtype=bool)


def davi(env, model, deadline, args, start):
    """DeepCubeA-style DAVI with a cached state buffer.

    We build a fixed buffer of states ONCE and precompute every state's child
    encodings. The training loop then only (a) re-evaluates children with the
    current target net, (b) takes the Bellman backup, (c) runs several SGD
    epochs over the cached parents. No env calls in the loop -> fast iterations.
    """
    rng = random.Random(args.seed)
    target = ValueNet()
    target.load_state_dict(model.state_dict())
    target.eval()

    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()
    solved_key = common.state_key(common.to_jsonable(env.solved_state()))

    # --- one-time buffer + child cache ---
    t0 = time.time()
    buffer = sample_states(env, args.buffer_size, args.max_walk, rng)
    parent_tokens = np.stack([common.encode_tokens(env, s) for s in buffer])
    ctok, owner, is_goal = expand_children(env, buffer, solved_key)
    if ctok is None:
        print("  WARNING: no children expanded; aborting DAVI")
        return [], 0
    B = len(buffer)
    print(f"  buffer={B} states, {ctok.shape[0]} children, built in {time.time()-t0:.1f}s")

    history = []
    it = 0
    mb = args.batch_size
    while time.time() < deadline:
        # (a) relabel children with the frozen target net
        jchild = net_values(target, ctok)
        jchild = np.where(is_goal, 0.0, jchild)
        cost = (1.0 + jchild).astype(np.float32)
        # (b) Bellman backup: per-parent min over children
        targets = np.full(B, np.inf, dtype=np.float32)
        np.minimum.at(targets, owner, cost)
        finite = np.isfinite(targets)
        idx_all = np.where(finite)[0]
        y_all = targets[finite]

        # (c) several SGD epochs over cached parents
        model.train()
        last_loss = 0.0
        for _ in range(args.epochs_per_relabel):
            if time.time() >= deadline:
                break
            perm = np.random.permutation(len(idx_all))
            for s in range(0, len(perm), mb):
                if time.time() >= deadline:
                    break
                sel = idx_all[perm[s:s + mb]]
                if len(sel) < 2:
                    continue
                dense, cv, tv = to_tensors(parent_tokens[sel])
                ysel = torch.from_numpy(targets[sel])
                pred = model(dense, cv, tv)
                loss = loss_fn(pred, ysel)
                opt.zero_grad()
                loss.backward()
                opt.step()
                last_loss = float(loss.item())

        if not np.isfinite(last_loss):
            print("  WARNING: non-finite loss; resyncing target net")
            target.load_state_dict(model.state_dict())
            continue
        history.append(last_loss)

        it += 1
        if it % args.target_update == 0:
            target.load_state_dict(model.state_dict())
        print(f"  davi relabel={it} loss={last_loss:.4f} "
              f"mean_target={float(y_all.mean()):.2f} max_target={float(y_all.max()):.1f} "
              f"elapsed={time.time()-start:.0f}s")

    return history, it


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_limit", type=int,
                        default=int(os.environ.get("TRAIN_TIME_LIMIT", TIME_LIMIT_DEFAULT)))
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--buffer_size", type=int, default=6000)
    parser.add_argument("--max_walk", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs_per_relabel", type=int, default=5)
    parser.add_argument("--target_update", type=int, default=3)
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

    model = ValueNet()

    print("fitting V via DAVI...")
    history, iters = davi(env, model, deadline, args, start)
    print(f"DAVI: {iters} iterations")

    torch.save(
        {"state_dict": model.state_dict(),
         "config": {"env_id": env_id, "iters": iters,
                    "history_tail": history[-20:]}},
        MODEL_PATH,
    )
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "env_id": env_id,
            "method": "davi",
            "iters": iters,
            "wall_time_sec": time.time() - start,
        }, f, indent=2)

    print(f"train.py done in {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
