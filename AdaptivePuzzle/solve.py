"""Inference: load model.pt, run A* (with V as heuristic), write CSV."""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

import gym
import common
from common import state_key
from model import ValueNet
from search import solve_astar


TIME_LIMIT_DEFAULT = 1 * 60
SAFETY_MARGIN = 10
MODEL_PATH = "model.pt"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = ValueNet()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def make_v_fn(env, model):
    if model is None:
        return lambda states: np.zeros(len(states), dtype=np.float32)

    def v_fn(states):
        tokens = np.stack([common.encode_tokens(env, s) for s in states])
        B, N, _ = tokens.shape
        parts = common.split_token_features(tokens.reshape(B * N, -1))
        dense = torch.from_numpy(parts["dense"].reshape(B, N, -1))
        cv = torch.from_numpy(parts["content_value"].reshape(B, N))
        tv = torch.from_numpy(parts["target_value"].reshape(B, N))
        with torch.no_grad():
            return model(dense, cv, tv).cpu().numpy()

    return v_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input_states.jsonl")
    parser.add_argument("--output", default="output_actions.csv")
    parser.add_argument("--time_limit", type=int,
                        default=int(os.environ.get("SOLVE_TIME_LIMIT", TIME_LIMIT_DEFAULT)))
    args = parser.parse_args()

    start = time.time()
    deadline = start + args.time_limit - SAFETY_MARGIN
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    env = gym.make_env()
    instances = load_jsonl(args.input)
    print(f"loaded {len(instances)} instances")

    model = load_model()
    print(f"model loaded: {model is not None}")

    env.reset()
    solved_k = state_key(env.get_state())
    v_fn = make_v_fn(env, model)

    n = len(instances)
    solved = 0

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance_id", "actions"])
        writer.writeheader()

        for i, inst in enumerate(instances):
            iid = inst["instance_id"]
            if time.time() >= deadline:
                writer.writerow({"instance_id": iid, "actions": ""})
                continue

            remaining = n - i
            inst_deadline = time.time() + max(0.5, (deadline - time.time()) / remaining)

            try:
                sol = solve_astar(env, inst["state"], solved_k, v_fn, inst_deadline)
            except Exception as e:
                print(f"  {iid} failed: {repr(e)}")
                sol = None

            actions = sol or []
            writer.writerow({"instance_id": iid, "actions": " ".join(actions)})
            if actions:
                solved += 1

            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{n} solved={solved} elapsed={time.time()-start:.0f}s")

    print(f"final: solved {solved}/{n}, time {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
