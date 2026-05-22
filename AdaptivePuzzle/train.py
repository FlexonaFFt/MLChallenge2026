import argparse
import os
import random

import numpy as np
import torch

import gym
from core.agent import Agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_limit", type=int,
                        default=int(os.environ.get("TRAIN_TIME_LIMIT", 3000)))
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

    env = gym.make_env()
    print(f"env_id={gym.ENV_ID}")

    agent = Agent(env)
    agent.train(
        time_limit=args.time_limit,
        seed=args.seed,
        num_pairs=args.num_pairs,
        max_walk=args.max_walk,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
