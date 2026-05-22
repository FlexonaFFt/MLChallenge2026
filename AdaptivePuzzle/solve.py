import argparse
import csv
import json
import os

import torch

import gym
from core.agent import Agent


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input_states.jsonl")
    parser.add_argument("--output", default="output_actions.csv")
    parser.add_argument("--time_limit", type=int,
                        default=int(os.environ.get("SOLVE_TIME_LIMIT", 1500)))
    args = parser.parse_args()

    torch.set_num_threads(min(8, os.cpu_count() or 1))

    env = gym.make_env()
    instances = load_jsonl(args.input)
    print(f"loaded {len(instances)} instances")

    agent = Agent(env)
    results = agent.solve_all(instances, args.time_limit)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance_id", "actions"])
        writer.writeheader()
        for iid, actions in results:
            writer.writerow({"instance_id": iid, "actions": " ".join(actions)})


if __name__ == "__main__":
    main()
