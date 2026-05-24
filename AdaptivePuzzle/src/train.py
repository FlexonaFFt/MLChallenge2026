"""Hidden-game adaptation: spend the train budget building a BACKWARD DISTANCE
TABLE (BFS outward from the goal), storing exact distance-to-goal for every
reachable state within the budget. solve.py then forward-searches into the
table and descends it to the goal -> optimal solutions that crack deep
instances. Generic: uses only the gym interface.
"""

import argparse
import json
import os
import time

import numpy as np

import gym
import search


TIME_LIMIT_DEFAULT = 50 * 60
SAFETY_MARGIN = 90          # leave time to serialize a large table
TABLE_PATH = "btable.npz"
META_PATH = "meta.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_limit", type=int,
                        default=int(os.environ.get("TRAIN_TIME_LIMIT", TIME_LIMIT_DEFAULT)))
    parser.add_argument("--max_states", type=int,
                        default=int(os.environ.get("TABLE_MAX_STATES", 40_000_000)))
    args = parser.parse_args()

    start = time.time()
    deadline = start + args.time_limit - SAFETY_MARGIN

    env = gym.make_env()
    env_id = getattr(gym, "ENV_ID", "unknown")
    env.reset()
    solved_state = env.get_state()
    print(f"env_id={env_id}  building backward table (budget {args.time_limit}s, "
          f"max_states {args.max_states})...")

    t0 = time.time()
    table, depth = search.build_backward_table(
        env, solved_state, deadline, max_states=args.max_states,
    )
    print(f"  table: {len(table)} states, max depth ~{depth}, {time.time()-t0:.0f}s")

    t1 = time.time()
    search.save_backward_table(table, TABLE_PATH)
    print(f"  saved {TABLE_PATH} in {time.time()-t1:.0f}s")

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "env_id": env_id,
            "method": "backward_table",
            "table_states": int(len(table)),
            "table_depth": int(depth),
            "wall_time_sec": time.time() - start,
        }, f, indent=2)

    print(f"train.py done in {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
