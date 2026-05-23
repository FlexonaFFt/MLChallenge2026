import argparse
import json
import random

import numpy as np

import gym


def state_key(state):
    if isinstance(state, dict):
        parts = []
        for k in sorted(state.keys()):
            v = state[k]
            if isinstance(v, list):
                parts.append(json.dumps(v, sort_keys=True).encode())
            elif isinstance(v, np.ndarray):
                parts.append(v.tobytes())
            else:
                parts.append(str(v).encode())
        return b"||".join(parts)
    return json.dumps(state, sort_keys=True).encode()


def generate_instances(num_instances, scramble_lengths, seed):
    rng = random.Random(seed)
    env = gym.make_env()
    rows = []
    seen = set()

    i = 0
    attempts = 0
    max_attempts = num_instances * 1000

    while len(rows) < num_instances:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Too many attempts while generating unique states")

        L = rng.choice(scramble_lengths)
        local_seed = rng.randint(0, 10**9)

        state, actions = env.scramble(length=L, seed=local_seed, no_backtrack=True)

        if env.is_solved():
            continue

        js_state = gym.to_jsonable(state)
        key = state_key(js_state)
        if key in seen:
            continue
        seen.add(key)

        rows.append({
            "instance_id": f"{gym.ENV_ID}_{i:06d}",
            "env_id": gym.ENV_ID,
            "state": js_state,
            "solved_state": gym.to_jsonable(env.solved_state()),
            "baseline_length": int(len(actions)),
        })
        i += 1

    return rows


def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_instances", type=int,
                        default=getattr(gym, "NUM_INSTANCES_DEFAULT", 1000))
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--scramble_lengths", default="")
    parser.add_argument("--public_output", default="input_states.jsonl")
    args = parser.parse_args()

    if args.scramble_lengths.strip():
        lengths = [int(x) for x in args.scramble_lengths.split(",")]
    else:
        lengths = list(getattr(gym, "SCRAMBLE_LENGTHS_DEFAULT", [20, 30, 40]))

    rows = generate_instances(
        num_instances=args.num_instances,
        scramble_lengths=lengths,
        seed=args.seed,
    )

    save_jsonl(rows, args.public_output)

    print(f"env_id={gym.ENV_ID}")
    print(f"instances={len(rows)} -> {args.public_output}")


if __name__ == "__main__":
    main()
