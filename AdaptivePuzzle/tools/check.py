import argparse
import csv
import json
import math
import os
import time
from typing import Dict, List

import numpy as np

import gym


TIME_LIMIT = 600


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_verdict(text):
    with open("verdict.txt", "w", encoding="utf-8") as f:
        f.write(text)


def write_score(obj):
    with open("score.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_submission_csv(path) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    sub = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("submission csv has no header")

        if "instance_id" not in reader.fieldnames or "actions" not in reader.fieldnames:
            raise ValueError("submission.csv must contain columns: instance_id, actions")

        for row in reader:
            iid = (row.get("instance_id") or "").strip()
            if not iid:
                continue

            actions_str = (row.get("actions") or "").strip()
            actions = actions_str.split() if actions_str else []
            sub[iid] = actions

    return sub


def validate_solution(env, initial_state, actions, max_moves):
    try:
        env.set_state(initial_state)
    except Exception as e:
        return {
            "valid": False,
            "solved": False,
            "moves": len(actions),
            "error": f"set_state_error: {repr(e)}",
        }

    if len(actions) > max_moves:
        return {
            "valid": False,
            "solved": False,
            "moves": len(actions),
            "error": f"too_many_moves: {len(actions)} > {max_moves}",
        }

    for t, a in enumerate(actions):
        try:
            valid = env.valid_actions()
            if a not in valid:
                return {
                    "valid": False,
                    "solved": False,
                    "moves": len(actions),
                    "error": f"invalid_action at {t}: {a}",
                }
            env.step(a)
        except Exception as e:
            return {
                "valid": False,
                "solved": False,
                "moves": len(actions),
                "error": f"exception at {t}: {repr(e)}",
            }

    return {
        "valid": True,
        "solved": env.is_solved(),
        "moves": len(actions),
        "error": None,
    }


def score_instance(baseline, result, cap):
    if not result["valid"] or not result["solved"]:
        return 0.0, math.inf

    moves = result["moves"]

    if moves == 0:
        if baseline == 0:
            return 1.0, 1.0
        return 0.0, math.inf

    raw = baseline / moves
    score = min(cap, raw)
    ratio = moves / baseline if baseline > 0 else math.inf

    return score, ratio


def aggregate(details):
    if not details:
        return {
            "num_instances": 0,
            "mean_score": 0.0,
            "solved_rate": 0.0,
            "valid_rate": 0.0,
            "mean_ratio_on_solved": None,
        }

    n = len(details)
    solved = [x for x in details if x["solved"]]
    valid = [x for x in details if x["valid"]]
    ratios = [x["ratio"] for x in solved if math.isfinite(x["ratio"])]

    return {
        "score": float(np.mean([x["score"] for x in details])),
    }


def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input_states.jsonl")
    parser.add_argument("--submission", default="output_actions.csv")
    parser.add_argument("--score_name", default="unknown")
    parser.add_argument("--max_factor", type=float, default=4.0)
    parser.add_argument("--max_add", type=int, default=20)
    parser.add_argument("--score_cap", type=float, default=2.0)
    parser.add_argument("--details_output", default="details.jsonl")
    args = parser.parse_args()

    try:
        if time.time() - start > TIME_LIMIT:
            write_verdict("time limit exceeded")
            write_score({})
            return

        instances = load_jsonl(args.input)
        submission = load_submission_csv(args.submission)
        env = gym.make_env()

        details = []

        for inst in instances:
            if time.time() - start > TIME_LIMIT:
                write_verdict("time limit exceeded")
                write_score({})
                return

            iid = inst["instance_id"]
            actions = submission.get(iid, [])

            baseline = int(inst.get("baseline_length", 1000))
            max_moves = int(args.max_factor * baseline + args.max_add)

            result = validate_solution(env, inst["state"], actions, max_moves=max_moves)
            score, ratio = score_instance(baseline, result, cap=args.score_cap)

            details.append({
                "instance_id": iid,
                "baseline_length": baseline,
                "moves": result["moves"],
                "valid": result["valid"],
                "solved": result["solved"],
                "score": score,
                "ratio": ratio,
                "error": result["error"],
            })

        score_obj = {
            "env_id": gym.ENV_ID,
            "score_name": args.score_name,
            **aggregate(details),
        }

        write_score(score_obj)

        with open(args.details_output, "w", encoding="utf-8") as f:
            for row in details:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        write_verdict("success")

    except Exception as e:
        write_verdict(f"error: {repr(e)}")
        write_score({})


if __name__ == "__main__":
    main()
