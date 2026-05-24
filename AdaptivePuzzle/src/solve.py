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
from search import (
    solve_astar, solve_bidirectional, build_linear_solver, build_manhattan_h,
    make_misplaced_h, solve_with_table, BackwardTable,
)


TABLE_PATH = "btable.npz"


TIME_LIMIT_DEFAULT = 25 * 60
SAFETY_MARGIN = 30
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


# ---------------------------------------------------------------------------
# Worker setup (used both by the multiprocessing pool and the sequential
# fallback). Each worker owns its own env + model so nothing is shared across
# processes. torch is pinned to 1 thread per worker to avoid oversubscription.
# ---------------------------------------------------------------------------

_W: dict = {}


def _init_worker(budget: float, global_deadline: float):
    torch.set_num_threads(1)
    env = gym.make_env()
    env.reset()
    model = load_model()
    _W["env"] = env
    _W["solved_state"] = common.to_jsonable(env.get_state())
    _W["v_fn"] = make_v_fn(env, model)
    # Cheap misplaced-count heuristic: faster than the NN and empirically a
    # better A* guide on rotation/permutation puzzles. Used as the general
    # A* heuristic (toggle->GF2 and sliding->Manhattan paths are unaffected).
    _W["cheap_h"] = make_misplaced_h(env)
    _W["budget"] = budget
    _W["gdl"] = global_deadline
    _W["w"] = float(os.environ.get("ASTAR_WEIGHT", "1.0"))
    # Detect a GF(2)/XOR (Lights-Out-like) puzzle once; if so we solve those
    # instances exactly and instantly instead of searching.
    _W["lin"] = build_linear_solver(env)
    # Detect a single-blank sliding puzzle once; if so use an admissible
    # graph-Manhattan heuristic in A* (optimal, strong) instead of the net.
    _W["slide_h"] = None if _W["lin"] is not None else build_manhattan_h(env)
    # Backward distance table built in train.py (BFS from goal). If present,
    # forward-search into it + descend -> optimal solutions, cracks deep ones.
    _W["table"] = BackwardTable.load(TABLE_PATH)


def _solve_one(inst):
    iid = inst["instance_id"]
    now = time.time()
    if now >= _W["gdl"]:
        return iid, []
    inst_deadline = min(now + _W["budget"], _W["gdl"])
    try:
        # Phase 0: exact GF(2) solve for linear (Lights-Out-like) puzzles.
        sol = None
        if _W["lin"] is not None:
            sol = _W["lin"].solve(_W["env"], inst["state"])
        # Phase 0.5: sliding puzzle -> A* with admissible graph-Manhattan
        # heuristic (optimal, far stronger than the learned net here).
        if sol is None and _W["slide_h"] is not None:
            sol = solve_astar(
                _W["env"], inst["state"], _W["solved_state"], _W["slide_h"],
                inst_deadline, max_nodes=2_000_000, weight=1.0,
            )
        # Phase 0.7: forward-search into the precomputed backward table, then
        # descend it to the goal (optimal; cracks deep instances). Primary for
        # rotation/permutation puzzles when a table was built in train.
        if sol is None and _W["table"] is not None:
            # half the remaining budget; if the table misses, bidi/A* still run
            table_deadline = now + 0.5 * (inst_deadline - now)
            sol = solve_with_table(
                _W["env"], inst["state"], _W["solved_state"], _W["table"],
                _W["cheap_h"], table_deadline,
            )
        # Phase 1: bidirectional BFS (shortest path -> best ratio). Give it most
        # of the per-instance budget, then fall back to V-guided A*.
        if sol is None:
            bidi_deadline = now + 0.6 * (inst_deadline - now)
            sol = solve_bidirectional(
                _W["env"], inst["state"], _W["solved_state"], bidi_deadline
            )
        if sol is None:
            sol = solve_astar(
                _W["env"], inst["state"], _W["solved_state"], _W["cheap_h"], inst_deadline,
                weight=_W["w"],
            )
    except Exception as e:
        print(f"  {iid} failed: {repr(e)}")
        sol = None
    return iid, list(sol or [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input_states.jsonl")
    parser.add_argument("--output", default="output_actions.csv")
    parser.add_argument("--time_limit", type=int,
                        default=int(os.environ.get("SOLVE_TIME_LIMIT", TIME_LIMIT_DEFAULT)))
    args = parser.parse_args()

    start = time.time()
    deadline = start + args.time_limit - SAFETY_MARGIN

    instances = load_jsonl(args.input)
    n = len(instances)
    print(f"loaded {n} instances")
    print(f"model present: {os.path.exists(MODEL_PATH)}")

    # Fair per-instance compute budget: with W workers over the wall window,
    # each instance may use up to W * window / n seconds of CPU time.
    workers = max(1, min(8, os.cpu_count() or 1))
    window = max(0.0, deadline - time.time())
    budget = max(0.5, workers * window / n) if n else 0.5
    print(f"workers={workers} per-instance budget={budget:.1f}s")

    results: dict = {}
    ran_parallel = False

    if workers > 1 and n > 1:
        try:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")  # safe with torch on every platform
            pool = ctx.Pool(
                processes=workers,
                initializer=_init_worker,
                initargs=(budget, deadline),
            )
            # HARD WALL GUARD: stop collecting at `deadline` no matter what the
            # workers are doing, then force-kill the pool. This guarantees
            # solve.py finishes before the grader's hard time-limit kill
            # (SAFETY_MARGIN covers terminate + CSV write). Without this the
            # pool teardown can hang on a stuck worker -> TL.
            try:
                it = pool.imap_unordered(_solve_one, instances, chunksize=1)
                done = 0
                while done < n:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        print("  wall deadline reached; stopping collection")
                        break
                    try:
                        iid, acts = it.next(timeout=remaining)
                    except mp.TimeoutError:
                        print("  wall deadline reached (timeout); stopping collection")
                        break
                    except StopIteration:
                        break
                    results[iid] = acts
                    done += 1
                    if done % 50 == 0:
                        nsolved = sum(1 for v in results.values() if v)
                        print(f"  {done}/{n} solved={nsolved} elapsed={time.time()-start:.0f}s")
            finally:
                pool.terminate()
                pool.join()
            ran_parallel = True
        except Exception as e:
            print(f"parallel pool failed ({e!r}); falling back to sequential")
            results = {}

    if not ran_parallel:
        _init_worker(budget, deadline)
        for inst in instances:
            if time.time() >= deadline:
                break
            iid, acts = _solve_one(inst)
            results[iid] = acts

    solved = sum(1 for v in results.values() if v)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance_id", "actions"])
        writer.writeheader()
        for inst in instances:
            iid = inst["instance_id"]
            writer.writerow({"instance_id": iid, "actions": " ".join(results.get(iid, []))})

    print(f"final: solved {solved}/{n}, time {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
