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
    build_packed_engine, packed_bidirectional, _flatten_state,
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
    # Packed-state engine: env.step-free search for state-independent
    # permutation/toggle puzzles (~20-60x node throughput). None on blank/slide
    # puzzles -> existing env path runs unchanged (zero regression).
    try:
        _W["packed"] = build_packed_engine(env, _W["solved_state"])
    except Exception:
        _W["packed"] = None


def _verify_solution(env, start_state, actions) -> bool:
    """Replay `actions` from `start_state` through the env; True iff it solves.
    Cheap safety net for the packed-engine path on unseen hidden puzzles."""
    try:
        env.set_state(start_state)
        for a in actions:
            if a not in env.valid_actions():
                return False
            env.step(a)
        return bool(env.is_solved())
    except Exception:
        return False


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
        # Phase 0.8: packed-state bidirectional BFS (env.step-free, ~20-60x
        # faster -> reaches far deeper). Optimal. Runs on state-independent
        # permutation/toggle puzzles; None otherwise.
        if sol is None and _W["packed"] is not None:
            bidi_deadline = now + 0.4 * (inst_deadline - now)
            try:
                start_vec = _flatten_state(inst["state"]).astype(np.int16)
                cand = packed_bidirectional(_W["packed"], start_vec, bidi_deadline)
                # verify by env replay (guards against any flatten/order mismatch
                # on an unseen hidden puzzle): only accept a genuinely solving path
                if cand is not None and _verify_solution(
                    _W["env"], inst["state"], cand):
                    sol = cand
            except Exception:
                sol = None
        # Phase 1: bidirectional BFS (shortest path -> best ratio). Give it most
        # of the per-instance budget, then fall back to V-guided A*. Skipped when
        # the packed engine already ran (it supersedes env-bidi).
        if sol is None and _W["packed"] is None:
            bidi_deadline = now + 0.4 * (inst_deadline - now)
            sol = solve_bidirectional(
                _W["env"], inst["state"], _W["solved_state"], bidi_deadline
            )
        # Split the time that is left after the table/bidi phases between an
        # OPTIMAL A* (weight 1 -> shortest path -> best ratio) and a GREEDY
        # best-effort phase. Deadlines are measured from the CURRENT time, not
        # the stale per-instance `now`, so each gets a real, fair slice.
        if sol is None:
            rem = time.time()
            opt_deadline = rem + 0.5 * (inst_deadline - rem)
            sol = solve_astar(
                _W["env"], inst["state"], _W["solved_state"], _W["cheap_h"],
                opt_deadline, weight=_W["w"],
            )
        # Best-effort: an empty row scores 0; any VALID path scores baseline/moves
        # (> 0, uncapped) even if long. A moderately greedy A* (weight ~3, the
        # measured sweet spot) reaches the goal/table on hard deep instances that
        # optimal A* cannot in budget. Only runs when every optimal phase missed,
        # so it never shortens/worsens an already-solved instance.
        if not sol:
            gw = float(os.environ.get("GREEDY_WEIGHT", "3.0"))
            if _W["table"] is not None:
                sol = solve_with_table(
                    _W["env"], inst["state"], _W["solved_state"], _W["table"],
                    _W["cheap_h"], inst_deadline, weight=gw,
                )
            if not sol:
                sol = solve_astar(
                    _W["env"], inst["state"], _W["solved_state"], _W["cheap_h"],
                    inst_deadline, weight=gw,
                )
    except Exception as e:
        print(f"  {iid} failed: {repr(e)}")
        sol = None
    return iid, list(sol or [])


def _run_pool(instances, budget, pass_deadline, workers, start):
    """Run _solve_one over `instances` with per-instance `budget`, collecting
    results until `pass_deadline` (HARD WALL: stop collecting and force-kill the
    pool at the deadline no matter what workers are doing -> never overruns the
    grader's hard kill). Returns {iid: actions}. Sequential fallback if the pool
    can't start."""
    results: dict = {}
    n = len(instances)
    if n == 0:
        return results
    if workers > 1 and n > 1:
        try:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")  # safe with torch on every platform
            pool = ctx.Pool(
                processes=workers,
                initializer=_init_worker,
                initargs=(budget, pass_deadline),
            )
            try:
                it = pool.imap_unordered(_solve_one, instances, chunksize=1)
                done = 0
                while done < n:
                    remaining = pass_deadline - time.time()
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
            return results
        except Exception as e:
            print(f"parallel pool failed ({e!r}); falling back to sequential")
            results = {}
    _init_worker(budget, pass_deadline)
    for inst in instances:
        if time.time() >= pass_deadline:
            break
        iid, acts = _solve_one(inst)
        results[iid] = acts
    return results


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
    # Memory guard. The backward table is loaded via mmap, so it lives in the
    # shared OS page cache and is counted ONCE regardless of worker count — it is
    # NOT multiplied by the number of workers. Per-worker PRIVATE memory is the
    # A* open/closed sets, bounded by the node caps (~1GB/worker worst case). So
    # 8 workers + a few-GB table stays well under the 32GB CI limit, and using
    # all 8 cores ~4x's the compute on the hard instances that actually miss the
    # table. We only step down for a truly huge table. Override via WORKERS env.
    if os.environ.get("WORKERS"):
        workers = max(1, int(os.environ["WORKERS"]))
    elif os.path.exists(TABLE_PATH):
        try:
            tbl_gb = os.path.getsize(TABLE_PATH) / (1024 ** 3)
            if tbl_gb > 12.0:
                workers = min(workers, 4)
            print(f"backward table size: {tbl_gb:.2f} GB -> workers={workers}")
        except Exception:
            pass
    # Two-pass budgeting. The backward table solves most instances in ms, so a
    # single fixed per-instance budget wastes the freed time on idle workers.
    # PASS 1 gives every instance a modest slice (breadth, capped to 60% of the
    # window so pass 2 always has time). PASS 2 hands ALL the leftover time to
    # the few still-unsolved instances (depth) — exactly the hard, high-value
    # deep ones where a longer greedy search converts a 0 into points.
    window = max(0.0, deadline - time.time())
    budget1 = max(0.5, workers * window / n) if n else 0.5
    pass1_deadline = time.time() + 0.6 * window
    print(f"workers={workers} pass1 budget={budget1:.1f}s window={window:.0f}s")
    results = _run_pool(instances, budget1, pass1_deadline, workers, start)

    unsolved = [inst for inst in instances if not results.get(inst["instance_id"])]
    rem = deadline - time.time()
    if unsolved and rem > 1.0:
        # Fair division of the remaining time across the unsolved instances.
        budget2 = max(budget1, workers * rem / len(unsolved))
        print(f"  pass2: {len(unsolved)} unsolved, budget={budget2:.1f}s, rem={rem:.0f}s")
        results.update(_run_pool(unsolved, budget2, deadline, workers, start))

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
