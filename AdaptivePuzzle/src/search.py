"""Search: bidirectional BFS (meet-in-the-middle) + forward A* with V heuristic."""

import heapq
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from common import state_key, to_jsonable


def _key(jsonable_state) -> str:
    """Canonical key for an already-jsonable state (matches common.state_key)."""
    return json.dumps(jsonable_state, sort_keys=True)


def solve_bidirectional(
    env,
    initial_state: Any,
    solved_state: Any,
    deadline: float,
    max_nodes: int = 150_000,
) -> Optional[List[str]]:
    """Meet-in-the-middle BFS. Returns a shortest action list or None.

    Forward wave expands with env.step. Backward wave expands predecessors:
    for state s and action b valid at s, s' = step(s, b) is a predecessor of s,
    connected by the forward action a = inverse_action(b) (a is valid at s' and
    step(s', a) == s by reversibility). The two waves meet on a shared state key.
    """
    start = to_jsonable(initial_state)
    goal = to_jsonable(solved_state)
    sk = _key(start)
    gk = _key(goal)
    if sk == gk:
        return []

    # key -> (parent_key, forward action leading INTO this key from the start side)
    fwd: Dict[str, Tuple[Optional[str], Optional[str]]] = {sk: (None, None)}
    # key -> (next_key toward goal, forward action FROM this key toward goal)
    bwd: Dict[str, Tuple[Optional[str], Optional[str]]] = {gk: (None, None)}

    f_frontier: List[Tuple[str, Any]] = [(sk, start)]
    b_frontier: List[Tuple[str, Any]] = [(gk, goal)]
    nodes = 2

    while f_frontier and b_frontier:
        if time.time() >= deadline or nodes >= max_nodes:
            return None

        # Always expand the smaller frontier to keep the waves balanced.
        expand_forward = len(f_frontier) <= len(b_frontier)
        frontier = f_frontier if expand_forward else b_frontier
        this_side = fwd if expand_forward else bwd
        other_side = bwd if expand_forward else fwd
        next_frontier: List[Tuple[str, Any]] = []

        for key, state in frontier:
            if time.time() >= deadline:
                return None
            try:
                env.set_state(state)
                valid = env.valid_actions()
            except Exception:
                continue
            for mv in valid:
                try:
                    env.set_state(state)
                    env.step(mv)
                    ns = to_jsonable(env.get_state())
                except Exception:
                    continue
                nk = _key(ns)
                if nk in this_side:
                    continue
                if expand_forward:
                    # forward edge: key --mv--> nk
                    this_side[nk] = (key, mv)
                else:
                    # backward: nk is a predecessor of `key`; forward edge nk --a--> key
                    try:
                        a = env.inverse_action(mv)
                    except Exception:
                        continue
                    this_side[nk] = (key, a)
                nodes += 1
                if nk in other_side:
                    return _stitch(nk, fwd, bwd)
                next_frontier.append((nk, ns))
                if nodes >= max_nodes:
                    return None

        if expand_forward:
            f_frontier = next_frontier
        else:
            b_frontier = next_frontier

    return None


def _stitch(meet_key: str, fwd, bwd) -> List[str]:
    """Build start -> meet (from fwd) + meet -> goal (from bwd), both forward."""
    head: List[str] = []
    cur = meet_key
    while True:
        pk, a = fwd.get(cur, (None, None))
        if pk is None or a is None:
            break
        head.append(a)
        cur = pk
    head.reverse()

    tail: List[str] = []
    cur = meet_key
    while True:
        nk, a = bwd.get(cur, (None, None))
        if nk is None or a is None:
            break
        tail.append(a)
        cur = nk

    return head + tail


def solve_astar(
    env,
    initial_state: Any,
    solved_key: str,
    v_fn: Optional[Callable[[List[Any]], np.ndarray]],
    deadline: float,
    max_nodes: int = 50_000,
    expand_batch: int = 32,
) -> Optional[List[str]]:
    """A* with f = g + V(s). Returns action list or None."""
    start = to_jsonable(initial_state)
    start_k = state_key(start)
    if start_k == solved_key:
        return []

    parents: Dict[str, Tuple[Optional[str], Optional[str]]] = {start_k: (None, None)}
    g_score: Dict[str, int] = {start_k: 0}

    h0 = float(v_fn([start])[0]) if v_fn else 0.0
    counter = 0
    open_heap = [(h0, counter, 0, start_k, start)]
    expanded = 0

    while open_heap:
        if time.time() >= deadline or expanded >= max_nodes:
            return None

        # Pop a batch of nodes to expand.
        batch = []
        while open_heap and len(batch) < expand_batch:
            batch.append(heapq.heappop(open_heap))

        # Expand: collect unique children.
        children = []  # (child_key, child_state, parent_key, action, g)
        seen = set()
        for _f, _c, g, sk, state in batch:
            if sk == solved_key:
                return _reconstruct(sk, parents)
            try:
                env.set_state(state)
                valid = env.valid_actions()
            except Exception:
                continue
            for a in valid:
                try:
                    env.set_state(state)
                    env.step(a)
                    ns = to_jsonable(env.get_state())
                except Exception:
                    continue
                nsk = state_key(ns)
                ng = g + 1
                if g_score.get(nsk, 1 << 30) <= ng or nsk in seen:
                    continue
                seen.add(nsk)
                children.append((nsk, ns, sk, a, ng))

        if not children:
            continue

        # Goal short-circuit on any child.
        for nsk, ns, psk, a, ng in children:
            if nsk == solved_key:
                parents[nsk] = (psk, a)
                return _reconstruct(nsk, parents)

        # Batched V over children.
        h_vals = v_fn([c[1] for c in children]) if v_fn else np.zeros(len(children), np.float32)

        for (nsk, ns, psk, a, ng), h in zip(children, h_vals):
            parents[nsk] = (psk, a)
            g_score[nsk] = ng
            counter += 1
            heapq.heappush(open_heap, (ng + float(h), counter, ng, nsk, ns))
            expanded += 1
            if expanded >= max_nodes:
                break

    return None


def _reconstruct(end_k: str, parents) -> List[str]:
    actions = []
    cur = end_k
    while True:
        pk, a = parents.get(cur, (None, None))
        if pk is None or a is None:
            break
        actions.append(a)
        cur = pk
    actions.reverse()
    return actions
