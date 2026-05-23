"""Search: GF(2) linear solver + bidirectional BFS + forward A* with V heuristic."""

import heapq
import json
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from common import state_key, to_jsonable


def _key(jsonable_state) -> str:
    """Canonical key for an already-jsonable state (matches common.state_key)."""
    return json.dumps(jsonable_state, sort_keys=True)


# =====================================================================
# GF(2) linear solver (Lights-Out-like / TOGGLE puzzles)
# =====================================================================
#
# A puzzle is "linear over GF(2)" when every action is an involution that
# flips a fixed (state-independent) set of binary cells. Then pressing a set
# of actions XORs their masks, order-independent. To reach the goal we solve
#   XOR of pressed action masks  ==  (content XOR target)   over GF(2),
# which Gaussian elimination does exactly and instantly.


def _changed_mask(before: List[int], after: List[int]) -> int:
    m = 0
    for i, (b, a) in enumerate(zip(before, after)):
        if b != a:
            m |= (1 << i)
    return m


class LinearGF2Solver:
    """Precomputed GF(2) basis over the action masks; solves any instance fast."""

    def __init__(self, actions: List[str], masks: List[int]):
        self.actions = actions
        # Reduced basis: list of (vec, combo) where `vec` is a mask in row-echelon
        # form (unique leading bit) and `combo` is the action-index bitmask that
        # XORs to `vec`.
        basis: List[Tuple[int, int]] = []
        for i, m in enumerate(masks):
            vec, combo = m, (1 << i)
            for bvec, bcombo in basis:
                lead = bvec.bit_length() - 1
                if (vec >> lead) & 1:
                    vec ^= bvec
                    combo ^= bcombo
            if vec:
                basis.append((vec, combo))
                basis.sort(key=lambda t: t[0].bit_length(), reverse=True)
        self.basis = basis

    def solve_target(self, b: int) -> Optional[List[str]]:
        """Return actions whose masks XOR to b, or None if unsolvable."""
        x, combo = b, 0
        for bvec, bcombo in self.basis:
            lead = bvec.bit_length() - 1
            if (x >> lead) & 1:
                x ^= bvec
                combo ^= bcombo
        if x != 0:
            return None
        return [self.actions[i] for i in range(len(self.actions)) if (combo >> i) & 1]

    def solve(self, env, state) -> Optional[List[str]]:
        enc = env.encode_state(state)
        content = enc["content_values"]
        target = enc["target_values"]
        if len(content) != len(target):
            return None
        b = _changed_mask(target, content)  # bits where current != goal
        return self.solve_target(b)


def build_linear_solver(env) -> Optional["LinearGF2Solver"]:
    """Detect a GF(2)/XOR puzzle and precompute a solver, else return None.

    Conservative: requires constant action set, binary cell values, each action
    an involution with a state-independent flip mask. Permutation puzzles (values
    not in {0,1}) are rejected immediately, so this never misfires on them.
    """
    try:
        env.reset()
        enc0 = env.encode_state()
        content0 = enc0["content_values"]
        target0 = enc0["target_values"]
        n = len(content0)
        if n == 0 or n > 4096:
            return None
        if any(v not in (0, 1) for v in content0):
            return None
        if any(v not in (0, 1) for v in target0):
            return None

        actions = list(env.valid_actions())
        if not actions:
            return None

        # Action masks from the solved state + involution check.
        masks: List[int] = []
        for a in actions:
            env.reset()
            before = env.encode_state()["content_values"]
            env.step(a)
            after = env.encode_state()["content_values"]
            masks.append(_changed_mask(before, after))
            env.step(a)  # involution: applying twice must return
            if env.encode_state()["content_values"] != before:
                return None

        # Verify action set is state-independent and masks are state-independent
        # from a scrambled state.
        env.reset()
        rng = random.Random(0)
        for _ in range(12):
            valid = env.valid_actions()
            if not valid:
                break
            env.step(rng.choice(valid))
        if list(env.valid_actions()) != actions:
            return None
        scr_vals = env.encode_state()["content_values"]
        if any(v not in (0, 1) for v in scr_vals):
            return None
        for a, m in zip(actions, masks):
            st = env.get_state()
            before = env.encode_state()["content_values"]
            env.step(a)
            after = env.encode_state()["content_values"]
            if _changed_mask(before, after) != m:
                return None
            env.set_state(st)

        env.reset()
        return LinearGF2Solver(actions, masks)
    except Exception:
        return None


# Content-type codes (mirror gym.py): EMPTY=0, NUM=1, COLOR=2, MASKED=3
_C_EMPTY = 0
_C_NUM = 1


def build_manhattan_h(env) -> Optional[Callable[[List[Any]], np.ndarray]]:
    """Detect a single-blank SWAP (sliding) puzzle and return an admissible
    graph-Manhattan heuristic h(states)->np.ndarray, else None.

    Signature: exactly one EMPTY cell, distinct NUM pieces, each move swaps the
    blank with a neighbour. Then each move displaces exactly one piece by one
    edge of the cell-adjacency graph, so summing each piece's graph distance to
    its goal cell is admissible -> A* finds OPTIMAL (shortest) solutions.
    Returns None for anything else (rotation/color/multi-blank) -> no misfire.
    """
    try:
        import collections
        env.reset()
        enc = env.encode_state()
        ct, cv = enc["content_types"], enc["content_values"]
        tt, tv = enc["target_types"], enc["target_values"]
        N = len(ct)
        if N == 0 or N > 4096:
            return None
        if sum(1 for t in ct if t == _C_EMPTY) != 1:
            return None
        piece_vals = [cv[i] for i in range(N) if ct[i] == _C_NUM]
        if not piece_vals or len(set(piece_vals)) != len(piece_vals):
            return None
        goal_cell = {tv[i]: i for i in range(N) if tt[i] == _C_NUM}
        if any(v not in goal_cell for v in piece_vals):
            return None

        def blank_of(e):
            for i, t in enumerate(e["content_types"]):
                if t == _C_EMPTY:
                    return i
            return None

        # Discover cell adjacency: each move shifts the blank to a neighbour.
        adj = collections.defaultdict(set)
        rng = random.Random(0)
        env.reset()
        before = env.encode_state()["content_values"]
        cur = blank_of(env.encode_state())
        for _ in range(4000):
            valid = env.valid_actions()
            if not valid:
                break
            env.step(rng.choice(valid))
            after = env.encode_state()["content_values"]
            nb = blank_of(env.encode_state())
            if nb is None:
                return None
            # STRICT: a genuine slide changes EXACTLY two cells (blank + one
            # piece swap) and the blank is one of them. Anything else (a move
            # that shifts several pieces) makes graph-Manhattan inadmissible,
            # so we bail and let the generic bidi/A* path handle this puzzle.
            changed = [i for i in range(N) if before[i] != after[i]]
            if len(changed) != 2 or cur not in changed or nb not in changed:
                return None
            adj[cur].add(nb)
            adj[nb].add(cur)
            before = after
            cur = nb
        if not adj:
            return None

        # All-pairs shortest path on the (small) cell graph.
        dist = [[-1] * N for _ in range(N)]
        for src in range(N):
            d = dist[src]
            d[src] = 0
            q = collections.deque([src])
            while q:
                u = q.popleft()
                for w in adj[u]:
                    if d[w] < 0:
                        d[w] = d[u] + 1
                        q.append(w)

        gc = dict(goal_cell)
        env.reset()

        def h_fn(states):
            out = np.empty(len(states), dtype=np.float32)
            for k, s in enumerate(states):
                e = env.encode_state(s)
                ec, ev = e["content_types"], e["content_values"]
                tot = 0
                for i in range(N):
                    if ec[i] == _C_NUM:
                        g = gc.get(ev[i])
                        if g is not None:
                            dd = dist[i][g]
                            if dd > 0:
                                tot += dd
                out[k] = tot
            return out

        return h_fn
    except Exception:
        return None


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
    weight: float = 1.0,
) -> Optional[List[str]]:
    """Weighted A* with f = g + weight * V(s). weight>1 trades solution length
    for speed/coverage (greedier toward the goal). Returns action list or None."""
    start = to_jsonable(initial_state)
    start_k = state_key(start)
    if start_k == solved_key:
        return []

    parents: Dict[str, Tuple[Optional[str], Optional[str]]] = {start_k: (None, None)}
    g_score: Dict[str, int] = {start_k: 0}

    h0 = float(v_fn([start])[0]) if v_fn else 0.0
    counter = 0
    open_heap = [(weight * h0, counter, 0, start_k, start)]
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
            heapq.heappush(open_heap, (ng + weight * float(h), counter, ng, nsk, ns))
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
