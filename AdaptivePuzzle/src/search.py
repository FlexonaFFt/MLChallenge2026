"""Search: GF(2) linear solver + bidirectional BFS + forward A* with V heuristic."""

import hashlib
import heapq
import json
import os
import random
import resource
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from common import state_key, to_jsonable


def _rss_bytes() -> int:
    """Current process peak RSS in bytes (ru_maxrss is KB on Linux, bytes on macOS)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru if sys.platform == "darwin" else ru * 1024


def state_hash(s) -> int:
    """Stable 64-bit hash of a native env state (consistent ACROSS processes,
    unlike Python's randomized hash). Used as the key in the backward table."""
    return int.from_bytes(hashlib.blake2b(_fast_key(s), digest_size=8).digest(), "big")


def _key(jsonable_state) -> str:
    """Canonical key for an already-jsonable state (matches common.state_key)."""
    return json.dumps(jsonable_state, sort_keys=True)


def _fast_key(s) -> bytes:
    """Fast, unique, hashable key for a NATIVE env state (ndarray / dict / int).
    ~10-50x cheaper than json.dumps. Always derive from env.get_state() so the
    representation is consistent across start/goal/children."""
    t = type(s)
    if t is np.ndarray:
        return s.tobytes()
    if t is dict:
        return b"|".join(k.encode() + b"=" + _fast_key(v) for k, v in sorted(s.items()))
    if t is list or t is tuple:
        return b"[" + b",".join(_fast_key(x) for x in s) + b"]"
    return repr(s).encode()


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


def _linear_conflicts(groups: dict) -> int:
    """Classic (admissible) linear-conflict count: per line, the minimum number
    of tiles to remove so the rest are in goal order, via greedy removal of the
    most-conflicting tile. Caller adds 2 per removal. `groups` maps line ->
    list of (current_pos, goal_pos) for tiles whose goal line == this line.
    """
    total = 0
    for tiles in groups.values():
        k = len(tiles)
        if k < 2:
            continue
        pairs = []
        for a in range(k):
            ca, ga = tiles[a]
            for b in range(a + 1, k):
                cb, gb = tiles[b]
                if (ca - cb) * (ga - gb) < 0:  # current vs goal order reversed
                    pairs.append((a, b))
        if not pairs:
            continue
        removed = [False] * k
        while True:
            rc = [0] * k
            any_conf = False
            for a, b in pairs:
                if not removed[a] and not removed[b]:
                    rc[a] += 1
                    rc[b] += 1
                    any_conf = True
            if not any_conf:
                break
            worst = max(range(k), key=lambda i: rc[i])
            removed[worst] = True
            total += 1
    return total


def _flatten_state(s) -> np.ndarray:
    """Flatten a NATIVE env state to a 1-D int array for cheap comparison."""
    if isinstance(s, np.ndarray):
        return s.ravel()
    if isinstance(s, dict):
        parts = [_flatten_state(s[k]) for k in sorted(s)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    if isinstance(s, (list, tuple)):
        parts = [_flatten_state(x) for x in s]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    return np.array([s])


def make_misplaced_h(env) -> Callable[[List[Any]], np.ndarray]:
    """Cheap, generic heuristic: number of cells differing from the goal.
    Primary path flattens the NATIVE state and compares to the goal vector
    (~2.25x faster than going through encode_state). Falls back to an
    encode_state comparison if a state can't be flattened consistently.
    No NN -> guides A* on rotation/permutation puzzles at least as well."""
    env.reset()
    g_enc = env.encode_state()
    gtt = np.asarray(g_enc["target_types"])
    gtv = np.asarray(g_enc["target_values"])
    try:
        goal_flat = _flatten_state(env.get_state()).astype(np.int64)
        use_fast = goal_flat.size > 0
    except Exception:
        goal_flat = None
        use_fast = False

    def h(states):
        out = np.empty(len(states), dtype=np.float32)
        if use_fast:
            try:
                for i, s in enumerate(states):
                    f = _flatten_state(s).astype(np.int64)
                    if f.shape != goal_flat.shape:
                        raise ValueError("shape mismatch")
                    out[i] = np.count_nonzero(f != goal_flat)
                return out
            except Exception:
                pass  # fall through to the robust encode_state path
        for i, s in enumerate(states):
            e = env.encode_state(s)
            ct = np.asarray(e["content_types"])
            cv = np.asarray(e["content_values"])
            out[i] = np.count_nonzero((ct != gtt) | (cv != gtv))
        return out

    return h


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

        # Try to recover a 2D grid from cell positions so we can add Linear
        # Conflict (admissible booster on top of Manhattan). If positions don't
        # form a clean grid, lc stays disabled and we use plain Manhattan.
        positions = enc["positions"]
        xs = sorted({round(p[0], 6) for p in positions})
        ys = sorted({round(p[1], 6) for p in positions})
        xrank = {x: i for i, x in enumerate(xs)}
        yrank = {y: i for i, y in enumerate(ys)}
        cell_col = [xrank[round(positions[i][0], 6)] for i in range(N)]
        cell_row = [yrank[round(positions[i][1], 6)] for i in range(N)]
        lc_enabled = (len({(cell_row[i], cell_col[i]) for i in range(N)}) == N)
        goal_rc = {v: (cell_row[gc[v]], cell_col[gc[v]]) for v in gc} if lc_enabled else {}

        env.reset()

        def h_fn(states):
            out = np.empty(len(states), dtype=np.float32)
            for k, s in enumerate(states):
                e = env.encode_state(s)
                ec, ev = e["content_types"], e["content_values"]
                tot = 0
                # group tiles by their current row / column for Linear Conflict
                rows = {} if lc_enabled else None
                cols = {} if lc_enabled else None
                for i in range(N):
                    if ec[i] != _C_NUM:
                        continue
                    v = ev[i]
                    g = gc.get(v)
                    if g is not None:
                        dd = dist[i][g]
                        if dd > 0:
                            tot += dd
                    if lc_enabled and v in goal_rc:
                        cr, cc = cell_row[i], cell_col[i]
                        gr, gc_ = goal_rc[v]
                        # a tile is a row-LC candidate if it's in its goal row now
                        if cr == gr:
                            rows.setdefault(cr, []).append((cc, gc_))
                        if cc == gc_:
                            cols.setdefault(cc, []).append((cr, gr))
                if lc_enabled:
                    tot += 2 * _linear_conflicts(rows) + 2 * _linear_conflicts(cols)
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
    max_nodes: int = 300_000,
) -> Optional[List[str]]:
    """Meet-in-the-middle BFS. Returns a shortest action list or None.

    Forward wave expands with env.step. Backward wave expands predecessors:
    for state s and action b valid at s, s' = step(s, b) is a predecessor of s,
    connected by the forward action a = inverse_action(b) (a is valid at s' and
    step(s', a) == s by reversibility). The two waves meet on a shared state key.

    States are kept in their NATIVE env form and keyed via _fast_key (bytes) —
    no json / to_jsonable in the hot loop.
    """
    env.set_state(initial_state)
    start = env.get_state()
    env.set_state(solved_state)
    goal = env.get_state()
    sk = _fast_key(start)
    gk = _fast_key(goal)
    if sk == gk:
        return []

    # key -> (parent_key, forward action leading INTO this key from the start side)
    fwd: Dict[bytes, Tuple[Optional[bytes], Optional[str]]] = {sk: (None, None)}
    # key -> (next_key toward goal, forward action FROM this key toward goal)
    bwd: Dict[bytes, Tuple[Optional[bytes], Optional[str]]] = {gk: (None, None)}

    f_frontier: List[Tuple[bytes, Any]] = [(sk, start)]
    b_frontier: List[Tuple[bytes, Any]] = [(gk, goal)]
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
                    ns = env.get_state()
                except Exception:
                    continue
                nk = _fast_key(ns)
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


def _misplaced_vs(env, state, ref_types, ref_vals) -> int:
    e = env.encode_state(state)
    ct = np.asarray(e["content_types"])
    cv = np.asarray(e["content_values"])
    return int(np.count_nonzero((ct != ref_types) | (cv != ref_vals)))


def solve_bidir_astar(
    env,
    initial_state: Any,
    solved_state: Any,
    deadline: float,
    max_nodes: int = 2_000_000,
    weight: float = 2.0,
) -> Optional[List[str]]:
    """Informed meet-in-the-middle: two best-first waves (f = g + weight*h),
    forward guided toward the goal, backward toward the start, both using the
    cheap misplaced-count heuristic. Halves effective search depth (d -> d/2),
    which is what unidirectional A* lacks on deep rotation puzzles.

    Backward expansion uses inverse actions (reversibility). Native states +
    _fast_key. Returns an action list or None."""
    env.set_state(initial_state)
    start = env.get_state()
    senc = env.encode_state(start)
    start_t = np.asarray(senc["content_types"]); start_v = np.asarray(senc["content_values"])
    env.set_state(solved_state)
    goal = env.get_state()
    genc = env.encode_state(goal)
    goal_t = np.asarray(genc["target_types"]); goal_v = np.asarray(genc["target_values"])

    sk = _fast_key(start)
    gk = _fast_key(goal)
    if sk == gk:
        return []

    fwd: Dict[bytes, Tuple[Optional[bytes], Optional[str]]] = {sk: (None, None)}
    bwd: Dict[bytes, Tuple[Optional[bytes], Optional[str]]] = {gk: (None, None)}
    cf = cb = 0
    # heaps: (f, counter, g, key, state)
    f_heap = [(0.0, 0, 0, sk, start)]
    b_heap = [(0.0, 0, 0, gk, goal)]
    nodes = 2

    while f_heap and b_heap:
        if time.time() >= deadline or nodes >= max_nodes:
            return None

        forward = len(f_heap) <= len(b_heap)
        heap = f_heap if forward else b_heap
        this_side = fwd if forward else bwd
        other_side = bwd if forward else fwd
        _f, _c, g, key, state = heapq.heappop(heap)

        try:
            env.set_state(state)
            valid = env.valid_actions()
        except Exception:
            continue
        for mv in valid:
            try:
                env.set_state(state)
                env.step(mv)
                ns = env.get_state()
            except Exception:
                continue
            nk = _fast_key(ns)
            if nk in this_side:
                continue
            if forward:
                this_side[nk] = (key, mv)
                h = _misplaced_vs(env, ns, goal_t, goal_v)
            else:
                try:
                    a = env.inverse_action(mv)
                except Exception:
                    continue
                this_side[nk] = (key, a)
                h = _misplaced_vs(env, ns, start_t, start_v)
            nodes += 1
            if nk in other_side:
                return _stitch(nk, fwd, bwd)
            ng = g + 1
            if forward:
                cf += 1
                heapq.heappush(f_heap, (ng + weight * h, cf, ng, nk, ns))
            else:
                cb += 1
                heapq.heappush(b_heap, (ng + weight * h, cb, ng, nk, ns))
            if nodes >= max_nodes:
                return None

    return None


def solve_astar(
    env,
    initial_state: Any,
    solved_state: Any,
    v_fn: Optional[Callable[[List[Any]], np.ndarray]],
    deadline: float,
    max_nodes: int = 1_000_000,
    expand_batch: int = 32,
    weight: float = 1.0,
) -> Optional[List[str]]:
    """Weighted A* with f = g + weight * V(s). weight>1 trades solution length
    for speed/coverage (greedier toward the goal). Returns action list or None.

    Native states + _fast_key (bytes) keys — no json/to_jsonable in the loop."""
    env.set_state(solved_state)
    solved_key = _fast_key(env.get_state())
    env.set_state(initial_state)
    start = env.get_state()
    start_k = _fast_key(start)
    if start_k == solved_key:
        return []

    parents: Dict[bytes, Tuple[Optional[bytes], Optional[str]]] = {start_k: (None, None)}
    g_score: Dict[bytes, int] = {start_k: 0}

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
                    ns = env.get_state()
                except Exception:
                    continue
                nsk = _fast_key(ns)
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


def _reconstruct(end_k, parents) -> List[str]:
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


# ---------------------------------------------------------------------------
# Backward distance table ("precomputed backward half"): BFS from the goal in
# TRAIN, storing exact distance-to-goal for every state within reach. In SOLVE
# we forward-search until we hit a table state, then descend the table to the
# goal -> optimal solutions that crack deep instances unidirectional A* can't.
# Generic (uses env.step / valid_actions / get_state) and reuses the otherwise
# idle 50-min train budget.
# ---------------------------------------------------------------------------

def build_backward_table(env, solved_state, deadline, max_states=40_000_000,
                         mem_cap_bytes=None):
    """BFS outward from the goal. Returns (dict {state_hash: depth}, max_depth).

    Stops on: time deadline, max_states, OR a memory cap (RSS) — the last is the
    key safety guard: high-branching puzzles blow up memory fast, so we halt
    before OOM. A partial table is still EXACT for the states it contains."""
    if mem_cap_bytes is None:
        mem_cap_bytes = int(float(os.environ.get("TABLE_MEM_CAP_GB", "22")) * (1024 ** 3))
    env.set_state(solved_state)
    goal = env.get_state()
    table = {state_hash(goal): 0}
    frontier = [goal]
    depth = 0
    since_check = 0

    def over_budget():
        return (time.time() >= deadline or len(table) >= max_states
                or _rss_bytes() >= mem_cap_bytes)

    while frontier and not over_budget():
        depth += 1
        nxt = []
        stop = False
        for st in frontier:
            try:
                env.set_state(st)
                valid = env.valid_actions()
            except Exception:
                continue
            for a in valid:
                try:
                    env.set_state(st)
                    env.step(a)
                    ns = env.get_state()
                except Exception:
                    continue
                h = state_hash(ns)
                if h not in table:
                    table[h] = depth
                    nxt.append(ns)
                    since_check += 1
                    if since_check >= 20_000:
                        since_check = 0
                        if over_budget():
                            stop = True
                            break
            if stop:
                break
        frontier = nxt
    return table, depth


def save_backward_table(table: dict, path: str):
    """Store as sorted uint64 hashes + uint8 depths (compact, fast to load,
    cheap per-worker memory vs a Python dict)."""
    keys = np.fromiter(table.keys(), dtype=np.uint64, count=len(table))
    depths = np.fromiter(table.values(), dtype=np.uint16, count=len(table))
    order = np.argsort(keys)
    np.savez(path, keys=keys[order], depths=depths[order].astype(np.uint16))


class BackwardTable:
    """Compact, process-local lookup over sorted hash keys (binary search)."""

    def __init__(self, keys: np.ndarray, depths: np.ndarray):
        self.keys = keys
        self.depths = depths

    @classmethod
    def load(cls, path: str) -> Optional["BackwardTable"]:
        try:
            d = np.load(path)
            return cls(d["keys"], d["depths"])
        except Exception:
            return None

    def get(self, h: int) -> int:
        i = int(np.searchsorted(self.keys, np.uint64(h)))
        if i < self.keys.size and int(self.keys[i]) == h:
            return int(self.depths[i])
        return -1


def _descend_table(env, meet_state, table: "BackwardTable") -> Optional[List[str]]:
    """From a state known to be in the table, greedily step to a depth-1
    neighbour until the goal. Returns the forward actions meet->goal."""
    acts: List[str] = []
    cur = meet_state
    d = table.get(state_hash(cur))
    if d < 0:
        return None
    guard = 0
    while d > 0:
        env.set_state(cur)
        moved = False
        for a in env.valid_actions():
            env.set_state(cur)
            env.step(a)
            ns = env.get_state()
            if table.get(state_hash(ns)) == d - 1:
                acts.append(a)
                cur = ns
                d -= 1
                moved = True
                break
        if not moved:
            return None
        guard += 1
        if guard > 100000:
            return None
    return acts


def solve_with_table(
    env, initial_state, solved_state, table: "BackwardTable",
    h_fn, deadline: float, max_nodes: int = 3_000_000,
) -> Optional[List[str]]:
    """Forward weighted-A* (cheap heuristic) until a state is found in the
    backward table, then descend the table to the goal. Returns actions/None."""
    env.set_state(initial_state)
    start = env.get_state()
    if table.get(state_hash(start)) >= 0:
        return _descend_table(env, start, table)

    sk = _fast_key(start)
    parents = {sk: (None, None)}
    g = {sk: 0}
    h0 = float(h_fn([start])[0]) if h_fn else 0.0
    cnt = 0
    heap = [(h0, 0, 0, sk, start)]

    while heap:
        if time.time() >= deadline or len(g) >= max_nodes:
            return None
        _f, _c, gg, k, st = heapq.heappop(heap)
        try:
            env.set_state(st)
            valid = env.valid_actions()
        except Exception:
            continue
        children = []
        for a in valid:
            try:
                env.set_state(st)
                env.step(a)
                ns = env.get_state()
            except Exception:
                continue
            nk = _fast_key(ns)
            if nk in g:
                continue
            g[nk] = gg + 1
            parents[nk] = (k, a)
            if table.get(state_hash(ns)) >= 0:
                fwd = _reconstruct(nk, parents)
                tail = _descend_table(env, ns, table)
                if tail is not None:
                    return fwd + tail
                # false hit (hash collision): keep searching
            children.append((nk, ns))
        if not children:
            continue
        hs = h_fn([c[1] for c in children]) if h_fn else np.zeros(len(children))
        for (nk, ns), h in zip(children, hs):
            cnt += 1
            heapq.heappush(heap, (g[nk] + float(h), cnt, g[nk], nk, ns))
    return None
