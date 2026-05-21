import heapq
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from utils.common import state_key, to_jsonable
from core.encoder import StateEncoder


class Searcher:
    def __init__(
        self,
        env,
        encoder: StateEncoder,
        model,
        max_nodes: int = 50_000,
        expand_batch: int = 32,
    ):
        self.env = env
        self.encoder = encoder
        self.model = model
        self.max_nodes = max_nodes
        self.expand_batch = expand_batch

        env.reset()
        self.solved_key = state_key(to_jsonable(env.get_state()))

    def _v_fn(self, states: List[Any]) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(states), dtype=np.float32)
        tokens = np.stack([self.encoder.encode_tokens(self.env, s) for s in states])
        B, N, _ = tokens.shape
        dense, cv, tv = self.encoder.to_tensors(tokens, B, N)
        with torch.no_grad():
            return self.model(dense, cv, tv).cpu().numpy()

    def solve(self, state: Any, deadline: float) -> Optional[List[str]]:
        start = to_jsonable(state)
        start_k = state_key(start)
        if start_k == self.solved_key:
            return []

        parents: Dict[str, Tuple[Optional[str], Optional[str]]] = {start_k: (None, None)}
        g_score: Dict[str, int] = {start_k: 0}

        h0 = float(self._v_fn([start])[0])
        counter = 0
        open_heap = [(h0, counter, 0, start_k, start)]
        expanded = 0

        while open_heap:
            if time.time() >= deadline or expanded >= self.max_nodes:
                return None

            batch = []
            while open_heap and len(batch) < self.expand_batch:
                batch.append(heapq.heappop(open_heap))

            children = []
            seen = set()
            for _f, _c, g, sk, st in batch:
                if sk == self.solved_key:
                    return self._reconstruct(sk, parents)
                try:
                    self.env.set_state(st)
                    valid = self.env.valid_actions()
                except Exception:
                    continue
                for a in valid:
                    try:
                        self.env.set_state(st)
                        self.env.step(a)
                        ns = to_jsonable(self.env.get_state())
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

            for nsk, ns, psk, a, ng in children:
                if nsk == self.solved_key:
                    parents[nsk] = (psk, a)
                    return self._reconstruct(nsk, parents)

            h_vals = self._v_fn([c[1] for c in children])

            for (nsk, ns, psk, a, ng), h in zip(children, h_vals):
                parents[nsk] = (psk, a)
                g_score[nsk] = ng
                counter += 1
                heapq.heappush(open_heap, (ng + float(h), counter, ng, nsk, ns))
                expanded += 1
                if expanded >= self.max_nodes:
                    break

        return None

    def _reconstruct(self, end_k: str, parents: dict) -> List[str]:
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


class BidirectionalSearcher:
    """Bidirectional A* (forward) + Dijkstra (backward) searcher.

    Forward frontier uses V(s) as heuristic; backward frontier uses g-only
    (Dijkstra from solved state via inverse actions). Two frontiers meet in
    the middle, cutting effective depth roughly in half.
    """

    def __init__(
        self,
        env,
        encoder: StateEncoder,
        model,
        max_nodes: int = 200_000,
        expand_batch: int = 16,
    ):
        self.env = env
        self.encoder = encoder
        self.model = model
        self.max_nodes = max_nodes
        self.expand_batch = expand_batch

        env.reset()
        self.solved_key = state_key(to_jsonable(env.get_state()))
        self.solved_state = to_jsonable(env.get_state())

    def _v_fn(self, states: List[Any]) -> np.ndarray:
        if self.model is None or not states:
            return np.zeros(len(states), dtype=np.float32)
        tokens = np.stack([self.encoder.encode_tokens(self.env, s) for s in states])
        B, N, _ = tokens.shape
        dense, cv, tv = self.encoder.to_tensors(tokens, B, N)
        with torch.no_grad():
            return self.model(dense, cv, tv).cpu().numpy()

    def solve(self, state: Any, deadline: float) -> Optional[List[str]]:
        start = to_jsonable(state)
        start_k = state_key(start)
        if start_k == self.solved_key:
            return []

        # fwd_par[k] = (parent_k, action)  — action takes parent → k
        fwd_g: Dict[str, int] = {start_k: 0}
        fwd_par: Dict[str, Tuple[Optional[str], Optional[str]]] = {start_k: (None, None)}
        h0 = float(self._v_fn([start])[0])
        counter = 0
        fwd_heap = [(h0, counter, start_k, start)]

        # bwd_par[k] = (parent_k, inv_action) — inv_action takes k → parent_k (toward goal)
        bwd_g: Dict[str, int] = {self.solved_key: 0}
        bwd_par: Dict[str, Tuple[Optional[str], Optional[str]]] = {self.solved_key: (None, None)}
        bwd_heap = [(0, counter, self.solved_key, self.solved_state)]

        best_cost = 1 << 30
        best_meeting: Optional[str] = None
        total_expanded = 0

        while total_expanded < self.max_nodes and time.time() < deadline:
            if not fwd_heap and not bwd_heap:
                break

            # --- expand forward batch ---
            fwd_children = []
            for _ in range(self.expand_batch):
                if not fwd_heap:
                    break
                f, _c, sk, st = heapq.heappop(fwd_heap)
                g = fwd_g.get(sk, 1 << 30)
                if f > g + 1e-6 + 1:  # stale entry (f was ng+h, g may have improved)
                    # re-check: f = g_at_push + h; if current g < g_at_push it's stale
                    pass  # we use g_score as authority; just proceed
                total_expanded += 1

                if sk in bwd_g:
                    cost = g + bwd_g[sk]
                    if cost < best_cost:
                        best_cost = cost
                        best_meeting = sk

                try:
                    self.env.set_state(st)
                    valid = self.env.valid_actions()
                except Exception:
                    continue
                for a in valid:
                    try:
                        self.env.set_state(st)
                        self.env.step(a)
                        ns = to_jsonable(self.env.get_state())
                    except Exception:
                        continue
                    nsk = state_key(ns)
                    ng = g + 1
                    if ng < fwd_g.get(nsk, 1 << 30):
                        fwd_g[nsk] = ng
                        fwd_par[nsk] = (sk, a)
                        fwd_children.append((nsk, ns, ng))

            if fwd_children:
                h_vals = self._v_fn([c[1] for c in fwd_children])
                for (nsk, ns, ng), h in zip(fwd_children, h_vals):
                    counter += 1
                    heapq.heappush(fwd_heap, (ng + float(h), counter, nsk, ns))

            # --- expand backward batch ---
            for _ in range(self.expand_batch):
                if not bwd_heap:
                    break
                g, _c, sk, st = heapq.heappop(bwd_heap)
                if g > bwd_g.get(sk, 1 << 30):
                    continue  # stale
                total_expanded += 1

                if sk in fwd_g:
                    cost = fwd_g[sk] + g
                    if cost < best_cost:
                        best_cost = cost
                        best_meeting = sk

                try:
                    self.env.set_state(st)
                    valid = self.env.valid_actions()
                except Exception:
                    continue
                for a in valid:
                    try:
                        inv_a = self.env.inverse_action(a)
                        self.env.set_state(st)
                        self.env.step(a)
                        ns = to_jsonable(self.env.get_state())
                    except Exception:
                        continue
                    nsk = state_key(ns)
                    ng = g + 1
                    if ng < bwd_g.get(nsk, 1 << 30):
                        bwd_g[nsk] = ng
                        # inv_a applied at nsk brings us back to sk (closer to goal)
                        bwd_par[nsk] = (sk, inv_a)
                        counter += 1
                        heapq.heappush(bwd_heap, (ng, counter, nsk, ns))

            # Termination: both frontiers cannot improve best_cost anymore
            if best_meeting is not None:
                fwd_min_f = fwd_heap[0][0] if fwd_heap else best_cost
                bwd_min_g = bwd_heap[0][0] if bwd_heap else best_cost
                if fwd_min_f + bwd_min_g >= best_cost:
                    break

        if best_meeting is None:
            return None
        return self._reconstruct_bidir(best_meeting, fwd_par, bwd_par)

    def _reconstruct_bidir(self, meeting_k: str, fwd_par: dict, bwd_par: dict) -> List[str]:
        # start → meeting
        fwd_actions: List[str] = []
        cur = meeting_k
        while True:
            pk, a = fwd_par.get(cur, (None, None))
            if pk is None or a is None:
                break
            fwd_actions.append(a)
            cur = pk
        fwd_actions.reverse()

        # meeting → goal: follow bwd_par, each inv_action moves us toward goal
        bwd_actions: List[str] = []
        cur = meeting_k
        while True:
            pk, a = bwd_par.get(cur, (None, None))
            if pk is None or a is None:
                break
            bwd_actions.append(a)
            cur = pk

        return fwd_actions + bwd_actions
