import heapq
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from utils.common import to_jsonable
from core.encoder import StateEncoder


def _fast_key(state) -> str:
    """Compact JSON key — sort_keys handles dicts; separators=(',',':') saves space."""
    return json.dumps(to_jsonable(state), sort_keys=True, separators=(',', ':'))


class Searcher:
    """Unidirectional A* with undo-trick expansion and inline solved check."""

    def __init__(
        self,
        env,
        encoder: StateEncoder,
        model,
        max_nodes: int = 300_000,
        expand_batch: int = 32,
    ):
        self.env = env
        self.encoder = encoder
        self.model = model
        self.max_nodes = max_nodes
        self.expand_batch = expand_batch

        env.reset()
        solved_raw = env.get_state()
        self.solved_key = _fast_key(solved_raw)
        self.solved_state = to_jsonable(solved_raw)

    def _v_fn(self, states: List[Any]) -> np.ndarray:
        if self.model is None or not states:
            return np.zeros(len(states), dtype=np.float32)
        tokens = np.stack([self.encoder.encode_tokens(self.env, s) for s in states])
        B, N, _ = tokens.shape
        dense, cv, tv = self.encoder.to_tensors(tokens, B, N)
        with torch.no_grad():
            return self.model(dense, cv, tv).cpu().numpy()

    def _expand(self, sk, st, g, g_score, parents):
        """Expand node once with undo trick; returns [(nsk, ns, ng)]."""
        try:
            self.env.set_state(st)
            valid = self.env.valid_actions()
        except Exception:
            return []

        children = []
        for a in valid:
            try:
                inv_a = self.env.inverse_action(a)
            except Exception:
                inv_a = None
            try:
                self.env.step(a)
                ns_raw = self.env.get_state()
                ns = to_jsonable(ns_raw)
                nsk = _fast_key(ns_raw)
                ng = g + 1
                if ng < g_score.get(nsk, 1 << 30):
                    g_score[nsk] = ng
                    parents[nsk] = (sk, a)
                    children.append((nsk, ns, ng))
            except Exception:
                pass
            finally:
                try:
                    if inv_a is not None:
                        self.env.step(inv_a)
                    else:
                        self.env.set_state(st)
                except Exception:
                    try:
                        self.env.set_state(st)
                    except Exception:
                        pass
        return children

    def solve(self, state: Any, deadline: float) -> Optional[List[str]]:
        start = to_jsonable(state)
        start_k = _fast_key(start)
        if start_k == self.solved_key:
            return []

        parents: Dict[str, Tuple] = {start_k: (None, None)}
        g_score: Dict[str, int] = {start_k: 0}

        h0 = float(self._v_fn([start])[0])
        counter = 0
        heap = [(h0, counter, 0, start_k, start)]
        expanded = 0

        while heap and time.time() < deadline and expanded < self.max_nodes:
            batch = []
            for _ in range(self.expand_batch):
                if not heap:
                    break
                batch.append(heapq.heappop(heap))

            children_all = []
            for _f, _c, g, sk, st in batch:
                if g > g_score.get(sk, 1 << 30):
                    continue
                if sk == self.solved_key:
                    return self._reconstruct(sk, parents)
                expanded += 1
                ch = self._expand(sk, st, g, g_score, parents)
                for nsk, ns, ng in ch:
                    if nsk == self.solved_key:
                        return self._reconstruct(nsk, parents)
                children_all.extend(ch)

            if not children_all:
                continue
            h_vals = self._v_fn([c[1] for c in children_all])
            for (nsk, ns, ng), h in zip(children_all, h_vals):
                counter += 1
                heapq.heappush(heap, (ng + float(h), counter, ng, nsk, ns))

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


class BeamSearcher:
    """Greedy beam search — finds any solution quickly, even suboptimally."""

    def __init__(self, env, encoder: StateEncoder, model, beam_width: int = 64):
        self.env = env
        self.encoder = encoder
        self.model = model
        self.beam_width = beam_width

        env.reset()
        self.solved_key = _fast_key(env.get_state())

    def _v_fn(self, states: List[Any]) -> np.ndarray:
        if self.model is None or not states:
            return np.zeros(len(states), dtype=np.float32)
        tokens = np.stack([self.encoder.encode_tokens(self.env, s) for s in states])
        B, N, _ = tokens.shape
        dense, cv, tv = self.encoder.to_tensors(tokens, B, N)
        with torch.no_grad():
            return self.model(dense, cv, tv).cpu().numpy()

    def solve(self, state: Any, deadline: float, max_steps: int = 400) -> Optional[List[str]]:
        start = to_jsonable(state)
        start_k = _fast_key(start)
        if start_k == self.solved_key:
            return []

        # beam: list of (state_jsonable, actions_so_far)
        beam = [(start, [])]
        visited = {start_k}

        for _ in range(max_steps):
            if time.time() >= deadline:
                return None

            candidates = []
            for s, path in beam:
                try:
                    self.env.set_state(s)
                    valid = self.env.valid_actions()
                except Exception:
                    continue

                for a in valid:
                    try:
                        inv_a = self.env.inverse_action(a)
                    except Exception:
                        inv_a = None
                    try:
                        self.env.step(a)
                        ns_raw = self.env.get_state()
                        nsk = _fast_key(ns_raw)
                        if nsk == self.solved_key:
                            return path + [a]
                        if nsk not in visited:
                            visited.add(nsk)
                            candidates.append((to_jsonable(ns_raw), path + [a]))
                    except Exception:
                        pass
                    finally:
                        try:
                            if inv_a is not None:
                                self.env.step(inv_a)
                            else:
                                self.env.set_state(s)
                        except Exception:
                            try:
                                self.env.set_state(s)
                            except Exception:
                                pass

            if not candidates:
                break

            h_vals = self._v_fn([c[0] for c in candidates])
            ranked = sorted(zip(h_vals.tolist(), range(len(candidates))), key=lambda x: x[0])
            beam = [candidates[i] for _, i in ranked[: self.beam_width]]

        return None


class BidirectionalSearcher:
    """Bidirectional A* (forward) + Dijkstra (backward) with undo-trick expansion."""

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
        solved_raw = env.get_state()
        self.solved_key = _fast_key(solved_raw)
        self.solved_state = to_jsonable(solved_raw)

    def _v_fn(self, states: List[Any]) -> np.ndarray:
        if self.model is None or not states:
            return np.zeros(len(states), dtype=np.float32)
        tokens = np.stack([self.encoder.encode_tokens(self.env, s) for s in states])
        B, N, _ = tokens.shape
        dense, cv, tv = self.encoder.to_tensors(tokens, B, N)
        with torch.no_grad():
            return self.model(dense, cv, tv).cpu().numpy()

    def _expand_fwd(self, sk, st, g, fwd_g, fwd_par):
        try:
            self.env.set_state(st)
            valid = self.env.valid_actions()
        except Exception:
            return []
        children = []
        for a in valid:
            try:
                inv_a = self.env.inverse_action(a)
            except Exception:
                inv_a = None
            try:
                self.env.step(a)
                ns_raw = self.env.get_state()
                ns = to_jsonable(ns_raw)
                nsk = _fast_key(ns_raw)
                ng = g + 1
                if ng < fwd_g.get(nsk, 1 << 30):
                    fwd_g[nsk] = ng
                    fwd_par[nsk] = (sk, a)
                    children.append((nsk, ns, ng))
            except Exception:
                pass
            finally:
                try:
                    if inv_a is not None:
                        self.env.step(inv_a)
                    else:
                        self.env.set_state(st)
                except Exception:
                    try:
                        self.env.set_state(st)
                    except Exception:
                        pass
        return children

    def _expand_bwd(self, sk, st, g, bwd_g, bwd_par):
        try:
            self.env.set_state(st)
            valid = self.env.valid_actions()
        except Exception:
            return []
        children = []
        for a in valid:
            try:
                inv_a = self.env.inverse_action(a)
            except Exception:
                inv_a = None
            try:
                self.env.step(a)
                ns_raw = self.env.get_state()
                ns = to_jsonable(ns_raw)
                nsk = _fast_key(ns_raw)
                ng = g + 1
                if ng < bwd_g.get(nsk, 1 << 30):
                    bwd_g[nsk] = ng
                    # inv_a takes us from nsk back toward sk (goal-ward)
                    bwd_par[nsk] = (sk, inv_a)
                    children.append((nsk, ns, ng))
            except Exception:
                pass
            finally:
                try:
                    if inv_a is not None:
                        self.env.step(inv_a)
                    else:
                        self.env.set_state(st)
                except Exception:
                    try:
                        self.env.set_state(st)
                    except Exception:
                        pass
        return children

    def solve(self, state: Any, deadline: float) -> Optional[List[str]]:
        start = to_jsonable(state)
        start_k = _fast_key(start)
        if start_k == self.solved_key:
            return []

        fwd_g: Dict[str, int] = {start_k: 0}
        fwd_par: Dict[str, Tuple] = {start_k: (None, None)}
        h0 = float(self._v_fn([start])[0])
        counter = 0
        fwd_heap = [(h0, counter, 0, start_k, start)]

        bwd_g: Dict[str, int] = {self.solved_key: 0}
        bwd_par: Dict[str, Tuple] = {self.solved_key: (None, None)}
        counter += 1
        bwd_heap = [(0, counter, self.solved_key, self.solved_state)]

        best_cost = 1 << 30
        best_meeting: Optional[str] = None
        total_expanded = 0

        while total_expanded < self.max_nodes and time.time() < deadline:
            if not fwd_heap and not bwd_heap:
                break

            # ---- forward batch ----
            fwd_children: List[Tuple] = []
            for _ in range(self.expand_batch):
                if not fwd_heap:
                    break
                f, _c, g, sk, st = heapq.heappop(fwd_heap)
                if g > fwd_g.get(sk, 1 << 30):
                    continue
                total_expanded += 1

                if sk in bwd_g:
                    cost = g + bwd_g[sk]
                    if cost < best_cost:
                        best_cost = cost
                        best_meeting = sk

                ch = self._expand_fwd(sk, st, g, fwd_g, fwd_par)
                fwd_children.extend(ch)

            # inline checks
            for nsk, ns, ng in fwd_children:
                if nsk == self.solved_key:
                    return self._reconstruct_bidir(nsk, fwd_par, bwd_par)
                if nsk in bwd_g:
                    cost = ng + bwd_g[nsk]
                    if cost < best_cost:
                        best_cost = cost
                        best_meeting = nsk

            if fwd_children:
                h_vals = self._v_fn([c[1] for c in fwd_children])
                for (nsk, ns, ng), h in zip(fwd_children, h_vals):
                    counter += 1
                    heapq.heappush(fwd_heap, (ng + float(h), counter, ng, nsk, ns))

            # ---- backward batch ----
            bwd_children: List[Tuple] = []
            for _ in range(self.expand_batch):
                if not bwd_heap:
                    break
                g, _c, sk, st = heapq.heappop(bwd_heap)
                if g > bwd_g.get(sk, 1 << 30):
                    continue
                total_expanded += 1

                if sk in fwd_g:
                    cost = fwd_g[sk] + g
                    if cost < best_cost:
                        best_cost = cost
                        best_meeting = sk

                ch = self._expand_bwd(sk, st, g, bwd_g, bwd_par)
                bwd_children.extend(ch)

            for nsk, ns, ng in bwd_children:
                if nsk in fwd_g:
                    cost = fwd_g[nsk] + ng
                    if cost < best_cost:
                        best_cost = cost
                        best_meeting = nsk

            for nsk, ns, ng in bwd_children:
                counter += 1
                heapq.heappush(bwd_heap, (ng, counter, nsk, ns))

            # termination
            if best_meeting is not None:
                fwd_min_f = fwd_heap[0][0] if fwd_heap else best_cost
                bwd_min_g = bwd_heap[0][0] if bwd_heap else best_cost
                if fwd_min_f + bwd_min_g >= best_cost:
                    break

        if best_meeting is None:
            return None
        return self._reconstruct_bidir(best_meeting, fwd_par, bwd_par)

    def _reconstruct_bidir(self, meeting_k: str, fwd_par: dict, bwd_par: dict) -> List[str]:
        fwd_actions: List[str] = []
        cur = meeting_k
        while True:
            pk, a = fwd_par.get(cur, (None, None))
            if pk is None or a is None:
                break
            fwd_actions.append(a)
            cur = pk
        fwd_actions.reverse()

        bwd_actions: List[str] = []
        cur = meeting_k
        while True:
            pk, a = bwd_par.get(cur, (None, None))
            if pk is None or a is None:
                break
            bwd_actions.append(a)
            cur = pk

        return fwd_actions + bwd_actions
