"""Minimal, low-risk post-optimization.

We only remove adjacent (a, inverse(a)) pairs. This is:
- pure string manipulation + env.inverse_action (no env.step, no replay, no
  search) -> negligible CPU, zero memory growth;
- algebraically guaranteed not to change correctness, so we don't even need a
  verifier pass;
- typically saves a few percent on long solutions where various search phases
  produce small undo cycles.

Heavier passes (window-shorten with bidi-BFS) were intentionally removed: they
hold the whole state trajectory in memory and risked OOM on hidden puzzles.
"""

from typing import List


def cancel_inverses(env, actions: List[str]) -> List[str]:
    if not actions:
        return actions
    cur = actions
    # Iterate to fixpoint: removing a pair can create a new adjacency.
    while True:
        out: List[str] = []
        i = 0
        L = len(cur)
        changed = False
        while i < L:
            if i + 1 < L:
                try:
                    inv = env.inverse_action(cur[i])
                except Exception:
                    inv = None
                if inv is not None and inv == cur[i + 1]:
                    i += 2
                    changed = True
                    continue
            out.append(cur[i])
            i += 1
        cur = out
        if not changed:
            return cur


def optimize(env, initial_state, actions: List[str], deadline=None, window=None) -> List[str]:
    """Safe wrapper. Extra args ignored (kept for API compatibility with the
    previous heavier version)."""
    if not actions:
        return actions
    try:
        return cancel_inverses(env, actions)
    except Exception:
        return actions
