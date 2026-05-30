"""A synthetic PURE-PERMUTATION puzzle that mimics a hidden Rubik-like game,
built on the same gym.py conventions (NUM pieces, ROTATE actions with
map_from/map_to, fixed state-independent action set). Used to validate that
PackedEngine activates and that packed bidirectional search massively
out-throughputs the env.step path on the class we expect to gain from.

Layout: R overlapping rings over a flat board of N distinct numbered cells.
Each ring is an ordered list of cell indices; action rotates that ring by +/-1.
All actions are fixed permutations of contents -> state-independent.
"""
import numpy as np

# action type code mirrors gym.ACTION_ROTATE
ACTION_ROTATE = 1

N_CELLS = 24
# three overlapping rings (share cells -> nontrivial group, like Hungarian rings)
RINGS = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8, 9, 10, 11],
    [8, 9, 10, 11, 12, 13, 14, 0],
]


class SynthPermEnv:
    env_id = "synth_perm"

    def __init__(self):
        self.n = N_CELLS
        self.goal = np.arange(self.n, dtype=np.int32)
        self.board = self.goal.copy()
        self._actions = []
        self._perm = {}   # action -> gather array (new = old[gather])
        for ri, ring in enumerate(RINGS):
            ring = np.asarray(ring)
            for d, suf in ((+1, "P"), (-1, "M")):
                a = f"R{ri}{suf}"
                self._actions.append(a)
                g = np.arange(self.n)
                rolled = np.roll(ring, d)
                # contents move along the ring: cell rolled[k] receives from ring[k]
                g[rolled] = ring
                self._perm[a] = g

    def reset(self, seed=None):
        self.board = self.goal.copy()
        return self.get_state()

    def get_state(self):
        return self.board.copy()

    def set_state(self, state):
        self.board = np.asarray(state, dtype=np.int32).copy()

    def solved_state(self):
        return self.goal.copy()

    def is_solved(self):
        return np.array_equal(self.board, self.goal)

    def valid_actions(self):
        return list(self._actions)

    def inverse_action(self, a):
        return a[:-1] + ("M" if a.endswith("P") else "P")

    def step(self, a):
        self.board = self.board[self._perm[a]]
        return self.get_state()

    def encode_state(self, state=None):
        b = self.board if state is None else np.asarray(state)
        return {
            "content_types": [1] * self.n,
            "content_values": b.tolist(),
            "target_types": [1] * self.n,
            "target_values": self.goal.tolist(),
        }

    def encode_actions(self, actions=None, state=None):
        acts = self._actions if actions is None else list(actions)
        map_from, map_to, affected, atypes = [], [], [], []
        for a in acts:
            g = self._perm[a]
            mt = np.where(g != np.arange(self.n))[0]
            mf = g[mt]
            map_to.append(mt.tolist())
            map_from.append(mf.tolist())
            affected.append(sorted(set(mt.tolist()) | set(mf.tolist())))
            atypes.append(ACTION_ROTATE)
        return {
            "actions": acts,
            "action_types": atypes,
            "axes": [-1] * len(acts),
            "indices": [0] * len(acts),
            "directions": [0] * len(acts),
            "affected": affected,
            "map_from": map_from,
            "map_to": map_to,
        }

    def scramble(self, length, seed=None, no_backtrack=True):
        import random
        rng = random.Random(seed)
        self.reset()
        prev = None
        for _ in range(length):
            va = self.valid_actions()
            if no_backtrack and prev is not None:
                inv = self.inverse_action(prev)
                va = [x for x in va if x != inv] or va
            a = rng.choice(va)
            self.step(a)
            prev = a
        return self.get_state(), None


def make_env():
    return SynthPermEnv()
