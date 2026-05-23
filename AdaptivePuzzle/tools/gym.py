import math
import os
import random
from typing import Optional, List

import numpy as np

# =========================================================
# Select environment here
# Override at runtime via the ENV_ID environment variable.
# =========================================================

_DEFAULT_ENV_ID = "game_15_2d"
ENV_ID = os.environ.get("ENV_ID", _DEFAULT_ENV_ID)

# =========================================================
# Common constants
# =========================================================

CONTENT_EMPTY = 0
CONTENT_NUM = 1
CONTENT_COLOR = 2
CONTENT_MASKED = 3

ACTION_SWAP = 0
ACTION_ROTATE = 1
ACTION_TOGGLE = 2
ACTION_PERMUTE = 3

AXIS_NONE = -1
AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2

DIRECTION_NEG = -1
DIRECTION_NONE = 0
DIRECTION_POS = 1

NUM_INSTANCES_DEFAULT = 1000
PUBLIC_FRACTION = 0.3


# =========================================================
# Per-env constants
# =========================================================

# 2D Fifteen
FIFTEEN2D_H = 4
FIFTEEN2D_W = 4
FIFTEEN2D_SCRAMBLE_LENGTHS = [40, 60, 80, 100]

# Lights Out
LIGHTS_H = 7
LIGHTS_W = 7
LIGHTS_SCRAMBLE_LENGTHS = [15, 20, 25, 30]

# Rotate & Slide
ROTATE_SLIDE_N = 6
ROTATE_SLIDE_H = 6
ROTATE_SLIDE_SCRAMBLE_LENGTHS = [60, 80, 100, 150]

def get_default_scramble_lengths():
    return {
        "game_15_2d": FIFTEEN2D_SCRAMBLE_LENGTHS,
        "toggle_lights": LIGHTS_SCRAMBLE_LENGTHS,
        "cylinder_game": ROTATE_SLIDE_SCRAMBLE_LENGTHS,
    }[ENV_ID]


SCRAMBLE_LENGTHS_DEFAULT = get_default_scramble_lengths()


def get_meta():
    if ENV_ID == "game_15_2d":
        return {"h": FIFTEEN2D_H, "w": FIFTEEN2D_W}
    if ENV_ID == "toggle_lights":
        return {"h": LIGHTS_H, "w": LIGHTS_W}
    if ENV_ID == "cylinder_game":
        return {"n": ROTATE_SLIDE_N, "h": ROTATE_SLIDE_H}
    return {}


META = get_meta()


# =========================================================
# Helpers
# =========================================================

def to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.int64, np.int32, np.int8)):
        return int(x)
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    return x


def norm01(i, n):
    if n <= 1:
        return 0.0
    return float(i) / float(n - 1)


def inverse_perm(p):
    return np.argsort(p)


class BasePuzzleEnv:
    env_id = "base"

    def reset(self, seed: Optional[int] = None):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError

    def solved_state(self):
        raise NotImplementedError

    def is_solved(self):
        raise NotImplementedError

    def valid_actions(self):
        raise NotImplementedError

    def inverse_action(self, action):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def encode_state(self, state=None):
        raise NotImplementedError

    def encode_actions(self, actions=None, state=None):
        if actions is None:
            if state is None:
                actions = self.valid_actions()
            else:
                old = self.get_state()
                self.set_state(state)
                actions = self.valid_actions()
                self.set_state(old)
        return self._encode_actions_impl(actions, state=state)

    def _encode_actions_impl(self, actions, state=None):
        raise NotImplementedError

    def scramble(self, length, seed=None, no_backtrack=True):
        rng = random.Random(seed)
        self.reset(seed)
        actions = []
        prev = None

        for _ in range(length):
            valid = self.valid_actions()
            if no_backtrack and prev is not None:
                try:
                    inv = self.inverse_action(prev)
                    valid = [a for a in valid if a != inv] or self.valid_actions()
                except Exception:
                    pass

            a = rng.choice(valid)
            self.step(a)
            actions.append(a)
            prev = a

        return self.get_state(), actions


# =========================================================
# 2D Fifteen
# =========================================================

class Fifteen2DEnv(BasePuzzleEnv):
    env_id = "game_15_2d"

    def __init__(self):
        self.h = FIFTEEN2D_H
        self.w = FIFTEEN2D_W
        self.n = self.h * self.w

        self.goal = np.arange(1, self.n + 1, dtype=np.int32)
        self.goal[-1] = 0
        self.goal = self.goal.reshape(self.h, self.w)

        self.board = self.goal.copy()
        self.blank = (self.h - 1, self.w - 1)

    def reset(self, seed=None):
        self.board = self.goal.copy()
        self.blank = (self.h - 1, self.w - 1)
        return self.get_state()

    def get_state(self):
        return self.board.copy()

    def set_state(self, state):
        self.board = np.array(state, dtype=np.int32).reshape(self.h, self.w)
        pos = np.argwhere(self.board == 0)
        if len(pos) != 1:
            raise ValueError("state must contain exactly one blank")
        self.blank = tuple(pos[0])

    def solved_state(self):
        return self.goal.copy()

    def is_solved(self):
        return np.array_equal(self.board, self.goal)

    def valid_actions(self):
        r, c = self.blank
        acts = []
        if c > 0:
            acts.append("X-")
        if c < self.w - 1:
            acts.append("X+")
        if r > 0:
            acts.append("Y-")
        if r < self.h - 1:
            acts.append("Y+")
        return acts

    def inverse_action(self, action):
        return {
            "X-": "X+",
            "X+": "X-",
            "Y-": "Y+",
            "Y+": "Y-",
        }[action]

    def step(self, action):
        r, c = self.blank
        nr, nc = r, c

        if action == "X-":
            nc -= 1
        elif action == "X+":
            nc += 1
        elif action == "Y-":
            nr -= 1
        elif action == "Y+":
            nr += 1
        else:
            raise ValueError(action)

        if not (0 <= nr < self.h and 0 <= nc < self.w):
            raise ValueError(action)

        self.board[r, c], self.board[nr, nc] = self.board[nr, nc], self.board[r, c]
        self.blank = (nr, nc)

        return self.get_state(), int(self.is_solved()), self.is_solved(), {}

    def _idx(self, r, c):
        return r * self.w + c

    def encode_state(self, state=None):
        board = self.board if state is None else np.array(state, dtype=np.int32).reshape(self.h, self.w)

        positions = []
        content_types = []
        content_values = []
        target_types = []
        target_values = []

        for r in range(self.h):
            for c in range(self.w):
                val = int(board[r, c])
                tgt = int(self.goal[r, c])

                positions.append([norm01(c, self.w), norm01(r, self.h), 0.0])

                if val == 0:
                    content_types.append(CONTENT_EMPTY)
                    content_values.append(0)
                else:
                    content_types.append(CONTENT_NUM)
                    content_values.append(val)

                if tgt == 0:
                    target_types.append(CONTENT_EMPTY)
                    target_values.append(0)
                else:
                    target_types.append(CONTENT_NUM)
                    target_values.append(tgt)

        return {
            "positions": positions,
            "content_types": content_types,
            "content_values": content_values,
            "target_types": target_types,
            "target_values": target_values,
        }

    def _encode_actions_impl(self, actions, state=None):
        if state is None:
            r, c = self.blank
        else:
            arr = np.array(state, dtype=np.int32).reshape(self.h, self.w)
            pos = np.argwhere(arr == 0)
            r, c = tuple(pos[0])

        action_types = []
        axes = []
        indices = []
        directions = []
        affected = []
        map_from = []
        map_to = []

        for a in actions:
            nr, nc = r, c
            if a == "X-":
                nc -= 1
                axis, direction = AXIS_X, DIRECTION_NEG
            elif a == "X+":
                nc += 1
                axis, direction = AXIS_X, DIRECTION_POS
            elif a == "Y-":
                nr -= 1
                axis, direction = AXIS_Y, DIRECTION_NEG
            elif a == "Y+":
                nr += 1
                axis, direction = AXIS_Y, DIRECTION_POS
            else:
                raise ValueError(a)

            i0 = self._idx(r, c)
            i1 = self._idx(nr, nc)

            action_types.append(ACTION_SWAP)
            axes.append(axis)
            indices.append(-1)
            directions.append(direction)
            affected.append([i0, i1])
            map_from.append([i0, i1])
            map_to.append([i1, i0])

        return {
            "actions": actions,
            "action_types": action_types,
            "axes": axes,
            "indices": indices,
            "directions": directions,
            "affected": affected,
            "map_from": map_from,
            "map_to": map_to,
        }


# =========================================================
# Lights Out
# =========================================================

class LightsOutEnv(BasePuzzleEnv):
    env_id = "toggle_lights"

    def __init__(self):
        self.h = LIGHTS_H
        self.w = LIGHTS_W
        self.goal = np.zeros((self.h, self.w), dtype=np.int8)
        self.board = self.goal.copy()

    def reset(self, seed=None):
        self.board = self.goal.copy()
        return self.get_state()

    def get_state(self):
        return self.board.copy()

    def set_state(self, state):
        self.board = np.array(state, dtype=np.int8).reshape(self.h, self.w)

    def solved_state(self):
        return self.goal.copy()

    def is_solved(self):
        return np.array_equal(self.board, self.goal)

    def valid_actions(self):
        return [f"{r}_{c}" for r in range(self.h) for c in range(self.w)]

    def inverse_action(self, action):
        return action

    def _toggle(self, r, c):
        if 0 <= r < self.h and 0 <= c < self.w:
            self.board[r, c] ^= 1

    def step(self, action):
        r, c = map(int, action.split("_"))
        self._toggle(r, c)
        self._toggle(r - 1, c)
        self._toggle(r + 1, c)
        self._toggle(r, c - 1)
        self._toggle(r, c + 1)
        return self.get_state(), int(self.is_solved()), self.is_solved(), {}

    def _cell_index(self, r, c):
        return r * self.w + c

    def encode_state(self, state=None):
        board = self.board if state is None else np.array(state, dtype=np.int8).reshape(self.h, self.w)

        positions = []
        content_types = []
        content_values = []
        target_types = []
        target_values = []

        for r in range(self.h):
            for c in range(self.w):
                positions.append([norm01(c, self.w), norm01(r, self.h), 0.0])
                content_types.append(CONTENT_NUM)
                content_values.append(int(board[r, c]))
                target_types.append(CONTENT_NUM)
                target_values.append(0)

        return {
            "positions": positions,
            "content_types": content_types,
            "content_values": content_values,
            "target_types": target_types,
            "target_values": target_values,
        }

    def _encode_actions_impl(self, actions, state=None):
        action_types = []
        axes = []
        indices = []
        directions = []
        affected = []
        map_from = []
        map_to = []

        for a in actions:
            r, c = map(int, a.split("_"))
            cells = []
            for rr, cc in [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                if 0 <= rr < self.h and 0 <= cc < self.w:
                    cells.append(self._cell_index(rr, cc))

            action_types.append(ACTION_TOGGLE)
            axes.append(AXIS_NONE)
            indices.append(self._cell_index(r, c))
            directions.append(DIRECTION_NONE)
            affected.append(cells)

            # Toggle is a value transform, not a permutation.
            map_from.append([])
            map_to.append([])

        return {
            "actions": actions,
            "action_types": action_types,
            "axes": axes,
            "indices": indices,
            "directions": directions,
            "affected": affected,
            "map_from": map_from,
            "map_to": map_to,
        }


# =========================================================
# Rotate & Slide
# =========================================================

class RotateSlideEnv(BasePuzzleEnv):
    env_id = "cylinder_game"

    def __init__(self):
        self.n = ROTATE_SLIDE_N
        self.h = ROTATE_SLIDE_H

        self.goal_main = np.zeros((self.h, self.n), dtype=np.int32)
        for c in range(self.n):
            self.goal_main[:, c] = c + 1

        self.goal_top_pos = 0
        self.goal_top_ball = 0

        self.main = self.goal_main.copy()
        self.top_pos = self.goal_top_pos
        self.top_ball = self.goal_top_ball

    def reset(self, seed=None):
        self.main = self.goal_main.copy()
        self.top_pos = self.goal_top_pos
        self.top_ball = self.goal_top_ball
        return self.get_state()

    def get_state(self):
        return {
            "main": self.main.copy(),
            "top_pos": int(self.top_pos),
            "top_ball": int(self.top_ball),
        }

    def set_state(self, state):
        self.main = np.array(state["main"], dtype=np.int32).reshape(self.h, self.n)
        self.top_pos = int(state["top_pos"])
        self.top_ball = int(state["top_ball"])

    def solved_state(self):
        return {
            "main": self.goal_main.copy(),
            "top_pos": int(self.goal_top_pos),
            "top_ball": int(self.goal_top_ball),
        }

    def _count_zeros(self):
        return int(np.sum(self.main == 0)) + (1 if self.top_ball == 0 else 0)

    def is_solved(self):
        return (
            self.top_pos == self.goal_top_pos
            and self.top_ball == self.goal_top_ball
            and np.array_equal(self.main, self.goal_main)
        )

    def valid_actions(self):
        acts = ["TL", "TR"]

        for l in range(self.h):
            acts.append(f"L_{l}")
            acts.append(f"R_{l}")

        c = self.top_pos

        for r in range(self.h):
            if self.main[r, c] == 0:
                continue
            if r == self.h - 1:
                if self.top_ball == 0:
                    acts.append(f"U_{r}")
            else:
                if self.main[r + 1, c] == 0:
                    acts.append(f"U_{r}")

        for r in range(1, self.h + 1):
            if r == self.h:
                if self.top_ball != 0 and self.main[r - 1, c] == 0:
                    acts.append(f"D_{r}")
            else:
                if self.main[r, c] != 0 and self.main[r - 1, c] == 0:
                    acts.append(f"D_{r}")

        return acts

    def inverse_action(self, action):
        if action == "TL":
            return "TR"
        if action == "TR":
            return "TL"
        if action.startswith("L_"):
            return action.replace("L_", "R_", 1)
        if action.startswith("R_"):
            return action.replace("R_", "L_", 1)
        if action.startswith("U_"):
            r = int(action.split("_")[1])
            return f"D_{r + 1}"
        if action.startswith("D_"):
            r = int(action.split("_")[1])
            return f"U_{r - 1}"
        raise ValueError(action)

    def step(self, action):
        if action == "TL":
            self.top_pos = (self.top_pos - 1) % self.n

        elif action == "TR":
            self.top_pos = (self.top_pos + 1) % self.n

        elif action.startswith("L_"):
            l = int(action.split("_")[1])
            self.main[l] = np.roll(self.main[l], -1)

        elif action.startswith("R_"):
            l = int(action.split("_")[1])
            self.main[l] = np.roll(self.main[l], 1)

        elif action.startswith("U_"):
            r = int(action.split("_")[1])
            c = self.top_pos
            if not (0 <= r < self.h):
                raise ValueError(action)
            if self.main[r, c] == 0:
                raise ValueError(action)

            if r == self.h - 1:
                if self.top_ball != 0:
                    raise ValueError(action)
                self.top_ball, self.main[r, c] = int(self.main[r, c]), 0
            else:
                if self.main[r + 1, c] != 0:
                    raise ValueError(action)
                self.main[r + 1, c], self.main[r, c] = self.main[r, c], 0

        elif action.startswith("D_"):
            r = int(action.split("_")[1])
            c = self.top_pos
            if not (1 <= r <= self.h):
                raise ValueError(action)

            if r == self.h:
                if self.top_ball == 0 or self.main[r - 1, c] != 0:
                    raise ValueError(action)
                self.main[r - 1, c], self.top_ball = self.top_ball, 0
            else:
                if self.main[r, c] == 0 or self.main[r - 1, c] != 0:
                    raise ValueError(action)
                self.main[r - 1, c], self.main[r, c] = self.main[r, c], 0
        else:
            raise ValueError(action)

        if self._count_zeros() != 1:
            raise RuntimeError("Invariant broken")

        return self.get_state(), int(self.is_solved()), self.is_solved(), {}

    def _main_index(self, r, c):
        return r * self.n + c

    def _top_index(self):
        return self.h * self.n

    def encode_state(self, state=None):
        if state is None:
            main = self.main
            top_pos = self.top_pos
            top_ball = self.top_ball
        else:
            main = np.array(state["main"], dtype=np.int32).reshape(self.h, self.n)
            top_pos = int(state["top_pos"])
            top_ball = int(state["top_ball"])

        positions = []
        content_types = []
        content_values = []
        target_types = []
        target_values = []

        for r in range(self.h):
            for c in range(self.n):
                theta = 2.0 * math.pi * c / self.n
                positions.append([math.cos(theta), math.sin(theta), norm01(r, self.h)])

                val = int(main[r, c])
                if val == 0:
                    content_types.append(CONTENT_EMPTY)
                    content_values.append(0)
                else:
                    content_types.append(CONTENT_COLOR)
                    content_values.append(val)

                target_types.append(CONTENT_COLOR)
                target_values.append(int(self.goal_main[r, c]))

        theta = 2.0 * math.pi * top_pos / self.n
        positions.append([math.cos(theta), math.sin(theta), 1.2])

        if top_ball == 0:
            content_types.append(CONTENT_EMPTY)
            content_values.append(0)
        else:
            content_types.append(CONTENT_COLOR)
            content_values.append(int(top_ball))

        target_types.append(CONTENT_EMPTY)
        target_values.append(0)

        return {
            "positions": positions,
            "content_types": content_types,
            "content_values": content_values,
            "target_types": target_types,
            "target_values": target_values,
        }

    def _encode_actions_impl(self, actions, state=None):
        if state is None:
            top_pos = self.top_pos
        else:
            top_pos = int(state["top_pos"])

        action_types = []
        axes = []
        indices = []
        directions = []
        affected = []
        map_from = []
        map_to = []

        top_idx = self._top_index()

        for a in actions:
            if a == "TL":
                action_types.append(ACTION_ROTATE)
                axes.append(AXIS_Z)
                indices.append(self.h)
                directions.append(DIRECTION_NEG)
                affected.append([top_idx])
                # geometry position changes, content stays in same top token
                map_from.append([top_idx])
                map_to.append([top_idx])

            elif a == "TR":
                action_types.append(ACTION_ROTATE)
                axes.append(AXIS_Z)
                indices.append(self.h)
                directions.append(DIRECTION_POS)
                affected.append([top_idx])
                map_from.append([top_idx])
                map_to.append([top_idx])

            elif a.startswith("L_"):
                l = int(a.split("_")[1])
                cells = [self._main_index(l, c) for c in range(self.n)]
                action_types.append(ACTION_ROTATE)
                axes.append(AXIS_Z)
                indices.append(l)
                directions.append(DIRECTION_NEG)
                affected.append(cells)
                map_from.append(cells)
                map_to.append(cells[-1:] + cells[:-1])

            elif a.startswith("R_"):
                l = int(a.split("_")[1])
                cells = [self._main_index(l, c) for c in range(self.n)]
                action_types.append(ACTION_ROTATE)
                axes.append(AXIS_Z)
                indices.append(l)
                directions.append(DIRECTION_POS)
                affected.append(cells)
                map_from.append(cells)
                map_to.append(cells[1:] + cells[:1])

            elif a.startswith("U_"):
                r = int(a.split("_")[1])
                action_types.append(ACTION_SWAP)
                axes.append(AXIS_Z)
                indices.append(r)
                directions.append(DIRECTION_POS)

                if r == self.h - 1:
                    a0 = self._main_index(r, top_pos)
                    a1 = top_idx
                else:
                    a0 = self._main_index(r, top_pos)
                    a1 = self._main_index(r + 1, top_pos)

                affected.append([a0, a1])
                map_from.append([a0, a1])
                map_to.append([a1, a0])

            elif a.startswith("D_"):
                r = int(a.split("_")[1])
                action_types.append(ACTION_SWAP)
                axes.append(AXIS_Z)
                indices.append(r)
                directions.append(DIRECTION_NEG)

                if r == self.h:
                    a0 = top_idx
                    a1 = self._main_index(r - 1, top_pos)
                else:
                    a0 = self._main_index(r, top_pos)
                    a1 = self._main_index(r - 1, top_pos)

                affected.append([a0, a1])
                map_from.append([a0, a1])
                map_to.append([a1, a0])

            else:
                raise ValueError(a)

        return {
            "actions": actions,
            "action_types": action_types,
            "axes": axes,
            "indices": indices,
            "directions": directions,
            "affected": affected,
            "map_from": map_from,
            "map_to": map_to,
        }

# =========================================================
# Factory
# =========================================================

def make_env():
    if ENV_ID == "game_15_2d":
        return Fifteen2DEnv()
    if ENV_ID == "toggle_lights":
        return LightsOutEnv()
    if ENV_ID == "cylinder_game":
        return RotateSlideEnv()

    raise ValueError(f"Unknown ENV_ID: {ENV_ID}")
