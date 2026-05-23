"""Generic beam search for any BasePuzzleEnv (15-puzzle, Lights Out, cylinder)."""
import numpy as np


def _key(state):
    if isinstance(state, dict):
        return tuple((k, _key(v)) for k, v in sorted(state.items()))
    if isinstance(state, np.ndarray):
        return state.tobytes()
    return state


def _score(env, state):
    env.set_state(state)
    enc = env.encode_state()
    return sum(c != t for c, t in zip(enc["content_values"], enc["target_values"]))


def beam_search(env, beam_width=8, max_depth=20):
    """Return actions that solve env, or the path to the closest-seen state."""
    start = env.get_state()
    start_score = _score(env, start)
    beam = [(start, [], start_score)]
    seen = {_key(start)}

    best_path = []
    best_score = start_score

    for _ in range(max_depth):
        candidates = []
        for state, path, _s in beam:
            env.set_state(state)
            actions = env.valid_actions()
            for action in actions:
                env.set_state(state)
                new_state, _, solved, _ = env.step(action)
                if solved:
                    env.set_state(start)
                    return path + [action]
                k = _key(new_state)
                if k in seen:
                    continue
                seen.add(k)
                candidates.append((new_state, path + [action], _score(env, new_state)))

        if not candidates:
            break
        candidates.sort(key=lambda x: x[2])
        if candidates[0][2] < best_score:
            best_score = candidates[0][2]
            best_path = candidates[0][1]
        beam = candidates[:beam_width]

    env.set_state(start)
    return best_path
