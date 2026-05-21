import random

import numpy as np

from utils.common import to_jsonable, TOKEN_FEAT_DIM
from core.encoder import StateEncoder


class DataCollector:
    def __init__(self, env, encoder: StateEncoder):
        self.env = env
        self.encoder = encoder

    def backward_walks(self, num_walks: int, min_len: int, max_len: int, seed: int = 0):
        rng = random.Random(seed)
        pairs = []
        for _ in range(num_walks):
            L = rng.randint(min_len, max_len)
            self.env.reset(seed=rng.randint(0, 10**9))
            for depth in range(1, L + 1):
                a = rng.choice(self.env.valid_actions())
                self.env.step(a)
                pairs.append((to_jsonable(self.env.get_state()), depth))
        return pairs

    def collect(self, num_pairs: int, max_walk: int, seed: int):
        pairs = self.backward_walks(
            num_walks=max(100, num_pairs // max(1, max_walk // 2)),
            min_len=1,
            max_len=max_walk,
            seed=seed,
        )
        random.Random(seed + 1).shuffle(pairs)
        pairs = pairs[:num_pairs]

        if not pairs:
            return (
                np.zeros((0, 1, TOKEN_FEAT_DIM), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        tokens = np.stack([self.encoder.encode_tokens(self.env, s) for s, _ in pairs])
        labels = np.array([d for _, d in pairs], dtype=np.float32)
        return tokens, labels
