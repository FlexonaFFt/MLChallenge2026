import numpy as np
import torch

from utils.common import VALUE_VOCAB, CONTENT_TYPES, TOKEN_FEAT_DIM, DENSE_DIM


class StateEncoder:
    def encode_tokens(self, env, state=None) -> np.ndarray:
        obs = env.encode_state(state)
        pos = np.asarray(obs["positions"], dtype=np.float32)
        ct = np.asarray(obs["content_types"], dtype=np.int64)
        cv = np.asarray(obs["content_values"], dtype=np.int64)
        tt = np.asarray(obs["target_types"], dtype=np.int64)
        tv = np.asarray(obs["target_values"], dtype=np.int64)

        n = len(ct)
        feat = np.zeros((n, TOKEN_FEAT_DIM), dtype=np.float32)
        feat[:, 0:3] = pos
        for t in range(CONTENT_TYPES):
            feat[:, 3 + t] = (ct == t).astype(np.float32)
        feat[:, 3 + CONTENT_TYPES] = np.clip(cv, 0, VALUE_VOCAB - 1).astype(np.float32)
        base = 3 + CONTENT_TYPES + 1
        for t in range(CONTENT_TYPES):
            feat[:, base + t] = (tt == t).astype(np.float32)
        feat[:, base + CONTENT_TYPES] = np.clip(tv, 0, VALUE_VOCAB - 1).astype(np.float32)
        match = ((ct == tt) & (cv == tv)).astype(np.float32)
        feat[:, -2] = match
        feat[:, -1] = 1.0 - match
        return feat

    def split_features(self, tokens: np.ndarray) -> dict:
        base = 3 + CONTENT_TYPES + 1
        dense = np.concatenate(
            [tokens[:, :3 + CONTENT_TYPES], tokens[:, base:base + CONTENT_TYPES], tokens[:, -2:]],
            axis=-1,
        ).astype(np.float32)
        return {
            "dense": dense,
            "content_value": tokens[:, 3 + CONTENT_TYPES].astype(np.int64),
            "target_value": tokens[:, base + CONTENT_TYPES].astype(np.int64),
        }

    def to_tensors(self, tokens: np.ndarray, B: int, N: int):
        parts = self.split_features(tokens.reshape(B * N, -1))
        dense = torch.from_numpy(parts["dense"].reshape(B, N, -1))
        cv = torch.from_numpy(parts["content_value"].reshape(B, N))
        tv = torch.from_numpy(parts["target_value"].reshape(B, N))
        return dense, cv, tv
