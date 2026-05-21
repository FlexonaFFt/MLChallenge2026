import json

import numpy as np


VALUE_VOCAB = 64
CONTENT_TYPES = 4
TOKEN_FEAT_DIM = 3 + CONTENT_TYPES + 1 + CONTENT_TYPES + 1 + 2
DENSE_DIM = 3 + CONTENT_TYPES + CONTENT_TYPES + 2


def to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    return x


def state_key(state) -> str:
    return json.dumps(to_jsonable(state), sort_keys=True)
