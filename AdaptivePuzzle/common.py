from utils.common import (  # noqa: F401
    VALUE_VOCAB,
    CONTENT_TYPES,
    TOKEN_FEAT_DIM,
    DENSE_DIM,
    to_jsonable,
    state_key,
)
from core.encoder import StateEncoder as _enc

_encoder = _enc()
encode_tokens = _encoder.encode_tokens
split_token_features = _encoder.split_features

from core.collector import DataCollector as _DataCollector  # noqa: E402


def backward_walks(env, num_walks, min_len, max_len, seed=0):
    return _DataCollector(env, _enc()).backward_walks(num_walks, min_len, max_len, seed)
