"""Точка входа: читает /workspace/input.pickle, пишет /workspace/output.json."""
import json
import pickle

from source.config import InferenceConfig
from source.engine import SchoolQAEngine


def main() -> None:
    with open("input.pickle", "rb") as f:
        rows = pickle.load(f)

    config = InferenceConfig()
    engine = SchoolQAEngine(config)
    result = engine.generate(rows)

    with open("output.json", "w") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
