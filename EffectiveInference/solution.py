"""Точка входа: читает /workspace/input.pickle, пишет /workspace/output.json."""
import json
import os
import pickle
import time

_START = time.time()  # старт процесса (включая загрузку весов) для wall-guard

from source.config import InferenceConfig
from source.engine import SchoolQAEngine


def main() -> None:
    with open("input.pickle", "rb") as f:
        rows = pickle.load(f)

    config = InferenceConfig()
    budget = os.environ.get("GEN_BUDGET_SEC")
    if budget:
        config.gen_budget_sec = float(budget)
    engine = SchoolQAEngine(config)
    result = engine.generate(rows, start_time=_START)

    with open("output.json", "w") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
