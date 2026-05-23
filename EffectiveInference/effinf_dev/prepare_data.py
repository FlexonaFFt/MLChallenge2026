"""Готовит данные для SFT и оценки из dataset_ml_challenge.parquet.

Запускается на Mac (нужны только pandas+pyarrow):

    ../../.venv/bin/python prepare_data.py

Производит в effinf_dev/data/:
  - splits.json              детерминированный held-out (eval) vs train
  - train_minimal.jsonl      messages с минимальным system-промптом
  - train_rich.jsonl         messages с богатым (текущим) system-промптом
  - eval.jsonl               held-out: {rid, query, reference} — system добавит генератор

Два train-набора нужны, чтобы сравнить промпты через прокси-судью (см. judge.py).
"""
import json
import os

import pandas as pd

from prompts import MINIMAL_SYSTEM, RICH_SYSTEM

HERE = os.path.dirname(__file__)
PARQUET = os.path.join(HERE, "..", "dataset_ml_challenge.parquet")
OUT_DIR = os.path.join(HERE, "data")

SEED = 0
N_EVAL = 800


def load() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET, engine="pyarrow")
    df["query"] = df["query"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()
    df = df[(df["query"] != "") & (df["answer"] != "")]
    df = df.drop_duplicates(subset=["query"]).reset_index(drop=True)
    return df


def write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def chat_record(system: str, query: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load()
    print(f"rows after clean/dedup: {len(df)}")

    shuffled = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    eval_df = shuffled.iloc[:N_EVAL].reset_index(drop=True)
    train_df = shuffled.iloc[N_EVAL:].reset_index(drop=True)
    print(f"train: {len(train_df)}  eval(held-out): {len(eval_df)}")

    write_jsonl(
        os.path.join(OUT_DIR, "train_minimal.jsonl"),
        [chat_record(MINIMAL_SYSTEM, r.query, r.answer) for r in train_df.itertuples()],
    )
    write_jsonl(
        os.path.join(OUT_DIR, "train_rich.jsonl"),
        [chat_record(RICH_SYSTEM, r.query, r.answer) for r in train_df.itertuples()],
    )
    write_jsonl(
        os.path.join(OUT_DIR, "eval.jsonl"),
        [
            {"rid": i, "query": r.query, "reference": r.answer}
            for i, r in enumerate(eval_df.itertuples())
        ],
    )

    with open(os.path.join(OUT_DIR, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "n_eval": N_EVAL,
                "n_train": len(train_df),
                "eval_queries": eval_df["query"].tolist(),
            },
            f,
            ensure_ascii=False,
        )

    print(f"written to {OUT_DIR}")


if __name__ == "__main__":
    main()
