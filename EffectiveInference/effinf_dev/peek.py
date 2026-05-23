"""Бесплатный ручной спот-чек: печатает несколько пар «ответ модели / эталон».

Без API — просто глазами оценить, похож ли стиль/полнота на эталон.

    python peek.py --candidates data/cand_minimal.jsonl --n 5
"""
import argparse
import json
import random


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--eval_jsonl", default="data/eval.jsonl")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    refs = {}
    with open(args.eval_jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            refs[r["rid"]] = r["reference"]

    with open(args.candidates, encoding="utf-8") as f:
        cands = [json.loads(line) for line in f]

    random.Random(args.seed).shuffle(cands)
    for c in cands[: args.n]:
        print("=" * 80)
        print("ВОПРОС:", c["query"])
        print("-" * 80)
        print("МОДЕЛЬ:\n", c["answer"])
        print("-" * 80)
        print("ЭТАЛОН:\n", refs[c["rid"]])
        print()


if __name__ == "__main__":
    main()
