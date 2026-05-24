"""Dry-run на Colab: проверить, что движок (8B-AWQ + speculative decoding)
ГРУЗИТСЯ без ошибок и УКЛАДЫВАЕТСЯ по времени — ДО сабмита.

Запуск из папки EffectiveInference/ (нужны скачанные weights/ и draft/):

    python dryrun.py --n 40

Печатает tok/s, среднюю длину ответа, экстраполяцию на 4000 запросов и
несколько примеров ответов. Если падает на инициализации LLM — значит конфиг
speculative decoding несовместим с этой версией vLLM (правим до сабмита).
"""
import argparse
import json
import time

from source.config import InferenceConfig
from source.engine import SchoolQAEngine


def load_questions(n: int) -> list[dict]:
    # берём из held-out, иначе из example
    for path in ("effinf_dev/data/eval.jsonl", "example-dataset.pickle"):
        try:
            if path.endswith(".jsonl"):
                with open(path, encoding="utf-8") as f:
                    rows = [json.loads(line) for line in f]
                return [{"rid": r["rid"], "question": r["query"]} for r in rows[:n]]
            else:
                import pickle
                with open(path, "rb") as f:
                    return pickle.load(f)[:n]
        except FileNotFoundError:
            continue
    raise SystemExit("нет ни eval.jsonl, ни example-dataset.pickle")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40)
    args = p.parse_args()

    rows = load_questions(args.n)
    cfg = InferenceConfig()
    print(f"spec_draft={cfg.spec_draft_model} | num_spec={cfg.num_speculative_tokens} "
          f"| max_model_len={cfg.max_model_len} | max_new={cfg.max_new_tokens}")

    t0 = time.time()
    engine = SchoolQAEngine(cfg)
    print(f"[engine loaded in {time.time()-t0:.1f}s]")

    t = time.time()
    res = engine.generate(rows)
    dt = time.time() - t

    gen = engine.tokenizer
    toks = sum(len(gen(r["answer"], add_special_tokens=False)["input_ids"]) for r in res)
    n = len(rows)
    print(f"\n=== {n} запросов за {dt:.1f}s | ~{toks/dt:.0f} tok/s | "
          f"средняя длина {toks/n:.0f} ток ===")
    print(f"ЭКСТРАПОЛЯЦИЯ на 4000: ~{dt/n*4000/60:.1f} мин "
          f"(лимит 15; на L4 будет быстрее, чем на T4)")
    for r in res[:3]:
        print("-" * 60)
        print("Q:", r.get("question", ""))
        print("A:", r["answer"][:400])


if __name__ == "__main__":
    main()
