"""Sanity-проход на сервере ПЕРЕД сабмитом: тайминг + длины + усечения + примеры.

Заменяет полноценный judge-гейт на первом заходе: ловит катастрофы (AWQ сломал
модель, зациклилась, ушла с языка, пустые ответы, упирается в кап) и проверяет
бюджет 15 мин — бесплатно и без судьи. Реальное качество подтвердит лидерборд.

Запуск из папки EffectiveInference/ (на L4/A5000 с весами):

    # текущая прод-модель (weights/):
    python dryrun.py --n 80

    # дообученный студент (после обучения и скачивания AWQ-весов):
    python dryrun.py --n 80 --model_dir ./student_qwen3_4b --show 15

Печатает: tok/s, экстраполяцию на 4000, перцентили длины ответа, сколько
ответов упёрлись в свой пер-категорийный кап (усечение!), разбивку по
категориям и N самых длинных ответов целиком — для глазной проверки формата.
"""
import argparse
import json
import time
from collections import defaultdict

from source.config import InferenceConfig
from source.engine import SchoolQAEngine


def load_questions(n: int, shuffle: bool, seed: int) -> list[dict]:
    for path in ("effinf_dev/data/eval.jsonl", "example-dataset.pickle"):
        try:
            if path.endswith(".jsonl"):
                with open(path, encoding="utf-8") as f:
                    rows = [json.loads(line) for line in f]
                rows = [{"rid": r["rid"], "question": r["query"]} for r in rows]
            else:
                import pickle
                with open(path, "rb") as f:
                    rows = pickle.load(f)
            if shuffle:
                import random
                random.Random(seed).shuffle(rows)
            return rows[:n]
        except FileNotFoundError:
            continue
    raise SystemExit("нет ни eval.jsonl, ни example-dataset.pickle")


def pct(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    return sorted_vals[min(int(q * len(sorted_vals)), len(sorted_vals) - 1)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--model_dir", default="", help="переопределить веса (студент)")
    p.add_argument("--show", type=int, default=12, help="сколько длинных ответов показать целиком")
    p.add_argument("--shuffle", action="store_true", help="случайная выборка из held-out")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = load_questions(args.n, args.shuffle, args.seed)
    cfg = InferenceConfig()
    if args.model_dir:
        cfg.model_dir = args.model_dir
    print(f"model_dir={cfg.model_dir} | dynamic_max_tokens={cfg.enable_dynamic_max_tokens} "
          f"| max_model_len={cfg.max_model_len}")

    t0 = time.time()
    engine = SchoolQAEngine(cfg)
    print(f"[engine loaded in {time.time()-t0:.1f}s]")

    t = time.time()
    res = engine.generate(rows)
    dt = time.time() - t

    tok = engine.tokenizer
    n = len(rows)
    qmap = {r["rid"]: r["question"] for r in rows}

    # на каждый ответ: категория, длина в токенах, кап, флаг усечения
    enriched = []
    for r in res:
        q = qmap[r["rid"]]
        ans = r["answer"]
        cat = engine._classify(q)
        cap = engine._max_tokens_for(q)
        ntok = len(tok(ans, add_special_tokens=False)["input_ids"]) if ans else 0
        truncated = ntok >= cap - 16          # вплотную к капу → вероятно обрезан
        empty = not ans.strip()
        enriched.append({
            "q": q, "a": ans, "cat": cat, "cap": cap,
            "ntok": ntok, "trunc": truncated, "empty": empty,
        })

    lens = sorted(e["ntok"] for e in enriched)
    total_tok = sum(lens)
    n_trunc = sum(e["trunc"] for e in enriched)
    n_empty = sum(e["empty"] for e in enriched)

    print(f"\n=== ТАЙМИНГ: {n} запросов за {dt:.1f}s | ~{total_tok/dt:.0f} tok/s ===")
    print(f"ЭКСТРАПОЛЯЦИЯ на 4000: ~{dt/n*4000/60:.1f} мин (лимит 15; на L4 ≈ как тут)")
    print(f"\n=== ДЛИНА ОТВЕТА (токены): "
          f"p50={pct(lens,.5)} p90={pct(lens,.9)} p95={pct(lens,.95)} max={lens[-1] if lens else 0} ===")
    print(f"усечены (упёрлись в кап): {n_trunc}/{n} ({n_trunc/n*100:.0f}%)  |  пустые: {n_empty}/{n}")

    # разбивка по категориям
    by = defaultdict(list)
    for e in enriched:
        by[e["cat"]].append(e)
    print("\n=== ПО КАТЕГОРИЯМ (n | кап | сред.длина | усечено) ===")
    for cat, es in sorted(by.items(), key=lambda x: -len(x[1])):
        avg = sum(e["ntok"] for e in es) / len(es)
        tr = sum(e["trunc"] for e in es)
        print(f"  {cat:<11} n={len(es):<4} cap={es[0]['cap']:<5} avg={avg:>5.0f}  усеч={tr}")

    # самые длинные ответы целиком — там виднее поломки формата/обрывы
    print(f"\n=== {args.show} САМЫХ ДЛИННЫХ ОТВЕТОВ (для глазной проверки) ===")
    for e in sorted(enriched, key=lambda e: -e["ntok"])[: args.show]:
        flag = " [УСЕЧЁН]" if e["trunc"] else ""
        print("=" * 70)
        print(f"[{e['cat']} | {e['ntok']} ток / кап {e['cap']}]{flag}")
        print("Q:", e["q"])
        print("A:", e["a"])


if __name__ == "__main__":
    main()
