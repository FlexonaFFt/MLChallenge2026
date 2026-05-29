"""A/B-харнесс: гоняет РЕАЛЬНЫЙ SchoolQAEngine по набору профилей конфига.

Главная идея: оффлайн == продакшн. Мы не дублируем логику промпта/семплинга,
а импортируем тот же source/engine.py + source/config.py, что едет в контейнер,
и меняем только поля InferenceConfig между профилями. Движок (4B-AWQ) грузится
ОДИН раз; профили переключаются мутацией config + сбросом кэша sampling.

Выход: на каждый профиль — data/cand_<profile>.jsonl ({rid, query, answer}),
плюс печать throughput и экстраполяции на 4000 запросов (контроль 15-мин лимита,
достоверно только на L4 — на T4 смотрим относительные числа).

Запуск (Kaggle/Colab, из папки effinf_dev):

    !python ab_eval.py --model_dir /kaggle/working/weights \
        --eval_jsonl data/eval.jsonl --limit 300 \
        --profiles baseline_672,sampling_qwen,sampling_mild,essay_headroom
"""
import argparse
import json
import os
import sys
import time

# дать доступ к ../source (тот же код, что в контейнере)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from source.config import InferenceConfig  # noqa: E402
from source.engine import SchoolQAEngine  # noqa: E402


# ---------------------------------------------------------------------------
# ПРОФИЛИ: имя -> dict переопределений полей InferenceConfig.
# baseline_672 = пусто => текущий дефолт (RICH-промпт + category few-shot +
# dynamic max_tokens), т.е. конфигурация, давшая 67.2. Добавляй свои гипотезы.
# ---------------------------------------------------------------------------
PROFILES: dict[str, dict] = {
    "baseline_672": {},
    # Qwen3 non-thinking рекомендованный сэмплинг — оживляет сочинения/объяснения.
    "sampling_qwen": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
    # мягче — ближе к greedy, меньше риск фактических ошибок.
    "sampling_mild": {"temperature": 0.6, "top_p": 0.9, "top_k": 20},
    # проверка запаса по обрезке длинных ответов (essay/default).
    "essay_headroom": {
        "category_max_tokens": {
            "essay": 2048, "default": 1536, "chemistry": 1100, "grammar": 1100,
            "physics": 1100, "math": 768, "morphology": 768, "english": 768,
        }
    },
}


def make_config(model_dir: str, overrides: dict) -> InferenceConfig:
    cfg = InferenceConfig(model_dir=model_dir, finetuned=False)
    # wall-guard выкл на eval: хотим полный прогон, тайминг считаем отдельно.
    cfg.enable_wall_guard = False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def run_profile(engine: SchoolQAEngine, name: str, overrides: dict,
                rows: list[dict], out_path: str) -> None:
    # переключаем профиль на живом инстансе движка
    for k, v in overrides.items():
        setattr(engine.config, k, v)
    engine._sp_cache.clear()

    t = time.time()
    result = engine.generate(rows)  # rows уже в формате {"rid","question"}
    elapsed = time.time() - t

    ans_by_rid = {r["rid"]: r["answer"] for r in result}
    n = len(rows)
    tot_chars = sum(len(a) for a in ans_by_rid.values())
    print(f"\n=== [{name}] {n} запросов за {elapsed:.1f}s "
          f"| {tot_chars/n:.0f} симв/ответ "
          f"| экстраполяция на 4000: ~{elapsed/n*4000/60:.1f} мин (только L4) ===")

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            rec = {"rid": r["rid"], "query": r["question"],
                   "answer": ans_by_rid[r["rid"]]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"candidates -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--eval_jsonl", default="data/eval.jsonl")
    p.add_argument("--limit", type=int, default=0, help="0 = все 800; иначе первые N")
    p.add_argument("--profiles", default="baseline_672",
                   help="через запятую; имена из PROFILES")
    p.add_argument("--out_dir", default="data")
    args = p.parse_args()

    names = [s.strip() for s in args.profiles.split(",") if s.strip()]
    unknown = [n for n in names if n not in PROFILES]
    if unknown:
        raise SystemExit(f"неизвестные профили: {unknown}\nдоступны: {list(PROFILES)}")

    with open(args.eval_jsonl, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f]
    if args.limit > 0:
        raw = raw[: args.limit]
    # eval хранит query/reference; движку нужен формат входа контеста.
    rows = [{"rid": r["rid"], "question": r["query"]} for r in raw]

    # движок грузится один раз (на baseline-конфиге); профили мутируют config.
    engine = SchoolQAEngine(make_config(args.model_dir, {}))

    for name in names:
        out_path = os.path.join(args.out_dir, f"cand_{name}.jsonl")
        run_profile(engine, name, PROFILES[name], rows, out_path)


if __name__ == "__main__":
    main()
