"""Локальный open-weights судья на vLLM (без API). Основной офлайн-сигнал.

Pairwise: для каждого вопроса показываем судье ответ модели и эталон в
случайном порядке, просим выбрать лучший. Метрика — win-rate против эталона:
(победы + 0.5*ничьи) / N. 0.5 = паритет с эталоном.

    python judge_local.py --candidates data/cand_minimal.jsonl

Судья по умолчанию Qwen2.5-3B-Instruct (влезает даже на T4). На L4/A100 можно
--judge_model Qwen/Qwen2.5-7B-Instruct. Это НЕ копия их reward-модели, но
честно ранжирует конфиги между собой, чтобы не жечь сабмиты.
"""
import argparse
import json
import random

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SYSTEM = "Ты — строгий и беспристрастный эксперт, оценивающий ответы на школьные вопросы."

TEMPLATE = """Вопрос школьника:
{query}

Ответ A:
{a}

Ответ B:
{b}

Сравни ответы по фактической правильности, полноте, ясности и корректности
оформления (формулы, шаги, итоговый ответ). Какой ответ лучше?
Ответь РОВНО одним токеном: A, B или TIE."""


def parse_verdict(text: str, cand_is_a: bool) -> str:
    t = text.strip().upper()
    if t.startswith("TIE") or "TIE" in t[:5]:
        return "tie"
    first = next((ch for ch in t if ch in "AB"), None)
    if first is None:
        return "tie"
    chose_a = first == "A"
    return "win" if chose_a == cand_is_a else "loss"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--eval_jsonl", default="data/eval.jsonl")
    p.add_argument("--judge_model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max_n", type=int, default=0, help="0 = все held-out")
    p.add_argument("--dtype", default="auto", help="T4 → float16")
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--out", default="", help="опц. jsonl с вердиктами")
    args = p.parse_args()

    refs = {}
    with open(args.eval_jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            refs[r["rid"]] = r["reference"]

    with open(args.candidates, encoding="utf-8") as f:
        cands = [json.loads(line) for line in f]
    if args.max_n > 0:
        cands = cands[: args.max_n]

    tok = AutoTokenizer.from_pretrained(args.judge_model, use_fast=True)

    prompts, meta = [], []
    for c in cands:
        rng = random.Random(c["rid"])
        cand_is_a = rng.random() < 0.5
        a, b = (c["answer"], refs[c["rid"]]) if cand_is_a else (refs[c["rid"]], c["answer"])
        user = TEMPLATE.format(query=c["query"], a=a, b=b)
        prompts.append(
            tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            )
        )
        meta.append((c["rid"], cand_is_a))

    llm = LLM(model=args.judge_model, dtype=args.dtype,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.9, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=4, top_k=-1)
    outputs = llm.generate(prompts, sampling_params=sp)

    results = []
    for (rid, cand_is_a), o in zip(meta, outputs):
        v = parse_verdict(o.outputs[0].text, cand_is_a)
        results.append({"rid": rid, "verdict": v})

    n = len(results)
    wins = sum(r["verdict"] == "win" for r in results)
    ties = sum(r["verdict"] == "tie" for r in results)
    losses = sum(r["verdict"] == "loss" for r in results)
    score = (wins + 0.5 * ties) / n if n else 0

    print(f"\n=== {args.candidates} vs эталон | судья {args.judge_model} (N={n}) ===")
    print(f"win={wins} tie={ties} loss={losses}")
    print(f"win-rate против эталона: {score:.3f}  (0.5 = паритет)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"вердикты → {args.out}")


if __name__ == "__main__":
    main()
