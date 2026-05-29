"""Судит сразу несколько cand_*.jsonl одним инстансом судьи и печатает таблицу.

Pairwise win-rate против эталона (как judge_local.py), но загрузка модели-судьи
один раз на все профили — экономит время на Kaggle. Метрика: (win+0.5*tie)/N,
0.5 = паритет с эталоном. Это НЕ копия контест-reward-модели, но честно
ранжирует профили между собой → шлём на лидерборд только победителя.

    !python judge_all.py --cands data/cand_baseline_672.jsonl,data/cand_sampling_qwen.jsonl \
        --judge_model Qwen/Qwen2.5-3B-Instruct --dtype float16
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
    if "TIE" in t[:5]:
        return "tie"
    first = next((ch for ch in t if ch in "AB"), None)
    if first is None:
        return "tie"
    return "win" if (first == "A") == cand_is_a else "loss"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cands", required=True, help="cand-файлы через запятую")
    p.add_argument("--eval_jsonl", default="data/eval.jsonl")
    p.add_argument("--judge_model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--dtype", default="auto", help="T4 -> float16")
    p.add_argument("--max_model_len", type=int, default=4096)
    args = p.parse_args()

    refs = {}
    with open(args.eval_jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            refs[r["rid"]] = r["reference"]

    tok = AutoTokenizer.from_pretrained(args.judge_model, use_fast=True)
    llm = LLM(model=args.judge_model, dtype=args.dtype,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.9, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=4, top_k=-1)

    files = [s.strip() for s in args.cands.split(",") if s.strip()]
    summary = []

    for path in files:
        with open(path, encoding="utf-8") as f:
            cands = [json.loads(line) for line in f]

        prompts, meta = [], []
        for c in cands:
            rng = random.Random(c["rid"])
            cand_is_a = rng.random() < 0.5
            a, b = ((c["answer"], refs[c["rid"]]) if cand_is_a
                    else (refs[c["rid"]], c["answer"]))
            user = TEMPLATE.format(query=c["query"], a=a, b=b)
            prompts.append(tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True))
            meta.append(cand_is_a)

        outputs = llm.generate(prompts, sampling_params=sp)
        verdicts = [parse_verdict(o.outputs[0].text, cia)
                    for cia, o in zip(meta, outputs)]
        n = len(verdicts)
        wins = verdicts.count("win")
        ties = verdicts.count("tie")
        losses = verdicts.count("loss")
        score = (wins + 0.5 * ties) / n if n else 0.0
        summary.append((path, score, wins, ties, losses, n))

    summary.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'='*70}\nРАНЖИРОВАНИЕ профилей (win-rate vs эталон, 0.5 = паритет)\n{'='*70}")
    print(f"{'профиль':<40}{'win-rate':>10}{'W/T/L':>18}")
    for path, score, w, t, l, n in summary:
        name = path.split("cand_")[-1].replace(".jsonl", "")
        print(f"{name:<40}{score:>10.3f}{f'{w}/{t}/{l}':>18}")


if __name__ == "__main__":
    main()
