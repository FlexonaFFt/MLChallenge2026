"""Прокси-судья: pairwise-сравнение ответов кандидата с эталоном через Claude API.

Запускается на Mac. Нужен ключ:

    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python judge.py --candidates data/cand_minimal.jsonl

Логика: для каждого rid берём ответ кандидата и эталон, случайно меняем их
местами (борьба с position bias), просим судью выбрать лучший. Метрика —
win-rate кандидата против эталона: (победы + 0.5*ничьи) / N. 0.5 = паритет
с эталоном, >0.5 = кандидат в среднем лучше эталона по мнению судьи.

Это НЕ копия их reward-модели, но даёт быстрый офлайн-сигнал для ранжирования
конфигов, чтобы не жечь 4 сабмита/сутки.
"""
import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

from anthropic import Anthropic

PROMPT = """Ты — строгий эксперт, оценивающий ответы на школьный вопрос.

Вопрос:
{query}

Ответ 1:
{a}

Ответ 2:
{b}

Сравни ответы по: фактической правильности, полноте, ясности объяснения и
корректности оформления (формулы, шаги решения, итоговый ответ). Какой ответ
лучше отвечает на вопрос школьника?

Ответь РОВНО одним словом без пояснений: "1", "2" или "TIE"."""


def judge_one(client, model, query, cand, ref, seed) -> str:
    """Возвращает 'win' | 'loss' | 'tie' для кандидата."""
    rng = random.Random(seed)
    cand_is_first = rng.random() < 0.5
    a, b = (cand, ref) if cand_is_first else (ref, cand)
    msg = client.messages.create(
        model=model,
        max_tokens=8,
        messages=[{"role": "user", "content": PROMPT.format(query=query, a=a, b=b)}],
    )
    text = msg.content[0].text.strip().upper()
    if "TIE" in text:
        return "tie"
    pick = re.search(r"[12]", text)
    if not pick:
        return "tie"
    chose_first = pick.group() == "1"
    return "win" if chose_first == cand_is_first else "loss"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--eval_jsonl", default="data/eval.jsonl")
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max_n", type=int, default=0, help="0 = все held-out")
    p.add_argument("--out", default="", help="опц. jsonl с по-элементными вердиктами")
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

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def work(c):
        verdict = judge_one(
            client, args.model, c["query"], c["answer"], refs[c["rid"]], c["rid"]
        )
        return {"rid": c["rid"], "verdict": verdict}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(work, cands))

    n = len(results)
    wins = sum(r["verdict"] == "win" for r in results)
    ties = sum(r["verdict"] == "tie" for r in results)
    losses = sum(r["verdict"] == "loss" for r in results)
    score = (wins + 0.5 * ties) / n

    print(f"\n=== {args.candidates} vs эталон (N={n}) ===")
    print(f"win={wins} tie={ties} loss={losses}")
    print(f"win-rate против эталона: {score:.3f}  (0.5 = паритет)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"вердикты → {args.out}")


if __name__ == "__main__":
    main()
