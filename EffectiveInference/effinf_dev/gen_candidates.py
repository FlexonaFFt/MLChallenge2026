"""Генерирует ответы кандидата на held-out (eval.jsonl) через vLLM.

Промпт/семплинг ДОСЛОВНО совпадают с рантаймом (source/engine.py):
chat-template, enable_thinking=False, greedy. Заодно печатает throughput
(tok/s) и оценку времени на 4000 запросов — для контроля 15-мин лимита.

На Colab:

    !python gen_candidates.py \
        --model_dir /content/drive/MyDrive/effinf/merged_minimal \
        --variant minimal \
        --eval_jsonl data/eval.jsonl \
        --out data/cand_minimal.jsonl

На T4 добавить --dtype float16 (bf16 не поддерживается).
Для базлайна (без дообучения) указать --model_dir на стоковые веса Qwen3-1.7B.
"""
import argparse
import json
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from prompts import MINIMAL_SYSTEM, RICH_SYSTEM


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--variant", choices=["minimal", "rich"], required=True)
    p.add_argument("--eval_jsonl", default="data/eval.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--dtype", default="auto", help="auto|float16|bfloat16 (T4 → float16)")
    p.add_argument("--limit", type=int, default=0, help="0 = все; иначе первые N (быстрая проба)")
    args = p.parse_args()

    system = MINIMAL_SYSTEM if args.variant == "minimal" else RICH_SYSTEM

    with open(args.eval_jsonl, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if args.limit > 0:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": r["query"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for r in rows
    ]

    llm = LLM(
        model=args.model_dir,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        seed=0,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, top_k=-1)

    t = time.time()
    outputs = llm.generate(prompts, sampling_params=sp)
    elapsed = time.time() - t

    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    n = len(rows)
    print(f"\n=== {n} запросов за {elapsed:.1f}s | {gen_tokens} ген-токенов "
          f"| {gen_tokens / elapsed:.0f} tok/s ===")
    print(f"средняя длина ответа: {gen_tokens / n:.0f} токенов")
    print(f"экстраполяция на 4000 запросов: ~{elapsed / n * 4000 / 60:.1f} мин "
          f"(NB: достоверно только на L4)")
    hit_cap = sum(len(o.outputs[0].token_ids) >= args.max_tokens for o in outputs)
    print(f"упёрлись в max_tokens={args.max_tokens}: {hit_cap}/{n} ({hit_cap/n*100:.1f}%)")

    with open(args.out, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outputs):
            rec = {"rid": r["rid"], "query": r["query"],
                   "answer": o.outputs[0].text.strip()}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"candidates → {args.out}")


if __name__ == "__main__":
    main()
