"""Rejection-sampling: N кандидатов на каждый train-запрос через vLLM.

Шаг 1 self-distill (RAFT). Берём уже дообученную SFT-модель, на подвыборке
train-запросов генерим по N разнообразных (temperature) ответов. На шаге отбора
(select_distill.py) эвристики + локальный судья выберут лучший, и из них
соберётся новый, «выше эталона» SFT-набор для дообучения.

Промпт/template ДОСЛОВНО как в рантайме (source/engine.py): minimal system,
add_generation_prompt, enable_thinking=False.

DATA-PARALLEL на 2×T4 (предпочтительно для маленькой модели — T4 без NVLink,
tensor-parallel упирался бы в PCIe): запускаем ДВА процесса, каждый на своей
карте (CUDA_VISIBLE_DEVICES) с --shard_id/--num_shards, потом склеиваем шарды.
4B fp16 = ~8GB, влезает в одну T4 целиком, межкарточного обмена ноль.

    # карта 0, первая половина запросов:
    CUDA_VISIBLE_DEVICES=0 python gen_distill.py --shard_id 0 --num_shards 2 \
        --model_dir work/merged_4b --out data/distill_cand_4b.0.jsonl --n 6 --limit 4000 &
    # карта 1, вторая половина:
    CUDA_VISIBLE_DEVICES=1 python gen_distill.py --shard_id 1 --num_shards 2 \
        --model_dir work/merged_4b --out data/distill_cand_4b.1.jsonl --n 6 --limit 4000 &
    wait
    cat data/distill_cand_4b.0.jsonl data/distill_cand_4b.1.jsonl > data/distill_cand_4b.jsonl

(оркестровка + склейка сделаны в kaggle_pipeline.ipynb). Для одной карты —
--num_shards 1. Выход (jsonl на запрос): {"id", "query", "reference", "candidates": [...]}.
"id" — глобальный индекс в train (стабилен между шардами).
"""
import argparse
import json
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MINIMAL_SYSTEM = "Отвечай на языке вопроса."


def load_pairs(path: str, limit: int) -> list[tuple[int, str, str]]:
    """train_*.jsonl (messages) → [(global_id, query, reference)]. minimal-вариант.

    global_id = позиция в train (а НЕ позиция внутри шарда), чтобы id оставался
    стабильным между data-parallel шардами и совпадал с порядком train_minimal.
    """
    pairs: list[tuple[int, str, str]] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            if "messages" in r:
                msgs = r["messages"]
                q = next(m["content"] for m in msgs if m["role"] == "user")
                a = next(m["content"] for m in msgs if m["role"] == "assistant")
            else:
                q, a = r["query"], r["answer"]
            pairs.append((i, q, a))
            if limit and len(pairs) >= limit:
                break
    return pairs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--train_jsonl", default="data/train_minimal.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=6, help="кандидатов на запрос")
    p.add_argument("--limit", type=int, default=4000, help="0 = все train-запросы")
    p.add_argument("--max_tokens", type=int, default=1280)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--dtype", default="float16", help="T4 → float16")
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                   help="1 для data-parallel шардов (по карте на процесс)")
    p.add_argument("--num_shards", type=int, default=1,
                   help="на сколько data-parallel процессов делим запросы")
    p.add_argument("--shard_id", type=int, default=0, help="индекс этого шарда [0..num_shards)")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = p.parse_args()

    pairs = load_pairs(args.train_jsonl, args.limit)
    if args.num_shards > 1:
        pairs = pairs[args.shard_id :: args.num_shards]   # чередование → равная нагрузка
        print(f"шард {args.shard_id}/{args.num_shards}: {len(pairs)} запросов")
    print(f"запросов: {len(pairs)}  × n={args.n} = {len(pairs) * args.n} генераций")

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": MINIMAL_SYSTEM},
             {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for _, q, _ in pairs
    ]

    llm = LLM(
        model=args.model_dir, dtype=args.dtype, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True, seed=0,
    )
    sp = SamplingParams(
        n=args.n, temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    t = time.time()
    outputs = llm.generate(prompts, sampling_params=sp)
    print(f"генерация заняла {time.time() - t:.0f}s")

    with open(args.out, "w", encoding="utf-8") as f:
        for (gid, q, ref), out in zip(pairs, outputs):
            cands = [o.text.strip() for o in out.outputs]
            rec = {"id": gid, "query": q, "reference": ref, "candidates": cands}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"кандидаты → {args.out}")


if __name__ == "__main__":
    main()
