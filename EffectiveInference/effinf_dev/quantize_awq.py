"""Квантизация смерженной модели в AWQ (int4) для инференса на L4.

Недостающее звено между merge_lora.py (отдаёт fp16) и посылкой: vLLM на L4
грузит AWQ быстрее и экономит память (текущая прод-модель — тоже AWQ).

    python quantize_awq.py \
        --model work/merged_qwen3_4b \
        --out   work/awq_qwen3_4b \
        --calib_jsonl data/train_minimal.jsonl

Калибровка — на РЕАЛЬНЫХ примерах из train (chat-формат, как в рантайме),
чтобы квантизация не «уплыла» на школьном домене. Результат кладётся в
weights/ посылки вместо стоковых AWQ-весов.

ПРИМ.: если autoawq не поддержит конкретную ревизию Qwen3 — альтернатива
llm-compressor (W4A16), API другой; см. README пайплайна.
"""
import argparse
import json
import random

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Тот же минимальный системный промпт, что в рантайме (source/config.py).
MINIMAL_SYSTEM = "Отвечай на языке вопроса."


def build_calib(path: str, tok, n: int, max_len: int) -> list[str]:
    """N примеров в виде полного chat-текста (system+user+assistant)."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    random.Random(0).shuffle(rows)
    calib: list[str] = []
    for r in rows:
        msgs = r["messages"] if "messages" in r else [
            {"role": "system", "content": MINIMAL_SYSTEM},
            {"role": "user", "content": r["query"]},
            {"role": "assistant", "content": r["answer"]},
        ]
        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        # грубо ограничиваем длину калибровочного семпла по символам
        calib.append(txt[: max_len * 4])
        if len(calib) >= n:
            break
    return calib


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="смерженная fp16-модель")
    p.add_argument("--out", required=True)
    p.add_argument("--calib_jsonl", required=True)
    p.add_argument("--calib_n", type=int, default=256)
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--w_bit", type=int, default=4)
    p.add_argument("--group_size", type=int, default=128)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    calib = build_calib(args.calib_jsonl, tok, args.calib_n, args.max_len)
    print(f"калибровка: {len(calib)} примеров")

    model = AutoAWQForCausalLM.from_pretrained(args.model, safetensors=True)
    quant_config = {
        "zero_point": True,
        "q_group_size": args.group_size,
        "w_bit": args.w_bit,
        "version": "GEMM",
    }
    model.quantize(tok, quant_config=quant_config, calib_data=calib)

    model.save_quantized(args.out)
    tok.save_pretrained(args.out)
    print(f"AWQ-модель сохранена в {args.out}")
    print("→ скопировать содержимое в weights/ посылки (заменив старые веса)")


if __name__ == "__main__":
    main()
