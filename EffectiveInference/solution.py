"""Бейзлайн-решение: vLLM + Qwen3-0.6B.

На входе:  /workspace/input.pickle
На выходе: /workspace/output.json
Веса:      /workspace/weights (предварительно скачиваются download_weights.py)
"""
import json
import os
import pickle

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


MODEL_DIR = "./weights"
MAX_NEW_TOKENS = 512
MAX_MODEL_LEN = 4096


def main() -> None:
    with open("input.pickle", "rb") as f:
        rows = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

    llm = LLM(
        model=MODEL_DIR,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
        tokenizer_mode="auto",
        seed=0,
    )

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["question"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in rows
    ]

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
        top_k=-1,
    )

    outputs = llm.generate(prompts, sampling_params=sampling)

    result = [
        {"rid": row["rid"], "answer": out.outputs[0].text.strip()}
        for row, out in zip(rows, outputs)
    ]

    with open("output.json", "w") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
