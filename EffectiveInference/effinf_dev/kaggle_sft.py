"""SFT школьных Q&A под Kaggle 2×T4 (Turing: fp16, без bf16/FlashAttn2).

Отличия от train_lora.py (тот заточен под одиночную A5000/Colab с device_map=auto):
  - DDP вместо наивного model-parallel: запускается через `accelerate launch
    --multi_gpu --num_processes 2`, каждый процесс держит свою копию модели на
    своей T4. device_map привязан к локальному рангу (НЕ "auto" — иначе слой
    размажется по картам и DDP сломается).
  - QLoRA 4-bit по умолчанию: 4B-база в nf4 ≈ 2.7GB → с запасом влезает в 16GB
    T4 вместе с активациями (grad-checkpointing) и оставляет место под KV/DDP.
    1.7B тоже грузим в 4-bit ради единообразия (можно --no_4bit, если есть VRAM).
  - fp16 жёстко (T4 не умеет bf16).

Маскирование промпта и chat-template ДОСЛОВНО как в train_lora.py / source/engine.py
(enable_thinking=False), иначе train/inference разъедутся.

Запуск (внутри notebook-ячейки):

    accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 \
        kaggle_sft.py \
        --train_jsonl data/train_minimal.jsonl \
        --output_dir work/lora_4b \
        --base_model Qwen/Qwen3-4B --epochs 2

На 1 GPU: `python kaggle_sft.py ...` (accelerate сам сведётся к одному процессу).
"""
import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

IGNORE = -100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen3-4B")
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max_samples", type=int, default=0,
                   help="0 = весь датасет; иначе первые N (быстрая итерация)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--neftune", type=float, default=5.0,
                   help="NEFTune noise alpha (0 = выкл). Дешёвый прирост качества SFT.")
    p.add_argument("--no_4bit", action="store_true",
                   help="грузить базу в fp16 вместо QLoRA nf4 (нужно больше VRAM)")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def local_device() -> int:
    """Индекс GPU текущего DDP-процесса (для device_map привязки к рангу).

    Возвращаем int (не строку): новый transformers делает torch.device(value), а
    torch.device('0') невалиден — нужен int 0 или 'cuda:0'.
    """
    try:
        from accelerate import PartialState
        return PartialState().local_process_index
    except Exception:
        return 0


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_dataset(records: list[dict], tokenizer, max_len: int) -> Dataset:
    """Токенизирует messages, маскируя всё до ответа ассистента (loss только по ответу)."""

    def encode(messages: list[dict]) -> dict:
        full_txt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        prompt_txt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        full = tokenizer(full_txt, add_special_tokens=False)["input_ids"]
        prompt = tokenizer(prompt_txt, add_special_tokens=False)["input_ids"]
        full = full[:max_len]
        prompt_len = min(len(prompt), len(full))
        labels = [IGNORE] * prompt_len + full[prompt_len:]
        return {"input_ids": full, "labels": labels, "length": len(full)}

    encoded = [encode(r["messages"]) for r in records]
    encoded = [e for e in encoded if any(l != IGNORE for l in e["labels"])]
    return Dataset.from_list(encoded)


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, labels, attn = [], [], []
        for x in batch:
            n = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_id] * n)
            labels.append(x["labels"] + [IGNORE] * n)
            attn.append([1] * len(x["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    if not args.no_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map={"": f"cuda:{local_device()}"},   # 'cuda:N' (НЕ голый int/str — новый transformers требует валидную device-строку); НЕ "auto"
        quantization_config=quant_cfg,
        attn_implementation="eager",        # T4 не поддерживает FlashAttention-2
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if quant_cfg is not None:
        model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    records = load_jsonl(args.train_jsonl)
    if args.max_samples > 0:
        records = records[: args.max_samples]
        print(f"subsampled to {len(records)} examples")
    ds = build_dataset(records, tokenizer, args.max_len)
    print(f"train examples after encode/filter: {len(ds)}")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        group_by_length=True,
        length_column_name="length",
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        neftune_noise_alpha=args.neftune if args.neftune > 0 else None,
        optim="paged_adamw_8bit" if quant_cfg is not None else "adamw_torch",
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=args.resume)

    # сохраняем только на главном процессе
    if trainer.is_world_process_zero():
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
