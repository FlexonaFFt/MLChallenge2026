"""LoRA SFT для Qwen3-1.7B на школьных Q&A. Рассчитан на Colab.

Colab-quickstart (в ячейке):

    !pip install -q -U transformers peft accelerate datasets bitsandbytes
    from google.colab import drive; drive.mount('/content/drive')
    # положить train_*.jsonl рядом или указать путь
    # быстрая итерация на T4 (~30-50 мин): подвыборка + 1 эпоха
    !python train_lora.py \
        --variant minimal \
        --train_jsonl data/train_minimal.jsonl \
        --output_dir /content/drive/MyDrive/effinf/lora_minimal \
        --max_samples 3000 --epochs 1

    # полный прогон (лучше на A100/L4): убрать --max_samples, --epochs 2


Маскирование промпта: loss считается ТОЛЬКО по токенам ответа ассистента
(токены system+user → label -100). Chat-template и enable_thinking=False
дословно совпадают с рантаймом (source/engine.py), иначе train/inference разъедутся.

dtype определяется автоматически: bf16 на Ampere+ (A100/L4), иначе fp16 (T4).
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
    p.add_argument("--variant", choices=["minimal", "rich"], required=True)
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen3-4B")
    p.add_argument("--eval_jsonl", default="", help="held-out для eval-loss (опц.)")
    p.add_argument("--eval_samples", type=int, default=300)
    p.add_argument("--resume", action="store_true",
                   help="докатить с последнего чекпоинта в output_dir после обрыва")
    p.add_argument("--load_4bit", action="store_true",
                   help="QLoRA: грузить базу в 4-bit (для 7-8B-учителя на 24GB)")
    # 1024 РЕЗАЛ длинные эталоны (бывают >1700 ток) → терялась supervision по
    # сочинениям/разборам. 2048 покрывает p99 и влезает на A5000 (LoRA, batch 1).
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="0 = весь датасет; иначе обучать на первых N (быстрая итерация).",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def pick_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def eval_records(path: str, variant: str, limit: int) -> list[dict]:
    """eval.jsonl ({rid,query,reference}) → messages с тем же системным промптом."""
    from prompts import MINIMAL_SYSTEM, RICH_SYSTEM
    system = MINIMAL_SYSTEM if variant == "minimal" else RICH_SYSTEM
    rows = load_jsonl(path)
    if limit > 0:
        rows = rows[:limit]
    return [
        {"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": r["query"]},
            {"role": "assistant", "content": r["reference"]},
        ]}
        for r in rows
    ]


def build_dataset(records: list[dict], tokenizer, max_len: int) -> Dataset:
    """Токенизирует messages, маскируя всё до ответа ассистента."""

    def encode(messages: list[dict]) -> dict:
        # tokenize=False + отдельная токенизация — chat-template уже добавляет
        # спец-токены, поэтому add_special_tokens=False. (tokenize=True в части
        # версий transformers возвращает Encoding, а не list[int].)
        full_txt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prompt_txt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full = tokenizer(full_txt, add_special_tokens=False)["input_ids"]
        prompt = tokenizer(prompt_txt, add_special_tokens=False)["input_ids"]
        full = full[:max_len]
        prompt_len = min(len(prompt), len(full))
        labels = [IGNORE] * prompt_len + full[prompt_len:]
        return {"input_ids": full, "labels": labels, "length": len(full)}

    encoded = [encode(r["messages"]) for r in records]
    # отбрасываем примеры, где ответ целиком обрезан max_len (loss пустой)
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
    dtype, use_fp16 = pick_dtype()
    print(f"dtype={dtype} fp16_flag={use_fp16}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    if args.load_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quant_cfg,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if args.load_4bit:
        model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    records = load_jsonl(args.train_jsonl)
    if args.max_samples > 0:
        records = records[: args.max_samples]
        print(f"subsampled to {len(records)} examples (fast mode)")
    ds = build_dataset(records, tokenizer, args.max_len)
    print(f"train examples after encode/filter: {len(ds)}")

    eval_ds = None
    if args.eval_jsonl:
        eval_ds = build_dataset(
            eval_records(args.eval_jsonl, args.variant, args.eval_samples),
            tokenizer, args.max_len,
        )
        print(f"eval examples: {len(eval_ds)}")

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
        eval_strategy="epoch" if eval_ds is not None else "no",
        per_device_eval_batch_size=max(1, args.batch),
        bf16=not use_fp16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        eval_dataset=eval_ds,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=args.resume)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
