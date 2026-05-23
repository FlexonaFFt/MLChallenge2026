"""Мержит LoRA-адаптер в базовую модель → полные safetensors для vLLM.

На Colab:

    !python merge_lora.py \
        --base_model Qwen/Qwen3-1.7B \
        --adapter /content/drive/MyDrive/effinf/lora_minimal \
        --out /content/drive/MyDrive/effinf/merged_minimal

vLLM грузит результат как обычную модель (без адаптеров в рантайме).
Этот же каталог потом кладётся в weights/ внутри посылки.
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)

    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"merged model saved to {args.out}")


if __name__ == "__main__":
    main()
