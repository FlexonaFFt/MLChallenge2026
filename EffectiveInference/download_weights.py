"""Скачивает Qwen3-0.6B с Hugging Face в ./weights.

Запустите ОДИН раз локально перед сборкой образа:

    pip install -U "huggingface_hub>=0.23"
    python3 download_weights.py

После этого папка ./weights попадает в docker image через `COPY . .` и
оказывается в /workspace/weights внутри контейнера. На проверяющем сервере
интернета нет (--network none), поэтому веса обязательно должны быть
внутри образа.
"""
from huggingface_hub import snapshot_download


REPO_ID = "Qwen/Qwen3-1.7B"
LOCAL_DIR = "weights"


def main() -> None:
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        allow_patterns=[
            "*.json",          # config.json, tokenizer_config.json, generation_config.json, ...
            "*.safetensors",   # веса
            "*.txt",           # vocab, merges
            "tokenizer*",      # tokenizer.json, tokenizer.model
            "*.jinja",         # chat template
        ],
    )
    print(f"weights downloaded to: {path}")


if __name__ == "__main__":
    main()
