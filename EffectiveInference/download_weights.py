"""Скачивает cyankiwi/Qwen3.5-4B-AWQ-4bit с Hugging Face в ./weights.

Запустите ОДИН раз локально перед сборкой образа:

    pip install -U "huggingface_hub>=0.23"
    python3 download_weights.py

После этого папка ./weights попадает в docker image через `COPY . .` и
оказывается в /workspace/weights внутри контейнера. На проверяющем сервере
интернета нет (--network none), поэтому веса обязательно должны быть
внутри образа.
"""
from huggingface_hub import snapshot_download

TARGET_REPO = "cyankiwi/Qwen3.5-4B-AWQ-4bit"

PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.txt",
    "*.jinja",
    "tokenizer*",
    ".*",
]


def dl(repo: str, out: str) -> None:
    path = snapshot_download(repo_id=repo, local_dir=out, allow_patterns=PATTERNS)
    print(f"{repo} -> {path}")


def main() -> None:
    dl(TARGET_REPO, "weights")


if __name__ == "__main__":
    main()
