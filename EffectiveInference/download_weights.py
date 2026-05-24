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

# Целевая (8B AWQ) + черновик для speculative decoding (0.6B, тот же токенайзер Qwen3).
TARGET_REPO = "Qwen/Qwen3-8B-AWQ"
DRAFT_REPO = "Qwen/Qwen3-0.6B"

PATTERNS = ["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.jinja"]


def dl(repo: str, out: str) -> None:
    path = snapshot_download(repo_id=repo, local_dir=out, allow_patterns=PATTERNS)
    print(f"{repo} -> {path}")


def main() -> None:
    dl(TARGET_REPO, "weights")
    dl(DRAFT_REPO, "draft")


if __name__ == "__main__":
    main()
