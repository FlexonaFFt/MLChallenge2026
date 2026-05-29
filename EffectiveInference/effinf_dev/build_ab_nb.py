"""Generate ab_eval.ipynb — offline A/B harness for Task C on Kaggle 2×T4.

Idea: offline == production. We run the REAL SchoolQAEngine (source/engine.py +
source/config.py — the exact code that ships in the container) over held-out
eval.jsonl under several InferenceConfig profiles, then judge each with a local
Qwen2.5-3B judge (pairwise win-rate vs reference). We submit ONLY the winner to
the leaderboard, so the 4 submits/day budget is spent on validated configs.

No training here — pure inference + judging. 4B-AWQ loads once for all profiles
(gen process), then the judge loads once for all profiles (judge process). Heavy
steps run as subprocesses so vLLM frees VRAM between them.

Attach as Kaggle Dataset: the EffectiveInference folder (with
dataset_ml_challenge.parquet + effinf_dev/ + source/). Settings: GPU T4 ×2,
Internet ON. Run: python3 build_ab_nb.py
"""
import json

OUT_PATHS = ["/Users/flexonafft/Downloads/ab_eval.ipynb",
             "/Users/flexonafft/MLChallenge2026/EffectiveInference/effinf_dev/ab_eval.ipynb"]
cells = []
def _l(s): return s.splitlines(keepends=True)
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": _l(s)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _l(s)})

md("""# Task C — оффлайн A/B-харнесс (Kaggle 2×T4)

**Цель:** выжать скор на промпте/decoding базового **Qwen3-4B-AWQ**, не тратя
попытки лидерборда. Гоняем **реальный** `SchoolQAEngine` (тот же код, что в
контейнере) по набору профилей конфига → судим локально → шлём только победителя.

**НЕ тренировка** — чистый инференс + судейство. Точка отсчёта — профиль
`baseline_672` (текущий дефолт = RICH-промпт + category few-shot + dynamic
max_tokens, давший 67.2). Все профили сравниваем с ним и между собой.

**Среда:** 2×T4 16GB (Turing → fp16, без bf16). 4B-AWQ грузится один раз на все
профили, потом судья — один раз на все. Тяжёлые шаги — отдельными процессами
(vLLM освобождает VRAM на выходе).

**Перед запуском:** Settings → Accelerator = **GPU T4 ×2**, Internet = **ON**.
Add Input → датасет с папкой `EffectiveInference` (нужны `effinf_dev/`, `source/`,
`dataset_ml_challenge.parquet`).""")

md("## 0. Зависимости (Qwen3 → transformers>=4.51, vllm>=0.8.5)")
code("""!pip install -q -U vllm==0.8.5.post1 transformers==4.51.3 \\
    accelerate==1.4.* datasets pyarrow autoawq
import transformers; print('transformers', transformers.__version__)
print('deps installed — ПЕРЕЗАПУСТИ ЯДРО (Run -> Restart) и запускай со следующей ячейки')""")

md("## 1. Поиск repo, рабочий каталог, конфиг A/B")
code("""import os, shutil, glob

# Автопоиск папки EffectiveInference в подключённых датасетах.
def find_src():
    for parquet in glob.glob('/kaggle/input/**/dataset_ml_challenge.parquet', recursive=True):
        d = os.path.dirname(parquet)
        if os.path.isdir(os.path.join(d, 'effinf_dev')):
            return d
    for dev in glob.glob('/kaggle/input/**/effinf_dev', recursive=True):
        return os.path.dirname(dev)
    raise FileNotFoundError('Не нашёл EffectiveInference в /kaggle/input — проверь Add Input')

SRC = find_src()
WORK = '/kaggle/working/EffectiveInference'
if not os.path.exists(WORK):
    shutil.copytree(SRC, WORK)
DEV = os.path.join(WORK, 'effinf_dev')
os.chdir(DEV)
print('cwd =', os.getcwd())

# --- параметры A/B ---
JUDGE_MODEL = 'Qwen/Qwen2.5-3B-Instruct'
EVAL_LIMIT  = 300   # быстрая итерация; для финального ранга поставь 0 (все 800)
PROFILES    = 'baseline_672,sampling_qwen,sampling_mild,essay_headroom'

os.makedirs('data', exist_ok=True)
!nvidia-smi --query-gpu=name,memory.total --format=csv""")

md("""## 2. Веса базового Qwen3-4B-AWQ

Качаем стоковые AWQ-веса (≈3 ГБ, int4) в `weights/`. Это та же модель, что даёт
67.2 — A/B меняет только промпт/decoding поверх неё.""")
code("""WEIGHTS = '/kaggle/working/weights'
if not os.path.exists(os.path.join(WEIGHTS, 'config.json')):
    from huggingface_hub import snapshot_download
    snapshot_download('Qwen/Qwen3-4B-AWQ', local_dir=WEIGHTS,
                      allow_patterns=['*.json', '*.safetensors', '*.txt', 'tokenizer*', '*.jinja'])
!du -sh $WEIGHTS && ls $WEIGHTS""")

md("""## 3. Подготовка held-out (eval.jsonl)

`prepare_data.py` пишет `data/eval.jsonl` (детерминированные 800 held-out:
query + reference). Если файл уже в репо — пропустится быстро.""")
code("""if not os.path.exists('data/eval.jsonl'):
    !python prepare_data.py
!wc -l data/eval.jsonl""")

md("""## 4. Генерация кандидатов по всем профилям (4B-AWQ, один процесс)

`ab_eval.py` грузит движок ОДИН раз и прогоняет каждый профиль, мутируя
`InferenceConfig`. На выходе `data/cand_<profile>.jsonl` + тайминг и
экстраполяция на 4000 (достоверно только на L4 — на T4 смотрим относительные
числа между профилями).""")
code("""!python ab_eval.py --model_dir $WEIGHTS \\
    --eval_jsonl data/eval.jsonl --limit $EVAL_LIMIT \\
    --profiles $PROFILES""")

md("""## 5. Судейство всех профилей (Qwen2.5-3B, один процесс)

`judge_all.py` грузит судью один раз и печатает ранжирование по win-rate против
эталона (0.5 = паритет). Это НЕ копия контест-reward-модели, но честно ранжирует
профили между собой.""")
code("""cands = ','.join(f'data/cand_{p}.jsonl' for p in PROFILES.split(','))
!python judge_all.py --cands $cands --judge_model $JUDGE_MODEL --dtype float16""")

md("""## 6. Что дальше

- Берём профиль с **наибольшим win-rate** (и без TL-риска по таймингу).
- Переносим его настройки в `source/config.py` (поля `temperature`/`top_p`/
  `top_k`/`category_max_tokens`), `finetuned=False`.
- Шлём на лидерборд ТОЛЬКО победителя — один сабмит вместо четырёх вслепую.
- Новые гипотезы добавляй в `PROFILES` (словарь в `ab_eval.py`) и перезапускай
  ячейки 4–5. Сначала на `EVAL_LIMIT=300`, финальный ранг — на `0` (все 800).

> Если два профиля близки (Δwin-rate < ~0.02 на 300) — перегони на полных 800,
> разница может быть шумом судьи.""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
for p in OUT_PATHS:
    try:
        with open(p, "w") as f:
            json.dump(nb, f, indent=1)
        print("wrote", p, "with", len(cells), "cells")
    except FileNotFoundError:
        print("skip (no dir):", p)
