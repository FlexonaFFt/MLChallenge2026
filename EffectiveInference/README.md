# Effective Inference On School Questions

Решение задачи C: быстрые ответы на школьные вопросы с помощью маленькой
локальной LLM. Проверяющая система запускает Docker-контейнер, передает
`input.pickle` и ожидает `output.json` с одним ответом на каждый `rid`.

## Скор

Лучший скор в локальном журнале посылок: **67.2**.

## Подход

- vLLM runtime поверх локальных весов из `./weights`.
- Классификатор категорий: математика, морфология, химия, английский, сочинения,
  грамматика и физика.
- Category-specific few-shot prompts для base-модели.
- Exact-match shortcut по `source/fewshot_pool.jsonl` для дублирующихся вопросов.
- Динамический `max_tokens` по категориям: меньше времени на короткие ответы,
  больше бюджета на сочинения и длинные объяснения.
- Wall-clock guard: около дедлайна остаток строк заполняется пустыми ответами,
  чтобы не потерять `rid`.
- Постпроцессинг для удаления `<think>`-блоков и типовых вводных фраз.

## Структура

```text
EffectiveInference/
├── Dockerfile
├── solution.py
├── download_weights.py
├── source/
│   ├── config.py
│   ├── engine.py
│   ├── retriever.py
│   └── fewshot_pool.jsonl
└── effinf_dev/          # эксперименты, подготовка данных и evaluation scripts
```

## Запуск

```bash
python download_weights.py
docker build -t effective-inference .
docker run --rm --gpus all \
  -v "$PWD/input.pickle":/workspace/input.pickle \
  -v "$PWD/out":/workspace/out \
  effective-inference
```

Веса не коммитятся в git. Ожидаемая runtime-директория:
`EffectiveInference/weights/`.
