# Модели

Используй локальные модели как инструменты для задачи, а не как замену валидации.

## Роли

| Роль | Модель | Использование |
|---|---|---|
| Основная локальная LLM | Qwen3.6 35B A3B UD_Q3_K_M | classification prompts, extraction, reranking, pseudo-labels |
| Альтернативная LLM | Gemma 4 26B A4B Q4_K_M | сравнение, ensemble, fallback |
| Быстрая LLM | YandexGPT-5-Lite-8B Q4_K_M | быстрые русские labels, дешевые проверки |
| Vision-language | Qwen3-VL-8B / Qwen3-VL-30B | задачи image+text |
| Embeddings | nomic-embed-text-v1.5 | retrieval, clustering, semantic features |

## Когда использовать

- Classification/extraction: используй строгие labels или JSON.
- Retrieval: сначала embeddings, потом generation.
- Pseudo-labeling: только после валидированного бейзлайна.
- Reranking: запускай LLM только на top candidates.
- Generation: до полного inference сделай parser и repair step.

## Когда не использовать

- Еще нет валидного бейзлайна.
- Метрика или output format непонятны.
- Полный inference не успеет завершиться.
- Prompt output нельзя надежно распарсить.

## Пример llama.cpp

Точные команды уточни у организаторов. Ожидаемый формат:

```bash
llama-server -m /models/MODEL.gguf -c 8192 --host 127.0.0.1 --port 8080
```

Увеличивай context только когда это реально нужно.

## Правила batch inference

- Сначала тестируй 20 строк, потом 200, потом весь датасет.
- Кэшируй ответы в JSONL.
- Сохраняй id, model, prompt version, raw output, parsed output, error.
- Не перезаписывай полезные cached outputs.

## Шаблоны prompt

Classification:

```text
Верни только один label из списка: {labels}

Текст:
{text}

Label:
```

Extraction:

```text
Верни только валидный JSON. Без markdown.

Schema:
{schema}

Текст:
{text}
```

Relevance:

```text
Верни только JSON: {"label":"relevant|not_relevant","confidence":0.0}

Query:
{query}

Document:
{document}
```
