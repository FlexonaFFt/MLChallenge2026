# Задачи

Быстро классифицируй задачу, затем сделай валидный бейзлайн.

## Первые 15 минут

Запиши эти факты в `STATE.md`:

- `INPUT:` таблица, текст, изображения, время, пары, документы, смешанный формат.
- `OUTPUT:` класс, вероятность, число, текст, JSON, ранжирование.
- `METRIC:` F1, AUC, logloss, RMSE, MAE, NDCG, MAP, exact match, кастомная.
- `SUBMISSION:` колонки, число строк, порядок id, допустимые значения.
- `VALIDATION:` stratified, group, time, random, custom.
- `LEAKAGE_RISK:` время, группы, дубликаты, подсказки таргета, test-like строки.

## Роутер

| Тип задачи | Первый бейзлайн | Потом попробовать | Следить за |
|---|---|---|---|
| Tabular classification | CatBoost/LightGBM/sklearn | encoding, ансамбли | target leakage |
| Tabular regression | CatBoost/LightGBM/sklearn | log target, robust loss | будущие данные |
| Text classification | TF-IDF + LogisticRegression | embeddings, LLM features | дубликаты |
| Pair matching | TF-IDF/embedding similarity | cross features, LLM judge | group leakage |
| Retrieval/QA | TF-IDF/BM25 top-k | embeddings, rerank, generate | слабый retrieval |
| Extraction | правила + LLM JSON | parser, voting | невалидный output |
| Ranking/recsys | простые candidate features | ranker, embeddings | query leakage |
| Time series | lag/rolling features | grouped models | random split |
| Image classification | pretrained embeddings/head | augment, ensemble | подсказки в файлах |
| Multimodal | отдельные text/image baseline | fusion, VLM | один огромный prompt |
| Generation | строгий prompt + parser | self-consistency | неограниченный текст |

## Правила бейзлайна

- Сначала сделай валидный submission, потом оптимизируй.
- Локально используй точную метрику, если возможно.
- Держи один стабильный validation split для сравнений.
- Меняй одну крупную вещь за один run.
- Останавливай идею после двух запусков без прироста, если она явно не недотестирована.

## Проверки leakage

- Один и тот же id/entity/user/document в train и validation.
- Time split нарушен future features.
- Target прямо или косвенно закодирован в тексте/колонках.
- Preprocessing обучен на train+test, хотя должен быть train-only.
- Дубликаты или near-duplicates текста/изображений между split.

## Проверки submission

- Те же строки, что в sample submission.
- Тот же порядок id, если правила не говорят иначе.
- Нет missing, inf, невалидных labels или неправильных dtypes.
- Predictions clipped/normalized, если метрика этого ожидает.
- Run, из которого сделан submission, записан в `RUNS.md`.
