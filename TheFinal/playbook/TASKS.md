# Tasks

Use this file to classify the task and pick the first working baseline.

## First 15 Minutes

Find:

- `INPUT:` table, text, image, time, pairs, documents, mixed.
- `OUTPUT:` class, probability, number, text, JSON, ranking.
- `METRIC:` F1, AUC, logloss, RMSE, MAE, NDCG, MAP, exact match, custom.
- `SUBMISSION:` required columns, row count, id order, value ranges.
- `LEAKAGE_RISK:` time, groups, duplicated entities, test-like rows.

## Decision Table

| Task shape | First baseline | Then try | Avoid early |
|---|---|---|---|
| Tabular classification | CatBoost/LightGBM/sklearn | target encoding, ensembles | deep nets |
| Tabular regression | CatBoost/LightGBM/sklearn | log target, robust loss | complex feature search |
| Text classification | TF-IDF + LogisticRegression | embeddings, LLM features | fine-tuning first |
| Pair matching | TF-IDF/embedding similarity | cross features, LLM judge | full RAG stack |
| Retrieval/RAG | BM25/TF-IDF retrieval | embeddings, rerank | generation before retrieval works |
| Information extraction | rules + local LLM JSON | validation parser, voting | free-form outputs |
| Ranking/recsys | simple candidate features | LGBM ranker, embeddings | giant neural ranker |
| Time series | lag/rolling features | grouped models, trend features | random split |
| Image classification | pretrained embeddings/classifier | augmentations, ensembles | training from scratch |
| Multimodal | separate text/image baselines | fusion features, VLM | one giant prompt only |
| Text generation | prompt baseline | self-consistency, validation parser | unbounded outputs |

## Tabular

### Baseline

- Read train/test/sample submission.
- Split with stratification for classification if possible.
- Use simple preprocessing.
- Try CatBoost/LightGBM if installed; otherwise sklearn.

### Leakage

- Duplicated ids between train/test.
- Target-derived columns.
- Future timestamps.
- Group overlap between train and validation.

## Text / NLP

### Baseline

- Clean only obvious broken text.
- TF-IDF word + char ngrams.
- LogisticRegression/LinearSVC/Ridge depending on metric.

### Improvements

- Local embeddings as features.
- LLM-generated labels/features.
- Pseudo-label high-confidence test rows only after a stable baseline.

### Leakage

- Same document in train/test.
- User/item/group overlap.
- Text containing labels or target hints.

## Retrieval / QA

### Baseline

- Split documents into chunks.
- Retrieve with TF-IDF/BM25.
- Answer from top chunks only.

### Improvements

- Embeddings with `nomic-embed-text` or another local embedder.
- Rerank top candidates.
- Generate answer with citations from retrieved chunks.

## Ranking

### Baseline

- Build query-candidate rows.
- Features: text similarity, exact matches, length, ids, categories.
- Optimize the real ranking metric if possible.

### Leakage

- Candidate order correlated with target.
- Same query across train/validation.
- Future interactions in train features.

## Time Series

### Baseline

- Time-based validation only.
- Lag features.
- Rolling mean/std/min/max.
- Calendar features.

### Leakage

- Random split.
- Rolling features that use future rows.
- Normalization fitted on all data.

## Computer Vision

### Baseline

- Use installed pretrained model or image embeddings if available.
- Freeze backbone first.
- Train a small classifier head.

### Leakage

- Same image with different resize/compression.
- Filename or folder target hints.
- Group leakage by source/user/product.

## Generation

### Baseline

- Prompt with strict output format.
- Add parser and repair step.
- Score locally on validation examples.

### Improvements

- Multiple prompts and voting.
- Smaller fast model for draft, bigger model for final.
- Rule-based postprocessing.

