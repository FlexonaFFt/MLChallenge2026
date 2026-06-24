# LLM Models

Use local LLMs as task components, not as magic solvers.

## Available Model Roles

| Role | Model | Use |
|---|---|---|
| Main local LLM | Qwen3.6 35B A3B UD_Q3_K_M | classification prompts, extraction, reranking, synthetic labels |
| Alternative LLM | Gemma 4 26B A4B Q4_K_M | compare outputs, ensemble, fallback |
| Fast LLM | YandexGPT-5-Lite-8B Q4_K_M | quick batch labels, Russian text, cheap checks |
| Vision-language | Qwen3-VL-8B / Qwen3-VL-30B | image+text tasks only |
| Embeddings | nomic-embed-text-v1.5 | retrieval, clustering, semantic features |

## When To Use LLMs

Use LLMs for:

- information extraction
- weak labels / pseudo-labels
- text normalization
- feature generation
- pair relevance judging
- answer generation when metric requires text
- reranking top candidates

Do not use LLMs first when:

- a TF-IDF or tabular baseline is not ready
- metric is unclear
- output format is not validated
- inference would not finish in time

## llama.cpp Server

Ask organizers for the exact command. Expected shape:

```bash
llama-server \
  -m /models/MODEL.gguf \
  -c 8192 \
  --host 127.0.0.1 \
  --port 8080
```

Increase context only when the task needs it. Larger context costs memory.

## Local API Pattern

Use OpenAI-compatible local endpoints when available:

```text
http://127.0.0.1:8080/v1
http://localhost:1234/v1
```

Use dummy API keys only for local tools that require a value.

## Batch Inference Rules

- Cache every response to JSONL.
- Include input id, prompt version, model name, raw output, parsed output, and error.
- Make prompts deterministic first.
- Prefer strict JSON for extraction/classification.
- Add a parser before running full test inference.
- Run 20 rows first, then 200, then full dataset.

## Prompt Templates

### Classification

```text
You are solving a competition task.
Return only one label from this list: {labels}

Text:
{text}

Label:
```

### JSON Extraction

```text
Extract fields from the text.
Return valid JSON only. No markdown.

Schema:
{schema}

Text:
{text}
```

### Pair Relevance

```text
Decide whether the document is relevant to the query.
Return JSON only: {"label": "relevant" or "not_relevant", "confidence": 0.0-1.0}

Query:
{query}

Document:
{document}
```

### Handoff Summary

```text
Summarize the useful plan for src code.
Include metric, baseline, exact implementation steps, commands, and risks.
Do not include brainstorming that is not actionable.
```

## Model Selection

- Need best local quality on text: start with Qwen3.6 35B.
- Need quick Russian labels: try YandexGPT-5-Lite-8B.
- Need second opinion: compare with Gemma 4 26B.
- Need retrieval/features: use embeddings before generation.
- Need image understanding: use Qwen3-VL only if images are part of the task.

