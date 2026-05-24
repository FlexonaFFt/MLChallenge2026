"""Офлайн-ретривер похожих обучающих примеров для динамического few-shot.

TF-IDF по символьным n-граммам (char_wb) — устойчив к русской морфологии и
не требует отдельной модели/эмбеддингов. Индекс строится из shipped-пула
source/fewshot_pool.jsonl при старте контейнера (несколько секунд на 10k).
"""
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class FewShotRetriever:
    def __init__(self, pool_path: str, ngram_range=(2, 4), min_df: int = 2) -> None:
        with open(pool_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        self.queries = [r["query"] for r in rows]
        self.answers = [r["answer"] for r in rows]
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram_range, min_df=min_df
        )
        # матрица L2-нормирована (norm='l2' по умолчанию) → dot = косинус
        self.matrix = self.vectorizer.fit_transform(self.queries)

    def topk(self, question: str, k: int, max_answer_chars: int) -> list[tuple[str, str]]:
        qv = self.vectorizer.transform([question])
        sims = (self.matrix @ qv.T).toarray().ravel()
        # берём с запасом, потом отфильтруем слишком длинные ответы
        pool = min(len(sims), max(k * 6, 30))
        cand = np.argpartition(-sims, pool - 1)[:pool]
        cand = cand[np.argsort(-sims[cand])]
        out: list[tuple[str, str]] = []
        for i in cand:
            if sims[i] <= 0:
                break
            ans = self.answers[i]
            if len(ans) <= max_answer_chars:
                out.append((self.queries[i], ans))
                if len(out) >= k:
                    break
        return out


def default_pool_path() -> str:
    return os.path.join(os.path.dirname(__file__), "fewshot_pool.jsonl")
