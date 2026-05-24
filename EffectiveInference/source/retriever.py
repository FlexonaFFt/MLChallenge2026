"""Поиск по обучающему пулу для no-GPU комбо.

ExactMatcher — точное совпадение нормализованного запроса → отдать эталон
напрямую (без модели). Безопасно: НЕ fuzzy, поэтому параметрические задачи
(математика/физика с разными числами) не путаются.

FewShotRetriever (TF-IDF) оставлен на будущее, sklearn импортируется лениво.
"""
import json
import os
import re

_WS = re.compile(r"\s+")
_EDGE = re.compile(r"^[\s\"'«»`.,:;!?()\-—–]+|[\s\"'«»`.,:;!?()\-—–]+$")


def normalize_query(q: str) -> str:
    q = q.replace("ё", "е").lower()
    q = _WS.sub(" ", q)
    q = _EDGE.sub("", q)
    return q


class ExactMatcher:
    def __init__(self, pool_path: str) -> None:
        self.by_query: dict[str, str] = {}
        with open(pool_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                key = normalize_query(r["query"])
                if key and key not in self.by_query:
                    self.by_query[key] = r["answer"]

    def lookup(self, question: str) -> str | None:
        return self.by_query.get(normalize_query(question))


class FewShotRetriever:
    def __init__(self, pool_path: str, ngram_range=(2, 4), min_df: int = 2) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer  # lazy
        with open(pool_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        self.queries = [r["query"] for r in rows]
        self.answers = [r["answer"] for r in rows]
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram_range, min_df=min_df
        )
        self.matrix = self.vectorizer.fit_transform(self.queries)

    def topk(self, question: str, k: int, max_answer_chars: int) -> list[tuple[str, str]]:
        import numpy as np
        qv = self.vectorizer.transform([question])
        sims = (self.matrix @ qv.T).toarray().ravel()
        pool = min(len(sims), max(k * 6, 30))
        cand = np.argpartition(-sims, pool - 1)[:pool]
        cand = cand[np.argsort(-sims[cand])]
        out: list[tuple[str, str]] = []
        for i in cand:
            if sims[i] <= 0:
                break
            if len(self.answers[i]) <= max_answer_chars:
                out.append((self.queries[i], self.answers[i]))
                if len(out) >= k:
                    break
        return out


def default_pool_path() -> str:
    return os.path.join(os.path.dirname(__file__), "fewshot_pool.jsonl")
