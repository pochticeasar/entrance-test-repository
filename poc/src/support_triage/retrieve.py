"""Поиск релевантных фрагментов базы знаний.

Упрощение PoC: TF-IDF + косинусная близость по 15 статьям в памяти. В целевой
архитектуре — векторная БД с ANN-индексом и эмбеддингами; интерфейс `Retriever.search`
при этом не меняется, меняется только реализация.
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schema import RetrievedDoc


class Retriever:
    def __init__(self, docs: list[dict]) -> None:
        self._docs = docs
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        corpus = [f"{d['title']} {d['text']}" for d in docs]
        self._matrix = self._vectorizer.fit_transform(corpus)

    @classmethod
    def from_file(cls, path: Path) -> "Retriever":
        docs = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    docs.append(json.loads(line))
        return cls(docs)

    def search(self, query: str, top_k: int = 2) -> list[RetrievedDoc]:
        scores = cosine_similarity(self._vectorizer.transform([query]), self._matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            RetrievedDoc(
                id=self._docs[i]["id"],
                title=self._docs[i]["title"],
                text=self._docs[i]["text"],
                score=float(scores[i]),
            )
            for i in ranked[:top_k]
        ]
