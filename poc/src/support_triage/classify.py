"""Классификация темы тикета.

Baseline: TF-IDF по символьным n-граммам + логистическая регрессия. Символьные
n-граммы выбраны сознательно — русская морфология и опечатки в чате ломают
пословную токенизацию, а лемматизатор тянуть в PoC не хочется.

Модель обучается на старте из data/tickets_train.jsonl (~70 синтетических
тикетов). В целевой архитектуре модель обучается офлайн на размеченных
исторических тикетах и загружается из артефакт-хранилища — см. docs/ml.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from .schema import Classification

MODEL_VERSION = "tfidf-logreg-v0.1"


def _build() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, C=10.0, class_weight="balanced"),
            ),
        ]
    )


class TopicClassifier:
    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    @classmethod
    def train(cls, dataset_path: Path) -> "TopicClassifier":
        texts, labels = _load(dataset_path)
        pipeline = _build()
        pipeline.fit(texts, labels)
        return cls(pipeline)

    def predict(self, text: str) -> Classification:
        """Возвращает тему и уверенность (максимум predict_proba)."""
        probabilities = self._pipeline.predict_proba([text])[0]
        best = int(probabilities.argmax())
        return Classification(
            topic=str(self._pipeline.classes_[best]),
            confidence=float(probabilities[best]),
        )


def cross_validate(dataset_path: Path, folds: int = 5) -> dict[str, float]:
    """Честная оценка baseline на мини-датасете.

    Метрика — macro-F1: классы несбалансированы по важности, и мы не хотим,
    чтобы качество на крупных темах маскировало провал на редких.
    """
    texts, labels = _load(dataset_path)
    scores = cross_val_score(_build(), texts, labels, cv=folds, scoring="f1_macro")
    return {"f1_macro_mean": float(scores.mean()), "f1_macro_std": float(scores.std())}


def _load(dataset_path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(row["topic"])
    return texts, labels
