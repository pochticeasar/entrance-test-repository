"""Оркестрация: собирает шаги в один проход по тикету.

Разделение путей, как в целевой архитектуре:

* «горячий» путь (redact → classify → risk → retrieve) — синхронный, бюджет 500 мс;
  его длительность замеряется отдельно и попадает в лог решения;
* «медленный» путь (генерация черновика) — вызывается только если решение
  предполагает текст ответа. В проде это отдельный воркер за очередью.
"""

from __future__ import annotations

import time
from pathlib import Path

from . import pii, policy
from .classify import MODEL_VERSION, TopicClassifier
from .generate import CircuitBreaker, Generator, TemplateGenerator
from .retrieve import Retriever
from .schema import Decision, Draft, Ticket, TriageResult

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class TriagePipeline:
    def __init__(
        self,
        classifier: TopicClassifier,
        retriever: Retriever,
        generator: Generator,
    ) -> None:
        self._classifier = classifier
        self._retriever = retriever
        self._generator = generator

    @classmethod
    def build(cls, generator: Generator | None = None, data_dir: Path = DATA_DIR):
        return cls(
            classifier=TopicClassifier.train(data_dir / "tickets_train.jsonl"),
            retriever=Retriever.from_file(data_dir / "kb.jsonl"),
            generator=generator or TemplateGenerator(),
        )

    def run(self, ticket: Ticket) -> TriageResult:
        started = time.perf_counter()

        # --- горячий путь ---
        redaction = pii.redact(ticket.text)
        classification = self._classifier.predict(redaction.text)
        risk = policy.assess_risk(redaction.text, classification, redaction.found)
        retrieved = self._retriever.search(redaction.text, top_k=2)
        top_score = retrieved[0].score if retrieved else 0.0

        llm_available = not (
            isinstance(self._generator, CircuitBreaker) and self._generator.is_open
        )
        decision, reason = policy.decide(
            classification, risk, top_score, llm_available=llm_available
        )
        hot_path_ms = (time.perf_counter() - started) * 1000

        # --- медленный путь ---
        if decision is Decision.ESCALATE:
            # Рискованный или low-confidence тикет: черновик не генерируем вовсе,
            # чтобы оператор не якорился на предложении модели.
            draft = Draft(text="", source="none")
        else:
            safe_ticket = Ticket(
                id=ticket.id, channel=ticket.channel, text=redaction.text
            )
            draft = self._generator.draft(safe_ticket, retrieved)
            if draft.degraded and decision is Decision.AUTO_REPLY:
                # LLM отвалился уже после решения — понижаем до suggest.
                decision, reason = Decision.SUGGEST, "llm_degraded_after_decision"

        return TriageResult(
            ticket_id=ticket.id,
            channel=ticket.channel,
            redacted_text=redaction.text,
            pii_found=redaction.found,
            classification=classification,
            risk=risk,
            retrieved=retrieved,
            decision=decision,
            reason=reason,
            draft=draft,
            hot_path_ms=hot_path_ms,
            total_ms=(time.perf_counter() - started) * 1000,
            model_version=MODEL_VERSION,
            policy_version=policy.POLICY_VERSION,
        )
