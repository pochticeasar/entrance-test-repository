"""Контракты между шагами пайплайна.

Каждый шаг принимает и возвращает объект из этого модуля — так шаги можно
тестировать и заменять независимо друг от друга.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Темы, по которым обучается классификатор (см. data/tickets_train.jsonl).
TOPICS = [
    "password_reset",
    "delivery_status",
    "faq_howto",
    "payment_failed",
    "refund_request",
    "account_deletion",
    "bug_report",
]

# Темы, по которым в принципе разрешён автоответ. Всё остальное — минимум suggest.
# Список намеренно узкий: расширять его — продуктовое решение, а не техническое.
SAFE_AUTO_TOPICS = {"password_reset", "delivery_status", "faq_howto"}


class Risk(str, Enum):
    LOW = "low"
    HIGH = "high"


class Decision(str, Enum):
    AUTO_REPLY = "auto_reply"   # ответ уходит пользователю без оператора
    SUGGEST = "suggest"         # черновик показывается оператору
    ESCALATE = "escalate"       # оператору, черновик не предлагается


@dataclass(frozen=True)
class Ticket:
    id: str
    channel: str
    text: str


@dataclass(frozen=True)
class Redaction:
    """Результат маскирования PII."""

    text: str
    found: tuple[str, ...]  # типы найденных сущностей: email, phone, card_number


@dataclass(frozen=True)
class Classification:
    topic: str
    confidence: float


@dataclass(frozen=True)
class RiskAssessment:
    level: Risk
    triggered_rules: tuple[str, ...]


@dataclass(frozen=True)
class RetrievedDoc:
    id: str
    title: str
    text: str
    score: float


@dataclass(frozen=True)
class Draft:
    text: str
    source: str  # llm | template | none
    degraded: bool = False  # True, если LLM был недоступен и сработал fallback


@dataclass
class TriageResult:
    ticket_id: str
    channel: str
    redacted_text: str
    pii_found: tuple[str, ...]
    classification: Classification
    risk: RiskAssessment
    retrieved: list[RetrievedDoc]
    decision: Decision
    reason: str
    draft: Draft
    hot_path_ms: float
    total_ms: float
    model_version: str
    policy_version: str
    extra: dict = field(default_factory=dict)
