"""Оценка риска и политика принятия решения.

Ключевое проектное решение: риск считается правилами, а не моделью.
Тема — обучаемая модель, её ошибка терпима (максимум — неверная очередь).
Риск — детерминированные правила: ошибка недопустима, решение нужно уметь
объяснить постфактум и менять за минуты без переобучения. Правила имеют право
veto: высокий риск запрещает автоответ независимо от уверенности модели.
"""

from __future__ import annotations

import re

from .schema import (
    SAFE_AUTO_TOPICS,
    Classification,
    Decision,
    Risk,
    RiskAssessment,
)

POLICY_VERSION = "policy-v0.1"

# Пороги. В проде живут в конфиге и меняются без релиза — это ручки, которыми
# бизнес регулирует баланс между автоматизацией и риском.
MIN_CONFIDENCE_FOR_AUTO = 0.60
MIN_RETRIEVAL_SCORE_FOR_AUTO = 0.18
MIN_CONFIDENCE_FOR_SUGGEST = 0.35

# Темы, по которым автоответ запрещён всегда — деньги и персональные данные.
ALWAYS_HUMAN_TOPICS = {"payment_failed", "refund_request", "account_deletion"}

# Типы PII, само наличие которых делает тикет «человеческим», независимо от темы.
# Логика не в приватности (текст всё равно замаскирован), а в том, что раскрытие
# платёжных реквизитов в переписке — это почти всегда платёжный контекст плюс
# инцидент безопасности, о котором пользователю нужно сказать отдельно.
# Контактные данные (email, телефон) сюда НЕ входят: они есть почти в каждом
# обращении с почтового канала, и запрет по ним убил бы автоматизацию целиком.
SENSITIVE_PII = {"card_number"}

_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "money_dispute",
        re.compile(
            r"списал|списан|двойн\w*\s+списан|верните деньги|вернуть деньги|"
            r"возврат\w*\s+(средств|денег)|компенсац",
            re.IGNORECASE,
        ),
    ),
    (
        "legal_threat",
        re.compile(r"в суд|подам в суд|иск\b|роспотребнадзор|жалоб\w*\s+в|прокуратур", re.IGNORECASE),
    ),
    (
        "personal_data_request",
        re.compile(
            r"удалит\w*\s+(мой\s+)?(аккаунт|профиль|учётн|учетн|персональн)|"
            r"отзыва\w*\s+согласие|персональн\w*\s+данн",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_injection",
        re.compile(
            r"игнорируй\w*\s+(все\s+)?(предыдущ|инструкц)|ignore\s+(all\s+)?previous|"
            r"ты больше не|забудь инструкц|system prompt|закрой тикет автоматически",
            re.IGNORECASE,
        ),
    ),
]


def assess_risk(
    text: str,
    classification: Classification,
    pii_found: tuple[str, ...] = (),
) -> RiskAssessment:
    """Правила поверх текста, предсказанной темы и найденных PII.

    `pii_found` передаётся отдельным аргументом, потому что к этому моменту
    текст уже замаскирован: номер карты в нём выглядит как `[CARD]` и текстовым
    правилом не ловится. Без этого аргумента наличие карты было бы невидимо
    для политики.
    """
    triggered = [name for name, pattern in _RULES if pattern.search(text)]
    if classification.topic in ALWAYS_HUMAN_TOPICS:
        triggered.append(f"topic:{classification.topic}")
    triggered += [f"sensitive_pii:{kind}" for kind in pii_found if kind in SENSITIVE_PII]
    level = Risk.HIGH if triggered else Risk.LOW
    return RiskAssessment(level=level, triggered_rules=tuple(triggered))


def decide(
    classification: Classification,
    risk: RiskAssessment,
    top_retrieval_score: float,
    llm_available: bool,
) -> tuple[Decision, str]:
    """Возвращает решение и человекочитаемую причину для аудита.

    Порядок проверок — это и есть приоритет: сначала запреты, потом разрешения.
    """
    if risk.level is Risk.HIGH:
        return Decision.ESCALATE, f"risk_rules_triggered: {', '.join(risk.triggered_rules)}"

    if classification.confidence < MIN_CONFIDENCE_FOR_SUGGEST:
        return Decision.ESCALATE, (
            f"confidence {classification.confidence:.2f} < {MIN_CONFIDENCE_FOR_SUGGEST}"
        )

    if not llm_available:
        # Деградация: черновик собирается из шаблона, автозакрытие выключено.
        return Decision.SUGGEST, "llm_unavailable: auto_reply disabled, template draft"

    if (
        classification.topic in SAFE_AUTO_TOPICS
        and classification.confidence >= MIN_CONFIDENCE_FOR_AUTO
        and top_retrieval_score >= MIN_RETRIEVAL_SCORE_FOR_AUTO
    ):
        return Decision.AUTO_REPLY, (
            f"safe_topic={classification.topic}, "
            f"confidence={classification.confidence:.2f}, "
            f"retrieval={top_retrieval_score:.2f}"
        )

    return Decision.SUGGEST, (
        f"topic={classification.topic}, confidence={classification.confidence:.2f}, "
        f"retrieval={top_retrieval_score:.2f} — below auto-reply bar"
    )
