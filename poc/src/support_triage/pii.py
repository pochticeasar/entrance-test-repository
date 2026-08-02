"""Маскирование персональных данных перед любой обработкой.

Упрощение PoC: регулярные выражения. В целевой архитектуре — NER-модель поверх
этого слоя (регулярки ловят структурированные сущности, NER — имена и адреса).
Важно не столько качество регулярок, сколько место шага: он стоит ДО вызова
внешнего LLM API, поэтому наружу сырой текст не уходит.
"""

from __future__ import annotations

import re

from .schema import Redaction

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    # Номер карты: 13–19 цифр, разделители допустимы только МЕЖДУ цифрами,
    # иначе паттерн съедает пробел после номера.
    ("card_number", re.compile(r"\b\d(?:[ -]?\d){12,18}\b")),
    ("phone", re.compile(r"\+?\d[\d ()-]{9,}\d")),
]

_PLACEHOLDER = {
    "email": "[EMAIL]",
    "card_number": "[CARD]",
    "phone": "[PHONE]",
}


def redact(text: str) -> Redaction:
    """Заменяет найденные PII на плейсхолдеры и возвращает список типов."""
    found: list[str] = []
    redacted = text
    for name, pattern in _PATTERNS:
        redacted, n = pattern.subn(_PLACEHOLDER[name], redacted)
        if n:
            found.append(name)
    return Redaction(text=redacted, found=tuple(found))
