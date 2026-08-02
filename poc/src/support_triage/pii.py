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


def _luhn_valid(digits: str) -> bool:
    """Контрольная сумма Луна — стандартная проверка номера карты.

    Нужна по двум причинам сразу. Во-первых, 16-значные номера заказов и
    трек-номера иначе маскируются как карты и эскалируют тикет впустую.
    Во-вторых, без неё любой набор из 16 цифр гарантирует попадание к
    оператору — то есть правило превращается в способ обойти очередь.
    Полностью обход это не закрывает (валидный номер можно сгенерировать),
    но поднимает планку с «набрать что попало» до осознанного действия.
    """
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact(text: str) -> Redaction:
    """Заменяет найденные PII на плейсхолдеры и возвращает список типов."""
    found: list[str] = []
    redacted = text

    def _card(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if not _luhn_valid(digits):
            return match.group(0)  # не карта — оставляем как есть
        found.append("card_number")
        return _PLACEHOLDER["card_number"]

    for name, pattern in _PATTERNS:
        if name == "card_number":
            redacted = pattern.sub(_card, redacted)
            continue
        redacted, n = pattern.subn(_PLACEHOLDER[name], redacted)
        if n:
            found.append(name)

    # Порядок типов не значим, но дубликаты мешают читать аудит-лог.
    return Redaction(text=redacted, found=tuple(dict.fromkeys(found)))
