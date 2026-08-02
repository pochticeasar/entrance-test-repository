"""Журнал автоматических решений.

ТЗ требует, чтобы все автоматические решения сохранялись и были аудируемы.
Пишем JSONL: одна строка — одно решение, с версиями модели и политики, чтобы
можно было ответить на вопрос «почему полгода назад тикет закрылся сам».

Сырой текст тикета в лог не попадает — только редактированный и хеш исходного.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .schema import TriageResult


def to_record(result: TriageResult, raw_text: str) -> dict:
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "ticket_id": result.ticket_id,
        "channel": result.channel,
        "input_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "redacted_text": result.redacted_text,
        "pii_found": list(result.pii_found),
        "topic": result.classification.topic,
        "confidence": round(result.classification.confidence, 4),
        "risk": result.risk.level.value,
        "triggered_rules": list(result.risk.triggered_rules),
        "retrieved": [
            {"id": d.id, "score": round(d.score, 4)} for d in result.retrieved
        ],
        "decision": result.decision.value,
        "reason": result.reason,
        "draft_source": result.draft.source,
        "draft_degraded": result.draft.degraded,
        "hot_path_ms": round(result.hot_path_ms, 2),
        "total_ms": round(result.total_ms, 2),
        "model_version": result.model_version,
        "policy_version": result.policy_version,
        **({"extra": result.extra} if result.extra else {}),
    }


def append(path: Path, result: TriageResult, raw_text: str) -> dict:
    record = to_record(result, raw_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


__all__ = ["append", "to_record", "asdict"]
