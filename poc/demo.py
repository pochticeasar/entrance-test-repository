"""Демонстрация PoC: happy path, risky path, fallback при отказе LLM.

Запуск:  python demo.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from support_triage import TriagePipeline, Ticket  # noqa: E402
from support_triage import audit  # noqa: E402
from support_triage.classify import cross_validate  # noqa: E402
from support_triage.generate import (  # noqa: E402
    CircuitBreaker,
    Generator,
    LLMUnavailable,
    TemplateGenerator,
    default_generator,
)
from support_triage.schema import Decision, Draft, RetrievedDoc  # noqa: E402

DATA_DIR = ROOT / "data"
LOG_PATH = ROOT / "logs" / "decisions.jsonl"
HOT_PATH_BUDGET_MS = 500

# Демо-набор намеренно перекошен в сторону рискованных кейсов, поэтому порог
# высокий. В проде это продуктовая величина из product.md, а не константа.
MAX_ESCALATION_RATE = 0.70


class AlwaysFailingGenerator(Generator):
    """Имитация недоступного LLM API для fallback-сценария."""

    def draft(self, ticket: Ticket, docs: list[RetrievedDoc]) -> Draft:
        raise LLMUnavailable("simulated outage: connection refused")


def load_demo_tickets() -> list[dict]:
    rows = []
    with (DATA_DIR / "demo_tickets.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def show(scenario: str, result, record: dict) -> None:
    icon = {
        Decision.AUTO_REPLY: "✅",
        Decision.SUGGEST: "📝",
        Decision.ESCALATE: "🚨",
    }[result.decision]

    print(f"\n{'─' * 78}")
    print(f"{icon}  [{scenario}] {result.ticket_id} · канал: {result.channel}")
    print(f"{'─' * 78}")
    print(f"  Текст (после PII-редакции): {result.redacted_text[:110]}...")
    if result.pii_found:
        print(f"  Найден и замаскирован PII : {', '.join(result.pii_found)}")
    print(
        f"  Тема                      : {result.classification.topic} "
        f"(confidence {result.classification.confidence:.2f})"
    )
    print(f"  Риск                      : {result.risk.level.value}", end="")
    print(f" [{', '.join(result.risk.triggered_rules)}]" if result.risk.triggered_rules else "")
    if result.retrieved:
        top = result.retrieved[0]
        print(f"  База знаний (top-1)       : {top.id} «{top.title}» score={top.score:.2f}")
    print(f"  РЕШЕНИЕ                   : {result.decision.value.upper()}")
    print(f"  Обоснование               : {result.reason}")
    if result.draft.text:
        label = "черновик (деградация)" if result.draft.degraded else f"черновик [{result.draft.source}]"
        first_line = result.draft.text.strip().splitlines()[0]
        print(f"  {label:<26}: {first_line[:100]}...")
    else:
        print("  Черновик                  : не генерировался — тикет уходит оператору")

    budget = "OK" if result.hot_path_ms < HOT_PATH_BUDGET_MS else "ПРЕВЫШЕН"
    print(
        f"  Горячий путь              : {result.hot_path_ms:.1f} мс "
        f"(бюджет {HOT_PATH_BUDGET_MS} мс — {budget}); всего {result.total_ms:.1f} мс"
    )
    print(f"  Запись в аудит-лог        : input_sha256={record['input_sha256'][:16]}…")


def main() -> int:
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    print("=" * 78)
    print("PoC: автоматизация обработки тикетов поддержки")
    print("=" * 78)

    generator = default_generator()
    kind = type(generator).__name__
    print(f"\nГенератор черновиков: {kind}", end="")
    print(
        "  (задан ANTHROPIC_API_KEY → реальный Claude)"
        if isinstance(generator, CircuitBreaker)
        else "  (ключ не задан → офлайн-шаблоны)"
    )

    metrics = cross_validate(DATA_DIR / "tickets_train.jsonl")
    print(
        f"Baseline-классификатор, 5-fold CV: macro-F1 = "
        f"{metrics['f1_macro_mean']:.3f} ± {metrics['f1_macro_std']:.3f}"
    )

    pipeline = TriagePipeline.build(generator=generator, data_dir=DATA_DIR)

    for row in load_demo_tickets():
        ticket = Ticket(id=row["id"], channel=row["channel"], text=row["text"])
        result = pipeline.run(ticket)
        record = audit.append(LOG_PATH, result, ticket.text)
        show(row["scenario"], result, record)

    # --- fallback: LLM недоступен ---
    print(f"\n{'=' * 78}")
    print("FALLBACK: LLM API недоступен (3 отказа подряд размыкают circuit breaker)")
    print("=" * 78)

    broken = CircuitBreaker(AlwaysFailingGenerator(), TemplateGenerator(), threshold=3)
    degraded_pipeline = TriagePipeline.build(generator=broken, data_dir=DATA_DIR)
    happy = load_demo_tickets()[0]

    for attempt in range(1, 5):
        ticket = Ticket(
            id=f"{happy['id']}-retry{attempt}",
            channel=happy["channel"],
            text=happy["text"],
        )
        result = degraded_pipeline.run(ticket)
        record = audit.append(LOG_PATH, result, ticket.text)
        state = "разомкнут" if broken.is_open else "замкнут"
        print(
            f"  попытка {attempt}: решение={result.decision.value:<10} "
            f"источник={result.draft.source:<9} деградация={result.draft.degraded} "
            f"| circuit breaker: {state}"
        )

    print(
        "\n  Итог: автозакрытие выключилось, тикеты продолжают "
        "классифицироваться и уходить оператору."
    )

    lines = LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]

    print(f"\n{'=' * 78}")
    print("РАСПРЕДЕЛЕНИЕ РЕШЕНИЙ")
    print("=" * 78)
    counts = Counter(r["decision"] for r in records)
    total = len(records)
    for decision in ("auto_reply", "suggest", "escalate"):
        n = counts.get(decision, 0)
        share = n / total
        bar = "█" * round(share * 40)
        print(f"  {decision:<11} {n:>3} ({share:>5.0%}) {bar}")

    # Все guardrails давят в сторону человека, поэтому нужна метрика и в
    # обратную сторону: система, эскалирующая почти всё, формально безопасна
    # и при этом бесполезна. Без этого «слишком осторожно» нечем заметить.
    escalation_rate = counts.get("escalate", 0) / total
    print(f"\n  Доля эскалаций: {escalation_rate:.0%} (порог избыточности {MAX_ESCALATION_RATE:.0%})")
    if escalation_rate > MAX_ESCALATION_RATE:
        print("  ⚠️  Система эскалирует больше, чем задумано: guardrails безопасны,")
        print("      но ценности не создают. Требуется разбор порогов и safe-list.")
    else:
        print("  ✅ В пределах ожидаемого.")

    print(f"\nАудит-лог: {total} решений записано в {LOG_PATH.relative_to(ROOT)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
