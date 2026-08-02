# Контейнер нужен ради воспроизводимости: PoC разрабатывался на Python 3.14,
# и колёса scikit-learn для свежих версий интерпретатора доступны не везде.
# 3.12 — консервативный выбор с гарантированными бинарными колёсами.
FROM python:3.12-slim

WORKDIR /app

# Зависимости отдельным слоем: правка кода не инвалидирует установку пакетов.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY tests/ ./tests/
COPY demo.py pytest.ini ./

# Тесты прогоняются на этапе сборки: образ не соберётся, если инварианты
# безопасности сломаны. Дешёвый способ не выпустить нерабочий контейнер.
RUN python -m pytest -q

CMD ["python", "demo.py"]
