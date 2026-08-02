"""PII не должен покидать периметр: тикет маскируется до всех остальных шагов."""

from support_triage.pii import redact


def test_email_is_masked():
    result = redact("Пишите на ivanov@example.com, жду ответа")
    assert "ivanov@example.com" not in result.text
    assert "[EMAIL]" in result.text
    assert "email" in result.found


def test_card_number_is_masked_without_eating_surrounding_text():
    result = redact("С карты 4111 1111 1111 1111 списали деньги")
    assert "4111" not in result.text
    assert "card_number" in result.found
    assert "списали деньги" in result.text


def test_phone_is_masked():
    result = redact("Мой телефон +7 916 123-45-67")
    assert "916" not in result.text
    assert "phone" in result.found


def test_clean_text_is_untouched():
    original = "Не приходит письмо для сброса пароля"
    result = redact(original)
    assert result.text == original
    assert result.found == ()


def test_non_card_16_digits_are_not_flagged_as_card():
    """Номер заказа из 16 цифр не проходит проверку Луна и не эскалирует тикет.

    Без этой проверки любой набор цифр помечался картой: и ложные
    срабатывания на номерах заказов, и способ обойти очередь к оператору.
    """
    result = redact("Номер заказа 1234 5678 9012 3456, подскажите статус")
    assert "card_number" not in result.found


def test_luhn_valid_card_is_still_masked():
    result = redact("Карта 4111 1111 1111 1111")
    assert "card_number" in result.found
    assert "[CARD]" in result.text
