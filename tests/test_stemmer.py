from entity_linker import normalize_text, stem_word


def test_stemming_rules():
    """Проверка корректного срезания падежных окончаний SnowballStemmer."""
    # Русский язык: проверка получения основы
    assert stem_word("акциями") == "акц"
    assert stem_word("токенами") == "токен"
    assert stem_word("приорбанке") == "приорбанк"

    # Английский язык
    assert stem_word("deposits") == "deposit"
    assert stem_word("tokens") == "token"


def test_text_normalization():
    """Проверка очистки от пунктуации, регистра и нормализации фразы целиком."""
    raw_query = "Куплю акции Приорбанка!.."
    normalized = normalize_text(raw_query)

    assert "акц" in normalized
    assert "приорбанк" in normalized
    assert "!" not in normalized
    assert "." not in normalized