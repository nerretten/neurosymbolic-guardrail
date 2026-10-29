import pytest
from nlp_engine import NLPEngine


@pytest.fixture(scope="module")
def nlp_engine(neo4j_driver):
    """Инициализация NLPEngine один раз на модуль для экономии времени инференса."""
    return NLPEngine(neo4j_driver)


def test_classify_intents(nlp_engine):
    """Проверка точности семантической классификации намерений пользователя."""
    # 1. Запрос на получение инвестиционной рекомендации
    intent_advice = nlp_engine.classify_intent("Какие активы сейчас лучше всего приобрести?")
    assert intent_advice == "INT_GET_ADVICE"

    # 2. Сравнение доходности или надежности
    intent_compare = nlp_engine.classify_intent("Сравни доходность банковского вклада и токенов")
    assert intent_compare == "INT_COMPARE"

    # 3. Запрос гарантии прибыли или доходности
    intent_yield = nlp_engine.classify_intent("Какая точная гарантированная доходность будет через месяц?")
    assert intent_yield == "INT_PREDICT_YIELD"

    # 4. Справочно-образовательный запрос
    intent_edu = nlp_engine.classify_intent("Что такое банковский депозит простыми словами?")
    assert intent_edu == "INT_EDUCATIONAL"


def test_extract_equity_entities(nlp_engine):
    """Проверка извлечения спанов и связывания с классом акций."""
    res = nlp_engine.extract("Какие акции лучше купить?")

    assert res["intent_id"] == "INT_GET_ADVICE"
    assert "CLASS_EQUITY" in res["class_ids"]
    assert len(res["raw_spans"]) > 0


def test_extract_deposit_asset(nlp_engine):
    """Проверка извлечения спанов вклада и привязки к конкретному активу."""
    res = nlp_engine.extract("Посоветуй, какой вклад открыть в Беларусбанке на год?")

    assert res["intent_id"] == "INT_GET_ADVICE"
    # Должен быть слинкован конкретный вклад Беларусбанка или общий класс вкладов
    linked_nodes = set(res["asset_ids"] + res["class_ids"])
    assert bool(linked_nodes.intersection({"BB_DEPOSIT_STABLE", "CLASS_DEPOSIT"}))


def test_extract_comparison_assets(nlp_engine):
    """Проверка одновременного извлечения вклада и токена при сравнении."""
    res = nlp_engine.extract("Может лучше переложить деньги со вклада Приора в токены БелАЗа?")

    assert res["intent_id"] == "INT_COMPARE"
    # Должен распознать токен БелАЗа
    assert "FINSTORE_BELAZ_01" in res["asset_ids"]
    assert len(res["raw_spans"]) >= 2