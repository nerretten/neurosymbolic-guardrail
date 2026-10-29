import pytest
from entity_linker import HybridEntityLinker


def test_entity_linker_sync(neo4j_driver):
    """Проверка динамической загрузки сущностей из базы Neo4j."""
    linker = HybridEntityLinker(neo4j_driver)
    # В init.cypher создается 10 активов и 8 классов (всего 18)
    assert len(linker.entities) == 18


def test_entity_linker_by_ticker(neo4j_driver):
    """Проверка точного матчинга по тикеру инструмента."""
    linker = HybridEntityLinker(neo4j_driver)

    res = linker.link("BTC")
    assert res is not None
    assert res["id"] == "CRYPTO_BTC"
    assert res["type"] == "Asset"
    assert res["confidence"] == 1.0


def test_entity_linker_fuzzy_mention(neo4j_driver):
    """Проверка нечеткого сопоставления с опечаткой и падежным окончанием."""
    linker = HybridEntityLinker(neo4j_driver)

    # Запрос с опечаткой и падежом: "токенами билаза" -> FINSTORE_BELAZ_01
    res = linker.link("Токены Белаза")
    assert res is not None
    assert res["id"] == "FINSTORE_BELAZ_01"
    assert res["type"] == "Asset"


def test_entity_linker_class_priority(neo4j_driver):
    """Проверка приоритета классов (AssetClass) при вводе одиночного термина."""
    linker = HybridEntityLinker(neo4j_driver)

    # Слово "акции" должно распознаваться как класс CLASS_EQUITY, а не конкретная бумага
    res = linker.link("акции")
    assert res is not None
    assert res["type"] == "AssetClass"
    assert res["id"] == "CLASS_EQUITY"


def test_entity_linker_noise_rejection(neo4j_driver):
    """Проверка отсечения несвязанного шума ниже порога similarity."""
    linker = HybridEntityLinker(neo4j_driver)

    res = linker.link("случайный бессмысленный текст")
    assert res is None