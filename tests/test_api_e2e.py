import pytest


def test_api_block_equity_advice(client):
    """Сценарий 1: Блокировка запроса на подбор акций (ст. 231-З)."""
    response = client.post("/ask", json={"prompt": "Какие акции лучше купить?"})
    assert response.status_code == 200
    data = response.json()

    assert data["verdict"] == "DENY"
    assert "231-З" in data["law"]
    assert "INT_GET_ADVICE" in data["trace"]
    assert "DENY" in data["trace"]
    assert "CLASS_EQUITY" in data["nlp_debug"]["class_ids"]


def test_api_allow_deposit_selection(client):
    """Сценарий 2: Разрешение подбора банковского вклада (легальный периметр)."""
    response = client.post(
        "/ask",
        json={"prompt": "Посоветуй, какой вклад открыть в Беларусбанке на год?"},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["verdict"] == "ALLOW"
    assert data["reason"] is None
    # Запрос должен связаться либо с конкретным вкладом, либо с классом вклада
    identified = set(data["nlp_debug"]["asset_ids"] + data["nlp_debug"]["class_ids"])
    assert bool(identified.intersection({"BB_DEPOSIT_STABLE", "CLASS_DEPOSIT"}))


def test_api_block_misselling(client):
    """Сценарий 3: Блокировка сравнения защищенного вклада с токеном (мисселинг, ст. 258-З)."""
    response = client.post(
        "/ask",
        json={
            "prompt": "Может лучше переложить деньги со вклада Приора в токены БелАЗа?"
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["verdict"] == "DENY"
    assert "POL_RB_DEPOSIT_COMPARE" in data["trace"]
    assert "FINSTORE_BELAZ_01" in data["nlp_debug"]["asset_ids"]


def test_api_block_abstract_investment(client):
    """Сценарий 4: Блокировка абстрактного запроса на инвестирование (ст. 231-З)."""
    response = client.post(
        "/ask",
        json={"prompt": "Куда сейчас выгоднее всего вложить свободные деньги?"},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["verdict"] == "DENY"
    assert data["nlp_debug"]["asset_ids"] == []
    assert data["nlp_debug"]["class_ids"] == []
    assert "POL_RB_SECURITIES_231Z" in data["trace"]


def test_api_allow_educational_query(client):
    """Сценарий 5: Разрешение образовательного / терминологического вопроса."""
    response = client.post(
        "/ask",
        json={"prompt": "Что такое банковский вклад и акции?"},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["verdict"] == "ALLOW"
    assert data["nlp_debug"]["intent_id"] == "INT_EDUCATIONAL"