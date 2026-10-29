import pytest
from rule_engine import RuleEngine


@pytest.fixture
def rule_engine(neo4j_driver):
    return RuleEngine(neo4j_driver)


def test_rule_abstract_advice_blocked(rule_engine):
    """Случай А: Абстрактный инвест-запрос без активов отсекается по ст. 231-З."""
    res = rule_engine.evaluate(
        intent_id="INT_GET_ADVICE",
        asset_ids=[],
        class_ids=[],
    )
    assert res["verdict"] == "DENY"
    assert res["policy"] == "POL_RB_SECURITIES_231Z"
    assert "231-З" in res["law"]
    assert res["trace"] == ["INT_GET_ADVICE", "POL_RB_SECURITIES_231Z", "DENY"]


def test_rule_transitive_equity_advice_blocked(rule_engine):
    """Случай Б: Транзитивный подъем CLASS_EQUITY -> CLASS_SECURITY -> Запрет."""
    res = rule_engine.evaluate(
        intent_id="INT_GET_ADVICE",
        asset_ids=[],
        class_ids=["CLASS_EQUITY"],
    )
    assert res["verdict"] == "DENY"
    assert res["policy"] == "POL_RB_SECURITIES_231Z"


def test_rule_misselling_deposit_vs_token_blocked(rule_engine):
    """Случай В: Сравнение гарантированного вклада и токена блокируется (ст. 258-З)."""
    res = rule_engine.evaluate(
        intent_id="INT_COMPARE",
        asset_ids=["PRIOR_DEPOSIT_PREMIUM", "FINSTORE_BELAZ_01"],
        class_ids=["CLASS_DEPOSIT", "CLASS_TOKEN"],
    )
    assert res["verdict"] == "DENY"
    assert res["policy"] == "POL_RB_DEPOSIT_COMPARE"
    assert "258-З" in res["law"]


def test_rule_guaranteed_yield_advertising_blocked(rule_engine):
    """Случай Г: Запрос гарантии доходности пресекается законом о рекламе."""
    res = rule_engine.evaluate(
        intent_id="INT_PREDICT_YIELD",
        asset_ids=["BCSE_PRIOR"],
        class_ids=["CLASS_EQUITY"],
    )
    assert res["verdict"] == "DENY"
    assert res["policy"] == "POL_RB_ADVERTISING_PROFIT"
    assert "рекламе" in res["law"]


def test_rule_deposit_advice_allowed(rule_engine):
    """Легальный периметр: Подбор банковского вклада разрешен (ALLOW)."""
    res = rule_engine.evaluate(
        intent_id="INT_GET_ADVICE",
        asset_ids=["BB_DEPOSIT_STABLE"],
        class_ids=["CLASS_DEPOSIT"],
    )
    assert res["verdict"] == "ALLOW"
    assert res["policy"] is None
    assert res["law"] is None
    assert res["trace"] == []


def test_rule_educational_query_allowed(rule_engine):
    """Справочный запрос по любым классам активов разрешен (ALLOW)."""
    res = rule_engine.evaluate(
        intent_id="INT_EDUCATIONAL",
        asset_ids=["BCSE_PRIOR", "CRYPTO_BTC"],
        class_ids=["CLASS_EQUITY", "CLASS_CRYPTO"],
    )
    assert res["verdict"] == "ALLOW"
    assert res["policy"] is None