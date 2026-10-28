import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from neo4j import Driver, GraphDatabase
from pydantic import BaseModel

from nlp_engine import NLPEngine
from rule_engine import RuleEngine

load_dotenv()

# Глобальные сервисы
driver: Driver | None = None
nlp_engine: NLPEngine | None = None
rule_engine: RuleEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global driver, nlp_engine, rule_engine

    # 1. Подключение к Neo4j
    bolt_port = os.getenv("NEO4J_BOLT_PORT", "7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "guardrail_secret_pass_2024")
    uri = f"bolt://localhost:{bolt_port}"

    driver = GraphDatabase.driver(uri, auth=(neo4j_user, neo4j_password))
    print("[App] Подключение к Neo4j установлено.")

    # 2. Инициализация NLP и Rule движков
    nlp_engine = NLPEngine(driver)
    rule_engine = RuleEngine(driver)
    print("[App] Модули Guardrail готовы к работе.")

    yield

    # Закрытие пула при остановке
    if driver:
        driver.close()
        print("[App] Соединение с Neo4j закрыто.")


app = FastAPI(title="Neuro-Symbolic Guardrail API", lifespan=lifespan)


class AskRequest(BaseModel):
    prompt: str


class GuardrailResponse(BaseModel):
    prompt: str
    verdict: str  # ALLOW / DENY
    reason: str | None = None
    law: str | None = None
    trace: list[str] = []
    nlp_debug: dict[str, Any]
    response_text: str


@app.post("/ask", response_model=GuardrailResponse)
def ask_endpoint(req: AskRequest) -> GuardrailResponse:
    assert nlp_engine is not None, "NLPEngine не инициализирован"
    assert rule_engine is not None, "RuleEngine не инициализирован"

    # 1. Извлечение интента и сущностей через NLP
    nlp_res = nlp_engine.extract(req.prompt)

    # 2. Проверка ограничений в Neo4j
    check_res = rule_engine.evaluate(
        intent_id=nlp_res["intent_id"],
        asset_ids=nlp_res["asset_ids"],
        class_ids=nlp_res["class_ids"],
    )

    # 3. Маршрутизация ответа
    if check_res["verdict"] == "DENY":
        return GuardrailResponse(
            prompt=req.prompt,
            verdict="DENY",
            reason=check_res["reason"],
            law=check_res["law"],
            trace=check_res["trace"],
            nlp_debug=nlp_res,
            response_text=f"Запрос заблокирован по требованиям законодательства РБ "
            f"({check_res['law']}): {check_res['reason']}",
        )

    # 4. Если ALLOW — здесь запрос пойдет в LLM
    mock_llm_response = (
        f"Запрос одобрен шлюзом безопасности. Ответ консультанта на: '{req.prompt}'"
    )
    return GuardrailResponse(
        prompt=req.prompt,
        verdict="ALLOW",
        nlp_debug=nlp_res,
        response_text=mock_llm_response,
    )
