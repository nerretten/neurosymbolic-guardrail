import os
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from neo4j import Driver, GraphDatabase

import main
from main import app

load_dotenv()


@pytest.fixture(scope="session")
def neo4j_driver() -> Driver:
    """Сессионная фикстура для подключения к Neo4j."""
    bolt_port = os.getenv("NEO4J_BOLT_PORT", "7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "guardrail_secret_pass_2024")
    uri = f"bolt://localhost:{bolt_port}"

    driver = GraphDatabase.driver(uri, auth=(neo4j_user, neo4j_password))
    # Проверяем доступность базы данных перед запуском тестов
    with driver.session() as session:
        session.run("RETURN 1")

    yield driver
    driver.close()


@pytest.fixture(scope="session")
def client(neo4j_driver: Driver):
    """Тестовый клиент FastAPI с инициализированным Lifespan."""
    with TestClient(app) as test_client:
        yield test_client