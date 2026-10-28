import os
import re
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from nltk.stem.snowball import SnowballStemmer
from rapidfuzz import fuzz

load_dotenv()

# Инициализируем стеммер для русского и английского языков
stemmer_ru = SnowballStemmer("russian")
stemmer_en = SnowballStemmer("english")


def stem_word(word: str) -> str:
    """Приводит слово к основе в зависимости от алфавита."""
    if re.search(r"[а-яА-ЯёЁ]", word):
        return str(stemmer_ru.stem(word))
    return str(stemmer_en.stem(word))


def normalize_text(text: str) -> str:
    """
    Очищает строку от знаков пунктуации, приводит к нижнему регистру
    и выполняет стемминг каждого слова (срезает падежные окончания).
    """
    words = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower())
    stems = [stem_word(w) for w in words]
    return " ".join(stems)


class HybridEntityLinker:
    def __init__(self, neo4j_driver: Driver) -> None:
        self.driver = neo4j_driver
        self.entities: list[dict[str, Any]] = []
        self._sync_with_graph()

    def _sync_with_graph(self) -> None:
        """
        Динамически выгружает все сущности Asset и AssetClass из базы Neo4j.
        Никаких захардкоженных списков в коде!
        """
        query = """
        MATCH (n) WHERE n:Asset OR n:AssetClass
        RETURN n.id AS id,
               labels(n)[0] AS type,
               n.name AS name,
               coalesce(n.ticker, '') AS ticker
        """
        with self.driver.session() as session:
            records = session.run(query).data()

        self.entities = []
        for r in records:
            # Сохраняем как оригинальное имя, так и его нормализованную основу
            norm_name = normalize_text(r["name"])
            ticker_clean = r["ticker"].strip().lower()

            self.entities.append(
                {
                    "id": r["id"],
                    "type": r["type"],
                    "name": r["name"],
                    "norm_name": norm_name,
                    "ticker": ticker_clean,
                }
            )

        print(
            f"[EntityLinker] Проиндексировано {len(self.entities)} сущностей из Neo4j."
        )

    def link(self, raw_mention: str, threshold: float = 65.0) -> dict[str, Any] | None:
        if not self.entities:
            return None

        query_norm = normalize_text(raw_mention)
        query_tokens = set(query_norm.split())

        best_entity = None
        highest_score = 0.0

        for ent in self.entities:
            ent_tokens = set(ent["norm_name"].split())

            # 1. Точное совпадение по тикеру
            if ent["ticker"] and ent["ticker"] in query_tokens:
                return {
                    "id": ent["id"],
                    "type": ent["type"],
                    "name": ent["name"],
                    "confidence": 1.0,
                }

            # 2. Прямое пословное совпадение (если слово целиком есть в названии)
            token_overlap = query_tokens.intersection(ent_tokens)
            is_exact_token_match = len(token_overlap) == len(query_tokens)

            # Базовые метрики сходства
            ts_score = fuzz.token_set_ratio(query_norm, ent["norm_name"])

            # partial_ratio используем только если длины строк сопоставимы,
            # чтобы короткое слово "акции" не матчилось на 100%
            # со всеми длинными названиями бумаг
            p_score = 0.0
            if (
                len(query_norm) >= 4
                and abs(len(query_norm) - len(ent["norm_name"])) < 10
            ):
                p_score = fuzz.partial_ratio(query_norm, ent["norm_name"])

            score = max(ts_score, p_score)

            # Бонус за совпадение базового класса,
            # если введен только один категориальный токен
            if is_exact_token_match and ent["type"] == "AssetClass":
                score += 15.0

            if score > highest_score:
                highest_score = score
                best_entity = ent

        if highest_score >= threshold and best_entity:
            return {
                "id": best_entity["id"],
                "type": best_entity["type"],
                "name": best_entity["name"],
                "confidence": round(min(highest_score, 100.0) / 100.0, 3),
            }

        return None


# --- Блок автономного тестирования модуля ---
if __name__ == "__main__":
    bolt_port = os.getenv("NEO4J_BOLT_PORT", "7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "guardrail_secret_pass_2024")

    uri = f"bolt://localhost:{bolt_port}"
    driver = GraphDatabase.driver(uri, auth=(neo4j_user, neo4j_password))

    try:
        linker = HybridEntityLinker(driver)

        test_cases = [
            "токенами билаза",
            "приарбанке",
            "депозиты",
            "акции",
            "биткоин",
            "случайный мусор",
        ]

        print("\n--- Проверка распознавания сущностей ---")
        for q in test_cases:
            res = linker.link(q)
            print(f"«{q}» -> {res}")

    finally:
        driver.close()
