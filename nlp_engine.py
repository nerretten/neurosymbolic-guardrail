import os
from typing import Any

import numpy as np
from gliner import GLiNER
from neo4j import Driver
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from entity_linker import HybridEntityLinker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GLINER_PATH = os.path.join(BASE_DIR, "weights", "gliner")
EMBEDDER_PATH = os.path.join(BASE_DIR, "weights", "embedder")

# Базовые описания интентов для семантической классификации моделью
INTENT_DESCRIPTIONS = {
    "INT_GET_ADVICE": "посоветуй какие активы или бумаги купить, куда вложить средства,"
    "инвестиционная рекомендация",
    "INT_COMPARE": "сравни доходность и надежность инструментов,"
    " что выгоднее переложить деньги",
    "INT_PREDICT_YIELD": "гарантия высокой доходности, "
    "точный прогноз роста цены и прибыли",
    "INT_EDUCATIONAL": "что такое банковский вклад или акции, "
    "объясни финансовые термины",
}


class NLPEngine:
    def __init__(self, neo4j_driver: Driver) -> None:
        print("[NLP] Загрузка локальных моделей из папки ./weights (оффлайн режим)...")

        # 1. Загрузка GLiNER строго с диска
        self.gliner = GLiNER.from_pretrained(GLINER_PATH, local_files_only=True)

        # 2. Загрузка SentenceTransformer строго с диска
        self.embedder = SentenceTransformer(EMBEDDER_PATH)

        # Предрассчитываем эмбеддинги для интентов
        self.intent_names = list(INTENT_DESCRIPTIONS.keys())
        intent_texts = list(INTENT_DESCRIPTIONS.values())
        self.intent_vectors = self.embedder.encode(
            intent_texts, normalize_embeddings=True
        )

        # 3. Линкер сущностей к графу
        self.linker = HybridEntityLinker(neo4j_driver)
        print("[NLP] Модели готовы к инференсу.")

    def classify_intent(self, user_text: str) -> str:
        """Нейросетевая классификация интента через косинусное сходство эмбеддингов."""
        user_vec = self.embedder.encode([user_text], normalize_embeddings=True)
        scores = cosine_similarity(user_vec, self.intent_vectors)[0]
        best_idx = int(np.argmax(scores))
        return self.intent_names[best_idx]

    def extract(self, text: str) -> dict[str, Any]:
        # Нейросетевое определение интента
        intent_id = self.classify_intent(text)

        # Извлечение спанов сущностей моделью GLiNER
        labels = [
            "акция",
            "банковский вклад",
            "облигация",
            "токен",
            "криптовалюта",
            "финансовая организация",
        ]
        entities = self.gliner.predict_entities(text, labels, threshold=0.35)

        asset_ids: set[str] = set()
        class_ids: set[str] = set()
        raw_spans = [e["text"] for e in entities]

        # Привязка извлеченных сущностей к узлам Neo4j
        for span in raw_spans:
            matched = self.linker.link(span)
            if matched:
                if matched["type"] == "Asset":
                    asset_ids.add(matched["id"])
                elif matched["type"] == "AssetClass":
                    class_ids.add(matched["id"])

        return {
            "intent_id": intent_id,
            "asset_ids": list(asset_ids),
            "class_ids": list(class_ids),
            "raw_spans": raw_spans,
        }
