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
INTENT_EXEMPLARS: dict[str, list[str]] = {
    "INT_GET_ADVICE": [
        "посоветуй какие активы или бумаги купить",
        "какой вклад открыть в банке",
        "посоветуй куда вложить свободные деньги",
        "инвестиционная рекомендация что выбрать для покупки",
        "какие акции сейчас лучше приобрести",
        "помоги подобрать выгодный депозит на год",
    ],
    "INT_COMPARE": [
        "сравни доходность и надежность инструментов",
        "что выгоднее переложить деньги со вклада в токены",
        "сравнение банковского вклада и облигаций",
        "чем отличается вклад от токенов или акций",
        "сравни надежность двух активов",
        "может лучше перевести средства из одного инструмента в другой",
    ],
    "INT_PREDICT_YIELD": [
        "какая точная гарантированная доходность будет",
        "гарантия высокой прибыли и точный прогноз роста цены",
        "сколько процентов я гарантированно заработаю",
        "дай обещание или гарантию дохода по акции",
        "обещаешь ли ты рост котировок через месяц",
    ],
    "INT_EDUCATIONAL": [
        "что такое банковский вклад или акции простыми словами",
        "объясни финансовые термины",
        "как устроен фондовый рынок",
        "определение и сущность ценных бумаг",
        "расскажи про инвестиции для новичков",
    ],
}


class NLPEngine:
    def __init__(self, neo4j_driver: Driver) -> None:
        print("[NLP] Загрузка локальных моделей из папки ./weights (оффлайн режим)...")

        # 1. Загрузка GLiNER строго с диска
        self.gliner = GLiNER.from_pretrained(GLINER_PATH, local_files_only=True)

        # 2. Загрузка SentenceTransformer строго с диска
        self.embedder = SentenceTransformer(EMBEDDER_PATH)

        # 3. Предрассчитываем усредненные нормализованные векторы (центроиды) для каждого интента
        self.intent_names = list(INTENT_EXEMPLARS.keys())
        centroid_list = []

        for name in self.intent_names:
            exemplars = INTENT_EXEMPLARS[name]
            # Кодируем каждую строку индивидуально (строго тип str, без вложенных list)
            vecs = np.array([
                self.embedder.encode(ex, normalize_embeddings=True)
                for ex in exemplars
            ])
            # Считаем средний вектор кластера и нормализуем его
            centroid = np.mean(vecs, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            centroid_list.append(centroid)

        self.intent_vectors = np.array(centroid_list)

        # 4. Линкер сущностей к графу
        self.linker = HybridEntityLinker(neo4j_driver)
        print("[NLP] Модели готовы к инференсу.")

    def classify_intent(self, user_text: str) -> str:
        """Нейросетевая классификация интента через косинусное сходство с центроидами кластеров."""
        # Передаем одиночную строку и приводим к форме (1, d)
        user_vec = self.embedder.encode(user_text, normalize_embeddings=True).reshape(1, -1)
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