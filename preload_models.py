import os

from gliner import GLiNER
from sentence_transformers import SentenceTransformer

# Создаем папку под веса
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
GLINER_DIR = os.path.join(WEIGHTS_DIR, "gliner")
EMBEDDER_DIR = os.path.join(WEIGHTS_DIR, "embedder")

os.makedirs(GLINER_DIR, exist_ok=True)
os.makedirs(EMBEDDER_DIR, exist_ok=True)

print("1/2. Скачивание GLiNER (для NER/извлечения сущностей)...")
# urchade/gliner_base использует стандартный токенизатор без конфликтов SentencePiece
gliner = GLiNER.from_pretrained("urchade/gliner_base")
gliner.save_pretrained(GLINER_DIR)
print(f"-> GLiNER сохранен в: {GLINER_DIR}")

print("\n2/2. Скачивание классификатора/эмбеддера (для интентов и линкера)...")
# Компактный мультиязычный энкодер для семантики интентов (~110 МБ)
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embedder.save_pretrained(EMBEDDER_DIR)
print(f"-> Эмбеддер сохранен в: {EMBEDDER_DIR}")

print("\nВсе модели успешно скачаны на диск. Сеть больше не понадобится!")
