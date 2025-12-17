# app/models/embedding.py
from sentence_transformers import SentenceTransformer
from app.settings import Settings
from app.logging import logger

class EmbeddingModel:

    _instance = None

    def __new__(cls):
        logger.info("Загрузка модели эмбеддингов (multilingual-e5-small) на CPU...")
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = SentenceTransformer(
                Settings.EMBEDDING_MODEL_PATH, device="cpu"
            )
        logger.success("✅ Модель эмбеддингов загружена и готова к использованию.")

        return cls._instance

    def encode(self, query: str):
        prefixed = "query: " + query
        return self.model.encode(prefixed, normalize_embeddings=True).tolist()
