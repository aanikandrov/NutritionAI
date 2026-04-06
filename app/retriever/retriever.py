import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class SemanticRetriever:
    def __init__(self, data_path: str, embedding_model: str = "all-MiniLM-L6-v2", top_k: int = 3):
        self.data_path = Path(data_path)
        self.top_k = top_k
        self.model = SentenceTransformer(embedding_model)
        self.products: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

        self._load_data()
        self._build_embeddings()

    def _load_data(self):
        """ Загружает продукты из JSON-файла """
        if not self.data_path.exists():
            logger.warning(f"Файл с данными не найден: {self.data_path}")
            self.products = []
            return

        with open(self.data_path, "r", encoding="utf-8") as f:
            self.products = json.load(f)
        logger.info(f"Загружено {len(self.products)} продуктов")

    def _build_embeddings(self):
        """ Строит эмбеддинги для названий продуктов """
        if not self.products:
            self.embeddings = None
            return

        texts = [p["name"] for p in self.products]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        logger.info(f"Построено {len(self.embeddings)} эмбеддингов")

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """ Возвращает top_k наиболее похожих продуктов """
        if not self.products or self.embeddings is None:
            logger.warning("Ретривер не инициализирован (нет данных)")
            return []

        query_emb = self.model.encode(query, normalize_embeddings=True)
        similarities = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]

        results = []
        for idx in top_indices:
            product = self.products[idx].copy()
            product["similarity"] = float(similarities[idx])
            results.append(product)

        logger.info(f"Найдено {len(results)} релевантных продуктов для запроса '{query}'")
        return results

# Глобальный экземпляр
retriever: Optional[SemanticRetriever] = None

def init_retriever(data_path: str, embedding_model: str = "all-MiniLM-L6-v2", top_k: int = 3):
    global retriever
    retriever = SemanticRetriever(data_path, embedding_model, top_k)

def get_retriever() -> Optional[SemanticRetriever]:
    return retriever