from typing import List, Optional
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from part1.search_engine import Document, SearchResult

class FAISSSearcher:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Инициализация индекса
        """
        self.model = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.index: Optional[faiss.Index] = None
        self.dimension: int = 384  # Размерность для 'all-MiniLM-L6-v2'

    def build_index(self, documents: List[Document]) -> None:
        self.documents = documents

        # Получение эмбеддингов для документов
        embeddings = self.model.encode([doc.text for doc in documents], convert_to_numpy=True)

        # Нормализация векторов
        faiss.normalize_L2(embeddings)

        # Создание quantizer и индекса
        n_clusters = 100  # Выберите количество кластеров
        quantizer = faiss.IndexFlatIP(self.dimension)  # Используем Inner Product
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, n_clusters)

        # Обучение индекса
        self.index.train(embeddings)

        # Добавление векторов в индекс
        self.index.add(embeddings)

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'index': faiss.serialize_index(self.index)
            }, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.index = faiss.deserialize_index(data['index'])

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        # Получение эмбеддинга для запроса
        query_embedding = self.model.encode(query, convert_to_numpy=True).reshape(1, -1)

        # Нормализация вектора
        faiss.normalize_L2(query_embedding)

        # Поиск в индексе
        distances, indices = self.index.search(query_embedding, top_k)

        # Построение результатов
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS возвращает -1 для отсутствующих значений
                continue
            results.append(SearchResult(
                document=self.documents[idx],
                score=float(distances[0][i])
            ))
        return results

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[SearchResult]]:
        # Получение эмбеддингов для запросов
        query_embeddings = self.model.encode(queries, convert_to_numpy=True)

        # Нормализация векторов
        faiss.normalize_L2(query_embeddings)

        # Поиск в индексе
        distances, indices = self.index.search(query_embeddings, top_k)

        # Построение результатов
        all_results = []
        for q_idx in range(len(queries)):
            results = []
            for i, idx in enumerate(indices[q_idx]):
                if idx == -1:
                    continue
                results.append(SearchResult(
                    document=self.documents[idx],
                    score=float(distances[q_idx][i])
                ))
            all_results.append(results)
        return all_results
