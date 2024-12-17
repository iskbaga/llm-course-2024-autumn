import heapq
import json
import pickle
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    id: str
    title: str
    text: str
    embedding: Optional[np.ndarray] = None


@dataclass
class SearchResult:
    doc_id: str
    score: float
    title: str
    text: str


def load_documents(path: str) -> List[Document]:
    """Загрузка документов из json файла"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [
        Document(
            id=article['id'],
            title=article['title'],
            text=article['text'],
            embedding=None
        )
        for article in data['articles']
    ]


class Indexer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, documents: List[Document]) -> None:
        """
        TODO: Реализовать индексацию документов
        1. Сохранить документы в self.documents
        2. Получить эмбеддинги для документов используя self.model.encode()
           Подсказка: для каждого документа нужно объединить title и text
        3. Сохранить эмбеддинги в self.embeddings
        """
        self.documents.extend(documents)
        for doc in documents:
            emb = self.model.encode(f"{doc.title}{doc.text}")
            doc.embedding = emb
        self.embeddings = np.array([doc.embedding for doc in documents])

    def save(self, path: str) -> None:
        """
        TODO: Реализовать сохранение индекса
        1. Сохранить self.documents и self.embeddings в pickle файл
        """
        try:
            with open(path, 'wb') as file:
                pickle.dump({'documents': self.documents, 'embeddings': self.embeddings}, file)
            print(f"Index successfully saved to {path}")
        except Exception as e:
            print(f"An error occurred while saving the index: {e}")

    def load(self, path: str) -> None:
        """
        TODO: Реализовать загрузку индекса
        1. Загрузить self.documents и self.embeddings из pickle файла
        """
        try:
            with open(path, 'rb') as file:
                data = pickle.load(file)
                self.documents = data.get('documents', {})
                self.embeddings = data.get('embeddings', {})
            print(f"Index successfully loaded from {path}")
        except FileNotFoundError:
            print(f"File not found: {path}")
        except Exception as e:
            print(f"An error occurred while loading the index: {e}")


class Searcher:
    def __init__(self, index_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        """
        TODO: Реализовать инициализацию поиска
        1. Загрузить индекс из index_path
        2. Инициализировать sentence-transformers
        """
        self.indexer = Indexer(model_name=model_name)
        self.indexer.load(index_path)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        TODO: Реализовать поиск документов
        1. Получить эмбеддинг запроса через self.model.encode()
        2. Вычислить косинусное сходство между запросом и документами
        3. Вернуть top_k наиболее похожих документов
        """
        top_k_heap = []

        for doc in self.indexer.documents:
            cos_score = float((cosine_similarity(self.indexer.model.encode(query).reshape(1, -1),
                                                 doc.embedding.reshape(1, -1))[0][0] + 1) / 2)
            if len(top_k_heap) < top_k:
                heapq.heappush(top_k_heap, (cos_score, doc.id, doc.title, doc.text))
            else:
                heapq.heappushpop(top_k_heap, (cos_score, doc.id, doc.title, doc.text))

        return [SearchResult(doc_id, cos_score, doc_title, doc_text) for (cos_score, doc_id, doc_title, doc_text) in
                top_k_heap]
