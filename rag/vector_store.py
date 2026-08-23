from typing import List, Optional
import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from src.rag.chunker import TextChunk


class SearchResult(BaseModel):
    chunk: TextChunk
    score: float
    retrieval_method: str = "vector"


class VectorStore:
    """In-memory Vector Store with dense semantic embeddings and cosine similarity index."""

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.chunks: List[TextChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
        self._is_fitted = False

    def add_chunks(self, new_chunks: List[TextChunk]) -> int:
        if not new_chunks:
            return 0
        self.chunks.extend(new_chunks)
        self._build_index()
        return len(self.chunks)

    def clear(self):
        self.chunks = []
        self.embeddings = None
        self.vectorizer = None
        self.svd = None
        self._is_fitted = False

    def _build_index(self):
        corpus = [c.text for c in self.chunks]
        if not corpus:
            return

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=2500,
            sublinear_tf=True
        )
        tfidf_mat = self.vectorizer.fit_transform(corpus)
        n_samples, n_features = tfidf_mat.shape

        n_components = min(self.embedding_dim, n_features - 1, n_samples - 1)
        if n_components >= 2:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            dense_vectors = self.svd.fit_transform(tfidf_mat)
        else:
            dense_vectors = tfidf_mat.toarray()

        norms = np.linalg.norm(dense_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.embeddings = dense_vectors / norms
        self._is_fitted = True

    def embed_query(self, query: str) -> Optional[np.ndarray]:
        if not self._is_fitted or not self.vectorizer:
            return None

        q_tfidf = self.vectorizer.transform([query])
        if self.svd is not None:
            q_dense = self.svd.transform(q_tfidf)
        else:
            q_dense = q_tfidf.toarray()

        norm = np.linalg.norm(q_dense)
        if norm == 0:
            return None
        return q_dense / norm

    def search(self, query: str, top_k: int = 4) -> List[SearchResult]:
        if not self._is_fitted or self.embeddings is None or not self.chunks:
            return []

        q_vec = self.embed_query(query)
        if q_vec is None:
            return []

        similarities = np.dot(self.embeddings, q_vec.T).ravel()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.0:
                results.append(SearchResult(
                    chunk=self.chunks[idx],
                    score=round(score, 4),
                    retrieval_method="dense_vector"
                ))
        return results

    def get_all_chunks(self) -> List[TextChunk]:
        return list(self.chunks)