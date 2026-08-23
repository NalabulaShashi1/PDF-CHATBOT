import math
import re
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np
from src.rag.chunker import TextChunk
from src.rag.vector_store import VectorStore, SearchResult


class BM25Retriever:
    """High-Performance Inverted-Index BM25 Okapi Sparse Keyword Ranker."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[TextChunk] = []
        self.doc_len: np.ndarray = np.array([])
        self.avgdl: float = 0.0
        self.idf: Dict[str, float] = {}
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}

    def fit(self, chunks: List[TextChunk]):
        self.chunks = chunks
        if not chunks:
            self.inverted_index = {}
            self.doc_len = np.array([])
            return

        n_docs = len(chunks)
        doc_lens = []
        self.inverted_index = {}
        doc_freqs: Dict[str, int] = {}

        for doc_idx, c in enumerate(chunks):
            tokens = self._tokenize(c.text)
            doc_lens.append(len(tokens))
            term_counts = Counter(tokens)
            
            for term, tf in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_idx, tf))
                doc_freqs[term] = doc_freqs.get(term, 0) + 1

        self.doc_len = np.array(doc_lens, dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_len)) if len(self.doc_len) > 0 else 1.0

        self.idf = {}
        for term, freq in doc_freqs.items():
            self.idf[term] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_k: int = 4) -> List[Tuple[int, float]]:
        if not self.chunks or len(self.doc_len) == 0:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        scores = np.zeros(len(self.chunks), dtype=np.float32)
        
        for token in q_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            postings = self.inverted_index.get(token, [])
            
            for doc_idx, tf in postings:
                d_len = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (d_len / max(self.avgdl, 1.0)))
                score_part = idf_val * ((tf * (self.k1 + 1.0)) / denom)
                scores[doc_idx] += score_part

        non_zero_indices = np.where(scores > 0)[0]
        if len(non_zero_indices) == 0:
            return []

        if len(non_zero_indices) <= top_k:
            ranked = non_zero_indices[np.argsort(-scores[non_zero_indices])]
        else:
            part = np.argpartition(-scores[non_zero_indices], top_k)[:top_k]
            ranked = non_zero_indices[part[np.argsort(-scores[non_zero_indices[part]])]]

        return [(int(idx), float(scores[idx])) for idx in ranked]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())


class HybridRetriever:
    """Hybrid Search Engine combining Dense Vector Search with Sparse BM25 Keyword Search."""

    def __init__(self, vector_store: VectorStore, alpha: float = 0.6):
        self.vector_store = vector_store
        self.bm25 = BM25Retriever()
        self.alpha = min(max(alpha, 0.0), 1.0)
        self._is_indexed = False

    def index(self, chunks: List[TextChunk]):
        self.vector_store.clear()
        self.vector_store.add_chunks(chunks)
        self.bm25.fit(chunks)
        self._is_indexed = True

    def retrieve(self, query: str, top_k: int = 4) -> List[SearchResult]:
        if not self._is_indexed:
            return []

        all_chunks = self.vector_store.get_all_chunks()
        if not all_chunks:
            return []

        dense_results = self.vector_store.search(query, top_k=min(top_k * 2, len(all_chunks)))
        bm25_raw = self.bm25.search(query, top_k=min(top_k * 2, len(all_chunks)))

        dense_score_map: Dict[str, float] = {}
        if dense_results:
            max_dense = max(r.score for r in dense_results)
            for r in dense_results:
                norm_score = r.score / max_dense if max_dense > 0 else 0.0
                dense_score_map[r.chunk.chunk_id] = norm_score

        sparse_score_map: Dict[str, float] = {}
        if bm25_raw:
            max_bm25 = max(s for _, s in bm25_raw)
            for idx, raw_score in bm25_raw:
                norm_score = raw_score / max_bm25 if max_bm25 > 0 else 0.0
                c_id = all_chunks[idx].chunk_id
                sparse_score_map[c_id] = norm_score

        combined_scores: Dict[str, float] = {}
        all_candidate_ids = set(dense_score_map.keys()) | set(sparse_score_map.keys())

        for c_id in all_candidate_ids:
            d_s = dense_score_map.get(c_id, 0.0)
            s_s = sparse_score_map.get(c_id, 0.0)
            fused_score = (self.alpha * d_s) + ((1.0 - self.alpha) * s_s)
            combined_scores[c_id] = fused_score

        chunk_map = {c.chunk_id: c for c in all_chunks}
        sorted_candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: List[SearchResult] = []
        for c_id, score in sorted_candidates:
            if c_id in chunk_map:
                results.append(SearchResult(
                    chunk=chunk_map[c_id],
                    score=round(score, 4),
                    retrieval_method="hybrid_dense_bm25"
                ))

        return results