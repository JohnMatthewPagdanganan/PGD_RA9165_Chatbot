# retrieval.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import re
import numpy as np
from sentence_transformers import SentenceTransformer

Hit = Dict[str, Any]

_WORDS = re.compile(r"\b\w+\b")

def bm25_tokenize(text: str) -> list[str]:
    return _WORDS.findall((text or "").lower())

def dense_query_for_bge_v15(q: str, short_tokens: int = 8) -> str:
    q = (q or "").strip()
    if not q:
        return q
    if len(q.split()) <= int(short_tokens):
        return "Represent this sentence for searching relevant passages: " + q
    return q

@dataclass
class Retriever:
    embedding_model: SentenceTransformer

    def retrieve_bm25(self, store, query: str, k: int = 25) -> List[Hit]:
        q_tokens = bm25_tokenize(query)
        scores = store.bm25.get_scores(q_tokens)
        top_idx = np.argsort(scores)[::-1][:max(k * 5, k)]

        results: List[Hit] = []
        chunks = store.chunks
        for i in top_idx:
            i = int(i)
            results.append({**chunks[i], "score": float(scores[i]), "idx": i})
        return results[:k]

    def retrieve_hnsw(self, store, query: str, k: int = 25) -> List[Hit]:
        q2 = dense_query_for_bge_v15(query, short_tokens=8)
        q_emb = self.embedding_model.encode([q2], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

        n = int(store.index.get_current_count())
        if n <= 0:
            return []

        pull_k = max(int(k * 8), int(k))
        pull_k = min(pull_k, n)

        try:
            store.index.set_ef(max(50, pull_k))
        except Exception:
            pass

        labels, distances = store.index.knn_query(q_emb, k=pull_k)

        results: List[Hit] = []
        chunks = store.chunks
        for idx, dist in zip(labels[0], distances[0]):
            i = int(idx)
            score = 1.0 - float(dist)
            results.append({**chunks[i], "score": score, "idx": i})

        return results[:k]