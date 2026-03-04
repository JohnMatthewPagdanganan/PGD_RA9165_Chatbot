# rerank_crossencoder.py

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

Hit = Dict[str, Any]


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank_hits(self, query: str, hits: List[Hit], top_n: int = 30) -> List[Hit]:
        if not hits:
            return []

        pairs = [(query, h["text"]) for h in hits]
        scores = self.model.predict(pairs)

        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)

        hits.sort(key=lambda x: x["rerank_score"], reverse=True)
        return hits[:top_n]


def diversify_by_page(hits: List[Hit], max_per_page: int = 2) -> List[Hit]:
    seen = {}
    out = []

    for h in hits:
        page = h.get("page")
        if page not in seen:
            seen[page] = 0

        if seen[page] < max_per_page:
            out.append(h)
            seen[page] += 1

    return out