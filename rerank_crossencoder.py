# rerank_crossencoder.py
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

Hit = Dict[str, Any]


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        enabled: bool = True,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")
        self.device = device
        self.enabled = enabled and (os.getenv("DISABLE_RERANKER", "0") != "1")

        self._model = None  # lazy-init

        # Avoid HF Xet edge-cases in some hosted environments
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        # Allow either token env var name
        if os.getenv("HF_TOKEN") and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
            os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    def _ensure_loaded(self):
        if self._model is not None:
            return

        from sentence_transformers import CrossEncoder

        kwargs = {}
        # ✅ CrossEncoder uses cache_dir (NOT cache_folder)
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        if self.device:
            kwargs["device"] = self.device

        self._model = CrossEncoder(self.model_name, **kwargs)

    def rerank_hits(self, query: str, hits: List[Hit], top_n: int = 30) -> List[Hit]:
        if not hits:
            return []

        if not self.enabled:
            hits = list(hits)
            hits.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
            return hits[:top_n]

        self._ensure_loaded()

        pairs = [(query, (h.get("text") or "")) for h in hits]
        scores = self._model.predict(pairs)

        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)

        hits.sort(key=lambda x: float(x.get("rerank_score", 0.0) or 0.0), reverse=True)
        return hits[:top_n]


def diversify_by_page(hits: List[Hit], max_per_page: int = 2) -> List[Hit]:
    seen = {}
    out: List[Hit] = []
    for h in hits:
        page = h.get("page")
        if page not in seen:
            seen[page] = 0
        if seen[page] < max_per_page:
            out.append(h)
            seen[page] += 1
    return out