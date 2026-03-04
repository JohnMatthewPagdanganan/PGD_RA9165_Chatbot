# rerank_crossencoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os
import inspect

from sentence_transformers import CrossEncoder

Hit = Dict[str, Any]


@dataclass
class CrossEncoderReranker:
    model_name: str
    device: Optional[str] = None   # Streamlit Cloud is usually CPU
    max_length: int = 512

    _model: Optional[CrossEncoder] = None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        kwargs = {}

        # device handling (safe)
        if self.device:
            kwargs["device"] = self.device

        # max_length supported by CrossEncoder in sentence-transformers
        kwargs["max_length"] = int(self.max_length)

        # HuggingFace cache arg name differs across versions:
        # some use cache_folder, some use cache_dir
        cache_path = os.getenv("SENTENCE_TRANSFORMERS_HOME") or os.getenv("HF_HOME") or None
        if cache_path:
            sig = inspect.signature(CrossEncoder.__init__)
            if "cache_folder" in sig.parameters:
                kwargs["cache_folder"] = cache_path
            elif "cache_dir" in sig.parameters:
                kwargs["cache_dir"] = cache_path

        # IMPORTANT: do NOT pass token/tokenizer_args/etc here.
        # HF auth is handled by HF_TOKEN/HUGGINGFACE_HUB_TOKEN env vars.
        self._model = CrossEncoder(self.model_name, **kwargs)

    def rerank_hits(self, query: str, hits: List[Hit], top_n: int = 30) -> List[Hit]:
        if not hits:
            return []

        self._ensure_loaded()

        pairs = [(query, (h.get("text") or "")) for h in hits]
        scores = self._model.predict(pairs)

        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)

        hits.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
        return hits[: int(top_n)]


def diversify_by_page(hits: List[Hit], max_per_page: int = 2) -> List[Hit]:
    seen = {}
    out = []
    for h in hits:
        page = h.get("page")
        seen[page] = seen.get(page, 0)
        if seen[page] < max_per_page:
            out.append(h)
            seen[page] += 1
    return out