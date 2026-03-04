# fusion_rrf.py
from __future__ import annotations
from typing import List, Dict, Any

Hit = Dict[str, Any]

def rrf_fusion_weighted(
    chunks: List[dict],
    dense_hits: List[Hit],
    sparse_hits: List[Hit],
    rrf_k: int = 60,
    w_dense: float = 0.9,
    w_sparse: float = 1.3,
) -> List[Hit]:
    """
    Weighted Reciprocal Rank Fusion (RRF).
    - Assumes each hit has: {"idx": int, "score": float, "source":..., "page":..., "text":...}
    - Fuses by internal chunk index (idx) within the SAME corpus.
    Returns list of hits with updated "score" (fused score).
    """
    fused_scores: dict[int, float] = {}

    def add_list(hits: List[Hit], w: float):
        for rank, h in enumerate(hits):
            cid = int(h["idx"])
            fused_scores[cid] = fused_scores.get(cid, 0.0) + w * (1.0 / (rrf_k + rank + 1))

    add_list(dense_hits, w_dense)
    add_list(sparse_hits, w_sparse)

    fused: List[Hit] = []
    for cid, score in fused_scores.items():
        # Rebuild a hit object from the canonical chunk record
        base = chunks[cid]
        fused.append({**base, "idx": cid, "score": float(score)})

    fused.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
    return fused