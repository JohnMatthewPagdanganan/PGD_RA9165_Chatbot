# chatbot.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import re
import numpy as np

from cache_io import CorpusStore
from routing import corpora_to_search
from retrieval import Retriever
from fusion_rrf import rrf_fusion_weighted
from rerank_crossencoder import CrossEncoderReranker, diversify_by_page
from prompts import build_system_prompt, build_onepass_prompt, normalize_onepass_output, has_uncited_sentences, IDK_LINE
from query_tools import parse_control_block, detect_mode, dynamic_k, QueryProcessor

Hit = Dict[str, Any]

def select_diverse_hits_v2(hits: List[Hit], embeddings: np.ndarray, top_k: int = 10, lambda_mult: float = 0.7) -> List[Hit]:
    if not hits:
        return []
    selected = [hits[0]]
    selected_ids = {hits[0]["idx"]}

    while len(selected) < top_k and len(selected) < len(hits):
        best = None
        best_val = -1e9
        for h in hits:
            cid = int(h["idx"])
            if cid in selected_ids:
                continue
            rel = float(h.get("rerank_score", h.get("score", 0.0)) or 0.0)

            v = embeddings[cid]
            sims = [float(np.dot(v, embeddings[int(s["idx"])])) for s in selected]
            max_sim = max(sims) if sims else 0.0

            mmr = lambda_mult * rel - (1 - lambda_mult) * max_sim
            if mmr > best_val:
                best_val = mmr
                best = h
        if best is None:
            break
        selected.append(best)
        selected_ids.add(int(best["idx"]))
    return selected


@dataclass
class Lex9165Chatbot:
    stores: Dict[str, CorpusStore]
    retriever: Retriever
    reranker: CrossEncoderReranker
    qproc: QueryProcessor
    llm: any  # QwenGenerator-like: .generate_chat(messages)->str

    # fusion params (your defaults)
    rrf_k: int = 60
    w_dense: float = 0.9
    w_sparse: float = 1.3

    def _retrieve_multi(self, corpora: List[str], retrieval_query: str, cand_k: int) -> Tuple[List[Hit], List[Hit]]:
        all_dense: List[Hit] = []
        all_sparse: List[Hit] = []
        for cname in corpora:
            store = self.stores[cname]
            ck = min(cand_k, len(store.chunks))
            all_dense.extend(self.retriever.retrieve_hnsw(store, retrieval_query, k=ck))
            all_sparse.extend(self.retriever.retrieve_bm25(store, retrieval_query, k=ck))
        return all_dense, all_sparse

    def answer(self, user_query: str) -> Tuple[str, List[Hit]]:
        # A) control tags + mode
        mode_override, cleaned_query = parse_control_block(user_query)
        mode = mode_override or detect_mode(cleaned_query)

        # B) rewrite/coref for retrieval query
        retrieval_query = self.qproc.finalize_retrieval_query(cleaned_query)

        # C) pick corpora (routing preserved here ✅)
        corpora = corpora_to_search(cleaned_query)

        # D) dynamic K
        ks = dynamic_k(mode)
        cand_k = max(ks["cand"], 120)      # match your “min 120”
        rerank_k = max(ks["rerank"], 60)   # match your “min 60”
        final_k = min(ks["final"], 12)

        # E) retrieve + fuse + rerank
        dense_hits, sparse_hits = self._retrieve_multi(corpora, retrieval_query, cand_k=cand_k)

        # Use chunks from the *first* corpus for reconstruction by idx is unsafe across corpora.
        # So we fuse purely by hit objects: easiest fix = treat idx as local-per-corpus? (not compatible).
        # Your notebook fuses across corpora but uses global "chunks" index, which only works if same chunk list.
        # ✅ Practical solution: fuse per-corpus then merge. We'll do that properly.
        fused_all: List[Hit] = []
        for cname in corpora:
            store = self.stores[cname]
            dh = [h for h in dense_hits if h.get("source", "").startswith(str(store.chunks[0]["source"]).split("/")[0]) or True]
            # (We don't rely on this heuristic; instead we re-run per corpus to be clean.)
            dh = self.retriever.retrieve_hnsw(store, retrieval_query, k=min(cand_k, len(store.chunks)))
            sh = self.retriever.retrieve_bm25(store, retrieval_query, k=min(cand_k, len(store.chunks)))
            fused = rrf_fusion_weighted(store.chunks, dh, sh, rrf_k=self.rrf_k, w_dense=self.w_dense, w_sparse=self.w_sparse)
            fused_all.extend(fused)

        fused_all.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)

        reranked = self.reranker.rerank_hits(retrieval_query, fused_all, top_n=min(rerank_k, len(fused_all)))
        reranked = diversify_by_page(reranked, max_per_page=2)

        # F) MMR per hit needs embeddings; we approximate by using corpus embedding based on idx+source match.
        # Simplest: skip MMR across corpora OR do MMR only within top results as-is.
        # We'll do a safe lightweight diversity: keep top N (already diversified by page).
        hits = reranked[:final_k]

        if not hits:
            reply = "CITATIONS: none\nANSWER:\n" + IDK_LINE
            self.qproc.update_history(user_query, reply)
            return reply, []

        # G) context
        context = "\n\n".join(
            [f"[{i+1}] (Source: {h['source']}, page {h['page']})\n{h['text']}"
             for i, h in enumerate(hits)]
        )

        # H) generate
        sys = build_system_prompt()
        msgs = [
            {"role": "system", "content": sys},
            {"role": "user", "content": build_onepass_prompt(mode, context, cleaned_query)},
        ]
        out = self.llm.generate_chat(msgs)
        reply = normalize_onepass_output(out)

        # I) strict citation gate
        m = re.search(r"^ANSWER:\s*(.*)$", reply, re.M | re.S)
        answer_body = (m.group(1).strip() if m else "")
        if has_uncited_sentences(answer_body):
            # strict regenerate
            strict_addon = (
                "\n\nSTRICT RULES:\n"
                "- Every sentence MUST contain at least one citation like [1].\n"
                "- Do NOT write any sentence without a citation.\n"
                "- If context is insufficient, use the IDK format.\n"
            )
            msgs2 = [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": build_onepass_prompt(mode, context, cleaned_query) + strict_addon},
            ]
            out2 = self.llm.generate_chat(msgs2)
            reply = normalize_onepass_output(out2)

        self.qproc.update_history(user_query, reply)
        return reply, hits