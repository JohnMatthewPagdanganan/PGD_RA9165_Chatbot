# main.py  (NO fastcoref version — copy/paste this whole file)
from pathlib import Path

from cache_io import load_all_stores
from retrieval import Retriever
from rerank_crossencoder import CrossEncoderReranker
from llm_together import TogetherLLM, TogetherConfig

from sentence_transformers import SentenceTransformer
from query_tools import QueryProcessor
from chatbot import Lex9165Chatbot
import os
import streamlit as st

# Avoid HF Xet token fetch path that often triggers 429 on shared IPs
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Provide HF auth for model downloads
HF_TOKEN = st.secrets.get("HF_TOKEN", None) or os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN


EMBED_MODEL  = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-large"

def build_bot(project_root: Path) -> Lex9165Chatbot:
    cache_root = project_root / "Cache"

    # you said you will no longer use jurisprudence_structured
    corpora = ["jurisprudence", "statutes_and_guidelines"]
    stores = load_all_stores(cache_root, corpora)

    embedding_model = SentenceTransformer(EMBED_MODEL, token=HF_TOKEN)
    retriever = Retriever(embedding_model=embedding_model)

    reranker = CrossEncoderReranker(RERANK_MODEL, cache_dir=str(project_root / ".hf_cache"), device="cpu")

    # Together API LLM
    llm = TogetherLLM(TogetherConfig(
        model="Qwen/Qwen2.5-7B-Instruct-turbo",
        temperature=0.0,
        max_tokens=350,
    ))

    # ✅ QueryProcessor without fastcoref
    qproc = QueryProcessor(
        llm=llm,
        max_history_turns=4
    )

    return Lex9165Chatbot(
        stores=stores,
        retriever=retriever,
        reranker=reranker,
        qproc=qproc,
        llm=llm,
    )

def main():
    root = Path(__file__).resolve().parent
    bot = build_bot(root)

    print("Lex9165 ready. Type 'exit' to quit.\n")
    while True:
        q = input("You> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        reply, hits = bot.answer(q)
        print("\nAssistant:\n", reply)
        print("\nSources:")
        for i, h in enumerate(hits, 1):
            rrf_sc = float(h.get("score", 0.0))
            ce_sc = h.get("rerank_score", None)
            if ce_sc is None:
                print(f"  [{i}] {h['source']} p.{h['page']}  rrf={rrf_sc:.3f}")
            else:
                print(f"  [{i}] {h['source']} p.{h['page']}  ce={float(ce_sc):.3f}  rrf={rrf_sc:.3f}")
        print("\n" + "-" * 80 + "\n")

if __name__ == "__main__":
    main()