# main.py
from pathlib import Path
import torch

from cache_io import load_all_stores
from retrieval import Retriever
from rerank_crossencoder import CrossEncoderReranker
from llm_together import TogetherLLM, TogetherConfig

# --- fastcoref patch + init (copied from your notebook) ---
from transformers import AutoConfig
_old_cfg = AutoConfig.from_pretrained
def _patched_cfg(*args, **kwargs):
    cfg = _old_cfg(*args, **kwargs)
    if hasattr(cfg, "attn_implementation"):
        cfg.attn_implementation = "eager"
    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = "eager"
    return cfg
AutoConfig.from_pretrained = _patched_cfg
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from fastcoref.coref_models.modeling_lingmess import LingMessModel
if not hasattr(LingMessModel, "all_tied_weights_keys"):
    LingMessModel.all_tied_weights_keys = {}
from fastcoref import LingMessCoref

from sentence_transformers import SentenceTransformer
from query_tools import QueryProcessor
from chatbot import Lex9165Chatbot

EMBED_MODEL  = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-large"
COREF_MODEL  = "biu-nlp/lingmess-coref"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"

def build_bot(project_root: Path) -> Lex9165Chatbot:
    cache_root = project_root / "Cache"

    # you said you will no longer use jurisprudence_structured
    corpora = ["jurisprudence", "statutes_and_guidelines"]
    stores = load_all_stores(cache_root, corpora)

    embedding_model = SentenceTransformer(EMBED_MODEL)
    retriever = Retriever(embedding_model=embedding_model)

    reranker = CrossEncoderReranker(RERANK_MODEL)

    # ✅ Together API LLM (replaces local Qwen)
    llm = TogetherLLM(TogetherConfig(
        model="Qwen/Qwen2.5-7B-Instruct-turbo",
        temperature=0.0,
        max_tokens=350,
    ))

    coref_model = LingMessCoref(
        model_name_or_path=COREF_MODEL,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    # ✅ QueryProcessor now uses Together LLM for rewrite
    qproc = QueryProcessor(
        coref_model=coref_model,
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