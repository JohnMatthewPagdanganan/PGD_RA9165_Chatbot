# cache_io.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import numpy as np
import hnswlib
from rank_bm25 import BM25Okapi

def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

@dataclass
class CorpusStore:
    name: str
    chunks: List[dict]
    embeddings: np.ndarray              # float32 (N, D)
    index: hnswlib.Index                # HNSW cosine
    bm25: BM25Okapi

def get_cache_paths(cache_root: Path, corpus_name: str) -> Dict[str, Path]:
    cache_dir = cache_root / corpus_name
    return {
        "CACHE_DIR": cache_dir,
        "CHUNKS_PATH": cache_dir / "chunks.cleaned.jsonl",
        "EMBED_PATH": cache_dir / "embeddings.npy",
        "HNSW_INDEX_PATH": cache_dir / "hnsw.index",
        "HNSW_META_PATH": cache_dir / "hnsw_meta.json",
        "BM25_PATH": cache_dir / "bm25_tokens.jsonl",
        "MANIFEST_PATH": cache_dir / "manifest.json",
    }

def cache_exists(paths: Dict[str, Path]) -> bool:
    needed = [
        paths["CHUNKS_PATH"], paths["EMBED_PATH"],
        paths["HNSW_INDEX_PATH"], paths["HNSW_META_PATH"],
        paths["BM25_PATH"]
    ]
    return all(p.exists() for p in needed)

def load_cache(paths: Dict[str, Path]) -> Tuple[List[dict], np.ndarray, hnswlib.Index, BM25Okapi]:
    chunks = read_jsonl(paths["CHUNKS_PATH"])
    embeddings = np.load(paths["EMBED_PATH"]).astype(np.float32)

    meta = read_json(paths["HNSW_META_PATH"])
    index = hnswlib.Index(space=meta["space"], dim=int(meta["dim"]))
    index.load_index(str(paths["HNSW_INDEX_PATH"]))
    index.set_ef(int(meta.get("ef", 50)))

    bm25_rows = read_jsonl(paths["BM25_PATH"])
    bm25_corpus = [r["toks"] for r in bm25_rows]
    bm25 = BM25Okapi(bm25_corpus)

    return chunks, embeddings, index, bm25

def load_all_stores(cache_root: Path, corpora: List[str]) -> Dict[str, CorpusStore]:
    stores: Dict[str, CorpusStore] = {}
    for name in corpora:
        paths = get_cache_paths(cache_root, name)
        if not cache_exists(paths):
            missing = [k for k, p in paths.items() if k.endswith("_PATH") and not p.exists()]
            raise FileNotFoundError(f"Cache missing for {name}: {missing} at {paths['CACHE_DIR']}")
        chunks, embeddings, index, bm25 = load_cache(paths)
        stores[name] = CorpusStore(name=name, chunks=chunks, embeddings=embeddings, index=index, bm25=bm25)
    return stores