import os
import json
import threading
import numpy as np
import faiss
import config
from embedder import get_embedding

_index = None
_metadata = None
_lock = threading.Lock()


def _load():
    global _index, _metadata
    index_file = os.path.join(config.FAISS_INDEX_PATH, "index.faiss")
    meta_file = os.path.join(config.FAISS_INDEX_PATH, "metadata.json")

    if not os.path.exists(index_file):
        raise FileNotFoundError("FAISS index not found. Please call /ingest first.")

    _index = faiss.read_index(index_file)
    with open(meta_file, "r", encoding="utf-8") as f:
        _metadata = json.load(f)


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    if _index is None:
        with _lock:
            if _index is None:
                _load()

    k = top_k if top_k is not None else config.TOP_K
    vec = np.array([get_embedding(query)], dtype="float32")
    faiss.normalize_L2(vec)

    scores, indices = _index.search(vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(_metadata):
            continue
        chunk = _metadata[idx].copy()
        chunk["score"] = round(float(score), 4)
        results.append(chunk)

    return results


def reload_index():
    global _index, _metadata
    with _lock:
        _index = None
        _metadata = None
        _load()
