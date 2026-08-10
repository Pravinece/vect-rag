import os
import json
import numpy as np
import faiss
import config
from embedder import get_embedding


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def load_documents(directory: str) -> list[dict]:
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            chunks = chunk_text(content, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            for i, chunk in enumerate(chunks):
                docs.append({
                    "source": filename,
                    "chunk_id": i,
                    "text": chunk,
                })
    return docs


def build_index(docs: list[dict]):
    print(f"Embedding {len(docs)} chunks...")
    embeddings = []
    for i, doc in enumerate(docs):
        print(f"  [{i+1}/{len(docs)}] {doc['source']} chunk {doc['chunk_id']}")
        vec = get_embedding(doc["text"])
        embeddings.append(vec)

    matrix = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(matrix)

    index = faiss.IndexFlatIP(matrix.shape[1])  # Inner product = cosine after normalize
    index.add(matrix)

    os.makedirs(config.FAISS_INDEX_PATH, exist_ok=True)
    faiss.write_index(index, os.path.join(config.FAISS_INDEX_PATH, "index.faiss"))

    with open(os.path.join(config.FAISS_INDEX_PATH, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"Index saved to '{config.FAISS_INDEX_PATH}/' with {index.ntotal} vectors.")
    return index, docs


def ingest():
    docs = load_documents(config.DOCUMENTS_DIR)
    if not docs:
        raise ValueError(f"No .txt files found in '{config.DOCUMENTS_DIR}/'")
    return build_index(docs)


if __name__ == "__main__":
    ingest()
