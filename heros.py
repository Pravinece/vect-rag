import os
from psycopg2.extras import execute_values
from embedder import get_embedding
from ingest import chunk_text
from db import create_source_table, build_vector_index, get_registry, upsert_registry, get_all_registry
from llm import ask_llm
import config


def _table_name(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    name = name.replace("-", "_").replace(" ", "_").lower()
    return f"docs_{name}"


def ingest_by_filename(conn, filename: str, description: str = None) -> dict:
    filepath = os.path.join(config.DOCUMENTS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"'{filename}' not found in '{config.DOCUMENTS_DIR}/'")

    # check registry
    registry = get_registry(conn, filename)
    is_update = registry is not None
    table_name = registry["table_name"] if is_update else _table_name(filename)

    # read and chunk file
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    chunks = chunk_text(content, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        raise ValueError(f"No content found in '{filename}'")

    # create table if new
    create_source_table(conn, table_name)

    # if update → clear old chunks
    if is_update:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table_name}")
        conn.commit()

    # embed and insert
    records = []
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}/{len(chunks)}] embedding chunk {i}")
        embedding = get_embedding(chunk)
        records.append((i, chunk, embedding))

    with conn.cursor() as cur:
        execute_values(cur,
            f"INSERT INTO {table_name} (chunk_id, content, embedding) VALUES %s",
            records,
            template="(%s, %s, %s::vector)"
        )
    conn.commit()

    # build vector index after insert
    build_vector_index(conn, table_name)

    # upsert registry
    upsert_registry(conn, filename, table_name, description, len(records), is_update)

    # fetch updated_at from registry
    updated = get_registry(conn, filename)

    return {
        "filename": filename,
        "table_name": table_name,
        "chunk_count": len(records),
        "is_update": is_update,
        "updated_at": updated["updated_at"],
    }

def search_pg(conn, query: str, top_k: int = 3) -> dict:
    registry = get_all_registry(conn)
    if not registry:
        raise ValueError("No sources found in registry. Please ingest first.")

    # build source list with descriptions for LLM router
    source_list = "\n".join(
        f"- {r['filename']}: {r['description'] or 'No description'}"
        for r in registry
    )

    # ask LLM to pick the most relevant source
    prompt = f"""You are a source router. Given the user query and available sources, return ONLY the most relevant filename, nothing else.

Sources:
{source_list}

Query: {query}

Return only the filename:"""

    matched_filename = ask_llm(prompt, "").strip()
    print(f"LLM selected source: {matched_filename}")
    # fallback — if LLM returns something not in registry, pick first
    registry_filenames = [r["filename"] for r in registry]
    if matched_filename not in registry_filenames:
        matched_filename = registry_filenames[0]

    # get table name for matched source
    matched = next(r for r in registry if r["filename"] == matched_filename)
    table_name = matched["table_name"]

    # vector search on matched cluster table
    embedding = get_embedding(query)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT chunk_id, content, 1 - (embedding <=> %s::vector) AS score
            FROM {table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (embedding, embedding, top_k))
        rows = cur.fetchall()

    return {
        "query": query,
        "matched_source": matched_filename,
        "results": [
            {"filename": matched_filename, "chunk_id": r[0], "content": r[1], "score": round(float(r[2]), 4)}
            for r in rows
        ]
    }
    
def chat_pg(conn, question: str, top_k: int = 3) -> dict:
    # reuse search_pg to get chunks from correct cluster
    search_result = search_pg(conn, question, top_k)

    if not search_result["results"]:
        raise ValueError("No relevant context found.")

    context = "\n\n---\n\n".join(
        f"[chunk {r['chunk_id']}]\n{r['content']}"
        for r in search_result["results"]
    )

    answer = ask_llm(question, context)

    return {
        "question": question,
        "answer": answer,
        "matched_source": search_result["matched_source"],
        "sources": search_result["results"],
    }


def ingest_description(conn, filename: str) -> dict:
    registry = get_registry(conn, filename)
    if not registry:
        raise FileNotFoundError(f"'{filename}' not found in registry. Please ingest first.")

    filepath = os.path.join(config.DOCUMENTS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"'{filename}' not found in '{config.DOCUMENTS_DIR}/'")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    if not content.strip():
        raise ValueError(f"No content found in file '{filename}'")

    # ask LLM to generate rich description from full content
    description = ask_llm(
        f"""Summarize the key topics, entities, names, and keywords in this document.
Be specific — include proper nouns, organizations, people, and events.
Keep it under 100 words.

{content}""", ""
    ).strip()

    # update registry with new description
    upsert_registry(conn, filename, registry["table_name"], description, registry["chunk_count"], is_update=True)

    updated = get_registry(conn, filename)
    return {
        "filename": filename,
        "table_name": registry["table_name"],
        "chunk_count": registry["chunk_count"],
        "is_update": True,
        "updated_at": updated["updated_at"],
    }
