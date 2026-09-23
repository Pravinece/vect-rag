from fastapi import FastAPI, HTTPException
from typing import Optional
import config
import retriever
from ingest import ingest
from llm import ask_llm
from contextlib import asynccontextmanager
from db import postgres
from pydantic import BaseModel
import heros
from schemas.herosSchema import IngestRequest, IngestResponse, HeroSearchRequest, SearchResponse, MetadataRequest, HeroChatRequest, HeroChatResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    postgres.connect()
    yield
    postgres.disconnect()


app = FastAPI(
    title="Vector RAG API",
    description="RAG pipeline using FAISS + Ollama (upgradeable to OpenAI/Azure)",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    top_k: int = config.TOP_K

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]

class SearchRequest(BaseModel):
    query: str
    top_k: int = config.TOP_K


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "running",
        "provider": config.PROVIDER,
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.LLM_MODEL,
    }


@app.post("/ingest")
def ingest_documents():
    """
    Reads all .txt files from the documents/ folder,
    chunks them, generates embeddings, and stores in FAISS.
    """
    try:
        index, docs = ingest()
        retriever.reload_index()
        return {"message": "Ingestion complete", "total_chunks": len(docs)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=list[dict])
def search(req: SearchRequest):
    """
    Returns top-k similar chunks from FAISS without LLM generation.
    Useful for debugging retrieval quality.
    """
    try:
        results = retriever.retrieve(req.query, req.top_k)
        return results
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Full RAG pipeline:
    1. Embed the user question
    2. Retrieve top-k chunks from FAISS
    3. Send context + question to LLM
    4. Return answer with sources
    """
    try:
        chunks = retriever.retrieve(req.question, req.top_k)
        if not chunks:
            raise HTTPException(status_code=404, detail="No relevant context found.")

        context = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in chunks
        )
        answer = ask_llm(req.question, context)

        sources = [{"source": c["source"], "chunk_id": c["chunk_id"], "score": c["score"]} for c in chunks]
        return ChatResponse(question=req.question, answer=answer, sources=sources)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Heros (PG Vector) Routes ───────────────────────────────────────────────────

@app.post("/heros/ingest", response_model=IngestResponse)
def heros_ingest(req: IngestRequest):
    """Ingest a single file by filename into its own pgvector cluster."""
    try:
        result = heros.ingest_by_filename(postgres.conn, req.filename, req.description)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/heros/generate-metadata", response_model=IngestResponse)
def heros_generate_metadata(req: MetadataRequest):
    """Generate description for a file using LLM and update registry."""
    try:
        result = heros.ingest_description(postgres.conn, req.filename)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/heros/search", response_model=SearchResponse)
def heros_search(req: HeroSearchRequest):
    """Search pgvector. LLM auto-detects the correct cluster."""
    try:
        results = heros.search_pg(postgres.conn, req.query, req.top_k)
        return results
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/heros/chat", response_model=HeroChatResponse)
def heros_chat(req: HeroChatRequest):
    """Full RAG pipeline on pgvector. LLM auto-detects cluster, retrieves context, returns answer."""
    try:
        result = heros.chat_pg(postgres.conn, req.question, req.top_k)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/status")
def status():
    """Check if FAISS index is loaded and how many vectors are stored."""
    import os, json
    meta_file = os.path.join(config.FAISS_INDEX_PATH, "metadata.json")
    if not os.path.exists(meta_file):
        return {"index_ready": False, "total_chunks": 0}
    with open(meta_file) as f:
        meta = json.load(f)
    return {"index_ready": True, "total_chunks": len(meta), "provider": config.PROVIDER}
