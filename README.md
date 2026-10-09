# Vector RAG API

RAG pipeline using FAISS + pgvector with support for multiple LLM providers.

---

## Setup

**1. Create & activate virtual environment**
```cmd
python -m venv venv
venv\Scripts\activate
```

**2. Install dependencies**
```cmd
pip install -r requirements.txt
```

**3. Configure `.env`**
```env
PROVIDER=gemini                          # gemini | ollama | openai | azure | groq | openrouter

# Gemini
GEMINI_API_KEY=<your-gemini-api-key>
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
GEMINI_LLM_MODEL=models/gemini-3.8-flash

# Database
DB_URL=postgresql://postgres:<password>@localhost:5432/ragdb
TELEGRAM_BOT_TOKEN=<your-bot-token>

# Chunking
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=3
EMBEDDING_DIM=768
```

---

## Run the App

**Activate venv (if not already)**
```cmd
venv\Scripts\activate
```

**Start the server**
```cmd
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**One-liner**
```cmd
venv\Scripts\activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI → http://localhost:8000/docs
- API Base   → http://localhost:8000

---

## List Available Models

Lists all models available for the currently configured `PROVIDER` in `.env`.

```cmd
venv\Scripts\activate && python modellist.py
```

**Output example (Gemini):**
```
Provider: gemini
-------------------------
Name: models/gemini-embedding-001
Description: ...

Name: models/gemini-3.8-flash
Description: ...
```

### Supported Providers

| PROVIDER | Requires |
|---|---|
| `gemini` | `GEMINI_API_KEY` |
| `ollama` | `LLM_URL` (local) |
| `openai` | `OPENAI_API_KEY` |
| `azure` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| `groq` | `GROQ_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |

---

## API Endpoints

### General
| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Server status + current provider info |
| GET | `/status` | FAISS index status + total chunks |
| POST | `/ingest` | Ingest all `.txt` files from `documents/` into FAISS |
| POST | `/search` | Semantic search on FAISS |
| POST | `/chat` | Full RAG chat using FAISS |

### Heros (pgvector)
| Method | Endpoint | Description |
|---|---|---|
| POST | `/heros/ingest` | Ingest a single file into pgvector |
| POST | `/heros/generate-metadata` | Auto-generate description via LLM |
| POST | `/heros/search` | Search pgvector (LLM auto-detects cluster) |
| POST | `/heros/chat` | Full RAG chat on pgvector |

### Telegram
| Method | Endpoint | Description |
|---|---|---|
| GET | `/telegram/documents` | List documents sent to the bot (from getUpdates) |
| POST | `/telegram/ingest` | Download file by `file_id` and ingest into pgvector |
