from dotenv import load_dotenv
import os

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "ollama")  # ollama | openai | azure

# Ollama (current)
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://172.23.198.77:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
LLM_URL = os.getenv("LLM_URL", "http://172.23.199.231:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:14b")

# OpenAI (future)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")

# Azure (future)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_LLM_DEPLOYMENT = os.getenv("AZURE_LLM_DEPLOYMENT")

DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "documents")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_store")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "3"))
