import requests
import config


def get_embedding(text: str) -> list[float]:
    if config.PROVIDER == "openai":
        return _openai_embedding(text)
    elif config.PROVIDER == "azure":
        return _azure_embedding(text)
    else:
        return _ollama_embedding(text)


def _ollama_embedding(text: str) -> list[float]:
    response = requests.post(
        f"{config.EMBEDDING_URL}/api/embeddings",
        json={"model": config.EMBEDDING_MODEL, "prompt": text},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def _openai_embedding(text: str) -> list[float]:
    # pip install openai  (when switching to openai)
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.embeddings.create(input=text, model=config.OPENAI_EMBEDDING_MODEL)
    return response.data[0].embedding


def _azure_embedding(text: str) -> list[float]:
    # pip install openai  (when switching to azure)
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=config.AZURE_OPENAI_API_KEY,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_version="2024-02-01",
    )
    response = client.embeddings.create(input=text, model=config.AZURE_EMBEDDING_DEPLOYMENT)
    return response.data[0].embedding
