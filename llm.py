import requests
import config


def ask_llm(question: str, context: str) -> str:
    prompt = f"""You are a helpful assistant. Use only the context below to answer the question.
If the answer is not in the context, say "I don't have information about that."

Context:
{context}

Question: {question}
Answer:"""

    if config.PROVIDER == "openai":
        return _openai_chat(prompt)
    elif config.PROVIDER == "azure":
        return _azure_chat(prompt)
    else:
        return _ollama_chat(prompt)


def _ollama_chat(prompt: str) -> str:
    response = requests.post(
        f"{config.LLM_URL}/api/generate",
        json={"model": config.LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"]


def _openai_chat(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=config.OPENAI_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _azure_chat(prompt: str) -> str:
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=config.AZURE_OPENAI_API_KEY,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_version="2024-02-01",
    )
    response = client.chat.completions.create(
        model=config.AZURE_LLM_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
