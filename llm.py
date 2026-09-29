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
    elif config.PROVIDER == "gemini":
        return _gemini_chat(prompt)
    elif config.PROVIDER == "groq":
        return _groq_chat(prompt)
    elif config.PROVIDER == "openrouter":
        return _openrouter_chat(prompt)
    else:
        return _ollama_chat(prompt)

def _gemini_chat(prompt: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=types.HttpOptions(api_version="v1")
    )
    response = client.models.generate_content(
        model=config.GEMINI_LLM_MODEL,
        contents=prompt
    )
    return response.text

def _groq_chat(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=config.GROQ_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def _openrouter_chat(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model=config.OPENROUTER_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

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
