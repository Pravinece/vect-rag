import config

print(f"LLM Provider: {config.LLM_PROVIDER}\n" + "-" * 25)

if config.LLM_PROVIDER == "gemini":
    from google import genai
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    for model in client.models.list():
        print(f"Name: {model.name}\nDescription: {model.description}\n")

elif config.LLM_PROVIDER == "groq":
    from groq import Groq
    client = Groq(api_key=config.GROQ_API_KEY)
    for model in client.models.list().data:
        print(f"Name: {model.id}")

elif config.LLM_PROVIDER == "openai":
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    for model in client.models.list().data:
        print(f"Name: {model.id}")

elif config.LLM_PROVIDER == "azure":
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=config.AZURE_OPENAI_API_KEY,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_version="2024-02-01",
    )
    for model in client.models.list().data:
        print(f"Name: {model.id}")

elif config.LLM_PROVIDER == "openrouter":
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    for model in client.models.list().data:
        print(f"Name: {model.id}")

elif config.LLM_PROVIDER == "ollama":
    import requests
    response = requests.get(f"{config.LLM_URL}/api/tags")
    for model in response.json().get("models", []):
        print(f"Name: {model['name']}")

else:
    print(f"Unknown provider: {config.LLM_PROVIDER}")
