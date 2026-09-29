from google import genai
import config

# Initialize the client
client = genai.Client(api_key=config.GEMINI_API_KEY)

print("Available Embedding Models:\n" + "-"*25)

# Iterate through the models and filter for embedding capabilities
for model in client.models.list():
    # The API returns camelCase action names like 'embedContent'
    # if 'embedContent' in model.supported_actions:
    print(f"Model Name: {model.name}")
    print(f"Description: {model.description}\n")