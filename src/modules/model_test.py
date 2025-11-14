import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)

print("List of models that support generateContent:\n")
for m in client.models.list():
    for action in m.supported_actions:
        if action == "generateContent":
            print(m.name)

print("List of models that support embedContent:\n")
for m in client.models.list():
    for action in m.supported_actions:
        if action == "embedContent":
            print(m.name)