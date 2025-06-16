import os

# General settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/chroma_db")

# OpenAI settings
# You can set this via an environment variable or replace "your_openai_api_key_here" with your actual key.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

GITHUB_API_KEY = os.getenv("GITHUB_API_KEY", "")
