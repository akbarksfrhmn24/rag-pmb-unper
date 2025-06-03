import os

# General settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/chroma_db")

# OpenAI settings
# You can set this via an environment variable or replace "your_openai_api_key_here" with your actual key.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")

GITHUB_API_KEY = os.getenv("GITHUB_API_KEY", "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")