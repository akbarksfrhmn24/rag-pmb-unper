from openai import OpenAI
from app.config import OPENAI_API_KEY
from app.config import GITHUB_API_KEY
import os

# Set your OpenAI API key from config.py
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_API_KEY,
)

def query_openai(prompt: str) -> str:
    """
    Query the OpenAI API using the o3-mini model.
    Adjust parameters like temperature or max_tokens as needed.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_tokens=4096,
            top_p=1,
            # max_completion_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI API request failed: {e}")
