from openai import OpenAI
from app.config import OPENAI_API_KEY
from app.config import GITHUB_API_KEY
import os

# Set your OpenAI API key from config.py
client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=GITHUB_API_KEY,
)

def query_openai(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": "Kamu adalah asisten AI yang membantu menjawab pertanyaan seputar Penerimaan Mahasiswa Baru (PMB) Universitas Perjuangan Tasikmalaya (Unper)."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI API request failed: {e}")
