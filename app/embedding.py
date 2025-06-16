# app/embedding.py
from sentence_transformers import SentenceTransformer

class LocalEmbedding:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    # a function to embed text
    def embed(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()

    # a function to embed a query
    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)

    # a function to embed a list of documents
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()
