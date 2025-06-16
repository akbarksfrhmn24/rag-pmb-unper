# app/query.py
from app.vector_store import get_vector_store
from app.llm_service import query_openai
from app.embedding import LocalEmbedding  # Assuming you're using LocalEmbedding for embeddings

def answer_query(question: str, k: int = 7) -> dict:
    # Retrieve the vector store (Chroma or your chosen vector store)
    vector_store = get_vector_store()
    
    # Instantiate the local embedding model for generating the query embedding
    embedding_model = LocalEmbedding()
    question_embedding = embedding_model.embed_query(question)
    
    # Perform similarity search using the query embedding
    results = vector_store.similarity_search_by_vector(question_embedding, k=k)
    
    # Build the prompt using retrieved context and citations
    context = ""
    citations = []
    for i, doc in enumerate(results):
        context += f"Context {i+1}: {doc.page_content}\n"
        citations.append(doc.metadata.get("source", "unknown"))
    
    prompt = (
        f"Kamu adalah Customer Service online yang bertugas membantu menjawab pertanyaan seputar Penerimaan Mahasiswa Baru (PMB) Universitas Perjuangan Tasikmalaya (Unper):\n\n"
        f"Context:\n{context}\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    
    # Query the OpenAI API using the updated function
    answer = query_openai(prompt)
    
    return {"answer": answer, "citations": citations}
