# app/main.py
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import time
import shutil
import uuid
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.ingestion import ingest_document
from app.query import answer_query
from pydantic import BaseModel

app = FastAPI(title="RAG System API", description="Answer questions on technical documentation.")

# In-memory metrics (could be replaced with a proper monitoring system)
metrics = {"token_usage": 0, "response_times": [], "success_count": 0, "failure_count": 0}

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", summary="Root endpoint")
async def read_root():
    return {"message": "Welcome to the RAG System API!"}

@app.post("/upload", summary="Upload and ingest a document")
async def upload_document(file: UploadFile = File(...)):
    # Save file locally
    file_id = uuid.uuid4().hex
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        ingest_document(file_path)
    except Exception as e:
        metrics["failure_count"] += 1
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")
    metrics["success_count"] += 1
    return {"message": "Document ingested successfully", "file": file.filename}

class QueryRequest(BaseModel):
    question: str

@app.post("/query", summary="Query the ingested documents")
async def query(request: QueryRequest):
    """
    Accepts a JSON body with a "question" field.
    Example request body:
    {
        "question": "How do I install the software?"
    }
    """
    start_time = time.time()
    try:
        # Use the question from the request body.
        result = answer_query(request.question)
        elapsed = time.time() - start_time
        metrics["response_times"].append(elapsed)
        metrics["success_count"] += 1
        return JSONResponse(content={"result": result, "metrics": metrics})
    except Exception as e:
        metrics["failure_count"] += 1
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/metrics", summary="Retrieve service metrics")
async def get_metrics():
    return JSONResponse(content=metrics)