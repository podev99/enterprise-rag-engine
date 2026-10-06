import os
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from document_loader import chunk_document
from vector_store import add_chunks_to_vector_store
from rag_engine import generate_rag_response

# Load environment variables
load_dotenv()

# API Key Security Configuration
API_KEY_NAME = "X-API-Key"
DEFAULT_API_KEY = "techcorp-secret-key-2026"
API_KEY_VALUE = os.getenv("API_KEY", DEFAULT_API_KEY)

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    """Validates the incoming X-API-Key header against configured key."""
    if api_key != API_KEY_VALUE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing X-API-Key header.",
        )
    return api_key


# Initialize FastAPI App
app = FastAPI(
    title="TechCorp Enterprise RAG Engine API",
    description="REST API interface for Enterprise Policy RAG Engine using Gemini API & ChromaDB.",
    version="1.0.0",
)

# CORS Middleware Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class QueryRequest(BaseModel):
    query: str
    top_k: int = 2


class SourceItem(BaseModel):
    chunk_id: int
    source: str
    distance: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceItem]


class IngestTextRequest(BaseModel):
    text: str
    filename: Optional[str] = "custom_policy.txt"


class IngestResponse(BaseModel):
    status: str
    filename: str
    chunks_created: int


# API Endpoints
@app.get("/health", tags=["System"])
async def health_check():
    """Public health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["RAG Engine"],
    dependencies=[Depends(verify_api_key)],
)
async def query_rag(request: QueryRequest):
    """Executes RAG pipeline for user query and returns answer with sources."""
    try:
        answer, sources = generate_rag_response(request.query, top_k=request.top_k)
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Data Ingestion"],
    dependencies=[Depends(verify_api_key)],
)
async def ingest_document(payload: IngestTextRequest):
    """Ingests raw text document into vector database."""
    try:
        chunks = chunk_document(payload.text)
        if not chunks:
            raise HTTPException(
                status_code=400, detail="Provided document produced 0 valid chunks."
            )

        add_chunks_to_vector_store(chunks)
        return IngestResponse(
            status="success",
            filename=payload.filename,
            chunks_created=len(chunks),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))