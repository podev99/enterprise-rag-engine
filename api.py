import os
import requests
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Depends, status, Request, BackgroundTasks
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
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    """Validates the incoming X-API-Key header against configured key."""
    if api_key != API_KEY_VALUE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing X-API-Key header.",
        )
    return api_key

def send_telegram_message(chat_id: int, text: str):
    """Sends a formatted message back to the Telegram chat using Telegram Bot API."""
    if not TELEGRAM_BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Error sending Telegram message: {e}")


def process_telegram_update(chat_id: int, user_query: str):
    """Processes RAG response asynchronously and sends result to Telegram."""
    try:
        answer, sources = generate_rag_response(user_query, top_k=2)
        formatted_reply = f"🤖 *ANSWER:*\n{answer}\n\n"

        if sources and "do not have enough information" not in answer.lower():
            formatted_reply += "📌 *SOURCES USED:*\n"
            for src in sources:
                formatted_reply += (
                    f"• File: `{src['source']}` | Chunk ID: `{src['chunk_id']}` "
                    f"(Distance: {src['distance']:.4f})\n"
                )
        else:
            formatted_reply += "📌 *SOURCES:* None (Out-of-Scope / Refused)"

        send_telegram_message(chat_id, formatted_reply)
    except Exception as e:
        send_telegram_message(chat_id, f"❌ Error processing query: {str(e)}")


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


@app.post("/telegram-webhook", tags=["Telegram Integration"])
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receives incoming Telegram updates via Webhook and delegates processing."""
    data = await request.json()

    if "message" in data and "text" in data["message"]:
        chat_id = data["message"]["chat"]["id"]
        user_query = data["message"]["text"]

        if user_query.strip() == "/start":
            send_telegram_message(
                chat_id,
                "👋 Welcome to TechCorp AI Support Bot!\n\nAsk me any question about company policies, products, or services.",
            )
        else:
            send_telegram_message(chat_id, "🔎 Querying Enterprise Knowledge Base...")
            background_tasks.add_task(process_telegram_update, chat_id, user_query)

    return {"status": "ok"}