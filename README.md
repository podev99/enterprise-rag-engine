# 🏢 TechCorp Enterprise RAG Engine & Telegram Support Chatbot

An enterprise-grade Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **Google Gemini API**, **ChromaDB**, and **Telegram Webhook Integration**. Designed to deliver accurate, context-grounded policy answers with zero hallucinations and explicit source lineage citations.

## 📐 System Architecture

```text
                                  +-----------------------+
                                  |   Telegram User UI    |
                                  +-----------+-----------+
                                              |
                                              | (Webhook HTTP POST)
                                              v
+------------------+   HTTP GET/POST    +-----------------------+
|  REST Clients    +------------------->|     FastAPI Service   |
| (Postman / Web)  |  (X-API-Key Auth)  |  (Render Cloud / Port)|
+------------------+                    +-----------+-----------+
                                                    |
                                                    | Background Task
                                                    v
                                        +-----------------------+
                                        |     RAG Core Engine   |
                                        +---+---------------+---+
                                            |               |
               Semantic Search (top_k=2)    |               | Prompt + Context (temp=0.0)
                                            v               v
                                  +-----------------+  +------------------+
                                  | Chroma VectorDB |  | Google Gemini    |
                                  | (Embeddings)    |  | API (GenAI SDK)  |
                                  +-----------------+  +------------------+
```
## ✨ Key Features
+ **Zero Hallucination Guardrails**: Uses `temperature=0.0` with explicit System Instructions to refuse out-of-scope queries gracefully.
+ **Source Lineage Citations**: Returns precise file references, Chunk IDs, and Cosine Distance scores for total transparency.
+ **REST API Packaging**: Built with FastAPI, featuring Pydantic schemas, CORS support, `X-API-Key` authentication, and interactive Swagger UI docs.
+ **Telegram Webhook Integration**: Non-blocking async updates via `BackgroundTasks` to respond instantly without HTTP timeouts.
+ **Production Ready**: Pre-configured for zero-downtime 24/7 cloud deployment on Render.

## 🛠️ Tech Stack
+ **Language**: Python 3.10
+ **Framework**: FastAPI, Uvicorn
+ **AI / LLM**: Google GenAI SDK (`gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash`)
+ **Vector Database**: ChromaDB
+ **Document Processing**: LangChain Text Splitters
+ **Integration**: Telegram Bot API (Webhook)

## 📁 Repository Structure
```text
enterprise-rag-engine/
│
├── api.py                 # FastAPI Web Application & Telegram Webhook Router
├── rag_engine.py          # Core RAG Logic (Gemini API + Anti-Hallucination Guardrails)
├── vector_store.py        # ChromaDB Client & Vector Search Operations
├── document_loader.py      # Text Extraction & Recursive Chunking
├── telegram_bot.py        # Standalone Telegram Bot Script (Long Polling)
├── company_policy.txt     # Enterprise Knowledge Base Source File
├── Procfile               # Cloud Deployment Start Command for Render
├── requirements.txt       # Project Dependencies
└── README.md              # Project Documentation
```

## 🚀 Quickstart Guide (Local Development)
### 1. Clone & Set Up Environment
```bash
git clone [https://github.com/your-username/enterprise-rag-engine.git](https://github.com/your-username/enterprise-rag-engine.git)
cd enterprise-rag-engine

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```bash
GEMINI_API_KEY=your_google_gemini_api_key
API_KEY=techcorp-secret-key-2026
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
```
### 3. Ingest Data & Run Application
```bash
# Ingest knowledge base into ChromaDB
python document_loader.py

# Launch FastAPI application server
uvicorn api:app --reload --port 8000
```

Swagger API Documentation is available at: `http://127.0.0.1:8000/docs`

## 📡 REST API Reference

### Headers

| Key | Value | Description |
| :--- | :--- | :--- |
| `X-API-Key` | `techcorp-secret-key-2026` | Required for authenticated endpoints |
| `Content-Type` | `application/json` | Request payload format |

## Endpoints

### 1. System Health Check

+ `GET /health` (Public)

+ Response:

```bash
JSON{
  "status": "healthy",
  "version": "1.0.0"
}
```

### 2. Query RAG Engine

+ `POST /query` (Authenticated)

+ Request Body:
```bash
JSON{
  "query": "What is the Starter Plan price?",
  "top_k": 2
}
```
+ Response:
```bash
JSON{
  "query": "What is the Starter Plan price?",
  "answer": "Based on the provided context, the Starter Plan price is $200/month.",
  "sources": [
    {
      "chunk_id": 4,
      "source": "company_policy.txt",
      "distance": 0.2571
    }
  ]
}
```
## ☁️ Cloud Deployment (Render)
1. Push the code to GitHub.
2. Log in to Render Dashboard and create a Web Service.
3. Set configuration:
+ Runtime: `Python 3`
+ Build Command: `pip install -r requirements.txt`
+ Start Command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
+ Add Environment Variables on Render: `GEMINI_API_KEY, API_KEY, TELEGRAM_BOT_TOKEN.`
+ Connect Telegram Webhook by visiting in browser:
```text
[https://api.telegram.org/bot](https://api.telegram.org/bot)/setWebhook?url=https://[.onrender.com/telegram-webhook](https://.onrender.com/telegram-webhook)
```
