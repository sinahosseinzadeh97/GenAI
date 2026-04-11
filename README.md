# QueryMind

AI-powered natural language interface for querying databases and documents.

## Architecture

```
User → Frontend (React)
         ↓
     FastAPI Backend
         ↓
   ┌─────┴──────┐
   │            │
SQL Gen      RAG Pipeline
(Anthropic)  (OpenAI Embeddings
              + Anthropic Generation)
   │            │
Database    Vector Store
(SQLite/PG) (PostgreSQL pgvector)
```

## Features

- Natural language → SQL query generation
- RAG pipeline for document Q&A (PDF ingestion)
- Persistent query history with CSV/JSON export
- API key authentication
- Rate limiting (20 req/min)
- Agent mode with tool use

## Requirements

- Python 3.11+
- Node.js 18+
- PostgreSQL (for production) or SQLite (for development)

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/sinahosseinzadeh97/GenAI.git
cd GenAI/QueryMind
```

### 2. Backend setup
```bash
cd querymind
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

### 3. Configure environment
```bash
cp .env.example .env
```
Edit `.env` and fill in:
```
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
QUERYMIND_API_KEY=your_secret_api_key
DATABASE_URL=sqlite:///./querymind.db
ALLOWED_ORIGINS=http://localhost:3000
HISTORY_RETENTION_DAYS=30
```

### 4. Start backend
```bash
uvicorn api.app:app --reload --port 8000
```

### 5. Frontend setup
```bash
cd ../frontend
cp .env.example .env
# Edit .env and set VITE_API_KEY to match backend QUERYMIND_API_KEY
npm install
npm run dev
```

## API Reference

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| /health | GET | ❌ | Service status |
| /query | POST | ✅ | Natural language query |
| /schema | GET | ✅ | Database schema |
| /history | GET | ✅ | Query history |
| /history/export/csv | GET | ✅ | Export history as CSV |
| /history/export/json | GET | ✅ | Export history as JSON |
| /history | DELETE | ✅ | Clear old history |
| /agent/chat | POST | ✅ | Agent mode |
| /rag/ingest | POST | ✅ | Ingest PDF document |
| /rag/search | POST | ✅ | Search documents |

## Authentication

All endpoints (except `/health`) require an API key in the request header:
```
X-API-Key: your_querymind_api_key
```

## Troubleshooting

**App fails to start:**
- Check all environment variables are set in `.env`
- Run `pip install -e .` inside the `querymind/` directory

**401 Unauthorized:**
- Make sure `X-API-Key` header matches `QUERYMIND_API_KEY` in `.env`

**429 Too Many Requests:**
- Rate limit is 20 requests per minute per IP
- Wait 60 seconds and retry

**Embeddings not working:**
- Make sure `OPENAI_API_KEY` is valid
- Check `/health` endpoint for service status

**RAG results empty:**
- Ingest a document first via `POST /rag/ingest`
- Make sure PostgreSQL with pgvector extension is running for production
```
