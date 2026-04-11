![Python](https://img.shields.io/badge/python-3.12+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-37%20passing-brightgreen)

# 🧠 QueryMind
### Natural Language → SQL · RAG · Agent

> A production-oriented AI platform that lets you query databases,
> search PDF contracts, and chat with an intelligent agent —
> all in real time.

![Agent Mode](docs/screenshot_agent.png)

---

## What is this?

QueryMind is a production-oriented system with three modes and three interfaces:

| Mode | What it does |
|------|-------------|
| 🗄️ SQL Mode | Translates natural language to SQL and executes it |
| 📄 RAG Mode | Ingests PDF contracts and answers questions with page citations |
| 🤖 Agent Mode | Autonomous agent that picks the right tool and answers with memory |

**Three interfaces:**
- **MCP Server** — connects Claude Desktop directly to your database
- **REST API** — FastAPI backend with SSE streaming
- **Web UI** — React + TypeScript + Tailwind frontend

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Three Modes                         │
├──────────────┬──────────────────┬───────────────────────────┤
│  SQL Mode    │    RAG Mode      │       Agent Mode          │
│  NL → SQL    │  PDF → Vectors   │  LLM + Tools + Memory     │
└──────┬───────┴────────┬─────────┴────────────┬──────────────┘
       │                │                       │
       ▼                ▼                       ▼
┌─────────────┐  ┌─────────────┐      ┌────────────────────┐
│  FastAPI    │  │  RAG API    │      │   Agent API        │
│  /query     │  │  /rag/*     │      │   /agent/chat      │
│  /schema    │  │  ingest     │      │   tool_use         │
│  /history   │  │  search     │      │   memory           │
└──────┬──────┘  │  extract    │      └────────┬───────────┘
       │         │  compare    │               │
       │         └──────┬──────┘               │
       │                │                      │
       └────────────────┼──────────────────────┘
                        ▼
          ┌─────────────────────────┐
          │   PostgreSQL + pgvector │
          │   SQLite (default)      │
          └─────────────────────────┘
                        │
                        ▼
          ┌─────────────────────────┐
          │   OpenAI API            │
          │   text-embedding-3-small│
          │   gpt-4o-mini           │
          └─────────────────────────┘
```

---

## Features

| Feature | Description |
|---|---|
| 🔤 Natural Language → SQL | Ask in plain English |
| ⚡ SSE Streaming | Results stream in real time |
| 🧠 Conversation Memory | Remembers previous questions |
| 💾 Query Cache | Same question → instant response |
| 💡 AI Insights | Explains what the data means |
| 🔍 Schema Watcher | Detects DB changes automatically |
| 📄 PDF Ingestion | Upload contracts, embed by page |
| 🔎 Semantic Search | pgvector similarity search with citations |
| 🤖 AI Agent | Autonomous tool selection + multi-turn memory |
| 🐘 PostgreSQL + pgvector | Vector storage for RAG |
| 🐳 Docker | One command to run everything |
| 🔌 MCP | Works natively with Claude Desktop |

---

## Tech Stack

**Backend**
- Python 3.12+
- FastAPI + uvicorn
- Pydantic v2
- MCP (Model Context Protocol)
- aiosqlite / asyncpg
- PyMuPDF (PDF parsing)
- pgvector (vector similarity)
- OpenAI SDK (embeddings + agent)

**Frontend**
- React 18 + TypeScript
- Vite + Tailwind CSS
- SSE streaming

**Infrastructure**
- Docker + docker-compose
- PostgreSQL 16 + pgvector
- pytest (37 tests)

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- Anthropic API key
- OpenAI API key (for RAG + Agent)

### 1. Clone the repo
```bash
git clone https://github.com/sinahosseinzadeh97/GenAI.git
cd GenAI/QueryMind
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your keys:
# ANTHROPIC_API_KEY=...
# OPENAI_API_KEY=...
```

### 3. Run everything
```bash
docker compose up
```

Open **http://localhost:3001**

---

## Current Limitations
- Single-tenant: no user accounts or login system yet
- OpenAI API key required for RAG Mode and Agent Mode
- Designed for databases up to ~50,000 rows
- Agent memory resets on container restart (Redis persistence coming in v0.2)
- Tested on: macOS 14, Ubuntu 22.04

## Tech Decisions
- **pypdf over PyMuPDF**: pypdf is MIT-licensed, safe for commercial use
- **SQLite default**: zero-config for local development; swap to postgres for production
- **SSE over WebSockets**: simpler infrastructure, works behind standard reverse proxies
- **Claude Haiku for SQL**: fast and cheap for structured generation tasks

---

## Three Modes

### 🗄️ SQL Mode
Ask natural language questions about your database.
```
How many users do we have?
Which products are low on stock?
What's the total revenue from delivered orders?
```

### 📄 RAG Mode
![RAG Mode](docs/screenshot_rag.png)

Upload PDF contracts and search them semantically.
- Upload via drag-and-drop UI or `POST /rag/ingest`
- Search returns results with **filename + page number** citations
- Supports bilingual documents (English + Italian)

### 🤖 Agent Mode
![Agent Mode](docs/screenshot_agent.png)

Chat with an autonomous agent that:
1. Understands your question
2. Picks the right tool automatically (`search_contracts`, `extract_field`, `compare_contracts`)
3. Calls the RAG pipeline internally
4. Returns a grounded answer with source citations
5. Remembers the conversation (multi-turn memory)

---

## API Endpoints

### Core
| Method | Endpoint | Description |
|---|---|---|
| POST | `/query` | Stream NL query as SSE |
| GET | `/schema` | Get database schema |
| GET | `/history` | Get conversation history |
| DELETE | `/history` | Clear conversation |
| GET | `/cache/stats` | Cache statistics |
| DELETE | `/cache` | Invalidate cache |
| GET | `/health` | Health check |

### RAG
| Method | Endpoint | Description |
|---|---|---|
| POST | `/rag/ingest` | Upload and index a PDF |
| POST | `/rag/search` | Semantic search across contracts |
| POST | `/rag/extract` | Extract specific fields |
| POST | `/rag/compare` | Compare clauses across contracts |

### Agent
| Method | Endpoint | Description |
|---|---|---|
| POST | `/agent/chat` | Chat with the contract agent |

**Agent request:**
```json
{
  "message": "When does the contract expire?",
  "session_id": "user_123"
}
```
**Agent response:**
```json
{
  "answer": "The contract expires on December 31, 2025.",
  "tools_used": ["search_contracts"],
  "sources": [
    {"filename": "contract.pdf", "page_number": 1}
  ]
}
```

---

## Project Structure

```
QueryMind/
├── querymind/
│   ├── api/              # FastAPI app + SSE endpoints
│   ├── agent/            # AI Agent with tool use + memory
│   │   ├── tools.py      # OpenAI function-calling schemas
│   │   ├── memory.py     # Session-based conversation memory
│   │   ├── orchestrator.py # Agent loop + tool dispatch
│   │   └── api/          # POST /agent/chat
│   ├── rag/              # RAG pipeline
│   │   ├── ingestion/    # PDF parser + embedder
│   │   ├── store/        # pgvector storage
│   │   ├── retrieval/    # similarity search
│   │   ├── generation/   # LLM answer generation
│   │   └── api/          # RAG endpoints
│   ├── cache/            # Query result cache
│   ├── database/         # SQLite + PostgreSQL engines
│   ├── memory/           # Conversation memory
│   ├── prompts/          # SQL generation prompts
│   ├── schemas/          # Pydantic models
│   ├── tools/            # MCP tools
│   └── server.py         # MCP server entry point
├── frontend/             # React + TypeScript + Tailwind
├── docs/                 # Screenshots
├── data/                 # SQLite database
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Connect to Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "querymind": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", "/absolute/path/to/QueryMind/data:/app/data",
        "-e", "DB_PATH=/app/data/querymind.db",
        "querymind"
      ]
    }
  }
}
```

---

## Run Tests

```bash
cd QueryMind
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
```

**37 tests — 0 failures**

---

## Environment Variables

```env
# Core
DB_PATH=querymind.db
DATABASE_TYPE=sqlite
ANTHROPIC_API_KEY=your_anthropic_api_key

# PostgreSQL (optional)
POSTGRES_DSN=postgresql://querymind:querymind_pass@postgres:5432/querymind

# RAG + Agent (OpenAI)
OPENAI_API_KEY=your_openai_api_key
BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v1
BEDROCK_LLM_MODEL=anthropic.claude-haiku-4-5-20251001
```

---

*Built with Python 3.12 · Pydantic v2 · MCP · FastAPI · pgvector · React*
