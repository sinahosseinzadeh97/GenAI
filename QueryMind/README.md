

# 🧠 QueryMind
### Natural Language → SQL via MCP

> Ask questions about your database in plain English.
> QueryMind translates them to SQL, executes them,
> and explains the results — all in real time.

![QueryMind UI](docs/screenshot_v2.png)

---

## What is this

QueryMind is a production-grade system with three interfaces:

- **MCP Server** — connects Claude Desktop directly to your database
- **REST API** — FastAPI backend with SSE streaming
- **Web UI** — React + TypeScript + Tailwind frontend

You ask: *"Which products are low on stock?"*
QueryMind returns the SQL, the rows, and an insight — instantly.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Two Modes                        │
├────────────────────┬────────────────────────────────┤
│   Claude Desktop   │        Browser                 │
│   (MCP protocol)   │   http://localhost:3001        │
└────────┬───────────┴──────────────┬─────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐      ┌──────────────────┐
│   MCP Server    │      │  FastAPI + SSE   │
│  (stdio/stdio)  │      │  :8000           │
└────────┬────────┘      └────────┬─────────┘
         │                        │
         └──────────┬─────────────┘
                    ▼
         ┌──────────────────┐
         │  execute_nl_query│  ← shared core
         │  + Memory        │
         │  + Cache         │
         │  + SchemaWatcher │
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ SQLite/PostgreSQL│
         └──────────────────┘
```

---

## Features

| Feature | Description |
|---|---|
| 🔤 Natural Language | Ask in plain English |
| ⚡ SSE Streaming | Results stream in real time |
| 🧠 Conversation Memory | Remembers previous questions |
| 💾 Query Cache | Same question → instant response |
| 💡 AI Insights | Explains what the data means |
| 🔍 Schema Watcher | Detects DB changes automatically |
| 🐘 PostgreSQL | Drop-in alternative to SQLite |
| 🐳 Docker | One command to run everything |
| 🔌 MCP | Works natively with Claude Desktop |

---

## Tech Stack

**Backend**
- Python 3.12+
- FastAPI + uvicorn
- Pydantic v2 (strict typing everywhere)
- MCP (Model Context Protocol)
- aiosqlite / asyncpg
- httpx (Anthropic API calls)

**Frontend**
- React 18 + TypeScript
- Vite + Tailwind CSS
- SSE streaming (no WebSocket needed)

**Infrastructure**
- Docker + docker-compose
- PostgreSQL 16 (optional)
- pytest + pytest-asyncio (37 tests)

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- Anthropic API key

### 1. Clone the repo
```bash
git clone https://github.com/sinahosseinzadeh97/GenAI
cd GenAI/QueryMind
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run everything
```bash
docker compose up
```

Open **http://localhost:3001**

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

> The `-i` flag is required for stdio MCP protocol.

---

## Import Your Own Data

```bash
# Import any CSV file
docker compose exec api python -m querymind import-csv \
  /app/data/yourfile.csv table_name
```

---

## Example Queries

```
How many users do we have?
Which products are low on stock?
What's the total revenue from delivered orders?
Show me all pending orders with customer names
Who spent the most money?
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/query` | Stream NL query as SSE |
| GET | `/schema` | Get database schema |
| GET | `/history` | Get conversation history |
| DELETE | `/history` | Clear conversation |
| GET | `/cache/stats` | Cache statistics |
| DELETE | `/cache` | Invalidate cache |
| GET | `/health` | Health check |

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

## Project Structure

```
QueryMind/
├── querymind/
│   ├── api/          # FastAPI app + SSE endpoints
│   ├── cache/        # Query result cache (TTL-based)
│   ├── database/     # SQLite + PostgreSQL engines + router
│   ├── memory/       # Conversation memory (multi-turn)
│   ├── prompts/      # SQL generation + insight prompts
│   ├── schemas/      # All Pydantic models
│   ├── tools/        # MCP tools (query, schema, import)
│   └── server.py     # MCP server entry point
├── frontend/         # React + TypeScript + Tailwind
├── data/             # SQLite database (volume mounted)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## PostgreSQL Setup (Optional)

```bash
# Start postgres
docker compose up postgres -d

# Update .env
DATABASE_TYPE=postgresql
POSTGRES_DSN=postgresql://querymind:querymind_pass@localhost:5432/querymind

# Run
docker compose up
```

---

## License

MIT

---

*Built with Python 3.12 · Pydantic v2 · MCP · FastAPI · React*

---
