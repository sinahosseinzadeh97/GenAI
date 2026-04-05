from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import httpx
from sse_starlette.sse import EventSourceResponse

from querymind.config import settings
from querymind.memory.conversation import ConversationMemory
from querymind.cache.query_cache import QueryCache
from querymind.database.schema_watcher import SchemaWatcher
from querymind.tools.query_tool import execute_nl_query
from querymind.schemas.models import QueryRequest, QueryError

_memory = ConversationMemory()
_cache = QueryCache(ttl_seconds=settings.cache_ttl_seconds)
_watcher = SchemaWatcher()

app = FastAPI(title="QueryMind API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    try:
        # Auto-seed if DB is empty
        from querymind.database.seed import seed_database
        import os
        if not os.path.exists(settings.db_path) or \
           os.path.getsize(settings.db_path) == 0:
            await seed_database()
        await _watcher.initialize()
    except Exception as e:
        print(f"Warning: startup error: {e}")

class QueryAPIRequest(BaseModel):
    nl_query: str
    max_rows: int = 50

@app.post("/query")
async def query_endpoint(req: QueryAPIRequest):
    """
    Stream query results as Server-Sent Events.
    Events in order:
      1. {"type": "status", "content": "Checking schema..."}
      2. {"type": "sql",    "content": "SELECT ..."}
      3. {"type": "rows",   "content": [...]}
      4. {"type": "insight","content": {...}}
      5. {"type": "done",   "content": null}
    On error:
      {"type": "error", "content": "message"}
    """
    async def event_generator():
        try:
            # Schema change check
            event = await _watcher.check_for_changes()
            if event:
                _cache.invalidate_all()
                _watcher.acknowledge_change()
                yield {"data": json.dumps({
                    "type": "status",
                    "content": "Schema change detected — cache cleared."
                })}

            yield {"data": json.dumps({
                "type": "status",
                "content": "Translating your question to SQL..."
            })}

            # We need to intercept the SQL before execution
            # Use a wrapper sampler that emits the SQL as SSE
            generated_sql: list[str] = []

            async def streaming_sampler(system: str, user: str) -> str:
                # First call = SQL generation
                # We cannot call real LLM here (no MCP session)
                # Use a direct Anthropic API call instead
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": settings.anthropic_api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json"
                        },
                        json={
                            "model": "claude-haiku-4-5-20251001",
                            "max_tokens": 1024,
                            "system": system,
                            "messages": [{"role": "user", "content": user}]
                        },
                        timeout=30.0
                    )
                    data = response.json()
                    result = data["content"][0]["text"]
                    generated_sql.append(result)
                    return result

            request = QueryRequest(
                nl_query=req.nl_query,
                max_rows=req.max_rows
            )

            result = await execute_nl_query(
                request,
                sampler_callback=streaming_sampler,
                memory=_memory,
                cache=_cache
            )

            if isinstance(result, QueryError):
                yield {"data": json.dumps({
                    "type": "error",
                    "content": result.message
                })}
                return

            # Emit SQL
            if generated_sql:
                yield {"data": json.dumps({
                    "type": "sql",
                    "content": generated_sql[0]
                })}
            elif getattr(result, "from_cache", False):
                yield {"data": json.dumps({
                    "type": "status",
                    "content": "Result served from cache."
                })}

            # Emit rows
            yield {"data": json.dumps({
                "type": "rows",
                "content": result.model_dump()
            })}

            # Emit insight if present
            if getattr(result, "insight", None):
                yield {"data": json.dumps({
                    "type": "insight",
                    "content": result.insight.model_dump()
                })}

            yield {"data": json.dumps({
                "type": "done",
                "content": None
            })}

        except Exception as e:
            yield {"data": json.dumps({
                "type": "error",
                "content": str(e)
            })}

    return EventSourceResponse(event_generator())

@app.get("/schema")
async def get_schema():
    from querymind.tools.schema_tool import get_db_schema
    ddl = await get_db_schema()
    return {"schema": ddl}

@app.get("/cache/stats")
async def cache_stats():
    return _cache.stats().model_dump()

@app.delete("/cache")
async def clear_cache():
    count = _cache.invalidate_all()
    return {"cleared": count}

@app.get("/history")
async def get_history():
    return {
        "turns": [t.model_dump() for t in _memory.get_turns()]
    }

@app.delete("/history")
async def clear_history():
    _memory.clear()
    return {"cleared": True}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "database_type": settings.database_type,
        "schema_hash": _watcher.get_status().current_hash[:8]
    }
