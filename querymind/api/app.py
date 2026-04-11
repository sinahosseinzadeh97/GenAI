from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import httpx
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

import os
import time

from querymind.config import settings
from querymind.memory.conversation import ConversationMemory
from querymind.cache.redis_cache import RedisQueryCache
from querymind.database.schema_watcher import SchemaWatcher
from querymind.tools.query_tool import execute_nl_query
from querymind.schemas.models import QueryRequest, QueryError
from querymind.rag.api.rag_routes import router as rag_router
from querymind.api.middleware.auth import get_api_key
from fastapi import Depends
from querymind.api.middleware.rate_limit import rate_limit_dependency
from querymind.api.health import router as health_router

from querymind.logging_config import setup_logging, StructuredLogger
setup_logging()
logger = StructuredLogger("querymind.api")

_memory = ConversationMemory()
_redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
_cache = RedisQueryCache(redis_url=_redis_url)
_watcher = SchemaWatcher()

app = FastAPI(title="QueryMind API", version="0.1.0")

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

app.add_middleware(SecurityHeadersMiddleware)

allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:3001"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type"],
)

app.include_router(rag_router)
from querymind.agent.api.agent_routes import router as agent_router
app.include_router(agent_router)
app.include_router(health_router)

@app.on_event("startup")
async def startup():
    try:
        # Create history table
        from querymind.database.history import Base, engine
        Base.metadata.create_all(bind=engine)
        
        # Load last 10 turns back into active memory
        from sqlalchemy.orm import Session
        with Session(engine) as db_session:
            await _memory.load_from_db(session_id="default", db=db_session, last_n=10)

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
async def query_endpoint(request: Request, req: QueryAPIRequest, api_key: str = Depends(get_api_key), _: None = Depends(rate_limit_dependency)):
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

            start_time = time.time()
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

            latency_ms = (time.time() - start_time) * 1000
            sql_executed = generated_sql[0] if generated_sql else "[cached]"
            rows_num = getattr(result, "row_count", 0)
            is_cached = getattr(result, "from_cache", False)
            logger.log_query(
                query=req.nl_query, 
                sql=sql_executed, 
                rows=rows_num, 
                latency_ms=latency_ms, 
                cached=is_cached
            )

        except Exception as e:
            logger.log_error("query_endpoint_error", str(e))
            yield {"data": json.dumps({
                "type": "error",
                "content": str(e)
            })}

    return EventSourceResponse(event_generator())

@app.get("/schema")
async def get_schema(api_key: str = Depends(get_api_key)):
    from querymind.tools.schema_tool import get_db_schema
    ddl = await get_db_schema()
    return {"schema": ddl}

@app.get("/cache/stats")
async def cache_stats(api_key: str = Depends(get_api_key)):
    return _cache.stats().model_dump()

@app.delete("/cache")
async def clear_cache(api_key: str = Depends(get_api_key)):
    count = _cache.invalidate_all()
    return {"cleared": count}

from fastapi import Query
from querymind.database.history_repo import get_history as db_get_history

@app.get("/history")
async def get_query_history(
    session_id: str | None = Query(None),
    limit: int = Query(20, le=100),
    offset: int = Query(0),
    status: str | None = Query(None),
    api_key: str = Depends(get_api_key)
):
    history = await db_get_history(
        session_id=session_id,
        limit=limit,
        offset=offset,
        status=status
    )
    return {
        "total": len(history),
        "offset": offset,
        "limit": limit,
        "items": history
    }

from datetime import datetime, timedelta, timezone

@app.delete("/history")
async def clear_history(
    session_id: str | None = Query(None),
    older_than_days: int = Query(30),
    api_key: str = Depends(get_api_key)
):
    from querymind.database.history import engine, QueryHistory
    from sqlalchemy.orm import Session
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    
    with Session(engine) as db:
        query = db.query(QueryHistory).filter(QueryHistory.created_at < cutoff)
        if session_id:
            query = query.filter(QueryHistory.session_id == session_id)
        
        deleted_count = query.delete()
        db.commit()
    
    return {"deleted": deleted_count, "cutoff_date": cutoff.isoformat()}

import csv
import io
from fastapi.responses import StreamingResponse

@app.get("/history/export/csv")
async def export_history_csv(
    session_id: str | None = Query(None),
    api_key: str = Depends(get_api_key)
):
    history = await db_get_history(session_id=session_id, limit=1000)
    
    output = io.StringIO()
    if history:
        writer = csv.DictWriter(output, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=query_history.csv"}
    )

@app.get("/history/export/json")
async def export_history_json(
    session_id: str | None = Query(None),
    api_key: str = Depends(get_api_key)
):
    history = await db_get_history(session_id=session_id, limit=1000)
    
    output = json.dumps(history, indent=2, default=str)
    return StreamingResponse(
        iter([output]),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=query_history.json"}
    )
