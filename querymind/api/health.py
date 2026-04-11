import os
import time
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()


async def check_redis() -> dict:
    try:
        import redis.asyncio as aioredis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        r = aioredis.from_url(redis_url, socket_connect_timeout=2)
        await r.ping()
        await r.aclose()
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def check_postgres() -> dict:
    try:
        import asyncpg
        dsn = os.getenv("POSTGRES_DSN")
        if not dsn:
            return {"status": "not_configured"}
        conn = await asyncpg.connect(dsn, timeout=3)
        await conn.close()
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def check_sqlite() -> dict:
    try:
        import aiosqlite
        db_path = os.getenv("DB_PATH", "data/querymind.db")
        async with aiosqlite.connect(db_path) as db:
            await db.execute("SELECT 1")
        return {"status": "healthy", "path": db_path}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def check_api_keys() -> dict:
    return {
        "anthropic": "configured" if os.getenv("ANTHROPIC_API_KEY") else "missing",
        "openai": "configured" if os.getenv("OPENAI_API_KEY") else "missing",
    }


@router.get("/health")
async def health_check():
    start = time.monotonic()

    redis_status = await check_redis()
    postgres_status = await check_postgres()
    sqlite_status = await check_sqlite()
    api_keys = check_api_keys()

    all_healthy = (
        redis_status["status"] == "healthy"
        and sqlite_status["status"] == "healthy"
        and api_keys["anthropic"] == "configured"
    )

    latency_ms = round((time.monotonic() - start) * 1000, 2)

    response = {
        "status": "healthy" if all_healthy else "degraded",
        "latency_ms": latency_ms,
        "dependencies": {
            "redis": redis_status,
            "postgres": postgres_status,
            "sqlite": sqlite_status,
            "api_keys": api_keys,
        },
        "version": "0.2.0"
    }

    status_code = 200 if all_healthy else 207
    return JSONResponse(content=response, status_code=status_code)
