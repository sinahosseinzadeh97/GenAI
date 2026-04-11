import os
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse

# Limits are configurable via environment variables
QUERY_LIMIT = os.getenv("RATE_LIMIT_QUERY", "30/minute")
RAG_LIMIT = os.getenv("RATE_LIMIT_RAG", "20/minute")
AGENT_LIMIT = os.getenv("RATE_LIMIT_AGENT", "20/minute")
INGEST_LIMIT = os.getenv("RATE_LIMIT_INGEST", "5/minute")

limiter = Limiter(key_func=get_remote_address)


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "message": str(exc.detail),
            "retry_after": "60 seconds"
        }
    )
