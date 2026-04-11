import os
import logging
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

EXCLUDED_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


async def api_key_middleware(request: Request, call_next):
    if request.url.path in EXCLUDED_PATHS:
        return await call_next(request)

    expected_key = os.getenv("API_SECRET_KEY")

    if not expected_key:
        # Dev mode: no key configured, allow all requests
        logger.debug("API_SECRET_KEY not set — running in open dev mode")
        return await call_next(request)

    api_key = request.headers.get("X-API-Key")

    if api_key != expected_key:
        logger.warning(f"Unauthorized request to {request.url.path} from {request.client.host}")
        return JSONResponse(
            status_code=401,
            content={
                "error": "Unauthorized",
                "message": "Missing or invalid API key. Add X-API-Key header."
            }
        )

    return await call_next(request)
