"""Request ID middleware for RAGForge Enterprise.

Every inbound HTTP request is assigned a unique trace identifier so that all
log lines emitted during the lifetime of that request can be correlated in
log-aggregation systems (Loki, ELK, CloudWatch Logs, etc.).

Behaviour
---------
* Reads the ``X-Request-ID`` request header if the upstream proxy/client
  already set one; otherwise, generates a new UUID v4.
* Stores the value on ``request.state.request_id`` so any code downstream
  (handlers, dependencies, background tasks) can access it without parsing
  headers again.
* Echoes the identifier back in an ``X-Request-ID`` response header, enabling
  client-side log correlation.
* After the response is returned, emits a single structured INFO log line with
  ``method``, ``path``, ``status_code``, ``duration_ms``, and ``request_id``
  as ``extra={}`` fields (consumed by the JSON formatter in
  :mod:`src.utils.logger`).
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from src.utils.logger import get_logger

log = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that attaches a request-scoped trace identifier.

    Args:
        app: The ASGI application to wrap.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: object) -> Response:  # type: ignore[override]
        """Process a single HTTP request/response cycle.

        Args:
            request:   The incoming Starlette request.
            call_next: Callable that forwards the request to the next handler.

        Returns:
            The response, augmented with the ``X-Request-ID`` header.
        """
        # Honour an upstream-set request ID (e.g. from an API gateway or load
        # balancer); generate a fresh UUID v4 when the header is absent.
        request_id: str = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start_ns: int = time.perf_counter_ns()

        # Forward the request and capture the response.
        response: Response = await call_next(request)  # type: ignore[arg-type]

        duration_ms: float = round((time.perf_counter_ns() - start_ns) / 1_000_000, 2)

        # Echo the identifier to the caller.
        response.headers["X-Request-ID"] = request_id

        log.info(
            "request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "request_id": request_id,
            },
        )

        return response
