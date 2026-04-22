"""Italian API localisation middleware for RAGForge Italia.

Responsibilities
----------------
1. **Accept-Language detection** — reads the ``Accept-Language`` request header;
   activates Italian localisation when the header includes ``it`` or ``it-IT``.
2. **X-Legal-Disclaimer header** — injects the mandatory Italian legal disclaimer
   on *every* response, regardless of locale::

       X-Legal-Disclaimer: Le risposte fornite hanno carattere informativo e
           non costituiscono parere legale.

   This is required by art. 4 D.Lgs. 70/2003 (e-commerce directive) and is
   standard practice for AI legal tools in Italy.
3. **Error message translation** — when a Starlette ``HTTPException`` is raised
   and the client requested Italian, the ``detail`` field in the JSON body is
   replaced with the corresponding Italian message from the catalogue.
4. **Content-Language response header** — set to ``it`` when translation is
   applied.

Middleware ordering (in ``main.py``)
-------------------------------------
Starlette applies middleware in **reverse** registration order::

    app.add_middleware(RequestIDMiddleware)       # outermost, runs first
    app.add_middleware(ItalianLocalisationMiddleware)  # second

So ``ItalianLocalisationMiddleware`` runs *after* ``RequestIDMiddleware``,
which ensures the request ID is available in logs emitted here.
"""

from __future__ import annotations

import json
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from src.italia.i18n.errors_it import ERROR_MESSAGES_IT, get_error_message_it
from src.utils.logger import get_logger

log = get_logger(__name__)

# The mandatory Italian legal disclaimer injected on every response.
LEGAL_DISCLAIMER_IT = (
    "Le risposte fornite hanno carattere informativo "
    "e non costituiscono parere legale."
)

# HTTP status codes → best-fit error catalogue keys.
_STATUS_TO_KEY: dict[int, str] = {
    400: "INVALID_REQUEST",
    401: "AUTHENTICATION_REQUIRED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    413: "REQUEST_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMIT_EXCEEDED",
    500: "INTERNAL_ERROR",
    503: "SERVICE_UNAVAILABLE",
    504: "GATEWAY_TIMEOUT",
}


def _wants_italian(accept_language: str | None) -> bool:
    """Return True if the Accept-Language header prefers Italian.

    Handles bare ``it``, ``it-IT``, ``it;q=0.9``, and complex lists such as
    ``en-US,en;q=0.9,it;q=0.8``.

    Args:
        accept_language: Raw ``Accept-Language`` header value, or ``None``.

    Returns:
        ``True`` when ``it`` or ``it-IT`` appears anywhere in the header.
    """
    if not accept_language:
        return False
    for part in accept_language.split(","):
        lang = part.strip().split(";")[0].strip().lower()
        if lang in ("it", "it-it"):
            return True
    return False


class ItalianLocalisationMiddleware(BaseHTTPMiddleware):
    """Starlette middleware for Italian API localisation.

    Args:
        app:             The ASGI application being wrapped.
        disclaimer_text: Override the default Italian legal disclaimer text.
            Leave as ``None`` to use the default from this module.
    """

    def __init__(
        self,
        app: ASGIApp,
        disclaimer_text: str | None = None,
    ) -> None:
        super().__init__(app)
        self._disclaimer = disclaimer_text or LEGAL_DISCLAIMER_IT

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[..., object],
    ) -> Response:
        """Process a single request/response cycle.

        Args:
            request:   Incoming Starlette request.
            call_next: Callable forwarding to the next handler.

        Returns:
            Response with ``X-Legal-Disclaimer`` (always) and optionally
            Italian-translated error body + ``Content-Language: it``.
        """
        italian = _wants_italian(request.headers.get("Accept-Language"))

        # Forward the request to the application.
        response: Response = await call_next(request)  # type: ignore[arg-type]

        # ── 1. Always inject the legal disclaimer ──────────────────────────────
        response.headers["X-Legal-Disclaimer"] = self._disclaimer

        # ── 2. Translate error bodies for Italian clients ──────────────────────
        if italian and response.status_code >= 400:
            response = await self._translate_error(response)

        return response

    async def _translate_error(self, response: Response) -> Response:
        """Replace the JSON ``detail`` field with an Italian message.

        Args:
            response: The original error response from the application.

        Returns:
            A new ``Response`` with the translated body, or the original
            response if the body is not JSON or cannot be parsed.
        """
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return response

        try:
            # Read the response body — starlette StreamingResponse must be
            # consumed before we can inspect it.
            body_bytes = b""
            async for chunk in response.body_iterator:  # type: ignore[attr-defined]
                body_bytes += chunk if isinstance(chunk, bytes) else chunk.encode()

            payload: dict[str, object] = json.loads(body_bytes)
            status_code: int = response.status_code

            # Look up the catalogue key for this status code.
            catalogue_key = _STATUS_TO_KEY.get(status_code, "INTERNAL_ERROR")

            # Preserve the original detail in a machine-readable field.
            original_detail = payload.get("detail", "")

            italian_msg = get_error_message_it(
                catalogue_key, detail=str(original_detail)
            )
            payload["detail"] = italian_msg
            payload["detail_original"] = original_detail

            new_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

            log.debug(
                "Translated error response to Italian",
                extra={
                    "status_code": status_code,
                    "catalogue_key": catalogue_key,
                },
            )

            # Rebuild the response — must update Content-Length.
            headers = dict(response.headers)
            headers["content-length"] = str(len(new_body))
            headers["content-language"] = "it"
            headers["X-Legal-Disclaimer"] = self._disclaimer

            return Response(
                content=new_body,
                status_code=status_code,
                headers=headers,
                media_type="application/json",
            )

        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Failed to translate error response body",
                extra={"error": str(exc)},
            )
            return response


def get_italian_error_response(
    key: str,
    status_code: int | None = None,
    **kwargs: object,
) -> Response:
    """Build a complete Italian-localised JSON error :class:`~starlette.responses.Response`.

    Convenience helper for use inside endpoint handlers that need to return
    an Italian error without raising an exception.

    Args:
        key:         Error code key from :data:`~src.italia.i18n.errors_it.ERROR_MESSAGES_IT`.
        status_code: HTTP status code override; defaults to the catalogue value.
        **kwargs:    Placeholder values passed to :func:`~src.italia.i18n.errors_it.get_error_message_it`.

    Returns:
        A :class:`~starlette.responses.Response` with ``Content-Type: application/json``,
        ``Content-Language: it``, and ``X-Legal-Disclaimer``.
    """
    from src.italia.i18n.errors_it import get_http_status_for_key

    msg = get_error_message_it(key, **kwargs)
    code = status_code if status_code is not None else get_http_status_for_key(key)
    body = json.dumps(
        {"detail": msg, "error_code": key},
        ensure_ascii=False,
    ).encode("utf-8")
    headers = {
        "content-language": "it",
        "X-Legal-Disclaimer": LEGAL_DISCLAIMER_IT,
    }
    return Response(
        content=body,
        status_code=code,
        headers=headers,
        media_type="application/json",
    )


# Export the catalogue keys so other modules can do exhaustive checks.
ITALIAN_ERROR_KEYS: frozenset[str] = frozenset(ERROR_MESSAGES_IT.keys())
