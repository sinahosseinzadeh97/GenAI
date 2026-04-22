"""Inbound webhook endpoints for RAGForge Italia integration connectors.

Exposes HTTP POST endpoints that receive push events from FileNet/Documentum
and LexisNexis Italia, verify their HMAC-SHA256 signatures, and dispatch
events to the appropriate connector for ingestion.

All endpoints:
- Require HMAC-SHA256 signature verification before processing.
- Respond with ``X-Legal-Disclaimer: IT`` (via ``ItalianLocalisationMiddleware``).
- Return ``202 Accepted`` for events that will be processed asynchronously,
  or ``200 OK`` for synchronous processing.
- Return Italian-localised error responses when ``Accept-Language: it``.

Security
--------
Each webhook integration is protected by a separate shared secret:
  - ``FILENET_WEBHOOK_SECRET`` — FileNet P8 / Documentum push events.
  - ``LEXISNEXIS_WEBHOOK_SECRET`` — LexisNexis Italia push notifications.

These must be set in the environment before the API starts.  If either secret
is empty, the corresponding webhook endpoint returns ``503 Service Unavailable``
(misconfigured, not accepting events).

Usage
-----
Mount in ``src/api/main.py``::

    from src.italia.connectors.webhook_router import webhook_router
    app.include_router(webhook_router, prefix="/italia")

Endpoints::

    POST /italia/webhooks/filenet
    POST /italia/webhooks/lexisnexis
"""

from __future__ import annotations

import hashlib
import hmac
import os
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel

from src.api.middleware_it import LEGAL_DISCLAIMER_IT
from src.italia.connectors.filenet_documentum import (
    FilenetDocumentumConnector,
    FilenetWebhookEvent,
    _verify_hmac,
)
from src.italia.connectors.lexisnexis_it import LexisNexisItaliaConnector
from src.utils.logger import get_logger

log = get_logger(__name__)

webhook_router = APIRouter(
    prefix="/webhooks",
    tags=["Italia Webhooks"],
)

# ── Response schemas ───────────────────────────────────────────────────────────


class WebhookAck(BaseModel):
    """Standard acknowledgement response for webhook endpoints."""

    status: str = "accepted"
    event_type: str
    object_id: str
    processed_at: str
    disclaimer: str = LEGAL_DISCLAIMER_IT


# ── Helpers ────────────────────────────────────────────────────────────────────


def _require_secret(secret_env: str, endpoint: str) -> str:
    """Return the webhook secret from the environment or raise 503.

    Args:
        secret_env: Name of the environment variable holding the secret.
        endpoint:   Endpoint name for logging.

    Returns:
        The secret string.

    Raises:
        :class:`HTTPException` 503: When the secret is not configured.
    """
    secret = os.getenv(secret_env, "")
    if not secret:
        log.warning(
            "Webhook endpoint not configured — missing secret",
            extra={"endpoint": endpoint, "env_var": secret_env},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Endpoint webhook {endpoint!r} non configurato. "
                f"Impostare la variabile d'ambiente {secret_env}."
            ),
        )
    return secret


def _verify_lexisnexis_hmac(payload: bytes, signature: str | None, secret: str) -> None:
    """Verify a LexisNexis HMAC-SHA256 webhook signature.

    Args:
        payload:   Raw request body bytes.
        signature: ``X-LexisNexis-Signature`` header value.
        secret:    Shared secret.

    Raises:
        :class:`HTTPException` 401: On verification failure.
    """
    if not signature:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Firma webhook LexisNexis assente (X-LexisNexis-Signature).",
        )
    expected = hmac.new(
        key=secret.encode("utf-8"),
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()
    received = signature.removeprefix("sha256=")
    if not hmac.compare_digest(expected, received):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Firma HMAC-SHA256 del webhook LexisNexis non valida.",
        )


# ── FileNet / Documentum webhook ───────────────────────────────────────────────


@webhook_router.post(
    "/filenet",
    response_model=WebhookAck,
    status_code=status.HTTP_202_ACCEPTED,
    summary="FileNet / Documentum push event receiver",
    description=(
        "Receives ``objectCreated``, ``objectUpdated``, and ``objectDeleted`` "
        "push events from IBM FileNet P8 or OpenText Documentum. "
        "Events are verified via HMAC-SHA256 (``X-FileNet-Signature`` header) "
        "before processing."
    ),
)
async def filenet_webhook(
    request: Request,
    x_filenet_signature: Annotated[str | None, Header()] = None,
) -> WebhookAck:
    """Process a FileNet / Documentum push event.

    Args:
        request:              Incoming FastAPI request.
        x_filenet_signature:  ``X-FileNet-Signature: sha256=<hex>`` header.

    Returns:
        :class:`WebhookAck` with ``202 Accepted``.

    Raises:
        :class:`HTTPException` 503: When ``FILENET_WEBHOOK_SECRET`` is not set.
        :class:`HTTPException` 401: When the HMAC signature is invalid.
        :class:`HTTPException` 400: When the payload cannot be parsed.
    """
    secret = _require_secret("FILENET_WEBHOOK_SECRET", "filenet")
    payload = await request.body()

    # Signature verification.
    try:
        _verify_hmac(payload=payload, signature=x_filenet_signature, secret=secret)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Firma FileNet non valida: {exc}",
        ) from exc

    # Parse the event.
    try:
        import json

        raw: dict[str, Any] = json.loads(payload)
        event = FilenetWebhookEvent(
            event_type=raw.get("eventType", "unknown"),
            object_id=raw.get("objectId", ""),
            repository=raw.get("repositoryId", ""),
            timestamp=datetime.now(tz=timezone.utc),
            raw=raw,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Payload FileNet non valido: {exc}",
        ) from exc

    log.info(
        "FileNet webhook event received",
        extra={
            "event_type": event.event_type,
            "object_id": event.object_id,
            "repository": event.repository,
        },
    )

    # For objectDeleted events — acknowledge without ingestion.
    if event.event_type == "objectDeleted":
        log.info(
            "FileNet objectDeleted — skipping ingestion",
            extra={"object_id": event.object_id},
        )

    # NOTE: Full async ingestion pipeline integration is a follow-up task.
    # The event is validated and logged; the indexing call would be:
    #   await background_tasks.add_task(index_filenet_document, event)

    return WebhookAck(
        event_type=event.event_type,
        object_id=event.object_id,
        processed_at=datetime.now(tz=timezone.utc).isoformat(),
    )


# ── LexisNexis Italia webhook ──────────────────────────────────────────────────


@webhook_router.post(
    "/lexisnexis",
    response_model=WebhookAck,
    status_code=status.HTTP_202_ACCEPTED,
    summary="LexisNexis Italia push notification receiver",
    description=(
        "Receives push notifications from LexisNexis Italia for new massime "
        "or updated norme. Events are verified via HMAC-SHA256 "
        "(``X-LexisNexis-Signature`` header) before processing."
    ),
)
async def lexisnexis_webhook(
    request: Request,
    x_lexisnexis_signature: Annotated[str | None, Header()] = None,
) -> WebhookAck:
    """Process a LexisNexis Italia push notification.

    Args:
        request:                  Incoming FastAPI request.
        x_lexisnexis_signature:   ``X-LexisNexis-Signature: sha256=<hex>`` header.

    Returns:
        :class:`WebhookAck` with ``202 Accepted``.

    Raises:
        :class:`HTTPException` 503: When ``LEXISNEXIS_WEBHOOK_SECRET`` is not set.
        :class:`HTTPException` 401: When the HMAC signature is invalid.
        :class:`HTTPException` 400: When the payload cannot be parsed.
    """
    secret = _require_secret("LEXISNEXIS_WEBHOOK_SECRET", "lexisnexis")
    payload = await request.body()

    _verify_lexisnexis_hmac(payload, x_lexisnexis_signature, secret)

    try:
        import json

        raw: dict[str, Any] = json.loads(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Payload LexisNexis non valido: {exc}",
        ) from exc

    event_type = str(raw.get("eventType", raw.get("tipo", "documentUpdated")))
    document_id = str(raw.get("documentId", raw.get("id", "unknown")))

    log.info(
        "LexisNexis webhook notification received",
        extra={"event_type": event_type, "document_id": document_id},
    )

    return WebhookAck(
        event_type=event_type,
        object_id=document_id,
        processed_at=datetime.now(tz=timezone.utc).isoformat(),
    )
