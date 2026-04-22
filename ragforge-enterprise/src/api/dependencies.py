"""Shared FastAPI dependencies for RAGForge Enterprise.

Currently provides:
    verify_api_key — enforces X-API-Key header authentication on protected
                     endpoints when ``settings.api_key`` is non-empty.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from src.config.settings import get_settings


async def verify_api_key(x_api_key: str = Header(default="", alias="X-API-Key")) -> None:
    """Validate the X-API-Key header against the configured secret.

    When ``settings.api_key`` is empty (the default), authentication is
    disabled so that local development works without any configuration.

    Args:
        x_api_key: Value of the ``X-API-Key`` request header (injected by
                   FastAPI via ``Header``).

    Raises:
        HTTPException: 401 Unauthorized when a key is configured and the
                       supplied value does not match.
    """
    settings = get_settings()
    if not settings.api_key:
        # If no key is configured, auth is disabled (dev mode)
        return
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
