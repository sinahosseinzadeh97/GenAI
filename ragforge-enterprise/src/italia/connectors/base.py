"""Abstract base class for all RAGForge Italia source connectors.

Every connector in ``src/italia/connectors/`` must subclass
:class:`BaseConnector` and implement :meth:`fetch`.  The base class
provides:

- A shared :mod:`httpx` session with connection pooling.
- A token-bucket rate limiter (``rate_limit_rps`` requests per second).
- Exponential back-off retry logic via :mod:`tenacity`.
- A generic ``_paginate`` helper for cursor/offset based APIs.
- Structured logging via the RAGForge logger.
- A ``_make_document`` factory that stamps :class:`~src.ingestion.loader.Document`
  objects with :class:`~src.italia.metadata.ItalianLegalMetadata` in
  ``metadata.extra``.

Design constraints
------------------
- ``fetch()`` must return ``list[Document]``.  Callers (CLI + pipeline)
  always receive typed :class:`~src.ingestion.loader.Document` objects,
  exactly as the existing ``DocumentLoader`` does.
- Italian metadata rides in ``Document.metadata.extra`` (a ``dict[str,Any]``)
  via :meth:`~src.italia.metadata.ItalianLegalMetadata.to_extra_dict`.
  This keeps the existing ingestion → chunking → indexing path unchanged.
- HTTP errors that are *permanent* (4xx, except 429) are raised immediately.
  Transient errors (5xx, timeouts, 429) are retried with back-off.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.ingestion.loader import Document, DocumentMetadata
from src.italia.metadata import ItalianLegalMetadata, TipoDocumento
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────


class ConnectorError(Exception):
    """Base exception for all connector failures."""


class ConnectorHTTPError(ConnectorError):
    """Raised when an HTTP request fails with a non-retryable status code."""

    def __init__(self, status_code: int, url: str, message: str = "") -> None:
        self.status_code = status_code
        self.url = url
        super().__init__(f"HTTP {status_code} from {url}: {message}")


class ConnectorRateLimitError(ConnectorError):
    """Raised when rate-limit retries are exhausted."""


class ConnectorParseError(ConnectorError):
    """Raised when response body cannot be parsed into Documents."""


# ── Token-bucket rate limiter ─────────────────────────────────────────────────


class _TokenBucket:
    """Thread-safe token-bucket rate limiter.

    Args:
        rate: Maximum requests per second.
    """

    def __init__(self, rate: float) -> None:
        self._rate = max(rate, 0.01)  # Guard against zero/negative.
        self._tokens: float = self._rate
        self._last: float = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until one token is available, then consume it."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            # Sleep for the time needed to accumulate one token.
            sleep_for = (1.0 - self._tokens) / self._rate
        time.sleep(sleep_for)
        with self._lock:
            self._tokens = 0.0


# ── Retry predicate helpers ───────────────────────────────────────────────────


def _is_transient_http(exc: BaseException) -> bool:
    """Return True for errors that warrant a retry."""
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (
        429,
        500,
        502,
        503,
        504,
    ):
        return True
    return False


# ── Abstract base connector ───────────────────────────────────────────────────

_RETRY_KWARGS = dict(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    reraise=True,
)


class BaseConnector(ABC):
    """Abstract base for all Italian legal source connectors.

    Args:
        rate_limit_rps: Maximum requests per second (default: 1.0).
        timeout:        HTTP request timeout in seconds (default: 30).
        headers:        Extra HTTP headers (e.g. ``Authorization``).
    """

    #: Override in subclasses to identify the source in logs and metadata.
    source_name: str = "unknown"

    def __init__(
        self,
        rate_limit_rps: float = 1.0,
        timeout: int = 30,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._bucket = _TokenBucket(rate_limit_rps)
        self._timeout = timeout
        default_headers = {
            "User-Agent": (
                "RAGForge-Italia/1.0 (legal research; "
                "contact: ragforge@example.com)"
            ),
            "Accept-Language": "it-IT,it;q=0.9,en;q=0.5",
        }
        if headers:
            default_headers.update(headers)
        self._client = httpx.Client(
            headers=default_headers,
            timeout=httpx.Timeout(float(timeout)),
            follow_redirects=True,
        )

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def fetch(self, **kwargs: Any) -> list[Document]:
        """Fetch documents from the source.

        All connectors share this signature.  Concrete implementations
        add source-specific keyword arguments (``codice``, ``from_date``,
        ``to_date``, ``limit``, etc.).

        Returns:
            List of :class:`~src.ingestion.loader.Document` objects with
            ``metadata.extra`` populated via
            :meth:`~src.italia.metadata.ItalianLegalMetadata.to_extra_dict`.

        Raises:
            :class:`ConnectorError`: On any unrecoverable failure.
        """

    # ── Shared HTTP helpers ───────────────────────────────────────────────────

    @retry(**_RETRY_KWARGS)  # type: ignore[arg-type]
    def _get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Rate-limited GET request with automatic retries.

        Args:
            url:    Absolute URL to request.
            params: Optional query-string parameters.

        Returns:
            :class:`httpx.Response` with a 2xx status code.

        Raises:
            :class:`ConnectorHTTPError`: On 4xx (non-429) responses.
            :class:`httpx.HTTPStatusError`: On 5xx/429 (triggers retry).
            :class:`httpx.TimeoutException`: On timeout (triggers retry).
        """
        self._bucket.acquire()
        log.debug(
            "HTTP GET",
            extra={"url": url, "params": params, "source": self.source_name},
        )
        response = self._client.get(url, params=params)
        if response.status_code in (429, 500, 502, 503, 504):
            response.raise_for_status()  # Let tenacity retry.
        if response.status_code >= 400:
            raise ConnectorHTTPError(
                status_code=response.status_code,
                url=url,
                message=response.text[:200],
            )
        return response

    def _paginate(
        self,
        url: str,
        params: dict[str, Any],
        page_key: str = "page",
        size_key: str = "size",
        page_size: int = 50,
        max_pages: int | None = None,
        results_key: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Generic page-by-page iterator for offset-based JSON APIs.

        Yields individual JSON result objects.  Stops when the server
        returns an empty results list or ``max_pages`` is reached.

        Args:
            url:         API endpoint URL.
            params:      Base query parameters (page/size keys will be added).
            page_key:    Name of the page-number query parameter.
            size_key:    Name of the page-size query parameter.
            page_size:   Number of items per page.
            max_pages:   Upper bound on pages fetched (``None`` = unlimited).
            results_key: JSON key containing the results array.  If ``None``,
                         the response is assumed to be the array itself.

        Yields:
            Individual JSON objects from the results array.
        """
        page = 0
        while True:
            if max_pages is not None and page >= max_pages:
                log.debug(
                    "Pagination limit reached",
                    extra={"source": self.source_name, "max_pages": max_pages},
                )
                break
            p = {**params, page_key: page, size_key: page_size}
            try:
                resp = self._get(url, params=p)
                data: Any = resp.json()
            except ConnectorHTTPError:
                break

            items: list[dict[str, Any]] = (
                data.get(results_key, []) if results_key else data
            )
            if not items:
                break

            yield from items
            if len(items) < page_size:
                break  # Last partial page.
            page += 1

    # ── Document factory ──────────────────────────────────────────────────────

    @staticmethod
    def _make_document(
        content: str,
        italian_meta: ItalianLegalMetadata,
        source_uri: str,
        filename: str | None = None,
    ) -> Document:
        """Build a :class:`~src.ingestion.loader.Document` for the existing pipeline.

        Args:
            content:       Raw text content of the document.
            italian_meta:  Populated :class:`~src.italia.metadata.ItalianLegalMetadata`.
            source_uri:    URL or URN string used as ``source_path``.
            filename:      Optional filename hint; defaults to last segment of
                           ``source_uri``.

        Returns:
            A :class:`~src.ingestion.loader.Document` with ``metadata.extra``
            carrying the Italian-specific fields (``it_*`` keys).
        """
        fname = filename or Path(source_uri.replace("://", "/")).name or source_uri
        extra = italian_meta.to_extra_dict()
        extra["source_uri"] = source_uri
        meta = DocumentMetadata(
            filename=fname,
            page_count=1,
            page_number=1,
            creation_date=datetime.now(tz=timezone.utc),
            loader_backend=italian_meta.fonte,
            extra={k: str(v) if not isinstance(v, (list, type(None))) else v for k, v in extra.items()},
        )
        return Document(
            content=content,
            metadata=meta,
            source_path=Path(source_uri),
            page_number=1,
        )

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "BaseConnector":
        return self

    def __exit__(self, *_: Any) -> None:
        self._client.close()

    def close(self) -> None:
        """Close the underlying HTTP client and free connections."""
        self._client.close()
