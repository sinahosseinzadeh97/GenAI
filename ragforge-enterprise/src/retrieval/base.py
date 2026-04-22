"""Abstract base class for all retrieval strategies.

Every concrete retriever — dense, sparse, or hybrid — must inherit from
:class:`BaseRetriever` and implement :meth:`retrieve`.  The base class
enforces a consistent logging contract so that latency and result-count
metrics are always captured.

Typical usage::

    from src.retrieval.base import BaseRetriever
    from src.vectorstore.schema import SearchResult

    class MyRetriever(BaseRetriever):
        def retrieve(
            self, query: str, top_k: int = 10, filters: dict | None = None
        ) -> list[SearchResult]:
            ...
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from src.utils.logger import get_logger
from src.vectorstore.schema import SearchResult

log = get_logger(__name__)


# ── Custom Exception ──────────────────────────────────────────────────────────


class RetrievalError(Exception):
    """Raised on any unrecoverable retrieval failure."""


# ── Abstract Base ─────────────────────────────────────────────────────────────


class BaseRetriever(ABC):
    """Abstract interface for retrieval strategies.

    All concrete retrievers must implement :meth:`retrieve`.  The
    :meth:`_timed_retrieve` helper automatically logs query, top_k,
    latency_ms, and result_count for every call.

    Example::

        class MyRetriever(BaseRetriever):
            def retrieve(self, query, top_k=10, filters=None):
                ...  # return List[SearchResult]
    """

    # ── Abstract method ───────────────────────────────────────────────────────

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve the top-k results for *query*.

        Args:
            query:   Natural-language query string.
            top_k:   Number of results to return.
            filters: Optional metadata filter map (passed to the backend).

        Returns:
            List of :class:`~src.vectorstore.schema.SearchResult` ordered
            by descending relevance score.

        Raises:
            RetrievalError: On any unrecoverable retrieval failure.
        """

    # ── Logged wrapper ────────────────────────────────────────────────────────

    def timed_retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[SearchResult], float]:
        """Call :meth:`retrieve` and return results with elapsed time in ms.

        This wrapper measures wall-clock latency and emits a structured
        INFO log entry that every retriever implementation benefits from
        automatically.

        Args:
            query:   Natural-language query string.
            top_k:   Number of results to return.
            filters: Optional metadata filter map.

        Returns:
            Tuple of (results, latency_ms).

        Raises:
            RetrievalError: Re-raised from :meth:`retrieve`.
        """
        t_start = time.perf_counter()
        results = self.retrieve(query=query, top_k=top_k, filters=filters)
        latency_ms = (time.perf_counter() - t_start) * 1000

        log.info(
            "Retrieval complete",
            extra={
                "retriever": type(self).__name__,
                "query": query[:120],
                "top_k": top_k,
                "result_count": len(results),
                "latency_ms": round(latency_ms, 2),
                "filters": filters,
            },
        )
        return results, latency_ms
