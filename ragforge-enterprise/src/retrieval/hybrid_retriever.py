"""Hybrid retriever using Reciprocal Rank Fusion (RRF).

:class:`HybridRetriever` combines a dense retriever and a sparse (BM25)
retriever by fusing their ranked result lists with the RRF formula:

    score(d) = Σ  1 / (k + rank(d))

where *k* is a smoothing constant (default 60) and the sum is taken over
each sub-retriever that returned the document.

Typical usage::

    from src.retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        rrf_k=60,
    )
    results = retriever.retrieve("invoice workflow", top_k=5)
"""

from __future__ import annotations

import time
from typing import Any

from src.retrieval.base import BaseRetriever, RetrievalError
from src.utils.logger import get_logger, log_exception
from src.vectorstore.schema import SearchResult

log = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """Reciprocal Rank Fusion combiner of dense and sparse retrievers.

    Fetches ``top_k * 2`` candidates from each sub-retriever, applies
    RRF, deduplicates by ``chunk_id``, sorts by combined RRF score, and
    returns the top-k results.

    Args:
        dense_retriever:  A :class:`~src.retrieval.dense_retriever.DenseRetriever`
                          (or any :class:`BaseRetriever`).
        sparse_retriever: A :class:`~src.retrieval.sparse_retriever.SparseRetriever`
                          (or any :class:`BaseRetriever`).
        rrf_k:            RRF smoothing constant (default 60).

    Example::

        hybrid = HybridRetriever(dense, sparse, rrf_k=60)
        results = hybrid.retrieve("payment terms", top_k=10)
    """

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever: BaseRetriever,
        rrf_k: int = 60,
    ) -> None:
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._rrf_k = rrf_k

        log.info(
            "HybridRetriever initialised",
            extra={
                "dense": type(dense_retriever).__name__,
                "sparse": type(sparse_retriever).__name__,
                "rrf_k": rrf_k,
            },
        )

    # ── RRF helper ────────────────────────────────────────────────────────────

    def _apply_rrf(
        self,
        ranked_lists: list[list[SearchResult]],
    ) -> dict[str, float]:
        """Compute Reciprocal Rank Fusion scores across multiple ranked lists.

        Args:
            ranked_lists: Each inner list is a ranked list from one retriever.
                          Lists need not be the same length.

        Returns:
            Dict mapping ``chunk_id`` → aggregated RRF score.
        """
        scores: dict[str, float] = {}
        for result_list in ranked_lists:
            for rank, result in enumerate(result_list, start=1):
                scores[result.chunk_id] = (
                    scores.get(result.chunk_id, 0.0) + 1.0 / (self._rrf_k + rank)
                )
        return scores

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Run dense + sparse retrieval, fuse with RRF, return top-k.

        Args:
            query:   Natural-language query string.
            top_k:   Number of final results to return.
            filters: Metadata filters forwarded to both sub-retrievers.

        Returns:
            List of :class:`~src.vectorstore.schema.SearchResult` sorted by
            descending RRF score.

        Raises:
            RetrievalError: When both sub-retrievers fail.
        """
        candidate_k = top_k * 2  # Fetch more candidates to improve fusion quality.

        # ── Dense retrieval ───────────────────────────────────────────────────
        t_dense = time.perf_counter()
        dense_results: list[SearchResult] = []
        dense_error: Exception | None = None
        try:
            dense_results = self._dense.retrieve(
                query=query, top_k=candidate_k, filters=filters
            )
        except RetrievalError as exc:
            dense_error = exc
            log.warning(
                "HybridRetriever: dense retrieval failed",
                extra={"error": str(exc)},
            )
        dense_latency_ms = (time.perf_counter() - t_dense) * 1000

        # ── Sparse retrieval ──────────────────────────────────────────────────
        t_sparse = time.perf_counter()
        sparse_results: list[SearchResult] = []
        sparse_error: Exception | None = None
        try:
            sparse_results = self._sparse.retrieve(
                query=query, top_k=candidate_k, filters=filters
            )
        except RetrievalError as exc:
            sparse_error = exc
            log.warning(
                "HybridRetriever: sparse retrieval failed",
                extra={"error": str(exc)},
            )
        sparse_latency_ms = (time.perf_counter() - t_sparse) * 1000

        if dense_error is not None and sparse_error is not None:
            raise RetrievalError(
                f"Both retrievers failed. Dense: {dense_error}. Sparse: {sparse_error}"
            )

        # ── Reciprocal Rank Fusion ────────────────────────────────────────────
        rrf_scores = self._apply_rrf(
            [r for r in [dense_results, sparse_results] if r]
        )

        # Build a lookup from chunk_id → SearchResult (prefer dense for metadata).
        result_map: dict[str, SearchResult] = {}
        for result in (dense_results or []) + (sparse_results or []):
            if result.chunk_id not in result_map:
                result_map[result.chunk_id] = result

        # Sort by RRF score descending and slice to top_k.
        fused: list[tuple[str, float]] = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        final_results: list[SearchResult] = []
        for rank, (chunk_id, rrf_score) in enumerate(fused, start=1):
            sr = result_map[chunk_id]
            final_results.append(
                SearchResult(
                    chunk_id=sr.chunk_id,
                    content=sr.content,
                    score=rrf_score,
                    metadata={**sr.metadata, "rrf_score": rrf_score},
                    rank=rank,
                )
            )

        log.info(
            "HybridRetriever fusion complete",
            extra={
                "query": query[:120],
                "dense_count": len(dense_results),
                "sparse_count": len(sparse_results),
                "fused_count": len(final_results),
                "dense_latency_ms": round(dense_latency_ms, 2),
                "sparse_latency_ms": round(sparse_latency_ms, 2),
                "rrf_k": self._rrf_k,
            },
        )

        return final_results
