"""CrossEncoder reranker using sentence-transformers.

:class:`CrossEncoderReranker` scores each (query, passage) pair with a
cross-encoder model and re-orders retrieval results by the new score.
It is intentionally decoupled from the retrieval strategy so it can be
plugged in after any retriever.

Typical usage::

    from src.retrieval.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query="invoice", results=hits, top_k=5)
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Any

from src.config.settings import get_settings
from src.utils.logger import get_logger, log_exception
from src.vectorstore.schema import SearchResult

_settings = get_settings()
log = get_logger(__name__, level=_settings.log_level)

# ── Model singleton ───────────────────────────────────────────────────────────

_ce_lock: Lock = Lock()
_ce_cache: dict[str, Any] = {}  # model_name → CrossEncoder instance


def _load_cross_encoder(model_name: str) -> Any:
    """Return a cached CrossEncoder instance for *model_name*.

    Thread-safe via double-checked locking.

    Args:
        model_name: sentence-transformers cross-encoder identifier.

    Returns:
        A ``CrossEncoder`` instance.

    Raises:
        RerankerError: When the model cannot be loaded.
    """
    if model_name in _ce_cache:
        return _ce_cache[model_name]

    with _ce_lock:
        if model_name in _ce_cache:
            return _ce_cache[model_name]
        try:
            from sentence_transformers.cross_encoder import (  # type: ignore[import-untyped]
                CrossEncoder,
            )

            log.info("Loading CrossEncoder model", extra={"model": model_name})
            t0 = time.perf_counter()
            model = CrossEncoder(model_name)
            elapsed = time.perf_counter() - t0
            log.info(
                "CrossEncoder model loaded",
                extra={"model": model_name, "load_time_seconds": round(elapsed, 3)},
            )
            _ce_cache[model_name] = model
            return model
        except ImportError as exc:
            raise RerankerError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
        except Exception as exc:
            raise RerankerError(
                f"Failed to load CrossEncoder model '{model_name}': {exc}"
            ) from exc


# ── Custom Exception ──────────────────────────────────────────────────────────


class RerankerError(Exception):
    """Raised on any unrecoverable reranking failure."""


# ── CrossEncoder Reranker ─────────────────────────────────────────────────────


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers.

    Scores (query, passage) pairs with a cross-encoder inference pass and
    re-orders the results.  A ``min_score`` threshold can be used to
    discard low-quality matches.

    Args:
        model_name:  HuggingFace cross-encoder model identifier.
        enabled:     When ``False``, :meth:`rerank` is a no-op and returns
                     the input unchanged (truncated to *top_k*).
        min_score:   Discard results with cross-encoder score below this
                     threshold.  Default ``0.0`` (keep all).

    Example::

        reranker = CrossEncoderReranker(enabled=True, min_score=0.1)
        final = reranker.rerank("shipping terms", results, top_k=5)
    """

    def __init__(
        self,
        model_name: str | None = None,
        enabled: bool | None = None,
        min_score: float | None = None,
    ) -> None:
        self._model_name: str = model_name or _settings.reranker_model
        self._enabled: bool = enabled if enabled is not None else _settings.reranker_enabled
        self._min_score: float = (
            min_score if min_score is not None else _settings.reranker_min_score
        )

        log.info(
            "CrossEncoderReranker initialised",
            extra={
                "model": self._model_name,
                "enabled": self._enabled,
                "min_score": self._min_score,
            },
        )

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Re-score and re-order *results* using the cross-encoder.

        When :attr:`_enabled` is ``False`` or *results* is empty, returns
        the input list (truncated to *top_k* if provided) unchanged.

        Args:
            query:   Natural-language query string.
            results: Retrieval results to rerank.
            top_k:   Maximum results to return after reranking.  ``None``
                     means return all results that pass the ``min_score``
                     filter.

        Returns:
            Reranked list of :class:`~src.vectorstore.schema.SearchResult`.
            Each result's ``score`` is replaced with the cross-encoder score,
            and ``metadata["rerank_score"]`` is set for downstream consumers.

        Raises:
            RerankerError: On cross-encoder inference failure.
        """
        if not self._enabled or not results:
            output = results[:top_k] if top_k is not None else results
            log.debug(
                "CrossEncoderReranker: skipped (disabled or empty results)",
                extra={"enabled": self._enabled, "input_count": len(results)},
            )
            return output

        input_count = len(results)
        t_start = time.perf_counter()

        try:
            model = _load_cross_encoder(self._model_name)

            # Build sentence pairs for batch inference.
            pairs: list[tuple[str, str]] = [
                (query, result.content) for result in results
            ]
            scores: list[float] = model.predict(pairs).tolist()

        except RerankerError:
            raise
        except Exception as exc:
            log_exception(log, "CrossEncoderReranker inference failed", exc)
            raise RerankerError(f"Cross-encoder reranking failed: {exc}") from exc

        latency_ms = (time.perf_counter() - t_start) * 1000

        # Pair results with their new scores and sort descending.
        scored: list[tuple[SearchResult, float]] = list(zip(results, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Apply min_score threshold, re-number ranks.
        output: list[SearchResult] = []
        for rank, (result, ce_score) in enumerate(scored, start=1):
            if self._min_score > 0.0 and ce_score < self._min_score:
                continue
            output.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=float(ce_score),
                    metadata={**result.metadata, "rerank_score": float(ce_score)},
                    rank=rank,
                )
            )

        if top_k is not None:
            output = output[:top_k]

        # Re-number final ranks after potential filtering.
        for i, r in enumerate(output, start=1):
            r.rank = i

        log.info(
            "CrossEncoderReranker complete",
            extra={
                "model": self._model_name,
                "input_count": input_count,
                "output_count": len(output),
                "latency_ms": round(latency_ms, 2),
                "min_score_threshold": self._min_score,
            },
        )

        return output
