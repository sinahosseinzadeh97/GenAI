"""Abstract base class for all embedding providers.

Every concrete embedder in RAGForge Enterprise inherits from :class:`BaseEmbedder`
and must implement :meth:`embed_single` and :meth:`embed_batch`.  The base class
handles L2 normalisation and structured logging, keeping concrete implementations
lean.

Typical usage::

    from src.embedding.bge_embedder import BGEEmbedder

    embedder = BGEEmbedder()
    vector   = embedder.embed_single("What is RAG?")
    vectors  = embedder.embed_batch(["text one", "text two"])
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.utils.logger import get_logger, log_exception

log = get_logger(__name__)


# ── Custom Exceptions ─────────────────────────────────────────────────────────


class EmbeddingError(Exception):
    """Raised on any unrecoverable embedding failure."""


# ── Abstract Base ─────────────────────────────────────────────────────────────


class BaseEmbedder(ABC):
    """Abstract interface for embedding providers.

    All concrete embedders must implement :meth:`embed_single`, the
    :attr:`dimension` property, and the :attr:`model_name` property.
    :meth:`embed_batch` is implemented here using :meth:`embed_single` with
    internal batching but **should** be overridden for efficiency.

    Args:
        normalize: If ``True`` (default), output vectors are L2-normalised to
            unit length before being returned.  Normalisation makes cosine
            similarity equivalent to a dot-product, which is faster at query
            time.
    """

    def __init__(self, normalize: bool = True) -> None:
        self._normalize = normalize

    # ── Abstract properties ───────────────────────────────────────────────────

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors produced by this model.

        Returns:
            Integer vector dimension.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier (e.g. ``"BAAI/bge-small-en-v1.5"``).

        Returns:
            Model name string.
        """

    # ── Abstract method ───────────────────────────────────────────────────────

    @abstractmethod
    def embed_single(self, text: str) -> list[float]:
        """Embed a single string and return its vector.

        Args:
            text: The text to embed.

        Returns:
            A list of floats with length equal to :attr:`dimension`.

        Raises:
            EmbeddingError: On any embedding failure.
        """

    # ── Concrete batch implementation ─────────────────────────────────────────

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Embed a list of texts in mini-batches.

        The default implementation calls :meth:`embed_single` sequentially;
        concrete subclasses **should** override this for efficiency.

        Args:
            texts:      List of text strings to embed.
            batch_size: Number of texts per mini-batch.  Ignored by this default
                        implementation but respected by subclass overrides.

        Returns:
            List of float vectors, one per input text.

        Raises:
            EmbeddingError: If any individual embedding call fails.
        """
        if not texts:
            return []

        log.info(
            "Starting batch embedding",
            extra={
                "model": self.model_name,
                "total_texts": len(texts),
                "batch_size": batch_size,
            },
        )

        t_start = time.perf_counter()
        results: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size

            log.debug(
                "Processing batch",
                extra={
                    "batch": batch_num,
                    "total_batches": total_batches,
                    "batch_size": len(batch),
                },
            )
            try:
                for text in batch:
                    results.append(self.embed_single(text))
            except EmbeddingError:
                raise
            except Exception as exc:
                log_exception(log, "Unexpected error during batch embedding", exc)
                raise EmbeddingError(f"Batch embedding failed at index {i}: {exc}") from exc

        elapsed = time.perf_counter() - t_start
        throughput = len(texts) / elapsed if elapsed > 0 else float("inf")

        log.info(
            "Batch embedding complete",
            extra={
                "model": self.model_name,
                "total_texts": len(texts),
                "duration_seconds": round(elapsed, 4),
                "throughput_per_sec": round(throughput, 2),
            },
        )
        return results

    # ── Normalisation helper ─────────────────────────────────────────────────

    @staticmethod
    def _l2_normalize(vector: list[float]) -> list[float]:
        """Apply L2 normalisation to *vector*.

        If the norm is zero (zero vector), the original vector is returned
        unchanged to avoid division-by-zero.

        Args:
            vector: Raw embedding vector.

        Returns:
            L2-normalised vector (unit length).
        """
        arr: np.ndarray[Any, np.dtype[np.float32]] = np.array(vector, dtype=np.float32)
        norm: float = float(np.linalg.norm(arr))
        if norm == 0.0:
            return vector
        normalised: np.ndarray[Any, np.dtype[np.float32]] = arr / norm
        return normalised.tolist()

    def _maybe_normalize(self, vector: list[float]) -> list[float]:
        """Return *vector* optionally L2-normalised based on instance config.

        Args:
            vector: Raw embedding vector.

        Returns:
            Optionally normalised vector.
        """
        return self._l2_normalize(vector) if self._normalize else vector
