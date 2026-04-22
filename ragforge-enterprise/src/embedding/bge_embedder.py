"""BGE embedder using BAAI/bge-small-en-v1.5 via sentence-transformers.

The BGE (BAAI General Embedding) model family uses different instruction
prefixes for passages (documents at index-time) vs. queries (at search-time):

- **Passage prefix**: ``"Represent this sentence for searching relevant passages: "``
- **Query prefix**:   ``"Represent this query for searching relevant passages: "``

This module exposes a singleton-safe :class:`BGEEmbedder` that loads the model
once per process and provides both single and batched embedding with structured
throughput logging.

Typical usage::

    from src.embedding.bge_embedder import BGEEmbedder

    embedder = BGEEmbedder()

    # Index-time (passage):
    vectors = embedder.embed_batch(texts, mode="passage")

    # Query-time:
    q_vector = embedder.embed_single("What is RAG?", mode="query")
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Any

from src.config.settings import get_settings
from src.embedding.base import BaseEmbedder, EmbeddingError
from src.utils.logger import get_logger, log_exception

_settings = get_settings()
log = get_logger(__name__, level=_settings.log_level)

# ── BGE instruction prefixes ──────────────────────────────────────────────────

_PASSAGE_PREFIX: str = "Represent this sentence for searching relevant passages: "
_QUERY_PREFIX: str = "Represent this query for searching relevant passages: "

# ── Model singleton ───────────────────────────────────────────────────────────

_model_lock: Lock = Lock()
_model_cache: dict[str, Any] = {}  # model_name → SentenceTransformer instance


def _load_model(model_name: str) -> Any:
    """Return the cached SentenceTransformer model for *model_name*.

    Thread-safe: only one thread loads the model; subsequent calls return the
    cached instance immediately.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        A ``SentenceTransformer`` instance.

    Raises:
        EmbeddingError: When the model cannot be loaded.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    with _model_lock:
        # Double-checked locking pattern.
        if model_name in _model_cache:
            return _model_cache[model_name]

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            log.info("Loading BGE model", extra={"model": model_name})
            t0 = time.perf_counter()
            model = SentenceTransformer(model_name)
            elapsed = time.perf_counter() - t0
            log.info(
                "BGE model loaded",
                extra={"model": model_name, "load_time_seconds": round(elapsed, 3)},
            )
            _model_cache[model_name] = model
            return model
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
        except Exception as exc:
            raise EmbeddingError(f"Failed to load BGE model '{model_name}': {exc}") from exc


# ── Concrete Embedder ─────────────────────────────────────────────────────────


class BGEEmbedder(BaseEmbedder):
    """Sentence-transformers embedder using BAAI/bge-small-en-v1.5.

    Supports both passage (indexing) and query (retrieval) modes via BGE
    instruction prefixes.  The underlying model is loaded once and cached for
    the lifetime of the process.

    Args:
        model_name:  HuggingFace model identifier to use.  Defaults to the
                     value in :func:`~src.config.settings.get_settings`.
        normalize:   Whether to L2-normalise output embeddings.
        batch_size:  Default mini-batch size for :meth:`embed_batch`.

    Attributes:
        _dim: Embedding dimension (inferred from model at init-time).

    Example::

        embedder = BGEEmbedder()
        vec = embedder.embed_single("Hello world", mode="passage")
        assert len(vec) == 384
    """

    def __init__(
        self,
        model_name: str | None = None,
        normalize: bool = True,
        batch_size: int = 32,
    ) -> None:
        super().__init__(normalize=normalize)
        self._model_name: str = model_name or _settings.embedding_model
        self._batch_size: int = batch_size
        self._dim: int | None = None  # Resolved on first embed call.

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """Return the HuggingFace model identifier.

        Returns:
            Model name string.
        """
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the embedding dimension.

        Triggers a model load on first access so that the dimension can be
        inferred from the loaded weights.

        Returns:
            Integer dimension (e.g. 384 for bge-small-en-v1.5).

        Raises:
            EmbeddingError: If the model cannot be loaded.
        """
        if self._dim is None:
            model = _load_model(self._model_name)
            self._dim = model.get_sentence_embedding_dimension()
        return self._dim  # type: ignore[return-value]

    # ── Core embedding methods ────────────────────────────────────────────────

    def embed_single(
        self,
        text: str,
        mode: str = "passage",
    ) -> list[float]:
        """Embed a single text string using BGE instruction prefix semantics.

        Args:
            text: The text to embed.
            mode: Either ``"passage"`` (indexing) or ``"query"`` (retrieval).
                  Determines which BGE instruction prefix is prepended.

        Returns:
            L2-normalised (if enabled) float vector of length :attr:`dimension`.

        Raises:
            EmbeddingError: On model failure.
        """
        prefix = _PASSAGE_PREFIX if mode == "passage" else _QUERY_PREFIX
        full_text = prefix + text

        try:
            model = _load_model(self._model_name)
            # normalize_embeddings=True lets sentence-transformers handle L2
            # normalisation; we optionally re-normalise with our own method for
            # consistency across embedder implementations.
            raw = model.encode(
                full_text,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,  # We handle normalisation ourselves.
            )
            # model.encode on a single string may return shape (dim,) or (1, dim)
            # depending on the sentence-transformers version. Normalise to 1-D.
            import numpy as _np  # noqa: PLC0415

            raw = _np.asarray(raw)
            if raw.ndim == 2:
                raw = raw[0]
            embedding: list[float] = raw.tolist()
            return self._maybe_normalize(embedding)
        except EmbeddingError:
            raise
        except Exception as exc:
            log_exception(log, "BGE embed_single failed", exc)
            raise EmbeddingError(f"BGE embedding failed: {exc}") from exc

    def embed_batch(  # type: ignore[override]
        self,
        texts: list[str],
        batch_size: int | None = None,
        mode: str = "passage",
    ) -> list[list[float]]:
        """Embed a list of texts in mini-batches with BGE instruction prefixes.

        Args:
            texts:      List of strings to embed.
            batch_size: Mini-batch size (defaults to :attr:`_batch_size`).
            mode:       Embedding mode – ``"passage"`` or ``"query"``.

        Returns:
            List of float vectors (one per text), each of length :attr:`dimension`.

        Raises:
            EmbeddingError: On model or encoding failure.
        """
        if not texts:
            return []

        bs = batch_size or self._batch_size
        prefix = _PASSAGE_PREFIX if mode == "passage" else _QUERY_PREFIX
        prefixed = [prefix + t for t in texts]

        log.info(
            "BGE batch embedding started",
            extra={
                "model": self._model_name,
                "total_texts": len(texts),
                "batch_size": bs,
                "mode": mode,
            },
        )

        t_start = time.perf_counter()

        try:
            model = _load_model(self._model_name)
            results: list[list[float]] = []

            for i in range(0, len(prefixed), bs):
                batch = prefixed[i : i + bs]
                t_batch = time.perf_counter()
                raw = model.encode(
                    batch,
                    batch_size=bs,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
                batch_elapsed = time.perf_counter() - t_batch
                batch_num = i // bs + 1
                total_batches = (len(prefixed) + bs - 1) // bs

                log.debug(
                    "BGE batch processed",
                    extra={
                        "batch": batch_num,
                        "total_batches": total_batches,
                        "texts_in_batch": len(batch),
                        "batch_duration_seconds": round(batch_elapsed, 4),
                    },
                )

                for vec in raw:
                    results.append(self._maybe_normalize(vec.tolist()))

            elapsed = time.perf_counter() - t_start
            throughput = len(texts) / elapsed if elapsed > 0 else float("inf")

            log.info(
                "BGE batch embedding complete",
                extra={
                    "model": self._model_name,
                    "total_texts": len(texts),
                    "duration_seconds": round(elapsed, 4),
                    "throughput_per_sec": round(throughput, 2),
                    "mode": mode,
                },
            )
            return results

        except EmbeddingError:
            raise
        except Exception as exc:
            log_exception(log, "BGE embed_batch failed", exc)
            raise EmbeddingError(f"BGE batch embedding failed: {exc}") from exc
