"""Dense retriever using BGE embeddings and Qdrant vector search.

:class:`DenseRetriever` embeds the query with the BGE query prefix and
calls :class:`~src.vectorstore.qdrant_store.QdrantStore` for cosine
similarity search.

Typical usage::

    from src.retrieval.dense_retriever import DenseRetriever
    from src.embedding.bge_embedder import BGEEmbedder
    from src.vectorstore.qdrant_store import QdrantStore

    retriever = DenseRetriever(
        embedder=BGEEmbedder(),
        store=QdrantStore(),
        collection_name="ragforge_docs",
    )
    results = retriever.retrieve("invoice processing", top_k=5)
"""

from __future__ import annotations

from typing import Any

from src.embedding.base import BaseEmbedder, EmbeddingError
from src.retrieval.base import BaseRetriever, RetrievalError
from src.utils.logger import get_logger, log_exception
from src.vectorstore.base import BaseVectorStore, VectorStoreError
from src.vectorstore.schema import SearchResult

log = get_logger(__name__)


class DenseRetriever(BaseRetriever):
    """Vector-similarity retriever backed by BGE + Qdrant.

    Embeds the query using the BGE query-mode prefix and forwards it to
    the vector store's cosine similarity search.  Metadata filters are
    passed directly to Qdrant.

    Args:
        embedder:        An :class:`~src.embedding.base.BaseEmbedder` instance.
        store:           A :class:`~src.vectorstore.base.BaseVectorStore` instance.
        collection_name: Qdrant collection to search in.

    Example::

        retriever = DenseRetriever(embedder, store, "ragforge_docs")
        results = retriever.retrieve("inventory management", top_k=10)
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        collection_name: str,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._collection_name = collection_name

        # Monkey-patch the store's active collection to match.
        # QdrantStore stores the collection at construction-time;
        # we override it here so we can support multi-collection routing.
        if hasattr(store, "_collection_name"):
            store._collection_name = collection_name  # type: ignore[attr-defined]

        log.info(
            "DenseRetriever initialised",
            extra={
                "embedder": embedder.model_name,
                "collection": collection_name,
            },
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Embed *query* and perform cosine similarity search in Qdrant.

        Args:
            query:   Natural-language query string.
            top_k:   Maximum number of results to return.
            filters: Payload filters forwarded to Qdrant (key=exact value).

        Returns:
            List of :class:`~src.vectorstore.schema.SearchResult` sorted by
            descending similarity score.

        Raises:
            RetrievalError: On embedding failure or vector store failure.
        """
        try:
            # Use the BGE query prefix for retrieval.
            query_vector = self._embedder.embed_single(query, mode="query")  # type: ignore[call-arg]
        except EmbeddingError as exc:
            log_exception(log, "DenseRetriever: embedding failed", exc)
            raise RetrievalError(f"Failed to embed query: {exc}") from exc
        except TypeError:
            # Fallback for embedders that don't accept 'mode' kwarg.
            try:
                query_vector = self._embedder.embed_single(query)
            except EmbeddingError as exc:
                raise RetrievalError(f"Failed to embed query: {exc}") from exc

        try:
            results: list[SearchResult] = self._store.search(
                query_vector=query_vector,
                top_k=top_k,
                filters=filters,
            )
        except VectorStoreError as exc:
            log_exception(log, "DenseRetriever: vector store search failed", exc)
            raise RetrievalError(f"Vector store search failed: {exc}") from exc

        # Ensure descending score order (Qdrant returns sorted, but be safe).
        results.sort(key=lambda r: r.score, reverse=True)

        # Re-number ranks.
        for i, result in enumerate(results, start=1):
            result.rank = i

        return results
