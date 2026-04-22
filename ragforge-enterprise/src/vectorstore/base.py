"""Abstract base class for all vector store backends.

Defines the minimum interface that every concrete vector store must implement.
This allows the indexing pipeline to be backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.vectorstore.schema import IndexedChunk


# ── Custom Exceptions ─────────────────────────────────────────────────────────


class VectorStoreError(Exception):
    """Raised on any unrecoverable vector store operation failure."""


# ── Upsert statistics ────────────────────────────────────────────────────────


@dataclass
class UpsertStats:
    """Statistics returned after a batch upsert operation.

    Attributes:
        total_inserted:   Number of new vectors inserted.
        total_updated:    Number of existing vectors updated.
        duration_seconds: Wall-clock time for the upsert operation.
    """

    total_inserted: int
    total_updated: int
    duration_seconds: float


# ── Abstract Base ─────────────────────────────────────────────────────────────


class BaseVectorStore(ABC):
    """Abstract interface for vector store backends.

    Concrete implementations must provide collection management, upsert, and
    search capabilities.  All methods must raise :class:`VectorStoreError` on
    unrecoverable failures.
    """

    @abstractmethod
    def create_collection(
        self,
        name: str,
        vector_size: int,
        distance: str = "COSINE",
    ) -> None:
        """Create a new vector collection (idempotent).

        Args:
            name:        Collection name.
            vector_size: Dimensionality of the vectors.
            distance:    Distance metric – ``"COSINE"``, ``"DOT"``, or
                         ``"EUCLIDEAN"``.

        Raises:
            VectorStoreError: On backend failure.
        """

    @abstractmethod
    def upsert_chunks(
        self,
        chunks: list[Any],
        batch_size: int = 100,
    ) -> UpsertStats:
        """Upsert a list of :class:`~src.vectorstore.schema.IndexedChunk` objects.

        Args:
            chunks:     Chunks to insert or update.
            batch_size: Number of points per upsert request.

        Returns:
            :class:`UpsertStats` with counts and timing.

        Raises:
            VectorStoreError: On backend failure.
        """

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Find the *top_k* most similar vectors.

        Args:
            query_vector: Dense query vector.
            top_k:        Number of results to return.
            filters:      Optional metadata filter map.

        Returns:
            List of :class:`~src.vectorstore.schema.SearchResult`, ranked by
            similarity (highest first).

        Raises:
            VectorStoreError: On backend failure.
        """

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Delete a collection and all its data.

        Args:
            name: Collection name.

        Raises:
            VectorStoreError: On backend failure.
        """

    @abstractmethod
    def collection_info(self, name: str) -> dict[str, Any]:
        """Return metadata about an existing collection.

        Args:
            name: Collection name.

        Returns:
            Dictionary with at least ``vectors_count``, ``status``, and
            ``vector_size`` keys.

        Raises:
            VectorStoreError: When the collection does not exist or the
                backend is unreachable.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Check connectivity to the vector store backend.

        Returns:
            ``True`` if the backend is reachable, ``False`` otherwise.
        """

    @abstractmethod
    def delete_chunks_by_source(self, collection_name: str, source_path: str) -> int:
        """Delete all points whose payload ``source_path`` matches *source_path*.

        Args:
            collection_name: Collection to delete from.
            source_path:     Value of the ``source_path`` payload field used as
                             the filter key.

        Returns:
            Number of points deleted.

        Raises:
            VectorStoreError: On backend failure.
        """

    @abstractmethod
    def source_exists(self, collection_name: str, source_path: str) -> bool:
        """Return ``True`` if at least one chunk with the given *source_path*
        payload value exists in *collection_name*.

        Uses a Qdrant ``count()`` call with an exact match filter so that
        the check is O(1) and does not fetch any vectors or payloads.

        Args:
            collection_name: Collection to inspect.
            source_path:     Value to look up in the ``source_path`` payload
                             field.

        Returns:
            ``True`` when the collection contains at least one matching point,
            ``False`` otherwise.

        Raises:
            VectorStoreError: On backend failure.
        """

    @abstractmethod
    def get_all_chunks(
        self,
        collection_name: str,
        batch_size: int = 100,
    ) -> list["IndexedChunk"]:
        """Page through every point in *collection_name* and return them as
        :class:`~src.vectorstore.schema.IndexedChunk` objects.

        Intended for populating an in-memory :class:`SparseRetriever` corpus
        at API startup so the BM25 index is not empty after the first restart.

        Args:
            collection_name: Collection to scroll through.
            batch_size:      Number of points fetched per scroll page.

        Returns:
            List of :class:`~src.vectorstore.schema.IndexedChunk` (may be
            empty if the collection does not exist or is empty).

        Raises:
            VectorStoreError: On unexpected backend failure.
        """
