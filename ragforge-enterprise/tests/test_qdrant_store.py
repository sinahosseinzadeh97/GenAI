"""Tests for QdrantStore using qdrant-client in-memory mode.

All tests use ``QdrantClient(":memory:")`` so no Docker instance is required.
Every public method on :class:`~src.vectorstore.qdrant_store.QdrantStore` is
covered: collection lifecycle, upsert, filtered search, and health check.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.vectorstore.base import UpsertStats, VectorStoreError
from src.vectorstore.schema import IndexedChunk, SearchResult
from src.vectorstore.qdrant_store import QdrantStore


# ── Helpers ───────────────────────────────────────────────────────────────────

_DIM = 8  # Small dimension for fast in-memory tests.

_COLLECTION = "test_col"


def _make_in_memory_store(collection: str = _COLLECTION) -> QdrantStore:
    """Return a QdrantStore backed by an in-memory Qdrant client.

    Args:
        collection: Default collection name for the store.

    Returns:
        Fully initialised :class:`QdrantStore`.
    """
    from qdrant_client import QdrantClient  # type: ignore[import-untyped]

    client = QdrantClient(":memory:")
    store = QdrantStore.__new__(QdrantStore)
    store._host = "memory"
    store._port = 0
    store._use_grpc = False
    store._collection_name = collection
    store._hnsw_m = 16
    store._hnsw_ef_construct = 100
    store._client = client
    return store


def _rand_vec(dim: int = _DIM, seed: int = 0) -> list[float]:
    """Return a deterministic pseudo-random vector of length *dim*.

    Args:
        dim:  Vector dimensionality.
        seed: NumPy RNG seed for reproducibility.

    Returns:
        List of floats.
    """
    import numpy as np

    rng = np.random.default_rng(seed=seed)
    vec: list[float] = rng.random(dim).tolist()
    return vec


def _make_chunk(
    content: str = "hello",
    seed: int = 0,
    meta: dict[str, Any] | None = None,
) -> IndexedChunk:
    """Build a throwaway :class:`IndexedChunk`.

    Args:
        content: Text content.
        seed:    Seed forwarded to :func:`_rand_vec`.
        meta:    Optional metadata override.

    Returns:
        :class:`IndexedChunk` ready to upsert.
    """
    return IndexedChunk(
        chunk_id=str(uuid.uuid4()),
        content=content,
        embedding=_rand_vec(seed=seed),
        metadata=meta or {"source_path": "test.pdf", "page_number": 1},
        indexed_at=datetime.now(tz=timezone.utc),
    )


# ── Tests: create_collection ──────────────────────────────────────────────────


class TestCreateCollection:
    """Tests for :meth:`QdrantStore.create_collection`."""

    def test_creates_new_collection(self) -> None:
        """A brand-new collection should be created without error."""
        store = _make_in_memory_store()
        store.create_collection(_COLLECTION, vector_size=_DIM)
        info = store.collection_info(_COLLECTION)
        assert info["vector_size"] == _DIM

    def test_idempotent_same_size(self) -> None:
        """Calling create_collection twice with the same size is a no-op."""
        store = _make_in_memory_store()
        store.create_collection(_COLLECTION, vector_size=_DIM)
        store.create_collection(_COLLECTION, vector_size=_DIM)  # Should not raise.
        info = store.collection_info(_COLLECTION)
        assert info["vector_size"] == _DIM

    def test_raises_on_size_mismatch(self) -> None:
        """Re-creating with a different vector size should raise VectorStoreError."""
        store = _make_in_memory_store()
        store.create_collection(_COLLECTION, vector_size=_DIM)
        with pytest.raises(VectorStoreError, match="size"):
            store.create_collection(_COLLECTION, vector_size=_DIM + 1)

    def test_default_distance_cosine(self) -> None:
        """Default metric should be COSINE."""
        store = _make_in_memory_store()
        store.create_collection(_COLLECTION, vector_size=_DIM)
        info = store.collection_info(_COLLECTION)
        assert "cosine" in info["distance"].lower()


# ── Tests: delete_collection ──────────────────────────────────────────────────


class TestDeleteCollection:
    """Tests for :meth:`QdrantStore.delete_collection`."""

    def test_deletes_existing_collection(self) -> None:
        """After deletion collection_info should raise."""
        store = _make_in_memory_store()
        store.create_collection(_COLLECTION, vector_size=_DIM)
        store.delete_collection(_COLLECTION)
        with pytest.raises(VectorStoreError):
            store.collection_info(_COLLECTION)


# ── Tests: upsert_chunks ──────────────────────────────────────────────────────


class TestUpsertChunks:
    """Tests for :meth:`QdrantStore.upsert_chunks`."""

    @pytest.fixture(autouse=True)
    def setup_collection(self) -> None:
        """Create a fresh in-memory store + collection for each test."""
        self.store = _make_in_memory_store()
        self.store.create_collection(_COLLECTION, vector_size=_DIM)

    def test_upsert_returns_stats(self) -> None:
        """upsert_chunks must return an UpsertStats with correct total."""
        chunks = [_make_chunk(seed=i) for i in range(5)]
        stats = self.store.upsert_chunks(chunks)
        assert isinstance(stats, UpsertStats)
        assert stats.total_inserted == 5
        assert stats.duration_seconds >= 0.0

    def test_upsert_empty_returns_zero_stats(self) -> None:
        """Upserting nothing should return zeroed-out stats."""
        stats = self.store.upsert_chunks([])
        assert stats.total_inserted == 0
        assert stats.total_updated == 0
        assert stats.duration_seconds == 0.0

    def test_upsert_increases_vector_count(self) -> None:
        """Collection vector count should reflect the upserted chunks."""
        chunks = [_make_chunk(seed=i) for i in range(3)]
        self.store.upsert_chunks(chunks)
        info = self.store.collection_info(_COLLECTION)
        assert info["vectors_count"] == 3

    def test_upsert_batches_correctly(self) -> None:
        """Large batches (> batch_size) should be handled across multiple calls."""
        chunks = [_make_chunk(f"text {i}", seed=i) for i in range(15)]
        stats = self.store.upsert_chunks(chunks, batch_size=4)
        assert stats.total_inserted == 15


# ── Tests: search ─────────────────────────────────────────────────────────────


class TestSearch:
    """Tests for :meth:`QdrantStore.search`."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Populate a collection with two distinct chunks."""
        self.store = _make_in_memory_store()
        self.store.create_collection(_COLLECTION, vector_size=_DIM)

        # Two chunks: one with known vector all-zeros-ish, one different.
        import numpy as np

        self.vec_a: list[float] = [1.0] + [0.0] * (_DIM - 1)
        self.vec_b: list[float] = [0.0] + [1.0] + [0.0] * (_DIM - 2)

        chunks = [
            IndexedChunk(
                chunk_id=str(uuid.uuid4()),
                content="chunk A",
                embedding=self.vec_a,
                metadata={"source_path": "a.pdf", "page_number": 1},
            ),
            IndexedChunk(
                chunk_id=str(uuid.uuid4()),
                content="chunk B",
                embedding=self.vec_b,
                metadata={"source_path": "b.pdf", "page_number": 2},
            ),
        ]
        self.store.upsert_chunks(chunks)

    def test_search_returns_list_of_results(self) -> None:
        """search must return a list of SearchResult objects."""
        results = self.store.search(self.vec_a, top_k=2)
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_top_k_respected(self) -> None:
        """Result count must not exceed top_k."""
        results = self.store.search(self.vec_a, top_k=1)
        assert len(results) == 1

    def test_search_best_match_first(self) -> None:
        """The closest chunk should be ranked first (rank=1)."""
        results = self.store.search(self.vec_a, top_k=2)
        assert results[0].rank == 1
        assert results[0].content == "chunk A"

    def test_search_score_range(self) -> None:
        """COSINE scores should be in a plausible range (near [−1, 1])."""
        results = self.store.search(self.vec_a, top_k=2)
        for r in results:
            assert -1.1 <= r.score <= 1.1

    def test_search_with_filter(self) -> None:
        """Filtered search should only return chunks from 'a.pdf'."""
        results = self.store.search(
            self.vec_a,
            top_k=10,
            filters={"source_path": "a.pdf"},
        )
        assert len(results) == 1
        assert results[0].content == "chunk A"

    def test_search_metadata_in_result(self) -> None:
        """SearchResult.metadata must include stored payload fields."""
        results = self.store.search(self.vec_a, top_k=1)
        assert "source_path" in results[0].metadata


# ── Tests: health_check ───────────────────────────────────────────────────────


class TestHealthCheck:
    """Tests for :meth:`QdrantStore.health_check`."""

    def test_health_check_healthy(self) -> None:
        """In-memory Qdrant client should pass the health check."""
        store = _make_in_memory_store()
        assert store.health_check() is True

    def test_health_check_fails_gracefully(self) -> None:
        """If get_collections raises, health_check must return False (not raise)."""
        store = _make_in_memory_store()
        store._client = MagicMock()
        store._client.get_collections.side_effect = RuntimeError("connection refused")
        assert store.health_check() is False
