"""End-to-end tests for IndexingPipeline.

Uses a mock embedder and an in-memory Qdrant store so the tests are
completely hermetic – no model downloads, no Docker.

Covers:
- Happy-path full pipeline run.
- Empty chunk input returns zeroed report.
- Embedding failure propagates as IndexingError.
- Upsert failure is captured in failed_chunks (partial-failure mode).
- IndexingReport string formatting.
- Auto-create collection behaviour.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.embedding.base import BaseEmbedder, EmbeddingError
from src.indexing.pipeline import FailedChunk, IndexingError, IndexingPipeline, IndexingReport
from src.ingestion.chunker import Chunk
from src.vectorstore.base import BaseVectorStore, UpsertStats, VectorStoreError
from src.vectorstore.schema import IndexedChunk, SearchResult


# ── Fakes ─────────────────────────────────────────────────────────────────────

_DIM = 8


class _FakeEmbedder(BaseEmbedder):
    """Deterministic fake embedder that returns zero-vectors."""

    def __init__(self, dim: int = _DIM, normalize: bool = False) -> None:
        super().__init__(normalize=normalize)
        self._dim = dim
        self.embed_batch_call_count = 0

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "fake-embedder-v0"

    def embed_single(self, text: str) -> list[float]:
        return [0.0] * self._dim

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        **_kwargs: Any,
    ) -> list[list[float]]:
        self.embed_batch_call_count += 1
        return [[0.0] * self._dim for _ in texts]


class _FailingEmbedder(_FakeEmbedder):
    """Embedder that unconditionally raises EmbeddingError."""

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        **_kwargs: Any,
    ) -> list[list[float]]:
        raise EmbeddingError("intentional embedding failure")


def _make_in_memory_store(collection: str = "pipe_test") -> "Any":
    """Return a QdrantStore wired to an in-memory Qdrant client.

    Args:
        collection: Default collection name.

    Returns:
        Fully initialised in-memory :class:`~src.vectorstore.qdrant_store.QdrantStore`.
    """
    from qdrant_client import QdrantClient  # type: ignore[import-untyped]

    from src.vectorstore.qdrant_store import QdrantStore

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


def _make_chunks(n: int = 5, strategy: str = "recursive") -> list[Chunk]:
    """Build *n* fake :class:`~src.ingestion.chunker.Chunk` objects.

    Args:
        n:        Number of chunks.
        strategy: ``strategy_used`` value stored in each chunk.

    Returns:
        List of :class:`~src.ingestion.chunker.Chunk`.
    """
    return [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            content=f"This is chunk number {i}.",
            metadata={"source_path": "fake.pdf", "page_number": i},
            strategy_used=strategy,
            token_count=10 + i,
        )
        for i in range(n)
    ]


# ── Tests: IndexingReport ─────────────────────────────────────────────────────


class TestIndexingReport:
    """Unit tests for :class:`IndexingReport` properties and formatting."""

    def test_total_failed_property(self) -> None:
        """total_failed must equal len(failed_chunks)."""
        fc = [FailedChunk("abc", "oops"), FailedChunk("xyz", "also oops")]
        report = IndexingReport(total_chunks_processed=10, total_indexed=8, failed_chunks=fc)
        assert report.total_failed == 2

    def test_success_rate_full(self) -> None:
        """100% rate when all chunks are indexed."""
        report = IndexingReport(total_chunks_processed=5, total_indexed=5)
        assert report.success_rate == 1.0

    def test_success_rate_partial(self) -> None:
        """Partial success should be reflected as a fraction."""
        report = IndexingReport(total_chunks_processed=4, total_indexed=3)
        assert abs(report.success_rate - 0.75) < 1e-9

    def test_success_rate_zero_chunks(self) -> None:
        """Report with no chunks should return 0.0 success rate (no ZeroDivisionError)."""
        report = IndexingReport(total_chunks_processed=0, total_indexed=0)
        assert report.success_rate == 0.0

    def test_str_contains_collection_name(self) -> None:
        """String representation must include the collection name."""
        report = IndexingReport(
            total_chunks_processed=3,
            total_indexed=3,
            collection_name="my_col",
        )
        assert "my_col" in str(report)

    def test_str_contains_embedder_model(self) -> None:
        """String representation must include the embedder model name."""
        report = IndexingReport(
            total_chunks_processed=3,
            total_indexed=3,
            embedder_model="BAAI/bge-small",
        )
        assert "BAAI/bge-small" in str(report)


# ── Tests: IndexingPipeline ───────────────────────────────────────────────────


class TestIndexingPipeline:
    """Integration-style tests for :class:`IndexingPipeline`."""

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_happy_path_indexes_all_chunks(self) -> None:
        """A standard run should index all chunks with zero failures."""
        chunks = _make_chunks(5)
        embedder = _FakeEmbedder()
        store = _make_in_memory_store()

        pipeline = IndexingPipeline(
            embedder=embedder,
            vector_store=store,
            collection_name="pipe_test",
            auto_create_collection=True,
        )
        report = pipeline.run(chunks)

        assert report.total_chunks_processed == 5
        assert report.total_indexed == 5
        assert report.total_failed == 0
        assert report.success_rate == 1.0

    def test_embedding_called_once(self) -> None:
        """embed_batch should be called exactly once per pipeline run."""
        chunks = _make_chunks(10)
        embedder = _FakeEmbedder()
        store = _make_in_memory_store()

        pipeline = IndexingPipeline(
            embedder=embedder,
            vector_store=store,
            collection_name="pipe_test",
            auto_create_collection=True,
        )
        pipeline.run(chunks)

        assert embedder.embed_batch_call_count == 1

    def test_report_timing_populated(self) -> None:
        """embedding_time_seconds and indexing_time_seconds must be non-negative."""
        report = IndexingPipeline(
            embedder=_FakeEmbedder(),
            vector_store=_make_in_memory_store(),
            collection_name="pipe_test",
            auto_create_collection=True,
        ).run(_make_chunks(3))

        assert report.embedding_time_seconds >= 0.0
        assert report.indexing_time_seconds >= 0.0

    def test_collection_name_in_report(self) -> None:
        """Report should carry the configured collection name."""
        report = IndexingPipeline(
            embedder=_FakeEmbedder(),
            vector_store=_make_in_memory_store("special_col"),
            collection_name="special_col",
            auto_create_collection=True,
        ).run(_make_chunks(2))

        assert report.collection_name == "special_col"

    def test_embedder_model_in_report(self) -> None:
        """Report should carry the embedder model identifier."""
        report = IndexingPipeline(
            embedder=_FakeEmbedder(),
            vector_store=_make_in_memory_store(),
            collection_name="pipe_test",
            auto_create_collection=True,
        ).run(_make_chunks(2))

        assert report.embedder_model == "fake-embedder-v0"

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_empty_chunks_returns_zeroed_report(self) -> None:
        """Calling run with zero chunks should return an all-zero report."""
        pipeline = IndexingPipeline(
            embedder=_FakeEmbedder(),
            vector_store=_make_in_memory_store(),
            collection_name="pipe_test",
        )
        report = pipeline.run([])

        assert report.total_chunks_processed == 0
        assert report.total_indexed == 0
        assert report.total_failed == 0

    # ── Failure modes ─────────────────────────────────────────────────────────

    def test_embedding_failure_raises_indexing_error(self) -> None:
        """Fatal embedding failure should bubble up as IndexingError."""
        pipeline = IndexingPipeline(
            embedder=_FailingEmbedder(),
            vector_store=_make_in_memory_store(),
            collection_name="pipe_test",
            auto_create_collection=True,
        )
        with pytest.raises(IndexingError, match="Embedding step failed"):
            pipeline.run(_make_chunks(3))

    def test_upsert_failure_captured_in_failed_chunks(self) -> None:
        """If upsert raises VectorStoreError the pipeline records all chunks as failed."""
        store = _make_in_memory_store()
        store._client = MagicMock()
        store._client.get_collections.return_value = MagicMock(collections=[])
        store._client.create_collection.return_value = None
        store._client.create_payload_index.return_value = None
        store._client.upsert.side_effect = RuntimeError("disk full")

        pipeline = IndexingPipeline(
            embedder=_FakeEmbedder(),
            vector_store=store,
            collection_name="pipe_test",
            auto_create_collection=True,
        )
        report = pipeline.run(_make_chunks(4))

        assert report.total_indexed == 0
        assert report.total_failed == 4
        assert report.success_rate == 0.0

    def test_auto_create_collection_triggered(self) -> None:
        """Pipeline should create the collection when auto_create_collection=True."""
        store = _make_in_memory_store("autocol")
        pipeline = IndexingPipeline(
            embedder=_FakeEmbedder(),
            vector_store=store,
            collection_name="autocol",
            auto_create_collection=True,
        )
        pipeline.run(_make_chunks(2))
        # Collection should exist post-run.
        info = store.collection_info("autocol")
        assert info["vectors_count"] == 2
