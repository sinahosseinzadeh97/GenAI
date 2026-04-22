"""Indexing pipeline orchestrating embed → store flow.

:class:`IndexingPipeline` accepts chunked documents from Phase 1, embeds them
in batches, constructs :class:`~src.vectorstore.schema.IndexedChunk` objects,
and upserts them into the configured vector store.

It returns an :class:`IndexingReport` dataclass summarising the entire
operation so that callers get deterministic end-to-end observability.

Typical usage::

    from src.indexing.pipeline import IndexingPipeline
    from src.embedding.bge_embedder import BGEEmbedder
    from src.vectorstore.qdrant_store import QdrantStore

    pipeline = IndexingPipeline(
        embedder=BGEEmbedder(),
        vector_store=QdrantStore(),
        collection_name="ragforge_docs",
    )
    report = pipeline.run(chunks)
    print(report)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.config.settings import get_settings
from src.embedding.base import BaseEmbedder, EmbeddingError
from src.ingestion.chunker import Chunk
from src.utils.logger import get_logger, log_exception
from src.vectorstore.base import BaseVectorStore, VectorStoreError
from src.vectorstore.schema import IndexedChunk

_settings = get_settings()
log = get_logger(__name__, level=_settings.log_level)


# ── Custom Exception ──────────────────────────────────────────────────────────


class IndexingError(Exception):
    """Raised on unrecoverable indexing pipeline failures."""


# ── Report dataclass ──────────────────────────────────────────────────────────


@dataclass
class FailedChunk:
    """Records a chunk that could not be indexed.

    Attributes:
        chunk_id: UUID of the failed chunk.
        reason:   Human-readable failure description.
    """

    chunk_id: str
    reason: str


@dataclass
class IndexingReport:
    """End-to-end summary of an indexing run.

    Attributes:
        total_chunks_processed:   Chunks received by the pipeline.
        total_indexed:            Chunks successfully stored in the vector store.
        failed_chunks:            List of :class:`FailedChunk` records.
        embedding_time_seconds:   Wall-clock time spent on embedding.
        indexing_time_seconds:    Wall-clock time spent on vector store upsert.
        avg_embedding_throughput: Chunks embedded per second.
        collection_name:          Name of the target collection.
        embedder_model:           Identifier of the embedder used.
    """

    total_chunks_processed: int
    total_indexed: int
    failed_chunks: list[FailedChunk] = field(default_factory=list)
    embedding_time_seconds: float = 0.0
    indexing_time_seconds: float = 0.0
    avg_embedding_throughput: float = 0.0
    collection_name: str = ""
    embedder_model: str = ""

    @property
    def total_failed(self) -> int:
        """Number of chunks that failed to index."""
        return len(self.failed_chunks)

    @property
    def success_rate(self) -> float:
        """Fraction of chunks successfully indexed (0.0–1.0)."""
        if self.total_chunks_processed == 0:
            return 0.0
        return self.total_indexed / self.total_chunks_processed

    def __str__(self) -> str:
        """Human-readable summary table.

        Returns:
            Formatted multi-line string summarising the report.
        """
        sep = "─" * 52
        lines = [
            "",
            "┌" + sep + "┐",
            "│{:^52}│".format("  RAGForge Indexing Report  "),
            "├" + sep + "┤",
            "│  {:<28} {:>20} │".format("Collection", self.collection_name[:20]),
            "│  {:<28} {:>20} │".format("Embedder", self.embedder_model[:20]),
            "├" + sep + "┤",
            "│  {:<28} {:>20,} │".format("Chunks processed", self.total_chunks_processed),
            "│  {:<28} {:>20,} │".format("Successfully indexed", self.total_indexed),
            "│  {:<28} {:>20,} │".format("Failed chunks", self.total_failed),
            "│  {:<28} {:>19.1%} │".format("Success rate", self.success_rate),
            "├" + sep + "┤",
            "│  {:<28} {:>18.3f}s │".format("Embedding time", self.embedding_time_seconds),
            "│  {:<28} {:>18.3f}s │".format("Indexing time", self.indexing_time_seconds),
            "│  {:<28} {:>16.1f}/s │".format(
                "Embedding throughput", self.avg_embedding_throughput
            ),
            "└" + sep + "┘",
            "",
        ]
        if self.failed_chunks:
            lines.append("Failed chunks:")
            for fc in self.failed_chunks[:10]:  # Show first 10.
                lines.append(f"  • {fc.chunk_id[:8]}… – {fc.reason}")
        return "\n".join(lines)


# ── Pipeline ──────────────────────────────────────────────────────────────────


class IndexingPipeline:
    """Orchestrates the embed → index flow.

    Steps:

    1. Accept ``List[Chunk]`` from Phase 1.
    2. Extract text strings for embedding.
    3. Call :meth:`~src.embedding.base.BaseEmbedder.embed_batch`.
    4. Combine :class:`~src.ingestion.chunker.Chunk` metadata with embeddings
       to produce :class:`~src.vectorstore.schema.IndexedChunk` objects.
    5. Call :meth:`~src.vectorstore.base.BaseVectorStore.upsert_chunks`.
    6. Return :class:`IndexingReport`.

    Args:
        embedder:        Concrete embedder instance.
        vector_store:    Concrete vector store instance.
        collection_name: Collection name to upsert into.  Defaults to the
                         settings value.
        embedding_batch_size: Mini-batch size forwarded to the embedder.
        upsert_batch_size:    Points per Qdrant upsert request.
        auto_create_collection: If ``True``, create the collection if it does
                                not exist.

    Raises:
        IndexingError: On unrecoverable pipeline failure.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        collection_name: str | None = None,
        embedding_batch_size: int | None = None,
        upsert_batch_size: int = 100,
        auto_create_collection: bool = True,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._collection_name: str = collection_name or _settings.qdrant_collection_name
        self._embedding_batch_size: int = (
            embedding_batch_size or _settings.embedding_batch_size
        )
        self._upsert_batch_size = upsert_batch_size
        self._auto_create = auto_create_collection

    def run(self, chunks: list[Chunk]) -> IndexingReport:
        """Execute the full embed → index pipeline.

        Args:
            chunks: Output of Phase 1 chunking.

        Returns:
            :class:`IndexingReport` with full statistics.

        Raises:
            IndexingError: On fatal embedding or storage failure.
        """
        if not chunks:
            log.warning("IndexingPipeline.run called with empty chunk list")
            return IndexingReport(
                total_chunks_processed=0,
                total_indexed=0,
                collection_name=self._collection_name,
                embedder_model=self._embedder.model_name,
            )

        log.info(
            "Indexing pipeline started",
            extra={
                "collection": self._collection_name,
                "embedder": self._embedder.model_name,
                "total_chunks": len(chunks),
            },
        )

        failed_chunks: list[FailedChunk] = []

        # ── Step 1: Auto-create collection ────────────────────────────────────
        if self._auto_create:
            try:
                self._vector_store.create_collection(
                    name=self._collection_name,
                    vector_size=self._embedder.dimension,
                    distance="COSINE",
                )
            except VectorStoreError as exc:
                raise IndexingError(
                    f"Failed to create collection '{self._collection_name}': {exc}"
                ) from exc

        # ── Step 2: Extract texts ─────────────────────────────────────────────
        texts: list[str] = [c.content for c in chunks]

        # ── Step 3: Embed ─────────────────────────────────────────────────────
        t_embed_start = time.perf_counter()
        try:
            embeddings: list[list[float]] = self._embedder.embed_batch(
                texts,
                batch_size=self._embedding_batch_size,
            )
        except EmbeddingError as exc:
            raise IndexingError(f"Embedding step failed: {exc}") from exc

        embedding_time = time.perf_counter() - t_embed_start
        throughput = len(chunks) / embedding_time if embedding_time > 0 else float("inf")

        # ── Step 4: Build IndexedChunks ───────────────────────────────────────
        indexed_chunks: list[IndexedChunk] = []
        for chunk, embedding in zip(chunks, embeddings):
            meta: dict[str, Any] = dict(chunk.metadata)
            meta["strategy_used"] = chunk.strategy_used
            meta["token_count"] = chunk.token_count
            indexed_chunks.append(
                IndexedChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    embedding=embedding,
                    metadata=meta,
                )
            )

        # ── Step 5: Upsert ────────────────────────────────────────────────────
        t_index_start = time.perf_counter()
        try:
            stats = self._vector_store.upsert_chunks(
                indexed_chunks,
                batch_size=self._upsert_batch_size,
            )
            total_indexed = stats.total_inserted + stats.total_updated
        except VectorStoreError as exc:
            log_exception(log, "Upsert step failed", exc)
            # Record all chunks as failed and return partial report.
            for chunk in chunks:
                failed_chunks.append(FailedChunk(chunk_id=chunk.chunk_id, reason=str(exc)))
            total_indexed = 0

        indexing_time = time.perf_counter() - t_index_start

        report = IndexingReport(
            total_chunks_processed=len(chunks),
            total_indexed=total_indexed,
            failed_chunks=failed_chunks,
            embedding_time_seconds=round(embedding_time, 4),
            indexing_time_seconds=round(indexing_time, 4),
            avg_embedding_throughput=round(throughput, 2),
            collection_name=self._collection_name,
            embedder_model=self._embedder.model_name,
        )

        log.info(
            "Indexing pipeline complete",
            extra={
                "total_processed": report.total_chunks_processed,
                "total_indexed": report.total_indexed,
                "total_failed": report.total_failed,
                "embedding_time_s": report.embedding_time_seconds,
                "indexing_time_s": report.indexing_time_seconds,
                "throughput_per_sec": report.avg_embedding_throughput,
            },
        )

        return report
