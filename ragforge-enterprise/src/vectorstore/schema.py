"""Shared data models for the vector store layer.

:class:`IndexedChunk` represents a chunk that has been embedded and stored in
the vector store.  :class:`SearchResult` carries the output of a similarity
search query.

These are plain dataclasses (not Pydantic models) to keep them lightweight and
compatible with the serialisation needs of Qdrant payloads.

Italian legal metadata
----------------------
:class:`~src.italia.metadata.ItalianLegalMetadata` and
:class:`~src.italia.metadata.TipoDocumento` are also re-exported from this
module for convenience.  Italian-specific fields travel inside the existing
``IndexedChunk.metadata`` dict under ``it_*`` keys (see
:meth:`~src.italia.metadata.ItalianLegalMetadata.to_extra_dict`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class IndexedChunk:
    """A text chunk together with its embedding vector and provenance metadata.

    Attributes:
        chunk_id:   UUID string matching :attr:`~src.ingestion.chunker.Chunk.chunk_id`.
        content:    Raw text content of the chunk.
        embedding:  Dense float vector produced by the embedder.
        metadata:   Dictionary of document-level and chunk-level fields
                    (``source_path``, ``filename``, ``page_number``,
                    ``strategy_used``, ``chunk_index``, etc.).
        indexed_at: UTC datetime of when this chunk was written to the store.
    """

    chunk_id: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    indexed_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )


@dataclass
class SearchResult:
    """A single hit returned by a vector similarity search.

    Attributes:
        chunk_id: UUID string of the matched chunk.
        content:  Text content of the matched chunk.
        score:    Similarity score in [0, 1] (COSINE distance converted to
                  similarity: ``1 - distance``).
        metadata: Payload fields stored alongside the vector.
        rank:     1-based position in the result list (1 = most similar).
    """

    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    rank: int = 1


# ── Italian legal metadata re-exports ──────────────────────────────────────────
# Re-exporting here keeps all schema types in one discoverable location while
# keeping the authoritative definitions in src.italia.metadata.

# ItalianLegalMetadata imported lazily to avoid circular imports — see src/italia/metadata.py

__all__ = [
    "IndexedChunk",
    "SearchResult",
    "ItalianLegalMetadata",
    "TipoDocumento",
]
