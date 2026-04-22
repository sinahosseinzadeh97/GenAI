"""Vector store sub-package for RAGForge Enterprise.

Exports the abstract :class:`BaseVectorStore`, concrete :class:`QdrantStore`,
and data models :class:`IndexedChunk` and :class:`SearchResult`.

Example::

    from src.vectorstore import QdrantStore, IndexedChunk, SearchResult
"""

from src.vectorstore.base import BaseVectorStore
from src.vectorstore.qdrant_store import QdrantStore
from src.vectorstore.schema import IndexedChunk, SearchResult

__all__ = ["BaseVectorStore", "QdrantStore", "IndexedChunk", "SearchResult"]
