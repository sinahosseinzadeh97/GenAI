"""Ingestion sub-package for RAGForge Enterprise.

Exposes the document loader, text cleaner, and chunker classes together
with their public exception hierarchy so consumers only need to import
from this package.

Example::

    from src.ingestion import DocumentLoader, DocumentCleaner, FixedSizeChunker
"""

from src.ingestion.chunker import (
    BaseChunker,
    Chunk,
    ChunkingError,
    EmbeddingError,
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
    TokenizerError,
    count_tokens,
)
from src.ingestion.cleaner import (
    CleaningError,
    CleaningStats,
    DocumentCleaner,
    LanguageDetectionError,
)
from src.ingestion.loader import (
    DirectoryNotFoundError,
    Document,
    DocumentLoader,
    DocumentMetadata,
    PDFCorruptedError,
    PDFExtractionError,
    PDFLoadError,
)

__all__ = [
    # chunker
    "Chunk",
    "ChunkingError",
    "TokenizerError",
    "EmbeddingError",
    "BaseChunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "count_tokens",
    # cleaner
    "CleaningError",
    "LanguageDetectionError",
    "CleaningStats",
    "DocumentCleaner",
    # loader
    "PDFLoadError",
    "PDFCorruptedError",
    "PDFExtractionError",
    "DirectoryNotFoundError",
    "DocumentMetadata",
    "Document",
    "DocumentLoader",
]
