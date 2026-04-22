"""Embedding sub-package for RAGForge Enterprise.

Exports the abstract :class:`BaseEmbedder` and concrete implementations so
that consumers only need to import from this package.

Example::

    from src.embedding import BGEEmbedder, OpenAIEmbedder, BaseEmbedder
    from src.embedding import ItalianLegalEmbedder, CrossEncoderReranker
"""

from src.embedding.base import BaseEmbedder
from src.embedding.bge_embedder import BGEEmbedder
from src.embedding.italian_embedder import CrossEncoderReranker, ItalianLegalEmbedder
from src.embedding.openai_embedder import OpenAIEmbedder

__all__ = [
    "BaseEmbedder",
    "BGEEmbedder",
    "OpenAIEmbedder",
    "ItalianLegalEmbedder",
    "CrossEncoderReranker",
]
