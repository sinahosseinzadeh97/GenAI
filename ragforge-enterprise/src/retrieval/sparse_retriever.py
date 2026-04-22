"""BM25 sparse retriever using the rank-bm25 library.

:class:`SparseRetriever` builds a BM25Okapi index from a corpus of text
chunks at construction time and uses it for keyword-based retrieval.
The tokenizer is a simple whitespace + punctuation stripper — no NLTK
dependency required.

Typical usage::

    from src.retrieval.sparse_retriever import SparseRetriever
    from src.ingestion.chunker import Chunk

    retriever = SparseRetriever(chunks=my_chunks)
    results = retriever.retrieve("invoice workflow", top_k=5)
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.retrieval.base import BaseRetriever, RetrievalError
from src.utils.logger import get_logger, log_exception
from src.vectorstore.schema import SearchResult

log = get_logger(__name__)

# Regex that strips everything that is not a letter or digit.
_NON_ALPHANUM = re.compile(r"[^a-zA-Z0-9\s]")


# ── Lightweight corpus item ───────────────────────────────────────────────────


@dataclass
class CorpusItem:
    """A lightweight representation of a text chunk stored in the BM25 index.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        content:  Raw text content.
        metadata: Arbitrary metadata dict.
    """

    chunk_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Tokenizer ─────────────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Tokenize *text* by lowercasing, removing punctuation, and splitting on whitespace.

    This deliberately avoids NLTK / spaCy dependencies so the retriever
    can be instantiated in any environment.

    Args:
        text: Raw text string.

    Returns:
        List of lowercase token strings (may be empty).
    """
    lowered = text.lower()
    stripped = _NON_ALPHANUM.sub(" ", lowered)
    tokens = stripped.split()
    # Filter single-character tokens to reduce noise.
    return [t for t in tokens if len(t) > 1]


# ── Sparse Retriever ──────────────────────────────────────────────────────────


class SparseRetriever(BaseRetriever):
    """BM25-based keyword retriever using rank-bm25 (Okapi BM25).

    Builds a BM25 index in memory from a list of :class:`CorpusItem`
    (or any object with ``chunk_id``, ``content``, and ``metadata``
    attributes).  The index is rebuildable via :meth:`rebuild_index`.

    Args:
        chunks: Initial corpus to index.  Can be a list of
                :class:`CorpusItem` or any objects with the same attributes.

    Raises:
        RetrievalError: If ``rank-bm25`` is not installed.

    Example::

        retriever = SparseRetriever(chunks=[CorpusItem("1", "hello world", {})])
        results = retriever.retrieve("hello", top_k=3)
    """

    def __init__(self, chunks: list[Any] | None = None) -> None:
        self._corpus: list[CorpusItem] = []
        self._bm25: Any = None

        if chunks:
            self.rebuild_index(chunks)
        else:
            log.info(
                "SparseRetriever initialised with empty corpus",
                extra={"vocabulary_size": 0, "corpus_size": 0},
            )

    # ── Index management ──────────────────────────────────────────────────────

    def rebuild_index(self, chunks: list[Any]) -> None:
        """(Re-)build the BM25 index from *chunks*.

        Replaces both the internal corpus and the BM25 index object.
        Safe to call multiple times (e.g. after ingesting new documents).

        Args:
            chunks: Objects with at minimum ``chunk_id``, ``content``,
                    and ``metadata`` attributes, or dicts with those keys.

        Raises:
            RetrievalError: When ``rank-bm25`` is not installed.
        """
        try:
            from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RetrievalError(
                "rank-bm25 is not installed. Run: pip install rank-bm25"
            ) from exc

        t_start = time.perf_counter()

        self._corpus = []
        tokenized_corpus: list[list[str]] = []

        for chunk in chunks:
            if isinstance(chunk, dict):
                item = CorpusItem(
                    chunk_id=str(chunk.get("chunk_id", uuid.uuid4())),
                    content=str(chunk.get("content", "")),
                    metadata=dict(chunk.get("metadata", {})),
                )
            else:
                item = CorpusItem(
                    chunk_id=str(getattr(chunk, "chunk_id", str(uuid.uuid4()))),
                    content=str(getattr(chunk, "content", "")),
                    metadata=dict(getattr(chunk, "metadata", {})),
                )
            self._corpus.append(item)
            tokenized_corpus.append(_tokenize(item.content))

        self._bm25 = BM25Okapi(tokenized_corpus)

        elapsed = time.perf_counter() - t_start
        # Estimate vocabulary size from the BM25 index's idf dict.
        vocab_size = len(getattr(self._bm25, "idf", {}))

        log.info(
            "BM25 index built",
            extra={
                "corpus_size": len(self._corpus),
                "vocabulary_size": vocab_size,
                "build_time_seconds": round(elapsed, 4),
            },
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Score all corpus documents against *query* and return the top-k.

        Args:
            query:   Natural-language query string.
            top_k:   Maximum number of results to return.
            filters: Metadata filter map applied *after* BM25 scoring.
                     Only results where ALL filter key/value pairs match
                     the result's metadata are kept.

        Returns:
            List of :class:`~src.vectorstore.schema.SearchResult` sorted by
            descending BM25 score.

        Raises:
            RetrievalError: When the index has not been built.
        """
        if self._bm25 is None or not self._corpus:
            raise RetrievalError(
                "BM25 index is empty. Call rebuild_index() with a non-empty corpus."
            )

        try:
            query_tokens = _tokenize(query)
            if not query_tokens:
                # All-stopword query: return empty list gracefully.
                log.warning(
                    "SparseRetriever: query tokenized to empty list",
                    extra={"query": query},
                )
                return []

            scores: list[float] = self._bm25.get_scores(query_tokens).tolist()

            # Get sorted indices (descending score).
            indexed_scores = sorted(
                enumerate(scores), key=lambda x: x[1], reverse=True
            )

            results: list[SearchResult] = []
            rank = 1
            for idx, score in indexed_scores:
                if len(results) >= top_k:
                    break
                item = self._corpus[idx]
                # Apply metadata filters.
                if filters and not self._matches_filters(item.metadata, filters):
                    continue
                results.append(
                    SearchResult(
                        chunk_id=item.chunk_id,
                        content=item.content,
                        score=float(score),
                        metadata=item.metadata,
                        rank=rank,
                    )
                )
                rank += 1

            return results

        except RetrievalError:
            raise
        except Exception as exc:
            log_exception(log, "SparseRetriever.retrieve failed", exc)
            raise RetrievalError(f"BM25 retrieval failed: {exc}") from exc

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check that all *filters* entries are present in *metadata*.

        Args:
            metadata: Chunk metadata dict.
            filters:  Required key/value pairs.

        Returns:
            ``True`` if all filter conditions are satisfied.
        """
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def corpus_size(self) -> int:
        """Number of documents in the current BM25 index.

        Returns:
            Integer corpus size.
        """
        return len(self._corpus)

    @property
    def vocabulary_size(self) -> int:
        """Estimated vocabulary size of the BM25 index.

        Returns:
            Integer vocabulary size (0 if index not yet built).
        """
        if self._bm25 is None:
            return 0
        return len(getattr(self._bm25, "idf", {}))
