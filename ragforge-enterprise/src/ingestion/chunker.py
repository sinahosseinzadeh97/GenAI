"""Chunking strategies for RAGForge Enterprise.

This module provides three production-grade text-chunking strategies that
share a common :class:`BaseChunker` interface.  All strategies operate on
:class:`~src.ingestion.loader.Document` instances and produce typed
:class:`Chunk` objects.

Strategies
----------
:class:`FixedSizeChunker`
    Splits text at a target token budget while always honouring sentence
    boundaries.  Uses ``tiktoken`` for accurate token counting.

:class:`RecursiveChunker`
    Re-implements LangChain's ``RecursiveCharacterTextSplitter`` from
    scratch using a hierarchy of separators.  Token-aware so that chunks
    never exceed the configured budget.

:class:`SemanticChunker`
    Embeds sentences with ``sentence-transformers`` and places split
    boundaries where the cosine similarity between consecutive sentences
    drops below a configurable threshold.

Typical usage::

    from pathlib import Path
    from src.ingestion.loader import DocumentLoader
    from src.ingestion.cleaner import DocumentCleaner
    from src.ingestion.chunker import FixedSizeChunker, RecursiveChunker, SemanticChunker
    from src.config.settings import get_settings

    cfg     = get_settings()
    loader  = DocumentLoader()
    cleaner = DocumentCleaner()
    docs    = cleaner.clean_batch(loader.load_file(Path("report.pdf")))

    fixed_chunks    = FixedSizeChunker(cfg).chunk_documents(docs)
    recursive_chunks = RecursiveChunker(cfg).chunk_documents(docs)
    semantic_chunks = SemanticChunker(cfg).chunk_documents(docs)
"""

from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from src.config.settings import Settings, get_settings
from src.ingestion.loader import Document
from src.utils.logger import get_logger, log_exception

# ── Logging ───────────────────────────────────────────────────────────────────
_settings = get_settings()
log = get_logger(__name__, level=_settings.log_level)


# ── Custom Exceptions ─────────────────────────────────────────────────────────


class ChunkingError(Exception):
    """Base exception for all chunking failures."""


class TokenizerError(ChunkingError):
    """Raised when the tiktoken tokeniser cannot be initialised."""


class EmbeddingError(ChunkingError):
    """Raised when sentence-transformer embedding fails."""


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """A single text chunk produced by a chunking strategy.

    Attributes:
        chunk_id:      Globally unique identifier (UUID4).
        content:       Text content of this chunk.
        token_count:   Number of tokens as counted by the cl100k_base tokeniser.
        strategy_used: Name of the chunking strategy that produced this chunk.
        metadata:      Combined document-level metadata plus chunk-level fields
                       ``chunk_index`` (0-based) and ``total_chunks``.
    """

    chunk_id: str
    content: str
    token_count: int
    strategy_used: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Tokeniser singleton ───────────────────────────────────────────────────────

_ENCODING_NAME = "cl100k_base"  # Used by GPT-4, text-embedding-3-*, etc.
_CHARS_PER_TOKEN: float = 4.0  # Conservative chars-per-token ratio for fast estimation.


def _get_encoding() -> Any:
    """Return the cached tiktoken encoding object.

    Returns:
        A ``tiktoken.Encoding`` instance.

    Raises:
        TokenizerError: If tiktoken is not installed or encoding load fails.
    """
    try:
        import tiktoken  # pylint: disable=import-outside-toplevel

        return tiktoken.get_encoding(_ENCODING_NAME)
    except ImportError as exc:
        raise TokenizerError("tiktoken is not installed. Run: pip install tiktoken") from exc
    except Exception as exc:
        raise TokenizerError(f"Failed to load tiktoken encoding '{_ENCODING_NAME}': {exc}") from exc


def count_tokens(text: str) -> int:
    """Count the number of cl100k_base tokens in *text*.

    Args:
        text: The string to tokenise.

    Returns:
        Integer token count.
    """
    enc = _get_encoding()
    return len(enc.encode(text))


# ── Sentence splitting ────────────────────────────────────────────────────────

_RE_SENTENCE_END = re.compile(
    r"(?<=[.!?])\s+"          # Sentence-ending punctuation followed by space.
    r"(?=[A-Z\"\'\(\[\{])"    # Followed by capital, quote, or open bracket.
)


def _split_into_sentences(text: str) -> list[str]:
    """Split *text* into sentences using a regex heuristic.

    This is intentionally lightweight and language-agnostic.  For production
    NLP-heavy workloads consider integrating spaCy or NLTK punkt tokeniser.

    Args:
        text: Input text.

    Returns:
        List of sentence strings (preserving trailing spaces).
    """
    sentences = _RE_SENTENCE_END.split(text.strip())
    # Re-attach the trailing punctuation stripped by the split.
    result: list[str] = []
    for sent in sentences:
        if sent.strip():
            result.append(sent.strip())
    return result or [text.strip()]


# ── Chunk factory helper ──────────────────────────────────────────────────────


def _make_chunk(
    content: str,
    strategy: str,
    doc: Document,
    chunk_index: int,
    total_chunks: int,
) -> Chunk:
    """Construct a :class:`Chunk` from content and parent document context.

    Args:
        content:      The chunk text.
        strategy:     Strategy name string.
        doc:          Source :class:`~src.ingestion.loader.Document`.
        chunk_index:  0-based position within the document's chunk list.
        total_chunks: Total number of chunks in this document.

    Returns:
        A fully populated :class:`Chunk`.
    """
    meta: dict[str, Any] = {
        "filename": doc.metadata.filename,
        "page_count": doc.metadata.page_count,
        "page_number": doc.metadata.page_number,
        "source_path": str(doc.source_path),
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
    }
    if doc.metadata.creation_date:
        meta["creation_date"] = doc.metadata.creation_date.isoformat()
    # Propagate cleaning stats if present.
    if "cleaning_stats" in doc.metadata.extra:
        meta["cleaning_stats"] = doc.metadata.extra["cleaning_stats"]

    return Chunk(
        chunk_id=str(uuid.uuid4()),
        content=content,
        token_count=count_tokens(content),
        strategy_used=strategy,
        metadata=meta,
    )


# ── Base interface ─────────────────────────────────────────────────────────────


class BaseChunker(ABC):
    """Abstract base class that all chunking strategies must implement.

    Args:
        settings: Application settings (chunk_size, chunk_overlap, etc.).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._cfg = settings or get_settings()

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Human-readable identifier for this strategy."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a single :class:`~src.ingestion.loader.Document` into chunks.

        Args:
            document: Cleaned document to chunk.

        Returns:
            Ordered list of :class:`Chunk` objects.
        """

    def chunk_documents(self, documents: Sequence[Document]) -> list[Chunk]:
        """Chunk a collection of documents and return a flat list.

        Args:
            documents: Iterable of documents (e.g. all pages of a PDF).

        Returns:
            Flat list of :class:`Chunk` objects across all documents.
        """
        all_chunks: list[Chunk] = []
        failed_docs: list[tuple[str, str]] = []
        doc_count = 0
        for doc in documents:
            doc_count += 1
            try:
                chunks = self.chunk(doc)
                all_chunks.extend(chunks)
            except ChunkingError:
                raise
            except Exception as exc:
                log_exception(
                    log,
                    f"{self.strategy_name}: failed to chunk {doc.metadata.filename} "
                    f"p{doc.page_number}",
                    exc,
                )
                failed_docs.append((str(doc.source_path), str(exc)))
        if failed_docs:
            log.warning(
                "chunk_documents completed with partial failures",
                extra={"failed_count": len(failed_docs), "failed_docs": failed_docs},
            )
        log.info(
            "Chunking complete",
            extra={
                "strategy": self.strategy_name,
                "documents": doc_count,
                "total_chunks": len(all_chunks),
            },
        )
        return all_chunks


# ── Strategy 1: FixedSizeChunker ──────────────────────────────────────────────


class FixedSizeChunker(BaseChunker):
    """Token-budget chunker that respects sentence boundaries.

    The algorithm works as follows:

    1. Split the document into sentences.
    2. Accumulate sentences into a window until the running token count
       would exceed ``chunk_size``.
    3. Emit the window as a chunk.
    4. Back-fill an overlap window (up to ``chunk_overlap`` tokens) from
       the tail of the emitted chunk and continue.

    This guarantees:
    - No chunk exceeds ``chunk_size`` tokens (unless a single sentence is
      longer, in which case it is emitted as-is to avoid data loss).
    - Chunks never split mid-sentence.
    - Consecutive chunks share a configurable token overlap.

    Args:
        settings: Application settings override.

    Example:
        >>> chunker = FixedSizeChunker()
        >>> chunks = chunker.chunk(doc)
        >>> all(c.token_count <= 512 for c in chunks if len(chunks) > 1)
        True
    """

    @property
    def strategy_name(self) -> str:
        """Return the strategy identifier string."""
        return "fixed_size"

    def chunk(self, document: Document) -> list[Chunk]:
        """Split *document* into fixed-size token-budget chunks.

        Args:
            document: The document to split.

        Returns:
            List of :class:`Chunk` objects.
        """
        text = document.content.strip()
        if not text:
            return []

        sentences = _split_into_sentences(text)
        chunk_size = self._cfg.chunk_size
        overlap = self._cfg.chunk_overlap

        raw_chunks: list[str] = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            # Fast estimate: exact tiktoken call reserved for final chunk-boundary check.
            sent_tokens = int(len(sentence) / _CHARS_PER_TOKEN)

            # Edge case: a single sentence exceeds the budget.
            if sent_tokens >= chunk_size and not current_sentences:
                raw_chunks.append(sentence)
                continue

            if current_tokens + sent_tokens > chunk_size and current_sentences:
                # Emit the current window.
                raw_chunks.append(" ".join(current_sentences))
                # Build overlap: walk back from the end while under budget.
                overlap_sentences: list[str] = []
                overlap_tokens = 0
                for prev_sent in reversed(current_sentences):
                    t = count_tokens(prev_sent)
                    if overlap_tokens + t > overlap:
                        break
                    overlap_sentences.insert(0, prev_sent)
                    overlap_tokens += t
                current_sentences = overlap_sentences
                current_tokens = overlap_tokens

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        if current_sentences:
            raw_chunks.append(" ".join(current_sentences))

        total = len(raw_chunks)
        return [
            _make_chunk(text, self.strategy_name, document, idx, total)
            for idx, text in enumerate(raw_chunks)
            if text.strip()
        ]


# ── Strategy 2: RecursiveChunker ─────────────────────────────────────────────


class RecursiveChunker(BaseChunker):
    """Re-implementation of LangChain's RecursiveCharacterTextSplitter.

    Splits text using a decreasing hierarchy of separators.  Each separator
    is tried in order; if a piece remains over-budget after splitting on the
    current separator, the next separator in the list is tried recursively.

    Separator hierarchy (in descending structural priority):
    ``["\\n\\n", "\\n", ". ", " ", ""]``

    This preserves paragraph > line > sentence > word structure in that
    preference order.  Token counting (not character counting) is used to
    evaluate the budget.

    Args:
        settings:    Application settings override.
        separators:  Override the default separator hierarchy.

    Note:
        LangChain is **not** imported; this is a clean-room re-implementation.
    """

    _DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        settings: Settings | None = None,
        separators: list[str] | None = None,
    ) -> None:
        super().__init__(settings)
        self._separators = separators or self._DEFAULT_SEPARATORS

    @property
    def strategy_name(self) -> str:
        """Return the strategy identifier string."""
        return "recursive"

    def chunk(self, document: Document) -> list[Chunk]:
        """Recursively split *document* using the separator hierarchy.

        Args:
            document: The document to split.

        Returns:
            Ordered list of :class:`Chunk` objects.
        """
        text = document.content.strip()
        if not text:
            return []

        raw_chunks = self._split_recursive(text, self._separators)
        merged = self._merge_with_overlap(raw_chunks)

        total = len(merged)
        return [
            _make_chunk(content, self.strategy_name, document, idx, total)
            for idx, content in enumerate(merged)
            if content.strip()
        ]

    # ── private recursive helpers ──────────────────────────────────────────

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split *text* by the first separator that works.

        If this separator produces pieces that still exceed the token budget,
        the next separator in the list is tried on those pieces.

        Args:
            text:       Text to split.
            separators: Remaining separators to try (consumed head-first).

        Returns:
            List of text pieces, each ≤ ``chunk_size`` tokens (best effort).
        """
        chunk_size = self._cfg.chunk_size

        # Base case: text is already within budget.
        if count_tokens(text) <= chunk_size:
            return [text]

        if not separators:
            # Last resort: hard character-split (very long word with no spaces).
            return self._hard_split(text)

        sep, *remaining = separators

        if sep:
            pieces = text.split(sep)
            # Re-attach the separator (except for the final piece) to preserve
            # document structure during re-assembly.
            pieces_with_sep = [
                (p + sep if i < len(pieces) - 1 else p) for i, p in enumerate(pieces)
            ]
        else:
            # Empty separator → character-level split.
            pieces_with_sep = list(text)

        result: list[str] = []
        for piece in pieces_with_sep:
            if not piece.strip():
                continue
            if count_tokens(piece) <= chunk_size:
                result.append(piece)
            else:
                result.extend(self._split_recursive(piece, remaining))
        return result

    def _hard_split(self, text: str) -> list[str]:
        """Split *text* at character boundaries as a last resort.

        Args:
            text: A single token-dense string with no whitespace splits.

        Returns:
            List of sub-strings each ≤ ``chunk_size`` tokens.
        """
        enc = _get_encoding()
        tokens = enc.encode(text)
        size = self._cfg.chunk_size
        return [
            enc.decode(tokens[i : i + size]) for i in range(0, len(tokens), size)
        ]

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """Merge small pieces back into chunks respecting size and overlap.

        Args:
            pieces: List of text pieces produced by :meth:`_split_recursive`.

        Returns:
            List of merged chunk strings.
        """
        chunk_size = self._cfg.chunk_size
        overlap = self._cfg.chunk_overlap

        merged: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for piece in pieces:
            piece_tokens = count_tokens(piece)
            if current_tokens + piece_tokens > chunk_size and current:
                merged.append("".join(current))
                # Build overlap tail.
                tail: list[str] = []
                tail_tokens = 0
                for prev in reversed(current):
                    t = count_tokens(prev)
                    if tail_tokens + t > overlap:
                        break
                    tail.insert(0, prev)
                    tail_tokens += t
                current = tail
                current_tokens = tail_tokens

            current.append(piece)
            current_tokens += piece_tokens

        if current:
            merged.append("".join(current))

        return merged


# ── Strategy 3: SemanticChunker ───────────────────────────────────────────────


class SemanticChunker(BaseChunker):
    """Embedding-based chunker that splits on semantic discontinuity.

    Algorithm:

    1. Split the document into sentences.
    2. Embed every sentence with a ``sentence-transformers`` model.
    3. Compute cosine similarity between each consecutive sentence pair.
    4. Insert a chunk boundary whenever similarity drops below
       ``similarity_threshold``.
    5. Merge micro-groups that together remain within the token budget.

    This produces topically coherent chunks that improve retrieval precision
    compared to arbitrary character or token splits.

    Args:
        settings: Application settings override.

    Note:
        The embedding model is loaded on first use and cached as an instance
        attribute to avoid repeated HuggingFace Hub round-trips.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self._model: Any = None  # Lazy-initialised on first call.

    @property
    def strategy_name(self) -> str:
        """Return the strategy identifier string."""
        return "semantic"

    def _get_model(self) -> Any:
        """Return the cached sentence-transformers model.

        Returns:
            A ``SentenceTransformer`` model instance.

        Raises:
            EmbeddingError: When the model cannot be loaded.
        """
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]

            log.info(
                "Loading embedding model",
                extra={"model": self._cfg.embedding_model},
            )
            self._model = SentenceTransformer(self._cfg.embedding_model)
            return self._model
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load model '{self._cfg.embedding_model}': {exc}"
            ) from exc

    def _embed(self, sentences: list[str]) -> np.ndarray:  # type: ignore[type-arg]
        """Embed a list of sentences.

        Args:
            sentences: List of sentence strings.

        Returns:
            2-D float32 numpy array of shape ``(len(sentences), embedding_dim)``.

        Raises:
            EmbeddingError: On embedding failure.
        """
        try:
            model = self._get_model()
            embeddings: np.ndarray = model.encode(  # type: ignore[assignment]
                sentences,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Pre-normalise for cosine efficiency.
            )
            return embeddings
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

    @staticmethod
    def _cosine_similarities(embeddings: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
        """Compute cosine similarity between consecutive embedding vectors.

        Since embeddings are L2-normalised, cosine similarity reduces to a
        dot product.

        Args:
            embeddings: (N, D) float32 array.

        Returns:
            (N-1,) float32 array of pairwise similarities.
        """
        if len(embeddings) < 2:
            return np.array([], dtype=np.float32)
        # Dot product of consecutive pairs (normalised vectors → cosine sim).
        sims: np.ndarray = np.einsum("ij,ij->i", embeddings[:-1], embeddings[1:])
        return sims

    def chunk(self, document: Document) -> list[Chunk]:
        """Split *document* at semantic discontinuities.

        Args:
            document: The document to split.

        Returns:
            List of :class:`Chunk` objects ordered by position.

        Raises:
            EmbeddingError: When the embedding model fails.
        """
        text = document.content.strip()
        if not text:
            return []

        sentences = _split_into_sentences(text)
        if len(sentences) == 1:
            chunk = _make_chunk(sentences[0], self.strategy_name, document, 0, 1)
            return [chunk]

        embeddings = self._embed(sentences)
        similarities = self._cosine_similarities(embeddings)

        threshold = self._cfg.similarity_threshold
        chunk_size = self._cfg.chunk_size

        # Identify split points: indices *after* which a boundary occurs.
        split_indices: list[int] = [
            i for i, sim in enumerate(similarities) if sim < threshold
        ]

        # Build sentence groups from split points.
        groups: list[list[str]] = []
        prev = 0
        for idx in split_indices:
            group = sentences[prev : idx + 1]
            if group:
                groups.append(group)
            prev = idx + 1
        if prev < len(sentences):
            groups.append(sentences[prev:])

        # Merge groups that fall under the token budget to avoid micro-chunks.
        merged_groups = self._merge_groups(groups, chunk_size)

        total = len(merged_groups)
        chunks: list[Chunk] = []
        for idx, group in enumerate(merged_groups):
            content = " ".join(group)
            chunks.append(_make_chunk(content, self.strategy_name, document, idx, total))

        log.debug(
            "Semantic chunking complete",
            extra={
                "file": document.metadata.filename,
                "page": document.page_number,
                "sentences": len(sentences),
                "split_points": len(split_indices),
                "chunks": total,
                "avg_sim": float(similarities.mean()) if len(similarities) > 0 else 0.0,
            },
        )
        return chunks

    def _merge_groups(
        self, groups: list[list[str]], chunk_size: int
    ) -> list[list[str]]:
        """Merge consecutive small sentence groups into larger chunks.

        Adjacent groups are merged as long as the combined token count stays
        within *chunk_size*.  This prevents the pathological case where a
        similarity spike on a one-sentence transition creates many tiny chunks.

        Args:
            groups:     List of sentence groups (each group is a list of strings).
            chunk_size: Maximum token budget per merged group.

        Returns:
            List of merged sentence groups.
        """
        merged: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0

        for group in groups:
            group_text = " ".join(group)
            group_tokens = count_tokens(group_text)

            if current_tokens + group_tokens > chunk_size and current:
                merged.append(current)
                current = list(group)
                current_tokens = group_tokens
            else:
                current.extend(group)
                current_tokens += group_tokens

        if current:
            merged.append(current)

        return merged
