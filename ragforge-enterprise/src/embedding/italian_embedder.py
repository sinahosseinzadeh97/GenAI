"""Italian Legal Embedder — RAGForge Italia Phase 2.1.

Wraps ``intfloat/multilingual-e5-large-instruct`` (primary) with
``paraphrase-multilingual-mpnet-base-v2`` as a drop-in fallback and exposes
a ``CrossEncoderReranker`` backed by ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1``
trained on the multilingual mMARCO corpus (which includes Italian).

Design decisions
----------------
- The Italian E5-instruct model requires a two-line Italian instruction prefix
  at query time and a ``"Passage: "`` prefix at indexing time.  These are baked
  into :meth:`embed_query` / :meth:`embed_passage` so callers never have to
  remember the protocol.
- Model loading is protected by a double-checked lock so the heavy weights are
  loaded only once per process even in a multithreaded FastAPI context.
- The fallback embedder is instantiated lazily only if the primary model fails
  to load, keeping the happy path free of unnecessary overhead.
- Cross-encoder reranking is a separate concern; the ``CrossEncoderReranker``
  is therefore its own class rather than a method on the embedder.

Typical usage::

    from src.embedding.italian_embedder import ItalianLegalEmbedder, CrossEncoderReranker

    embedder = ItalianLegalEmbedder()

    # At index time:
    passage_vec = embedder.embed_passage("L'art. 2043 c.c. prevede ...")

    # At query time:
    query_vec = embedder.embed_query("Quali sono i requisiti della responsabilità aquiliana?")

    # Rerank a shortlist:
    reranker = CrossEncoderReranker()
    ranked   = reranker.rerank(query, [candidate1, candidate2, candidate3])
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Any

from src.embedding.base import BaseEmbedder, EmbeddingError
from src.utils.logger import get_logger, log_exception

log = get_logger(__name__)

# ── Model identifiers ─────────────────────────────────────────────────────────

_PRIMARY_MODEL: str = "intfloat/multilingual-e5-large-instruct"
_FALLBACK_MODEL: str = "paraphrase-multilingual-mpnet-base-v2"
_RERANKER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# ── Instruction prefixes ──────────────────────────────────────────────────────

_ITALIAN_QUERY_INSTRUCTION: str = (
    "Instruct: Recupera documenti giuridici italiani pertinenti\nQuery: "
)
_PASSAGE_PREFIX: str = "Passage: "

# ── Thread-safe model singletons ─────────────────────────────────────────────

_st_lock: Lock = Lock()
_st_cache: dict[str, Any] = {}  # model_name → SentenceTransformer

_ce_lock: Lock = Lock()
_ce_cache: dict[str, Any] = {}  # model_name → CrossEncoder


def _load_sentence_transformer(model_name: str) -> Any:
    """Return a cached ``SentenceTransformer`` for *model_name*.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        A ``SentenceTransformer`` instance.

    Raises:
        EmbeddingError: When the model cannot be loaded.
    """
    if model_name in _st_cache:
        return _st_cache[model_name]

    with _st_lock:
        if model_name in _st_cache:
            return _st_cache[model_name]

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            log.info("Loading SentenceTransformer", extra={"model": model_name})
            t0 = time.perf_counter()
            model = SentenceTransformer(model_name)
            elapsed = time.perf_counter() - t0
            log.info(
                "SentenceTransformer loaded",
                extra={"model": model_name, "load_time_seconds": round(elapsed, 3)},
            )
            _st_cache[model_name] = model
            return model
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load SentenceTransformer '{model_name}': {exc}"
            ) from exc


def _load_cross_encoder(model_name: str) -> Any:
    """Return a cached ``CrossEncoder`` for *model_name*.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        A ``CrossEncoder`` instance.

    Raises:
        EmbeddingError: When the model cannot be loaded.
    """
    if model_name in _ce_cache:
        return _ce_cache[model_name]

    with _ce_lock:
        if model_name in _ce_cache:
            return _ce_cache[model_name]

        try:
            from sentence_transformers.cross_encoder import CrossEncoder  # type: ignore[import-untyped]

            log.info("Loading CrossEncoder", extra={"model": model_name})
            t0 = time.perf_counter()
            model = CrossEncoder(model_name)
            elapsed = time.perf_counter() - t0
            log.info(
                "CrossEncoder loaded",
                extra={"model": model_name, "load_time_seconds": round(elapsed, 3)},
            )
            _ce_cache[model_name] = model
            return model
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load CrossEncoder '{model_name}': {exc}"
            ) from exc


# ── ItalianLegalEmbedder ──────────────────────────────────────────────────────


class ItalianLegalEmbedder(BaseEmbedder):
    """Multilingual embedder optimised for Italian legal text.

    **Primary model**: ``intfloat/multilingual-e5-large-instruct`` (1024-dim)
    Uses the E5-instruct Italian query prefix at retrieval time and a neutral
    ``"Passage: "`` prefix at indexing time.

    **Fallback model**: ``paraphrase-multilingual-mpnet-base-v2`` (768-dim) is
    used transparently if the primary model fails to load (e.g. no GPU memory
    or network unavailable at startup).

    Args:
        model_name:   Override the primary model (defaults to
                      ``intfloat/multilingual-e5-large-instruct``).
        fallback_model: Override the fallback model.
        normalize:    L2-normalise output vectors (default ``True``).
        batch_size:   Mini-batch size for :meth:`embed_batch`.

    Example::

        embedder = ItalianLegalEmbedder()
        q_vec = embedder.embed_query("responsabilità contrattuale art. 1218 c.c.")
        p_vec = embedder.embed_passage("Art. 1218 c.c. Il debitore che non esegue ...")
    """

    QUERY_PREFIX: str = _ITALIAN_QUERY_INSTRUCTION
    PASSAGE_PREFIX: str = _PASSAGE_PREFIX

    def __init__(
        self,
        model_name: str = _PRIMARY_MODEL,
        fallback_model: str = _FALLBACK_MODEL,
        normalize: bool = True,
        batch_size: int = 16,
    ) -> None:
        super().__init__(normalize=normalize)
        self._primary_model_name: str = model_name
        self._fallback_model_name: str = fallback_model
        self._batch_size: int = batch_size
        self._active_model_name: str | None = None  # Resolved on first load.
        self._dim: int | None = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """Return the name of the *active* model (primary or fallback).

        Returns:
            Model identifier string.
        """
        return self._active_model_name or self._primary_model_name

    @property
    def dimension(self) -> int:
        """Return the embedding dimension of the active model.

        Triggers a model load on first access.

        Returns:
            Integer vector dimension (1024 for E5-large, 768 for mpnet).
        """
        if self._dim is None:
            model = self._get_model()
            self._dim = model.get_sentence_embedding_dimension()
        return self._dim  # type: ignore[return-value]

    # ── Italian-specific public API ───────────────────────────────────────────

    def embed_query(self, text: str) -> list[float]:
        """Embed an Italian legal *query* with the E5-instruct prefix.

        The prefix instructs the model to retrieve relevant Italian legal
        documents, exploiting its instruction-following training signal.

        Args:
            text: Raw Italian legal query string.

        Returns:
            L2-normalised (if enabled) embedding vector.

        Raises:
            EmbeddingError: If embedding fails even after fallback.
        """
        return self._embed(self.QUERY_PREFIX + text)

    def embed_passage(self, text: str) -> list[float]:
        """Embed an Italian legal *passage* for indexing.

        Args:
            text: Document chunk to index.

        Returns:
            L2-normalised (if enabled) embedding vector.

        Raises:
            EmbeddingError: If embedding fails even after fallback.
        """
        return self._embed(self.PASSAGE_PREFIX + text)

    # ── BaseEmbedder interface ────────────────────────────────────────────────

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text (passage mode, no prefix injection).

        For explicit control over prefix use :meth:`embed_query` or
        :meth:`embed_passage`.  This method satisfies the :class:`BaseEmbedder`
        contract and defaults to passage semantics.

        Args:
            text: Text to embed.

        Returns:
            Float embedding vector.
        """
        return self.embed_passage(text)

    def embed_batch(  # type: ignore[override]
        self,
        texts: list[str],
        batch_size: int | None = None,
        mode: str = "passage",
    ) -> list[list[float]]:
        """Embed a list of Italian legal texts in mini-batches.

        Args:
            texts:      Texts to embed.
            batch_size: Mini-batch size (default: instance ``_batch_size``).
            mode:       ``"passage"`` or ``"query"`` — determines prefix.

        Returns:
            List of float vectors, one per input text.

        Raises:
            EmbeddingError: On model failure.
        """
        if not texts:
            return []

        bs = batch_size or self._batch_size
        prefix = self.QUERY_PREFIX if mode == "query" else self.PASSAGE_PREFIX
        prefixed = [prefix + t for t in texts]

        log.info(
            "ItalianLegalEmbedder batch started",
            extra={
                "model": self.model_name,
                "total_texts": len(texts),
                "batch_size": bs,
                "mode": mode,
            },
        )

        t_start = time.perf_counter()
        results: list[list[float]] = []

        try:
            model = self._get_model()

            for i in range(0, len(prefixed), bs):
                batch = prefixed[i : i + bs]
                raw = model.encode(
                    batch,
                    batch_size=bs,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
                for vec in raw:
                    results.append(self._maybe_normalize(vec.tolist()))

            elapsed = time.perf_counter() - t_start
            throughput = len(texts) / elapsed if elapsed > 0 else float("inf")
            log.info(
                "ItalianLegalEmbedder batch complete",
                extra={
                    "model": self.model_name,
                    "total_texts": len(texts),
                    "duration_seconds": round(elapsed, 4),
                    "throughput_per_sec": round(throughput, 2),
                },
            )
            return results

        except EmbeddingError:
            raise
        except Exception as exc:
            log_exception(log, "ItalianLegalEmbedder batch failed", exc)
            raise EmbeddingError(f"Italian batch embedding failed: {exc}") from exc

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_model(self) -> Any:
        """Return the active SentenceTransformer, falling back if needed.

        Returns:
            A loaded ``SentenceTransformer`` instance.
        """
        if self._active_model_name is not None:
            return _load_sentence_transformer(self._active_model_name)

        try:
            model = _load_sentence_transformer(self._primary_model_name)
            self._active_model_name = self._primary_model_name
            return model
        except EmbeddingError:
            log.warning(
                "Primary Italian model unavailable, switching to fallback",
                extra={
                    "primary": self._primary_model_name,
                    "fallback": self._fallback_model_name,
                },
            )
            model = _load_sentence_transformer(self._fallback_model_name)
            self._active_model_name = self._fallback_model_name
            self._dim = None  # Invalidate cached dim — fallback has different size.
            return model

    def _embed(self, prefixed_text: str) -> list[float]:
        """Encode a single pre-prefixed string.

        Args:
            prefixed_text: Text already decorated with the correct prefix.

        Returns:
            (Optionally normalised) embedding vector.
        """
        try:
            import numpy as np  # noqa: PLC0415

            model = self._get_model()
            raw = model.encode(
                prefixed_text,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            raw = np.asarray(raw)
            if raw.ndim == 2:
                raw = raw[0]
            return self._maybe_normalize(raw.tolist())
        except EmbeddingError:
            raise
        except Exception as exc:
            log_exception(log, "ItalianLegalEmbedder _embed failed", exc)
            raise EmbeddingError(f"Italian embedding failed: {exc}") from exc


# ── CrossEncoderReranker ──────────────────────────────────────────────────────


class CrossEncoderReranker:
    """Cross-encoder reranker for Italian legal retrieval.

    Uses ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1``, a compact but
    high-quality cross-encoder trained on the multilingual mMARCO dataset
    which includes Italian judgments and statutes.

    The reranker scores ``(query, passage)`` pairs and returns candidates
    sorted by descending relevance, making it ideal as a second-stage ranker
    on top of a dense bi-encoder shortlist.

    Args:
        model_name: Override the cross-encoder model.
        top_k:      Maximum number of passages to return after reranking.

    Example::

        reranker = CrossEncoderReranker(top_k=5)
        ranked = reranker.rerank(
            query="responsabilità medica",
            passages=["Art. 2043 c.c. ...", "Cass. Civ. n. 123 ..."],
        )
        # ranked = [("Cass. Civ. n. 123 ...", 0.87), ("Art. 2043 c.c. ...", 0.73)]
    """

    def __init__(
        self,
        model_name: str = _RERANKER_MODEL,
        top_k: int = 10,
    ) -> None:
        self._model_name: str = model_name
        self.top_k: int = top_k

    @property
    def model_name(self) -> str:
        """Return the cross-encoder model identifier."""
        return self._model_name

    def rerank(
        self,
        query: str,
        passages: list[str],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Score and rerank *passages* for *query*.

        Args:
            query:    Italian legal query string.
            passages: List of candidate passage strings to rerank.
            top_k:    Return at most this many results (overrides instance default).

        Returns:
            List of ``(passage, score)`` tuples sorted by descending relevance.
            Scores are raw cross-encoder logits (higher is more relevant).

        Raises:
            EmbeddingError: If the cross-encoder model cannot be loaded.
        """
        if not passages:
            return []

        k = top_k if top_k is not None else self.top_k
        model = _load_cross_encoder(self._model_name)
        pairs = [(query, p) for p in passages]

        t0 = time.perf_counter()
        try:
            scores: list[float] = model.predict(pairs).tolist()
        except Exception as exc:
            log_exception(log, "CrossEncoderReranker.predict failed", exc)
            raise EmbeddingError(f"Cross-encoder reranking failed: {exc}") from exc

        elapsed = time.perf_counter() - t0
        ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)[:k]

        log.info(
            "CrossEncoderReranker complete",
            extra={
                "model": self._model_name,
                "num_passages": len(passages),
                "top_k": k,
                "duration_seconds": round(elapsed, 4),
                "top_score": round(ranked[0][1], 4) if ranked else None,
            },
        )
        return ranked

    def rerank_with_metadata(
        self,
        query: str,
        passages: list[dict[str, Any]],
        text_key: str = "content",
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank a list of passage *dicts* preserving all metadata fields.

        This is the production-ready variant: callers pass original retrieval
        result dicts (e.g. from Qdrant), and get back the same dicts enriched
        with a ``rerank_score`` field, sorted by descending relevance.

        Args:
            query:    Italian legal query string.
            passages: List of dicts, each containing a ``text_key`` field.
            text_key: Key in each dict that holds the passage text.
            top_k:    Return at most this many results.

        Returns:
            Re-ordered list of dicts, each with an added ``rerank_score`` field.

        Raises:
            EmbeddingError: On model failure.
            KeyError: If a passage dict is missing *text_key*.
        """
        texts = [p[text_key] for p in passages]
        ranked_pairs = self.rerank(query, texts, top_k=top_k)

        # Build a text → (score, original_dict) lookup.  If the same text appears
        # more than once, we process them in order so each gets its own score.
        text_to_original: dict[int, dict[str, Any]] = {i: p for i, p in enumerate(passages)}
        text_to_score: dict[str, float] = {text: score for text, score in ranked_pairs}

        result: list[dict[str, Any]] = []
        for text, score in ranked_pairs:
            # Find the first passage dict whose text matches.
            for idx, passage in text_to_original.items():
                if passage[text_key] == text:
                    enriched = {**passage, "rerank_score": score}
                    result.append(enriched)
                    del text_to_original[idx]
                    break

        _ = text_to_score  # Suppress unused-variable lint.
        return result
