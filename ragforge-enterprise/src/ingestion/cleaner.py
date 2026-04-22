"""Text cleaning and normalisation for RAGForge Enterprise.

The :class:`DocumentCleaner` processes raw :class:`~src.ingestion.loader.Document`
instances extracted by the loader, producing cleaned documents enriched with a
``cleaning_stats`` field in their metadata.

Design decisions
----------------
- Tables and structured lists are **flagged** in the returned metadata—not
  removed—so that downstream modules (e.g. a table-aware chunker or a
  specialised table-QA retriever) can make an informed routing decision.
- Language detection is performed after cleaning so that artefacts such as
  page numbers or boilerplate strings don't skew the detector.
- All regex patterns are compiled once at module level for performance.

Typical usage::

    from src.ingestion.loader import DocumentLoader
    from src.ingestion.cleaner import DocumentCleaner

    loader  = DocumentLoader()
    cleaner = DocumentCleaner()

    docs         = loader.load_file(Path("report.pdf"))
    cleaned_docs = [cleaner.clean(doc) for doc in docs]
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from src.config.settings import get_settings
from src.ingestion.loader import Document, DocumentMetadata
from src.utils.logger import get_logger, log_exception

# ── Logging ───────────────────────────────────────────────────────────────────
_settings = get_settings()
log = get_logger(__name__, level=_settings.log_level)


# ── Custom Exceptions ─────────────────────────────────────────────────────────


class CleaningError(Exception):
    """Base exception for all text-cleaning failures."""


class LanguageDetectionError(CleaningError):
    """Raised when language detection cannot produce a result."""


# ── Compiled patterns ─────────────────────────────────────────────────────────

# Standalone page numbers: "1", "- 1 -", "Page 1", "Page 1 of 12", etc.
_RE_PAGE_NUMBER = re.compile(
    r"(?mi)^[\s\-–—]*(?:page\s*)?\d+\s*(?:of\s*\d+)?[\s\-–—]*$"
)

# Running headers / footers: repeated short lines (≤ 80 chars) occurring on
# every page.  We detect them as lines that appear verbatim ≥ 3 times.
# (Detection is done per-document in clean_batch; single-doc path skips this.)

# Excess whitespace inside a line (multiple spaces → single space).
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")

# More than two consecutive newlines → two newlines (paragraph break).
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")

# Soft hyphens inserted by PDF engines at line breaks.
_RE_SOFT_HYPHEN = re.compile(r"\xad")  # U+00AD

# Ligature expansion map for common Latin ligatures missed by unicodedata.
_LIGATURE_MAP: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}

# Simple heuristic patterns for structural element detection.
# Each pattern compiled individually with re.MULTILINE to satisfy Python 3.14+
# which forbids inline (?m) flags anywhere except the very start of a pattern.
_RE_TABLE_PIPE = re.compile(r"^.*(?:\|.*){2,}.*$", re.MULTILINE)
_RE_TABLE_NUMS = re.compile(r"^[\s\d.,]+$", re.MULTILINE)
_RE_TABLE_SEP  = re.compile(r"^\s*[-+]{3,}", re.MULTILINE)
_RE_LIST_ITEM  = re.compile(r"^\s*(?:[•●▪▸◦\-\*]|\d+[.)]) ", re.MULTILINE)


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class CleaningStats:
    """Statistics produced by the cleaning pass.

    Attributes:
        original_char_count: Character count before cleaning.
        cleaned_char_count:  Character count after cleaning.
        removed_chars:       ``original_char_count - cleaned_char_count``.
        has_tables:          ``True`` if table-like patterns were detected.
        has_lists:           ``True`` if list-item patterns were detected.
        detected_language:   ISO 639-1 language code (e.g. ``"en"``), or
                             ``"unknown"`` when detection fails.
        language_confidence: Approximate confidence score in ``[0, 1]``.
    """

    original_char_count: int = 0
    cleaned_char_count: int = 0
    removed_chars: int = 0
    has_tables: bool = False
    has_lists: bool = False
    detected_language: str = "unknown"
    language_confidence: float = 0.0


# ── Internal helpers ──────────────────────────────────────────────────────────


def _expand_ligatures(text: str) -> str:
    """Replace common Unicode ligatures with their ASCII equivalents.

    Args:
        text: Input string potentially containing ligatures.

    Returns:
        String with ligatures replaced.
    """
    for ligature, replacement in _LIGATURE_MAP.items():
        text = text.replace(ligature, replacement)
    return text


def _normalise_unicode(text: str) -> str:
    """Apply NFC normalisation and soft-hyphen removal.

    Args:
        text: Input string.

    Returns:
        NFC-normalised string without soft hyphens.
    """
    text = unicodedata.normalize("NFC", text)
    text = _RE_SOFT_HYPHEN.sub("", text)
    return text


def _remove_page_numbers(text: str) -> str:
    """Strip standalone page-number lines.

    Args:
        text: Input text.

    Returns:
        Text with isolated page-number lines removed.
    """
    return _RE_PAGE_NUMBER.sub("", text)


def _collapse_whitespace(text: str) -> str:
    """Normalise whitespace: multi-spaces to one, 3+ newlines to two.

    Args:
        text: Input text.

    Returns:
        Whitespace-collapsed text.
    """
    text = _RE_MULTI_SPACE.sub(" ", text)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def _detect_structures(text: str) -> tuple[bool, bool]:
    """Detect whether *text* contains table-like or list-item patterns.

    Args:
        text: Cleaned document text.

    Returns:
        A ``(has_tables, has_lists)`` tuple of booleans.
    """
    has_tables = bool(
        _RE_TABLE_PIPE.search(text)
        or _RE_TABLE_NUMS.search(text)
        or _RE_TABLE_SEP.search(text)
    )
    has_lists = bool(_RE_LIST_ITEM.search(text))
    return has_tables, has_lists


def _detect_language(text: str) -> tuple[str, float]:
    """Detect the language of *text* using ``langdetect``.

    Args:
        text: The text sample to classify.

    Returns:
        A ``(language_code, confidence)`` tuple.  Returns
        ``("unknown", 0.0)`` when detection fails or ``langdetect`` is
        unavailable.

    Raises:
        LanguageDetectionError: Wrapped and logged internally; never
        propagated so that cleaning continues even when langdetect fails.
    """
    try:
        from langdetect import DetectorFactory, detect_langs  # type: ignore[import]

        # Seed for reproducibility.
        DetectorFactory.seed = 0
        results = detect_langs(text[:2000])  # Use first 2000 chars for speed.
        if results:
            top = results[0]
            return str(top.lang), float(top.prob)
        return "unknown", 0.0
    except Exception as exc:
        # langdetect raises LangDetectException for very short texts, etc.
        log_exception(log, "Language detection failed", exc)
        return "unknown", 0.0


def _remove_repeated_headers_footers(texts: list[str]) -> list[str]:
    """Remove lines that appear verbatim across ≥ 70 % of pages.

    This heuristic targets running headers and footers that are identical
    (or nearly so) on every page and therefore add no information.

    Args:
        texts: List of per-page raw strings (one per page).

    Returns:
        Cleaned list of per-page strings with boilerplate lines removed.
    """
    if len(texts) < 3:
        return texts  # Not enough pages to detect repetition.

    from collections import Counter  # pylint: disable=import-outside-toplevel

    line_counts: Counter[str] = Counter()
    for text in texts:
        # Consider only short lines; long lines are unlikely to be headers.
        for line in text.splitlines():
            stripped = line.strip()
            if 2 < len(stripped) <= 120:
                line_counts[stripped] += 1

    threshold = max(3, int(len(texts) * 0.7))
    boilerplate: set[str] = {line for line, cnt in line_counts.items() if cnt >= threshold}

    if not boilerplate:
        return texts

    log.debug(
        "Removing boilerplate lines",
        extra={"count": len(boilerplate), "examples": list(boilerplate)[:5]},
    )

    cleaned: list[str] = []
    for text in texts:
        filtered_lines = [
            ln for ln in text.splitlines() if ln.strip() not in boilerplate
        ]
        cleaned.append("\n".join(filtered_lines))
    return cleaned


# ── Public API ────────────────────────────────────────────────────────────────


class DocumentCleaner:
    """Clean and normalise :class:`~src.ingestion.loader.Document` objects.

    Each cleaning step is applied in a fixed, deterministic order:

    1. Unicode normalisation (NFC + ligature expansion + soft-hyphen removal).
    2. Page-number line removal.
    3. Whitespace normalisation.
    4. Structural detection (tables, lists) — **non-destructive**.
    5. Language detection.

    Running-header/footer removal requires multiple pages and is handled by
    :meth:`clean_batch`.

    Example:
        >>> cleaner = DocumentCleaner()
        >>> cleaned = cleaner.clean(doc)
        >>> cleaned.metadata.extra["cleaning_stats"]["detected_language"]
        'en'
    """

    def clean(self, document: Document) -> Document:
        """Apply the full cleaning pipeline to a single document.

        Args:
            document: A :class:`~src.ingestion.loader.Document` as returned
                      by the loader.

        Returns:
            A new :class:`~src.ingestion.loader.Document` with cleaned
            ``content`` and ``cleaning_stats`` injected into
            ``metadata.extra``.

        Raises:
            CleaningError: On unexpected errors within the cleaning pipeline.
        """
        try:
            return self._clean_text(document)
        except CleaningError:
            raise
        except Exception as exc:
            raise CleaningError(
                f"Unexpected error cleaning {document.source_path}: {exc}"
            ) from exc

    def clean_batch(self, documents: list[Document]) -> list[Document]:
        """Clean a batch of documents that originate from the **same PDF**.

        Compared to calling :meth:`clean` in a loop, this method additionally
        removes repeated header/footer lines detected across pages.

        Args:
            documents: Ordered list of :class:`~src.ingestion.loader.Document`
                       objects (one per page of the same PDF).

        Returns:
            Cleaned documents in the same order.
        """
        if not documents:
            return []

        # Pass 1 – individual cleaning (unicode, page numbers, whitespace).
        individually_cleaned = [self._clean_text(doc) for doc in documents]

        # Pass 2 – cross-page boilerplate removal.
        texts = [doc.content for doc in individually_cleaned]
        deduped_texts = _remove_repeated_headers_footers(texts)

        result: list[Document] = []
        for doc, new_text in zip(individually_cleaned, deduped_texts):
            if new_text != doc.content:
                # Re-run structural detection + stats after boilerplate removal.
                has_tables, has_lists = _detect_structures(new_text)
                lang, conf = _detect_language(new_text)
                stats = CleaningStats(
                    original_char_count=int(doc.metadata.extra.get("original_char_count", len(new_text))),
                    cleaned_char_count=len(new_text),
                    removed_chars=int(doc.metadata.extra.get("original_char_count", len(new_text))) - len(new_text),
                    has_tables=has_tables,
                    has_lists=has_lists,
                    detected_language=lang,
                    language_confidence=conf,
                )
                updated_extra = {**doc.metadata.extra, "cleaning_stats": _stats_to_dict(stats)}
                updated_meta = DocumentMetadata(
                    filename=doc.metadata.filename,
                    page_count=doc.metadata.page_count,
                    page_number=doc.metadata.page_number,
                    creation_date=doc.metadata.creation_date,
                    loader_backend=doc.metadata.loader_backend,
                    extra=updated_extra,
                )
                result.append(
                    Document(
                        content=new_text,
                        metadata=updated_meta,
                        source_path=doc.source_path,
                        page_number=doc.page_number,
                    )
                )
            else:
                result.append(doc)

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _clean_text(self, document: Document) -> Document:
        """Core cleaning pipeline for a single document.

        Args:
            document: The document to clean.

        Returns:
            A new :class:`~src.ingestion.loader.Document` with cleaned content.
        """
        original = document.content
        text = original

        # Step 1 – Unicode normalisation.
        text = _normalise_unicode(text)
        text = _expand_ligatures(text)

        # Step 2 – Page-number removal.
        text = _remove_page_numbers(text)

        # Step 3 – Whitespace normalisation.
        text = _collapse_whitespace(text)

        # Step 4 – Structural detection (flags only).
        has_tables, has_lists = _detect_structures(text)

        # Step 5 – Language detection.
        lang, conf = _detect_language(text) if text.strip() else ("unknown", 0.0)

        stats = CleaningStats(
            original_char_count=len(original),
            cleaned_char_count=len(text),
            removed_chars=len(original) - len(text),
            has_tables=has_tables,
            has_lists=has_lists,
            detected_language=lang,
            language_confidence=conf,
        )

        log.debug(
            "Document cleaned",
            extra={
                "file": document.metadata.filename,
                "page": document.page_number,
                "original_chars": stats.original_char_count,
                "cleaned_chars": stats.cleaned_char_count,
                "language": lang,
                "has_tables": has_tables,
                "has_lists": has_lists,
            },
        )

        updated_extra: dict[str, Any] = {
            **document.metadata.extra,
            "original_char_count": str(stats.original_char_count),
            "cleaning_stats": _stats_to_dict(stats),
        }
        updated_meta = DocumentMetadata(
            filename=document.metadata.filename,
            page_count=document.metadata.page_count,
            page_number=document.metadata.page_number,
            creation_date=document.metadata.creation_date,
            loader_backend=document.metadata.loader_backend,
            extra=updated_extra,
        )
        return Document(
            content=text,
            metadata=updated_meta,
            source_path=document.source_path,
            page_number=document.page_number,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _stats_to_dict(stats: CleaningStats) -> dict[str, Any]:
    """Serialise a :class:`CleaningStats` to a plain dict for metadata storage.

    Args:
        stats: The stats object to serialise.

    Returns:
        A JSON-serialisable dictionary.
    """
    return {
        "original_char_count": stats.original_char_count,
        "cleaned_char_count": stats.cleaned_char_count,
        "removed_chars": stats.removed_chars,
        "has_tables": stats.has_tables,
        "has_lists": stats.has_lists,
        "detected_language": stats.detected_language,
        "language_confidence": stats.language_confidence,
    }
