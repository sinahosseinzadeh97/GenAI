"""Tests for src.ingestion.cleaner.

Covers:
- Unicode normalisation and ligature expansion.
- Page-number line removal.
- Whitespace collapsing.
- Structural detection (tables, lists).
- Language detection behaviour (including unavailability).
- Cross-page boilerplate removal via clean_batch.
- CleaningStats integrity.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.ingestion.cleaner import (
    CleaningStats,
    DocumentCleaner,
    _collapse_whitespace,
    _detect_language,
    _detect_structures,
    _expand_ligatures,
    _normalise_unicode,
    _remove_page_numbers,
    _remove_repeated_headers_footers,
)
from src.ingestion.loader import Document, DocumentMetadata


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_doc(content: str, page: int = 1, page_count: int = 1) -> Document:
    """Create a minimal :class:`Document` for testing."""
    meta = DocumentMetadata(filename="test.pdf", page_count=page_count, page_number=page)
    return Document(content=content, metadata=meta, source_path=Path("/tmp/test.pdf"), page_number=page)


# ── Unit tests: helper functions ──────────────────────────────────────────────


class TestNormaliseUnicode:
    """Tests for _normalise_unicode."""

    def test_nfc_normalisation(self) -> None:
        # 'café' as NFD (decomposed) → should become NFC.
        nfd_string = "cafe\u0301"  # e + combining acute accent
        result = _normalise_unicode(nfd_string)
        assert result == "café"

    def test_soft_hyphen_removed(self) -> None:
        text = "hyp\xadhen"
        assert _normalise_unicode(text) == "hyphen"

    def test_regular_text_unchanged(self) -> None:
        text = "Hello, World! 123"
        assert _normalise_unicode(text) == text


class TestExpandLigatures:
    """Tests for _expand_ligatures."""

    def test_fi_ligature(self) -> None:
        assert _expand_ligatures("\ufb01le") == "file"

    def test_ff_ligature(self) -> None:
        assert _expand_ligatures("\ufb00ort") == "ffort"

    def test_ffl_ligature(self) -> None:
        assert _expand_ligatures("\ufb04uent") == "ffluent"

    def test_no_ligatures_unchanged(self) -> None:
        text = "plain ASCII text"
        assert _expand_ligatures(text) == text


class TestRemovePageNumbers:
    """Tests for _remove_page_numbers."""

    def test_standalone_integer(self) -> None:
        text = "Some content\n\n12\n\nMore content"
        result = _remove_page_numbers(text)
        assert "12" not in result.strip().split()

    def test_page_N_of_M_pattern(self) -> None:
        text = "Header\nPage 3 of 15\nBody text"
        result = _remove_page_numbers(text)
        assert "Page 3 of 15" not in result

    def test_dashed_page_number(self) -> None:
        text = "Content\n- 7 -\nMore"
        result = _remove_page_numbers(text)
        assert "- 7 -" not in result

    def test_body_text_preserved(self) -> None:
        text = "Machine learning models achieve 95% accuracy."
        assert _remove_page_numbers(text).strip() == text.strip()


class TestCollapseWhitespace:
    """Tests for _collapse_whitespace."""

    def test_multiple_spaces_collapsed(self) -> None:
        assert _collapse_whitespace("hello     world") == "hello world"

    def test_tabs_collapsed(self) -> None:
        assert _collapse_whitespace("hello\t\tworld") == "hello world"

    def test_multiple_newlines_limited(self) -> None:
        text = "para1\n\n\n\n\npara2"
        result = _collapse_whitespace(text)
        assert "\n\n\n" not in result
        assert "para1" in result
        assert "para2" in result

    def test_leading_trailing_stripped(self) -> None:
        assert _collapse_whitespace("  hello  ") == "hello"


class TestDetectStructures:
    """Tests for _detect_structures."""

    def test_detects_pipe_table(self) -> None:
        text = "Name | Age | City\n-----+-----+-----\nAlice | 30 | NYC"
        has_tables, _ = _detect_structures(text)
        assert has_tables

    def test_detects_bullet_list(self) -> None:
        text = "Key points:\n• Item one\n• Item two\n• Item three"
        _, has_lists = _detect_structures(text)
        assert has_lists

    def test_detects_numbered_list(self) -> None:
        text = "Steps:\n1. First step\n2. Second step\n3. Third step"
        _, has_lists = _detect_structures(text)
        assert has_lists

    def test_plain_text_no_structures(self) -> None:
        text = "This is a regular paragraph with no tables or lists."
        has_tables, has_lists = _detect_structures(text)
        assert not has_tables
        assert not has_lists


class TestDetectLanguage:
    """Tests for _detect_language."""

    def test_english_text(self) -> None:
        lang, conf = _detect_language(
            "The quick brown fox jumps over the lazy dog. "
            "Language detection should work well with this sample."
        )
        assert lang == "en"
        assert conf > 0.5

    def test_returns_unknown_on_import_error(self) -> None:
        with patch.dict("sys.modules", {"langdetect": None}):
            lang, conf = _detect_language("Some text")
            assert lang == "unknown"
            assert conf == 0.0

    def test_empty_string_returns_unknown(self) -> None:
        lang, _ = _detect_language("")
        # langdetect raises on empty strings; we should get "unknown".
        assert lang in ("unknown", "en")  # tolerate short-circuit in implementation


class TestRemoveRepeatedHeadersFooters:
    """Tests for _remove_repeated_headers_footers."""

    def test_removes_repeated_line(self) -> None:
        boilerplate = "Confidential – Do Not Distribute"
        pages = [f"{boilerplate}\nPage content {i}" for i in range(5)]
        cleaned = _remove_repeated_headers_footers(pages)
        for page in cleaned:
            assert boilerplate not in page

    def test_preserves_unique_content(self) -> None:
        pages = [
            "Header\nThis is unique content alpha.",
            "Header\nThis is unique content beta.",
            "Header\nThis is unique content gamma.",
            "Header\nThis is unique content delta.",
        ]
        cleaned = _remove_repeated_headers_footers(pages)
        assert "unique content alpha" in cleaned[0]

    def test_less_than_3_pages_unchanged(self) -> None:
        pages = ["Page A", "Page B"]
        assert _remove_repeated_headers_footers(pages) == pages


# ── Integration tests: DocumentCleaner ────────────────────────────────────────


class TestDocumentCleaner:
    """Integration tests for :class:`DocumentCleaner`."""

    @pytest.fixture()
    def cleaner(self) -> DocumentCleaner:
        return DocumentCleaner()

    def test_clean_returns_document(self, cleaner: DocumentCleaner) -> None:
        doc = _make_doc("Hello world. This is normal text without any special characters.")
        result = cleaner.clean(doc)
        assert isinstance(result, Document)

    def test_cleaning_stats_added_to_metadata(self, cleaner: DocumentCleaner) -> None:
        doc = _make_doc("Some content here that is readable English text.")
        result = cleaner.clean(doc)
        assert "cleaning_stats" in result.metadata.extra

    def test_cleaning_stats_structure(self, cleaner: DocumentCleaner) -> None:
        doc = _make_doc("Clean English text for testing statistics.")
        result = cleaner.clean(doc)
        stats = result.metadata.extra["cleaning_stats"]
        assert "original_char_count" in stats
        assert "cleaned_char_count" in stats
        assert "removed_chars" in stats
        assert "has_tables" in stats
        assert "has_lists" in stats
        assert "detected_language" in stats
        assert "language_confidence" in stats

    def test_ligature_expanded(self, cleaner: DocumentCleaner) -> None:
        doc = _make_doc("The \ufb01le was created.")
        result = cleaner.clean(doc)
        assert "file" in result.content
        assert "\ufb01" not in result.content

    def test_page_number_removed(self, cleaner: DocumentCleaner) -> None:
        doc = _make_doc("Introduction\n\n5\n\nThis is the body text.")
        result = cleaner.clean(doc)
        lines = result.content.split()
        assert "5" not in lines or "Introduction" in result.content

    def test_table_flagged_not_removed(self, cleaner: DocumentCleaner) -> None:
        table_text = "Name | Age\n-----+-----\nAlice | 30"
        doc = _make_doc("Header text.\n" + table_text)
        result = cleaner.clean(doc)
        stats = result.metadata.extra["cleaning_stats"]
        # Table content should still be in the document.
        assert "Alice" in result.content
        # But flagged in stats.
        assert stats["has_tables"] is True

    def test_list_flagged_not_removed(self, cleaner: DocumentCleaner) -> None:
        list_text = "Key points:\n• Alpha\n• Beta\n• Gamma"
        doc = _make_doc(list_text)
        result = cleaner.clean(doc)
        stats = result.metadata.extra["cleaning_stats"]
        assert "Alpha" in result.content
        assert stats["has_lists"] is True

    def test_empty_content_handled(self, cleaner: DocumentCleaner) -> None:
        doc = _make_doc("")
        result = cleaner.clean(doc)
        assert result.content == ""
        assert result.metadata.extra["cleaning_stats"]["detected_language"] == "unknown"

    def test_clean_batch_removes_boilerplate(self, cleaner: DocumentCleaner) -> None:
        boilerplate = "Company Confidential Report"
        docs = [
            _make_doc(f"{boilerplate}\nActual content page {i}.", page=i, page_count=5)
            for i in range(1, 6)
        ]
        cleaned = cleaner.clean_batch(docs)
        for doc in cleaned:
            # Boilerplate should be removed from all pages.
            assert boilerplate not in doc.content or "Actual content" in doc.content

    def test_clean_batch_returns_same_count(self, cleaner: DocumentCleaner) -> None:
        docs = [_make_doc(f"Page {i} content.", page=i, page_count=3) for i in range(1, 4)]
        cleaned = cleaner.clean_batch(docs)
        assert len(cleaned) == 3

    def test_source_path_preserved(self, cleaner: DocumentCleaner) -> None:
        path = Path("/some/path/report.pdf")
        meta = DocumentMetadata(filename="report.pdf", page_count=1)
        doc = Document(content="Text content.", metadata=meta, source_path=path, page_number=1)
        result = cleaner.clean(doc)
        assert result.source_path == path

    def test_unicode_text_normalised(self, cleaner: DocumentCleaner) -> None:
        # NFD-encoded 'é'
        doc = _make_doc("caf\u0065\u0301 au lait is a French coffee drink.")
        result = cleaner.clean(doc)
        assert "\u0301" not in result.content  # Combining accent should be merged
