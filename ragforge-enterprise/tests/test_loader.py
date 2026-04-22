"""Tests for src.ingestion.loader.

Covers:
- Successful single-file load (mocked).
- Batch directory loading.
- Graceful handling of corrupted PDFs.
- Metadata extraction.
- Empty page handling.
- DirectoryNotFoundError for missing paths.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.loader import (
    DirectoryNotFoundError,
    Document,
    DocumentLoader,
    DocumentMetadata,
    PDFCorruptedError,
    _parse_pdf_creation_date,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def loader() -> DocumentLoader:
    """Return a :class:`DocumentLoader` with fail_fast disabled."""
    return DocumentLoader(fail_fast=False)


@pytest.fixture()
def fail_fast_loader() -> DocumentLoader:
    """Return a :class:`DocumentLoader` with fail_fast enabled."""
    return DocumentLoader(fail_fast=True)


def _make_mock_pypdf_reader(pages: list[str], metadata: dict[str, str] | None = None) -> MagicMock:
    """Build a mock ``pypdf.PdfReader`` object.

    Args:
        pages:    List of per-page text strings.
        metadata: Optional metadata dict to attach.

    Returns:
        A ``MagicMock`` mimicking ``pypdf.PdfReader``.
    """
    reader = MagicMock()
    reader.metadata = metadata or {"/CreationDate": "D:20240101120000"}
    mock_pages = []
    for text in pages:
        page = MagicMock()
        page.extract_text.return_value = text
        mock_pages.append(page)
    reader.pages = mock_pages
    return reader


# ── Unit tests: _parse_pdf_creation_date ─────────────────────────────────────


class TestParsePdfCreationDate:
    """Tests for the PDF date-parsing helper."""

    def test_full_datetime_string(self) -> None:
        dt = _parse_pdf_creation_date("D:20230615143022+02'00'")
        assert dt is not None
        assert dt.year == 2023
        assert dt.month == 6
        assert dt.day == 15

    def test_date_only_string(self) -> None:
        dt = _parse_pdf_creation_date("D:20200101")
        assert dt is not None
        assert dt.year == 2020

    def test_none_input(self) -> None:
        assert _parse_pdf_creation_date(None) is None

    def test_empty_string(self) -> None:
        assert _parse_pdf_creation_date("") is None

    def test_unparseable_string(self) -> None:
        assert _parse_pdf_creation_date("not-a-date") is None


# ── Unit tests: DocumentLoader ────────────────────────────────────────────────


class TestDocumentLoaderSingleFile:
    """Tests for :meth:`DocumentLoader.load_file`."""

    @patch("src.ingestion.loader._extract_with_pypdf")
    def test_loads_single_page_pdf(self, mock_extract: MagicMock, loader: DocumentLoader, tmp_path: Path) -> None:
        """A single-page PDF with sufficient text loads correctly."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        mock_extract.return_value = [
            ("Hello world. This is a test document with enough text.", {"/CreationDate": "D:20240101120000"})
        ]

        docs = loader.load_file(pdf_path)

        assert len(docs) == 1
        assert "Hello world" in docs[0].content
        assert docs[0].page_number == 1
        assert docs[0].metadata.page_count == 1
        assert docs[0].metadata.filename == "test.pdf"

    @patch("src.ingestion.loader._extract_with_pypdf")
    def test_multipage_pdf(self, mock_extract: MagicMock, loader: DocumentLoader, tmp_path: Path) -> None:
        """A multi-page PDF returns one Document per page."""
        pdf_path = tmp_path / "multi.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        mock_extract.return_value = [
            ("Page one content with sufficient text here.", {}),
            ("Page two content with sufficient text here.", {}),
            ("Page three content with sufficient text here.", {}),
        ]

        docs = loader.load_file(pdf_path)
        assert len(docs) == 3
        assert docs[0].page_number == 1
        assert docs[2].page_number == 3

    @patch("src.ingestion.loader._extract_with_pypdf")
    @patch("src.ingestion.loader._extract_with_pdfplumber")
    def test_fallback_to_pdfplumber_on_thin_text(
        self,
        mock_plumber: MagicMock,
        mock_pypdf: MagicMock,
        loader: DocumentLoader,
        tmp_path: Path,
    ) -> None:
        """When pypdf returns thin text, pdfplumber fallback is triggered."""
        pdf_path = tmp_path / "complex.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        # pypdf returns very short text — triggers fallback.
        mock_pypdf.return_value = [("AB", {}), ("CD", {})]
        # pdfplumber returns richer text.
        mock_plumber.return_value = [
            "Page one with complex table content and sufficient characters.",
            "Page two with more content after pdfplumber extraction.",
        ]

        docs = loader.load_file(pdf_path)
        assert len(docs) == 2
        assert docs[0].metadata.loader_backend == "pdfplumber"

    @patch("src.ingestion.loader._extract_with_pypdf")
    def test_corrupted_pdf_raises_in_fail_fast(
        self,
        mock_extract: MagicMock,
        fail_fast_loader: DocumentLoader,
        tmp_path: Path,
    ) -> None:
        """PDFCorruptedError is raised when fail_fast=True."""
        pdf_path = tmp_path / "corrupted.pdf"
        pdf_path.write_bytes(b"not a pdf")
        mock_extract.side_effect = PDFCorruptedError("Cannot open")

        with pytest.raises(PDFCorruptedError):
            fail_fast_loader.load_file(pdf_path)

    @patch("src.ingestion.loader._extract_with_pypdf")
    def test_metadata_creation_date_parsed(
        self,
        mock_extract: MagicMock,
        loader: DocumentLoader,
        tmp_path: Path,
    ) -> None:
        """Creation date is correctly parsed from PDF metadata."""
        pdf_path = tmp_path / "dated.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        mock_extract.return_value = [
            ("Some text content here long enough.", {"CreationDate": "D:20230615143022"})
        ]

        docs = loader.load_file(pdf_path)
        assert docs[0].metadata.creation_date is not None
        assert docs[0].metadata.creation_date.year == 2023


class TestDocumentLoaderDirectory:
    """Tests for :meth:`DocumentLoader.load_directory`."""

    def test_raises_on_missing_directory(self, loader: DocumentLoader) -> None:
        """DirectoryNotFoundError is raised for non-existent directories."""
        with pytest.raises(DirectoryNotFoundError):
            loader.load_directory(Path("/nonexistent/fake/path"))

    @patch("src.ingestion.loader._extract_with_pypdf")
    def test_batch_loads_multiple_pdfs(
        self,
        mock_extract: MagicMock,
        loader: DocumentLoader,
        tmp_path: Path,
    ) -> None:
        """Batch loading returns documents from all PDFs in the directory."""
        for i in range(3):
            (tmp_path / f"doc{i}.pdf").write_bytes(b"%PDF-1.4 fake")

        mock_extract.return_value = [
            ("Sufficient text content for this document page.", {})
        ]

        docs = loader.load_directory(tmp_path)
        assert len(docs) == 3  # 3 PDFs × 1 page each

    @patch("src.ingestion.loader._extract_with_pypdf")
    def test_skips_failed_pdfs_when_not_fail_fast(
        self,
        mock_extract: MagicMock,
        loader: DocumentLoader,
        tmp_path: Path,
    ) -> None:
        """Failed PDFs are skipped and logged when fail_fast is False."""
        (tmp_path / "good.pdf").write_bytes(b"%PDF-1.4 fake")
        (tmp_path / "bad.pdf").write_bytes(b"not a pdf")

        def side_effect(path: Path) -> list[tuple[str, dict[str, str]]]:
            if "bad" in path.name:
                raise PDFCorruptedError("Corrupted")
            return [("Good document with enough text content.", {})]

        mock_extract.side_effect = side_effect

        docs = loader.load_directory(tmp_path)
        assert len(docs) == 1
        assert docs[0].metadata.filename == "good.pdf"

    def test_empty_directory_returns_empty_list(
        self,
        loader: DocumentLoader,
        tmp_path: Path,
    ) -> None:
        """An empty directory returns an empty document list."""
        docs = loader.load_directory(tmp_path)
        assert docs == []


class TestDocumentDataclass:
    """Tests for the Document dataclass structure."""

    def test_document_fields_accessible(self) -> None:
        """Document fields are accessible via attribute access."""
        meta = DocumentMetadata(
            filename="test.pdf",
            page_count=5,
            page_number=1,
        )
        doc = Document(
            content="Sample text.",
            metadata=meta,
            source_path=Path("/tmp/test.pdf"),
            page_number=1,
        )
        assert doc.content == "Sample text."
        assert doc.metadata.filename == "test.pdf"
        assert doc.metadata.page_count == 5
        assert doc.page_number == 1

    def test_metadata_extra_defaults_to_empty(self) -> None:
        """DocumentMetadata.extra defaults to an empty dict."""
        meta = DocumentMetadata(filename="a.pdf", page_count=1)
        assert meta.extra == {}
