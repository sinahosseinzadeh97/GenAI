"""PDF document loader for RAGForge Enterprise.

Provides :class:`DocumentLoader` which ingests individual PDFs or entire
directories of PDFs, returning typed :class:`Document` instances.

Design decisions
----------------
- **pypdf** is used as the primary extraction backend because it is fast,
  pure-Python, and handles the vast majority of text-layer PDFs without
  system dependencies.
- **pdfplumber** is used as an automatic fallback for pages whose pypdf
  extraction yields too little text (heuristic: < 20 characters/page).
  pdfplumber wraps pdfminer and produces better results for multi-column
  layouts, rotated text, and tables.
- Both libraries are wrapped so that all their exceptions surface as the
  custom :class:`PDFLoadError` hierarchy, keeping the calling code clean.

Typical usage::

    from src.ingestion.loader import DocumentLoader

    loader = DocumentLoader()
    docs   = loader.load_directory("data/sample_docs/")
    for doc in docs:
        print(doc.source_path, "→", len(doc.content), "chars")
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from src.config.settings import get_settings
from src.utils.logger import get_logger, log_exception

# ── Logging ───────────────────────────────────────────────────────────────────
_settings = get_settings()
log = get_logger(__name__, level=_settings.log_level)


# ── Custom Exceptions ─────────────────────────────────────────────────────────


class PDFLoadError(Exception):
    """Base exception for all PDF-loading failures."""


class PDFCorruptedError(PDFLoadError):
    """Raised when a PDF cannot be opened or is structurally invalid."""


class PDFExtractionError(PDFLoadError):
    """Raised when text extraction fails for a specific page."""


class DirectoryNotFoundError(PDFLoadError):
    """Raised when the supplied directory path does not exist."""


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass
class DocumentMetadata:
    """Structured metadata attached to every loaded document.

    Attributes:
        filename:      Basename of the source file (e.g. ``"report.pdf"``).
        page_count:    Total number of pages in the PDF.
        page_number:   1-based page index this content was extracted from.
                       ``None`` when the document represents the whole file
                       without per-page splits.
        creation_date: PDF creation date as a UTC-aware datetime, or ``None``
                       when the metadata field is absent/unparseable.
        loader_backend: Which library produced the text (``"pypdf"`` or
                        ``"pdfplumber"``).
        extra:         Arbitrary additional metadata from the PDF info dict.
    """

    filename: str
    page_count: int
    page_number: int | None = None
    creation_date: datetime | None = None
    loader_backend: str = "pypdf"
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class Document:
    """A single extracted document page or segment.

    Attributes:
        content:     Raw extracted text (pre-cleaning).
        metadata:    Structured :class:`DocumentMetadata`.
        source_path: Absolute path to the originating PDF.
        page_number: Convenience mirror of ``metadata.page_number``.
    """

    content: str
    metadata: DocumentMetadata
    source_path: Path
    page_number: int | None = None


# ── Internal helpers ──────────────────────────────────────────────────────────

_MIN_CHARS_PER_PAGE = 20  # Heuristic: below this, pypdf extraction is suspect


def _parse_pdf_creation_date(raw: str | None) -> datetime | None:
    """Attempt to parse a PDF ``/CreationDate`` string to a UTC datetime.

    PDF date strings follow the format ``D:YYYYMMDDHHmmSSOHH'mm'`` (ISO-like
    but not exactly ISO-8601).  We handle the common subset.

    Args:
        raw: Raw string from ``pdf.metadata.get("/CreationDate")``, or ``None``.

    Returns:
        A timezone-aware :class:`datetime` in UTC, or ``None`` on failure.
    """
    if not raw:
        return None
    # Strip the leading "D:" prefix common in PDF date strings.
    cleaned = raw.lstrip("D:").split("+")[0].split("-")[0].split("Z")[0]
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d"):
        try:
            return datetime.strptime(cleaned[:len(fmt.replace("%", "XX"))], fmt).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
    return None


def _extract_with_pypdf(path: Path) -> list[tuple[str, dict[str, str]]]:
    """Extract text from *path* using pypdf.

    Args:
        path: Absolute path to the PDF file.

    Returns:
        A list of ``(page_text, info_dict)`` tuples, one per page.
        ``info_dict`` contains the raw PDF metadata on page 0, empty for
        subsequent pages (it is the same for all and stored once).

    Raises:
        PDFCorruptedError: If pypdf cannot open or decrypt the file.
        PDFExtractionError: If text extraction fails on an individual page.
    """
    try:
        import pypdf  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise PDFLoadError("pypdf is not installed. Run: pip install pypdf") from exc

    try:
        reader = pypdf.PdfReader(str(path))
    except Exception as exc:
        raise PDFCorruptedError(f"pypdf could not open {path}: {exc}") from exc

    info: dict[str, str] = {}
    if reader.metadata:
        info = {k.lstrip("/"): str(v) for k, v in reader.metadata.items() if v is not None}

    pages: list[tuple[str, dict[str, str]]] = []
    for idx, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            raise PDFExtractionError(
                f"pypdf failed to extract page {idx + 1} of {path}: {exc}"
            ) from exc
        pages.append((text, info if idx == 0 else {}))
    return pages


def _extract_with_pdfplumber(path: Path) -> list[str]:
    """Extract text from *path* using pdfplumber (fallback).

    Args:
        path: Absolute path to the PDF file.

    Returns:
        A list of per-page text strings.

    Raises:
        PDFCorruptedError: If pdfplumber cannot open the file.
        PDFExtractionError: If text extraction fails on an individual page.
    """
    try:
        import pdfplumber  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise PDFLoadError("pdfplumber is not installed. Run: pip install pdfplumber") from exc

    try:
        pdf = pdfplumber.open(str(path))
    except Exception as exc:
        raise PDFCorruptedError(f"pdfplumber could not open {path}: {exc}") from exc

    pages: list[str] = []
    with pdf:
        for idx, page in enumerate(pdf.pages):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                raise PDFExtractionError(
                    f"pdfplumber failed to extract page {idx + 1} of {path}: {exc}"
                ) from exc
            pages.append(text)
    return pages


# ── Public API ────────────────────────────────────────────────────────────────


class DocumentLoader:
    """Load one or many PDF files into :class:`Document` instances.

    The loader tries **pypdf** first.  If any page produces fewer than
    :data:`_MIN_CHARS_PER_PAGE` characters, the whole file is re-extracted
    with **pdfplumber** so that complex layouts are handled gracefully.

    Args:
        fail_fast: When ``True``, the first PDF failure raises immediately.
                   When ``False`` (default), failures are logged and skipped,
                   allowing batch loads to continue.

    Example:
        >>> loader = DocumentLoader()
        >>> docs = loader.load_file(Path("report.pdf"))
        >>> len(docs)  # one Document per page
        12
    """

    def __init__(self, fail_fast: bool = False) -> None:
        self._fail_fast = fail_fast

    # ── Core loading logic ────────────────────────────────────────────────────

    def load_file(self, path: Path) -> list[Document]:
        """Load a single PDF file and return one :class:`Document` per page.

        Args:
            path: Path to the PDF file.

        Returns:
            A list of :class:`Document` objects (one per page, 1-indexed).

        Raises:
            PDFCorruptedError: When the PDF cannot be opened.
            PDFExtractionError: When page-level extraction fails.
        """
        path = path.resolve()
        log.info("Loading PDF", extra={"path": str(path)})

        pypdf_pages = _extract_with_pypdf(path)
        page_count = len(pypdf_pages)

        # Check whether pypdf's extraction is usable.
        weak_pages = sum(
            1 for text, _ in pypdf_pages if len(text.strip()) < _MIN_CHARS_PER_PAGE
        )
        use_fallback = weak_pages > 0

        if use_fallback:
            log.warning(
                "pypdf produced thin text on %d/%d pages; falling back to pdfplumber",
                weak_pages,
                page_count,
                extra={"path": str(path), "weak_pages": weak_pages},
            )
            plumber_pages = _extract_with_pdfplumber(path)
            # Re-merge: prefer pdfplumber text when it is richer.
            texts = [
                plumber if len(plumber) > len(pypdf) else pypdf
                for (pypdf, _), plumber in zip(pypdf_pages, plumber_pages)
            ]
            backend = "pdfplumber"
        else:
            texts = [text for text, _ in pypdf_pages]
            backend = "pypdf"

        # Retrieve metadata from the first page's info dict.
        _, first_info = pypdf_pages[0] if pypdf_pages else ("", {})
        creation_date = _parse_pdf_creation_date(first_info.get("CreationDate"))

        documents: list[Document] = []
        for page_idx, text in enumerate(texts, start=1):
            meta = DocumentMetadata(
                filename=path.name,
                page_count=page_count,
                page_number=page_idx,
                creation_date=creation_date,
                loader_backend=backend,
                extra={k: v for k, v in first_info.items() if k != "CreationDate"},
            )
            documents.append(
                Document(
                    content=text,
                    metadata=meta,
                    source_path=path,
                    page_number=page_idx,
                )
            )

        log.info(
            "Loaded PDF successfully",
            extra={
                "path": str(path),
                "pages": page_count,
                "backend": backend,
                "total_chars": sum(len(d.content) for d in documents),
            },
        )
        return documents

    def load_directory(self, directory: Path | str) -> list[Document]:
        """Batch-load all PDFs found in *directory* (non-recursive).

        Args:
            directory: Path to a directory containing PDF files.

        Returns:
            A flat list of :class:`Document` objects from all successfully
            loaded PDFs, preserving per-file page order.

        Raises:
            DirectoryNotFoundError: When *directory* does not exist.
        """
        dir_path = Path(directory).resolve()
        if not dir_path.is_dir():
            raise DirectoryNotFoundError(f"Directory not found: {dir_path}")

        pdf_files = sorted(dir_path.glob("*.pdf")) + sorted(dir_path.glob("*.PDF"))
        log.info(
            "Starting batch PDF load",
            extra={"directory": str(dir_path), "file_count": len(pdf_files)},
        )

        all_documents: list[Document] = []
        failed: list[str] = []

        for pdf_path in pdf_files:
            try:
                docs = self.load_file(pdf_path)
                all_documents.extend(docs)
            except PDFLoadError as exc:
                if self._fail_fast:
                    raise
                log_exception(log, f"Skipping {pdf_path.name} due to load error", exc)
                failed.append(pdf_path.name)

        log.info(
            "Batch load complete",
            extra={
                "loaded": len(pdf_files) - len(failed),
                "failed": len(failed),
                "failed_files": failed,
                "total_documents": len(all_documents),
            },
        )
        return all_documents

    # ── Convenience iterator ──────────────────────────────────────────────────

    def iter_directory(self, directory: Path | str) -> Iterator[Document]:
        """Yield :class:`Document` objects one at a time from *directory*.

        Memory-efficient alternative to :meth:`load_directory` when dealing
        with large corpora.

        Args:
            directory: Path to a directory containing PDF files.

        Yields:
            :class:`Document` instances page-by-page.

        Raises:
            DirectoryNotFoundError: When *directory* does not exist.
        """
        dir_path = Path(directory).resolve()
        if not dir_path.is_dir():
            raise DirectoryNotFoundError(f"Directory not found: {dir_path}")

        for pdf_path in sorted(dir_path.glob("*.pdf")) + sorted(dir_path.glob("*.PDF")):
            try:
                yield from self.load_file(pdf_path)
            except PDFLoadError as exc:
                if self._fail_fast:
                    raise
                log_exception(log, f"Skipping {pdf_path.name} due to load error", exc)


# ── CLI entry-point ───────────────────────────────────────────────────────────


def _cli() -> None:
    """Command-line interface for batch PDF loading.

    Usage::

        python -m src.ingestion.loader --input data/sample_docs/
    """
    parser = argparse.ArgumentParser(
        description="RAGForge Loader – batch-load PDFs and print document summaries."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=get_settings().data_dir,
        help="Directory containing PDF files (default: %(default)s).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on first PDF error instead of skipping.",
    )
    args = parser.parse_args()

    loader = DocumentLoader(fail_fast=args.fail_fast)
    try:
        docs = loader.load_directory(args.input)
    except DirectoryNotFoundError as exc:
        log.error("Directory not found", extra={"error": str(exc)})
        sys.exit(1)

    print(f"\n{'─' * 60}")
    print(f"  Loaded {len(docs)} document pages from {args.input}")
    print(f"{'─' * 60}")
    for doc in docs:
        print(
            f"  [{doc.metadata.filename}] "
            f"page {doc.page_number}/{doc.metadata.page_count}  "
            f"chars={len(doc.content)}"
        )
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    _cli()
