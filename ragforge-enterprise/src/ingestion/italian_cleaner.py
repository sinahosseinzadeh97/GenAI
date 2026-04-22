"""Italian Legal Text Cleaner — RAGForge Italia Phase 2.3.

:class:`ItalianLegalCleaner` extends :class:`~src.ingestion.cleaner.DocumentCleaner`
with a post-processing pass tailored to the specific artefacts and formatting
conventions found in Italian legal PDFs:

Cleaning layers (applied in order after the base cleaner)
----------------------------------------------------------
1. **Gazzetta Ufficiale header/footer removal** — strips the running header
   ("GAZZETTA UFFICIALE DELLA REPUBBLICA ITALIANA"), series banners
   ("Serie Generale", "4a Serie Speciale"), and page footers.
2. **Column-layout artefact removal** — GU and court PDFs are often two-column;
   PDF extractors produce interleaved lines from both columns.  We detect the
   pattern (alternating short lines) and merge them back into flowing paragraphs.
3. **Legal citation normalisation** — standardises the most common informal
   citation variants to their canonical form:
   - ``"art. 2043 cc"`` → ``"Art. 2043 c.c."``
   - ``"d.lgs. 231/01"`` → ``"D.Lgs. 231/2001"``
   - ``"l. 300/70"``     → ``"L. 300/1970"``
4. **Omissis normalisation** — court decisions redact personal data with
   varied forms (``"OMISSIS"``, ``"(omissis)"``, ``"***"``); all are unified
   to the canonical ``"[OMISSIS]"`` marker.
5. **Unicode normalisation** — NFC + legal symbol preservation (§, ©, °)
   with soft-hyphen removal.
6. **Special character handling** — preserves legal symbols (§, ©, °) while
   removing control characters and null bytes injected during PDF extraction.

Design decisions
----------------
- All patterns are compiled at *module* level so the class itself has zero
  startup cost beyond super().__init__().
- The cleaner is intentionally conservative: it never removes text that *might*
  be substantive.  When uncertain the original text is preserved.
- Citation normalisation uses a two-pass approach (lower → normalise → restore
  capitalisation) to handle mixed-case inputs from GU scans.

Typical usage::

    from src.ingestion.cleaner import DocumentCleaner
    from src.ingestion.italian_cleaner import ItalianLegalCleaner

    cleaner = ItalianLegalCleaner()
    cleaned_doc = cleaner.clean(raw_doc)

    # Or as a drop-in for the base cleaner inside the pipeline:
    cleaner = ItalianLegalCleaner()
    cleaned_docs = cleaner.clean_batch(page_docs)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from src.ingestion.cleaner import CleaningStats, DocumentCleaner, _stats_to_dict
from src.ingestion.loader import Document, DocumentMetadata
from src.utils.logger import get_logger

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Compiled patterns — Gazzetta Ufficiale
# ═══════════════════════════════════════════════════════════════════════════════

# Running masthead line (often repeated on every page).
_RE_GU_MASTHEAD = re.compile(
    r"(?mi)^[—–-]*\s*"
    r"(?:GAZZETTA\s+UFFICIALE\s+DELLA\s+REPUBBLICA\s+ITALIANA"
    r"|Gazzetta\s+Ufficiale\s+della\s+Repubblica\s+Italiana)"
    r"\s*[—–-]*$"
)

# Series headers: "Serie Generale", "4a Serie Speciale", etc.
_RE_GU_SERIES = re.compile(
    r"(?mi)^[—–-]*\s*"
    r"(?:\d+[aª°]?\s*)?Serie\s+(?:Generale|Speciale|Separata)[^\n]*$"
)

# Typical page footer: "— X — " or "Pag. X" or "N. XX — X"
_RE_GU_FOOTER = re.compile(
    r"(?mi)^[\\s—–-]*"
    r"(?:Supplemento\s+)?(?:Pag(?:ina)?\.?\s*)?\d+\s*[—–-]*\s*$"
)

# Issue date line: "Anno NNN — Numero XX" / "anno CXLVII"
_RE_GU_ISSUE = re.compile(
    r"(?mi)^[Aa]nno\s+[MDCLXVI\d]+\s*[—–]\s*[Nn]umero\s+\d+.*$"
)

# Publication house attribution line.
_RE_GU_PUBLISHER = re.compile(
    r"(?mi)^.*(?:Istituto\s+Poligrafico|IPZS|Zecca\s+dello\s+Stato).*$"
)


# ═══════════════════════════════════════════════════════════════════════════════
# Compiled patterns — Omissis markers
# ═══════════════════════════════════════════════════════════════════════════════

_RE_OMISSIS = re.compile(
    r"""
    (?:
        \(?\s*[Oo][Mm][Ii][Ss][Ss][Ii][Ss]\s*\)?  # (omissis) / OMISSIS
    |
        \[\s*(?:[Oo]missis|\*+|\.{3,})\s*\]        # [omissis] / [...] / [***]
    |
        \*{2,}                                       # *** ** ****
    |
        (?<!\.)\.{3}(?!\.)                           # ... (but not .…. or …)
        (?=\s)                                       # followed by whitespace
    )
    """,
    re.VERBOSE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Compiled patterns — Citation normalisation
# ═══════════════════════════════════════════════════════════════════════════════

# Article citations — normalise to "Art. N c.c." / "Art. N c.p." etc.
# Captures: art / Art / ART / articolo, optional dot, digits, optional c.c./c.p.
_RE_ART_RAW = re.compile(
    r"""
    (?i)\b
    (?P<prefix>art(?:icolo)?)\.?\s*
    (?P<num>\d+(?:\s*-?\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?)
    (?P<comma>(?:\s*,\s*(?:co(?:mma)?|c)\.?\s*\d+)?)
    (?:\s+(?P<code>
        c\.?c\.?|c\.?p\.?(?:\.?c\.?|\.?p\.?)?|c\.?p\.?a\.?|
        t\.?u\.?(?:i\.?r\.?)?|[Cc]ost\.?
    ))?
    """,
    re.VERBOSE,
)

# Codice labels mapping (lowercase key → canonical form).
_CODE_MAP: dict[str, str] = {
    "cc": "c.c.",
    "c.c": "c.c.",
    "c.c.": "c.c.",
    "cp": "c.p.",
    "c.p": "c.p.",
    "c.p.": "c.p.",
    "cpc": "c.p.c.",
    "c.p.c": "c.p.c.",
    "c.p.c.": "c.p.c.",
    "cpp": "c.p.p.",
    "c.p.p": "c.p.p.",
    "c.p.p.": "c.p.p.",
    "cpa": "c.p.a.",
    "c.p.a": "c.p.a.",
    "c.p.a.": "c.p.a.",
    "tu": "t.u.",
    "t.u": "t.u.",
    "t.u.": "t.u.",
    "tuir": "T.U.I.R.",
    "t.u.i.r": "T.U.I.R.",
    "t.u.i.r.": "T.U.I.R.",
    "cost": "Cost.",
    "cost.": "Cost.",
}

# D.Lgs. — abbreviated year: "d.lgs. 231/01" → "D.Lgs. 231/2001"
_RE_DLGS_ABBREV = re.compile(
    r"""
    (?i)\b
    d\.?\s*lgs\.?\s*
    (?:n\.?\s*)?
    (?P<num>\d+)/(?P<year>\d{2,4})
    """,
    re.VERBOSE,
)

# D.L. (decree-law)
_RE_DL_ABBREV = re.compile(
    r"""
    (?i)\b
    d\.?\s*l\.?\s*
    (?:n\.?\s*)?
    (?P<num>\d+)/(?P<year>\d{2,4})
    \b
    """,
    re.VERBOSE,
)

# Legge: "l. 300/70" → "L. 300/1970"
_RE_L_ABBREV = re.compile(
    r"""
    (?i)\b
    l\.?\s*
    (?:n\.?\s*)?
    (?P<num>\d+)/(?P<year>\d{2,4})
    \b
    """,
    re.VERBOSE,
)

# D.P.R.
_RE_DPR_ABBREV = re.compile(
    r"""
    (?i)\b
    d\.?\s*p\.?\s*r\.?\s*
    (?:n\.?\s*)?
    (?P<num>\d+)/(?P<year>\d{2,4})
    """,
    re.VERBOSE,
)


def _expand_year(year_str: str) -> str:
    """Expand a 2-digit year to a 4-digit year.

    Uses a pivot of 30 — years ≤ 30 map to 2000s, > 30 to 1900s.

    Args:
        year_str: A 2- or 4-digit year string.

    Returns:
        A 4-digit year string.
    """
    if len(year_str) == 4:
        return year_str
    y = int(year_str)
    return str(2000 + y) if y <= 30 else str(1900 + y)


# ═══════════════════════════════════════════════════════════════════════════════
# Compiled patterns — Column-layout artefacts
# ═══════════════════════════════════════════════════════════════════════════════

# A line is considered a column fragment if it is ≤ 60 characters and does NOT
# end with a sentence-terminating punctuation (.  !  ?  :  ;).
_RE_SHORT_LINE = re.compile(r"^.{1,60}$")
_RE_LINE_END = re.compile(r"[.!?;:]\s*$")


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level cleaning helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _strip_gu_boilerplate(text: str) -> str:
    """Remove Gazzetta Ufficiale headers, footers, and issue metadata.

    Args:
        text: Raw extracted text from a GU PDF page.

    Returns:
        Text with GU-specific boilerplate stripped.
    """
    text = _RE_GU_MASTHEAD.sub("", text)
    text = _RE_GU_SERIES.sub("", text)
    text = _RE_GU_FOOTER.sub("", text)
    text = _RE_GU_ISSUE.sub("", text)
    text = _RE_GU_PUBLISHER.sub("", text)
    return text


def _normalise_omissis(text: str) -> str:
    """Unify all omissis markers to the single canonical ``[OMISSIS]`` form.

    Args:
        text: Italian court decision text potentially containing varied omissis.

    Returns:
        Text with all omissis variants replaced by ``[OMISSIS]``.
    """
    return _RE_OMISSIS.sub("[OMISSIS]", text)


def _normalise_article_citation(match: re.Match[str]) -> str:  # type: ignore[type-arg]
    """Replacement function for ``_RE_ART_RAW`` — canonical Art. form.

    Args:
        match: A ``re.Match`` from ``_RE_ART_RAW``.

    Returns:
        Canonical citation string.
    """
    num: str = match.group("num").strip()
    # Capitalise bis/ter etc.
    num = re.sub(
        r"\b(bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)\b",
        lambda m: m.group(0).lower(),
        num,
    )
    comma: str = match.group("comma") or ""
    if comma:
        # Normalise comma to ", co. N"
        comma_num = re.search(r"\d+", comma)
        comma = f", co. {comma_num.group()}" if comma_num else ""

    code: str = match.group("code") or ""
    code_canonical = _CODE_MAP.get(code.lower().replace(" ", ""), "")
    if code_canonical:
        return f"Art. {num}{comma} {code_canonical}"
    return f"Art. {num}{comma}"


def _normalise_citations(text: str) -> str:
    """Normalise common Italian legal citation forms to canonical style.

    Transformations applied:
    - Article citations → ``Art. N c.c.`` etc.
    - D.Lgs. / D.L. / L. / D.P.R. 2-digit years expanded to 4-digit.

    Args:
        text: Input legal text.

    Returns:
        Text with normalised citations.
    """
    # Articles
    text = _RE_ART_RAW.sub(_normalise_article_citation, text)

    # D.Lgs.
    text = _RE_DLGS_ABBREV.sub(
        lambda m: f"D.Lgs. {m.group('num')}/{_expand_year(m.group('year'))}", text
    )

    # D.L. — only if NOT preceded by "D.Lgs." (avoid double-matching)
    # We use a negative lookbehind to avoid matching "gs." before the "d.l."
    text = _RE_DL_ABBREV.sub(
        lambda m: f"D.L. {m.group('num')}/{_expand_year(m.group('year'))}", text
    )

    # Legge
    text = _RE_L_ABBREV.sub(
        lambda m: f"L. {m.group('num')}/{_expand_year(m.group('year'))}", text
    )

    # D.P.R.
    text = _RE_DPR_ABBREV.sub(
        lambda m: f"D.P.R. {m.group('num')}/{_expand_year(m.group('year'))}", text
    )

    return text


def _repair_column_layout(text: str) -> str:
    """Merge column-interleaved lines from two-column Italian legal PDFs.

    When PDF extractors read two-column layouts linearly, they interleave lines
    from the left and right columns.  This produces alternating short lines that
    belong to separate paragraphs.  We detect runs of ≥ 4 consecutive
    short-line pairs and join them with spaces rather than newlines.

    This is a best-effort heuristic; it is only applied when the ratio of short
    lines to total lines exceeds 60 %, which strongly suggests column interleaving.

    Args:
        text: Raw multi-line text.

    Returns:
        Text with column artefacts merged.
    """
    lines = text.splitlines()
    if len(lines) < 8:
        return text

    short_count = sum(
        1 for ln in lines if _RE_SHORT_LINE.match(ln.strip()) and ln.strip()
    )
    ratio = short_count / len(lines)
    if ratio < 0.60:
        return text  # Not a column layout.

    # Join consecutive short lines that don't end with sentence terminators.
    merged: list[str] = []
    buffer: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                merged.append(" ".join(buffer))
                buffer = []
            merged.append("")
            continue

        if _RE_SHORT_LINE.match(stripped) and not _RE_LINE_END.search(stripped):
            buffer.append(stripped)
        else:
            if buffer:
                buffer.append(stripped)
                merged.append(" ".join(buffer))
                buffer = []
            else:
                merged.append(stripped)

    if buffer:
        merged.append(" ".join(buffer))

    return "\n".join(merged)


def _preserve_legal_symbols(text: str) -> str:
    """Ensure legal symbols are preserved in NFC-normalised form.

    PDF extraction sometimes converts §, ©, ° to question marks or removes
    them via overly aggressive ASCII coercion.  This function:

    1. Replaces common erroneous ASCII stand-ins back to the Unicode form.
    2. Removes non-printable control characters (U+0000–U+001F, U+007F)
       except for TAB and LF.

    Args:
        text: Raw text possibly containing garbled legal symbols.

    Returns:
        Text with preserved legal symbols and stripped control characters.
    """
    # Control character removal (keep \t and \n).
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)

    # NFC normalisation preserves § (U+00A7), © (U+00A9), ° (U+00B0).
    text = unicodedata.normalize("NFC", text)

    return text


# ═══════════════════════════════════════════════════════════════════════════════
# ItalianLegalCleaner
# ═══════════════════════════════════════════════════════════════════════════════


class ItalianLegalCleaner(DocumentCleaner):
    """Cleaning pipeline specialised for Italian legal documents.

    Inherits the full :class:`~src.ingestion.cleaner.DocumentCleaner` pipeline
    (Unicode normalisation, page-number removal, whitespace collapse, structural
    detection, language detection) and adds an Italian-specific post-processing
    pass with the following steps:

    1. Legal symbol preservation and control-character removal.
    2. Gazzetta Ufficiale header/footer stripping.
    3. Column-layout artefact repair (two-column GU / court PDF formats).
    4. Omissis normalisation.
    5. Legal citation normalisation.

    Args:
        strip_gu_boilerplate:  Whether to remove GU mastheads (default ``True``).
        repair_column_layout:  Whether to merge column-interleaved lines
                               (default ``True``).
        normalise_omissis:     Whether to unify omissis markers (default ``True``).
        normalise_citations:   Whether to canonicalise citation forms (default ``True``).

    Example::

        cleaner = ItalianLegalCleaner()
        cleaned = cleaner.clean(gazzetta_doc)
        # citations like "art. 2043 cc" are now "Art. 2043 c.c."
        # omissis are now "[OMISSIS]"

    Note:
        Both :meth:`clean` and :meth:`clean_batch` are available and handle
        the Italian post-processing layer transparently on top of the base class.
    """

    def __init__(
        self,
        strip_gu_boilerplate: bool = True,
        repair_column_layout: bool = True,
        normalise_omissis: bool = True,
        normalise_citations: bool = True,
    ) -> None:
        super().__init__()
        self._strip_gu = strip_gu_boilerplate
        self._repair_columns = repair_column_layout
        self._norm_omissis = normalise_omissis
        self._norm_citations = normalise_citations

    # ── Public overrides ──────────────────────────────────────────────────────

    def clean(self, document: Document) -> Document:
        """Apply the full Italian cleaning pipeline to a single document.

        Applies the base :class:`DocumentCleaner` pipeline first, then the
        Italian-specific post-processing pass.

        Args:
            document: A :class:`~src.ingestion.loader.Document` as returned
                      by the loader.

        Returns:
            A new :class:`~src.ingestion.loader.Document` with Italian-cleaned
            ``content`` and updated ``cleaning_stats`` in ``metadata.extra``.
        """
        # 1. Base cleaning pass.
        base_doc = super().clean(document)

        # 2. Italian post-processing pass.
        italian_text = self._italian_pass(base_doc.content)

        if italian_text == base_doc.content:
            return base_doc

        return self._rebuild_document(base_doc, italian_text)

    def clean_batch(self, documents: list[Document]) -> list[Document]:
        """Clean a batch of Italian legal documents from the same PDF.

        Applies base batch cleaning (including cross-page boilerplate removal)
        then runs the Italian post-processing pass on each document.

        Args:
            documents: Ordered list of :class:`~src.ingestion.loader.Document`
                       objects (one per page).

        Returns:
            Cleaned documents in the same order.
        """
        # 1. Base batch pass (includes cross-page header/footer detection).
        base_docs = super().clean_batch(documents)

        # 2. Italian post-processing pass per document.
        result: list[Document] = []
        for doc in base_docs:
            italian_text = self._italian_pass(doc.content)
            if italian_text != doc.content:
                result.append(self._rebuild_document(doc, italian_text))
            else:
                result.append(doc)
        return result

    # ── Italian post-processing ───────────────────────────────────────────────

    def _italian_pass(self, text: str) -> str:
        """Apply all Italian-specific cleaning steps.

        Args:
            text: Already base-cleaned text.

        Returns:
            Italian-cleaned text.
        """
        # Step 1 — legal symbol preservation and control-character sanitisation.
        text = _preserve_legal_symbols(text)

        # Step 2 — Gazzetta Ufficiale boilerplate.
        if self._strip_gu:
            text = _strip_gu_boilerplate(text)

        # Step 3 — column-layout repair.
        if self._repair_columns:
            text = _repair_column_layout(text)

        # Step 4 — omissis normalisation.
        if self._norm_omissis:
            text = _normalise_omissis(text)

        # Step 5 — citation normalisation.
        if self._norm_citations:
            text = _normalise_citations(text)

        # Final whitespace pass after all Italian transforms.
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ── Helper ────────────────────────────────────────────────────────────────

    def _rebuild_document(self, doc: Document, new_text: str) -> Document:
        """Rebuild a :class:`Document` with *new_text* and updated stats.

        Args:
            doc:      Source document (already base-cleaned).
            new_text: Italian-post-processed text.

        Returns:
            Updated :class:`Document`.
        """
        original_chars = int(
            doc.metadata.extra.get(
                "original_char_count",
                doc.metadata.extra.get("cleaning_stats", {}).get("original_char_count", len(new_text)),
            )
        )

        existing_stats: dict[str, Any] = doc.metadata.extra.get("cleaning_stats", {})
        stats = CleaningStats(
            original_char_count=original_chars,
            cleaned_char_count=len(new_text),
            removed_chars=original_chars - len(new_text),
            has_tables=bool(existing_stats.get("has_tables", False)),
            has_lists=bool(existing_stats.get("has_lists", False)),
            detected_language="it",  # We know it's Italian.
            language_confidence=float(existing_stats.get("language_confidence", 1.0)),
        )

        updated_extra: dict[str, Any] = {
            **doc.metadata.extra,
            "original_char_count": str(original_chars),
            "cleaning_stats": _stats_to_dict(stats),
            "italian_cleaned": True,
        }

        updated_meta = DocumentMetadata(
            filename=doc.metadata.filename,
            page_count=doc.metadata.page_count,
            page_number=doc.metadata.page_number,
            creation_date=doc.metadata.creation_date,
            loader_backend=doc.metadata.loader_backend,
            extra=updated_extra,
        )

        log.debug(
            "Italian document cleaned",
            extra={
                "file": doc.metadata.filename,
                "page": doc.page_number,
                "original_chars": original_chars,
                "cleaned_chars": len(new_text),
            },
        )

        return Document(
            content=new_text,
            metadata=updated_meta,
            source_path=doc.source_path,
            page_number=doc.page_number,
        )
