"""Regex-based extractor for legal entities.

:class:`RegexExtractor` complements :class:`~src.lexreview.extraction.ner.LegalNER`
with deterministic pattern matching for amounts, dates, party suffixes, and
jurisdiction references.  It is intentionally independent of any ML model.

Typical usage::

    rx = RegexExtractor()
    entities = rx.extract("This Agreement dated January 1, 2024 for $50,000...")
"""

from __future__ import annotations

import re

from src.lexreview.extraction.models import LegalEntities
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Compiled patterns ─────────────────────────────────────────────────────────

# Monetary amounts: $1,000 / USD 50,000.00 / EUR 1.000,00 / £500
_AMOUNT_RE = re.compile(
    r"""
    (?:
        (?:USD|EUR|GBP|CAD|AUD|JPY|CHF)\s*[\d,]+(?:\.\d{1,2})?  # currency code prefix
        | [\$£€¥]\s*[\d,]+(?:\.\d{1,2})?                          # symbol prefix
        | [\d,]+(?:\.\d{1,2})?\s*(?:USD|EUR|GBP|CAD|AUD)          # trailing code
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Dates: 2024-01-31 / January 1, 2024 / 01/31/2024 / 31 January 2024
_DATE_RE = re.compile(
    r"""
    (?:
        \d{4}-\d{2}-\d{2}                                               # ISO
        | (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|
             Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|
             Nov(?:ember)?|Dec(?:ember)?)
           \s+\d{1,2},?\s+\d{4}                                         # "January 1, 2024"
        | \d{1,2}\s+
           (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|
              Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|
              Nov(?:ember)?|Dec(?:ember)?)
           \s+\d{4}                                                      # "1 January 2024"
        | \d{1,2}/\d{1,2}/\d{2,4}                                      # MM/DD/YYYY
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Corporate party suffixes
_PARTY_RE = re.compile(
    r"""
    (?:
        [A-Z][A-Za-z\s&'\-,.]*?
        \s+
        (?:Corporation|Corp\.?|Incorporated|Inc\.?|Limited|Ltd\.?|
           LLC|L\.L\.C\.|LLP|L\.L\.P\.|PLLC|GmbH|S\.A\.|B\.V\.|Pty\.?)
    )
    """,
    re.VERBOSE,
)

# Jurisdiction references
_JURISDICTION_RE = re.compile(
    r"""
    (?:
        laws?\s+of\s+(?:the\s+)?(?:State\s+of\s+)?([A-Z][A-Za-z\s]+?)(?=\s*[,;.)])
        | State\s+of\s+([A-Z][A-Za-z\s]+?)(?=\s*[,;.)])
        | jurisdiction\s+of\s+(?:the\s+)?([A-Z][A-Za-z\s]+?)(?=\s*[,;.)])
        | (?:governed\s+by|subject\s+to)\s+(?:the\s+)?([A-Z][A-Za-z\s]+?)\s+law
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _dedupe(items: list[str]) -> list[str]:
    """Return *items* with duplicates removed, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        norm = item.strip()
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


class RegexExtractor:
    """Deterministic regex-based legal entity extractor.

    Extracts amounts, dates, party names (corporate suffix heuristic), and
    jurisdiction references from raw text.  No ML model required.

    Example::

        rx = RegexExtractor()
        entities = rx.extract(text)
        # LegalEntities(amounts=["$50,000.00"], dates=["January 1, 2024"], ...)
    """

    def extract(self, text: str) -> LegalEntities:
        """Extract legal entities from *text* using compiled regexes.

        Args:
            text: Raw document or clause text.

        Returns:
            :class:`~src.lexreview.extraction.models.LegalEntities`.
        """
        amounts = _dedupe(_AMOUNT_RE.findall(text))
        dates = _dedupe(_DATE_RE.findall(text))
        parties = _dedupe(_PARTY_RE.findall(text))

        # Jurisdiction: flatten groups from alternation
        jurisdictions: list[str] = []
        for match in _JURISDICTION_RE.finditer(text):
            for group in match.groups():
                if group:
                    jurisdictions.append(group.strip())
        jurisdictions = _dedupe(jurisdictions)

        log.debug(
            "RegexExtractor extraction complete",
            extra={
                "amounts": len(amounts),
                "dates": len(dates),
                "parties": len(parties),
                "jurisdictions": len(jurisdictions),
            },
        )
        return LegalEntities(
            parties=parties,
            dates=dates,
            amounts=amounts,
            jurisdictions=jurisdictions,
        )

    def merge(self, *entity_lists: LegalEntities) -> LegalEntities:
        """Merge multiple :class:`LegalEntities` objects into one (deduped).

        Args:
            *entity_lists: Any number of LegalEntities to combine.

        Returns:
            A single merged :class:`LegalEntities`.
        """
        parties: list[str] = []
        dates: list[str] = []
        amounts: list[str] = []
        jurisdictions: list[str] = []
        for le in entity_lists:
            parties.extend(le.parties)
            dates.extend(le.dates)
            amounts.extend(le.amounts)
            jurisdictions.extend(le.jurisdictions)
        return LegalEntities(
            parties=_dedupe(parties),
            dates=_dedupe(dates),
            amounts=_dedupe(amounts),
            jurisdictions=_dedupe(jurisdictions),
        )
