"""Clause detection for legal documents.

:class:`ClauseDetector` combines keyword density scoring and sentence
heuristics to identify and classify legal clauses without a trained ML model.

Supported clause types
----------------------
- indemnification
- termination
- limitation_of_liability
- confidentiality
- payment
- dispute_resolution
- force_majeure
- warranty
- governing_law

Typical usage::

    detector = ClauseDetector()
    clauses = detector.detect("This Agreement may be terminated upon 30 days notice...")
"""

from __future__ import annotations

import re

from src.lexreview.extraction.models import Clause
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Clause keyword catalogue ──────────────────────────────────────────────────

_CLAUSE_KEYWORDS: dict[str, list[str]] = {
    "indemnification": [
        "indemnif",
        "indemnity",
        "hold harmless",
        "defend and indemnif",
        "losses, damages",
        "costs and expenses",
    ],
    "termination": [
        "terminat",
        "expire",
        "expiration",
        "cancel",
        "cancellation",
        "notice of termination",
        "either party may terminate",
        "upon written notice",
    ],
    "limitation_of_liability": [
        "limitation of liability",
        "limit",
        "in no event",
        "aggregate liability",
        "maximum liability",
        "shall not exceed",
        "cap on liability",
        "consequential damages",
    ],
    "confidentiality": [
        "confidential",
        "non-disclosure",
        "nda",
        "proprietary information",
        "trade secret",
        "disclose",
        "disclosure",
    ],
    "payment": [
        "payment",
        "invoice",
        "due date",
        "pay",
        "fee",
        "compensation",
        "remuneration",
        "net 30",
        "net 60",
        "overdue",
    ],
    "dispute_resolution": [
        "arbitration",
        "mediation",
        "dispute",
        "controversy",
        "claim",
        "adr",
        "american arbitration",
        "tribunal",
    ],
    "force_majeure": [
        "force majeure",
        "act of god",
        "beyond the control",
        "unforeseeable",
        "natural disaster",
        "pandemic",
        "government action",
    ],
    "warranty": [
        "warrant",
        "representation",
        "as is",
        "without warranty",
        "merchantability",
        "fitness for a particular",
        "disclaim",
    ],
    "governing_law": [
        "governing law",
        "governed by",
        "subject to the laws",
        "jurisdiction",
        "venue",
        "applicable law",
        "choice of law",
    ],
}

# Sentence splitter — split on ". " followed by capital or on newlines
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|(?:\n\s*\n)")


def _score_sentence(sentence: str, keywords: list[str]) -> float:
    """Compute keyword-density confidence for a single sentence.

    Args:
        sentence: Raw sentence text (lower-cased inside).
        keywords: List of keyword strings for the clause type.

    Returns:
        A confidence score in [0.0, 1.0] based on hit density.
    """
    lower = sentence.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    # Normalise: 1 hit → 0.5, 2 hits → 0.75, 3+ → >0.85 (diminishing returns)
    if hits == 0:
        return 0.0
    return min(1.0, 1.0 - (1.0 / (hits + 1)))


class ClauseDetector:
    """Heuristic clause detector for legal documents.

    Splits text into sentences, scores each sentence against keyword catalogues
    for 9 standard legal clause types, and returns :class:`Clause` objects
    above a confidence threshold.

    Args:
        min_confidence:  Minimum score for a clause to be emitted (default 0.3).

    Example::

        detector = ClauseDetector(min_confidence=0.4)
        clauses = detector.detect(contract_text)
        for c in clauses:
            print(c.type, c.confidence)
    """

    def __init__(self, min_confidence: float = 0.3) -> None:
        self._min_confidence = min_confidence

    def detect(self, text: str) -> list[Clause]:
        """Detect and classify legal clauses in *text*.

        Args:
            text: Raw document or section text.

        Returns:
            List of :class:`~src.lexreview.extraction.models.Clause` objects
            ordered by start offset, above ``min_confidence``.
        """
        sentences = self._split_sentences(text)
        clauses: list[Clause] = []

        for sentence, start, end in sentences:
            best_type: str | None = None
            best_score = 0.0

            for clause_type, keywords in _CLAUSE_KEYWORDS.items():
                score = _score_sentence(sentence, keywords)
                if score > best_score:
                    best_score = score
                    best_type = clause_type

            if best_type is not None and best_score >= self._min_confidence:
                clauses.append(
                    Clause(
                        type=best_type,
                        text=sentence.strip(),
                        span=(start, end),
                        confidence=round(best_score, 4),
                    )
                )

        log.debug(
            "ClauseDetector complete",
            extra={
                "sentences_processed": len(sentences),
                "clauses_detected": len(clauses),
                "min_confidence": self._min_confidence,
            },
        )
        return clauses

    def _split_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """Split *text* into (sentence, start_char, end_char) tuples.

        Args:
            text: Raw document text.

        Returns:
            List of ``(sentence_str, start, end)`` triples.
        """
        results: list[tuple[str, int, int]] = []
        prev_end = 0
        for match in _SENTENCE_RE.finditer(text):
            chunk = text[prev_end : match.start()]
            if chunk.strip():
                results.append((chunk, prev_end, match.start()))
            prev_end = match.end()
        # Last segment
        remainder = text[prev_end:]
        if remainder.strip():
            results.append((remainder, prev_end, len(text)))
        return results
