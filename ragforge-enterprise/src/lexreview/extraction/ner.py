"""spaCy-powered Named Entity Recognition for legal documents.

:class:`LegalNER` wraps a spaCy pipeline extended with a domain-specific
``EntityRuler`` that adds ``PARTY``, ``JURISDICTION``, and ``CLAUSE_TYPE``
labels on top of the standard NER labels.

Typical usage::

    ner = LegalNER()
    entities = ner.extract("This Agreement is entered into between Acme Corp. ...")
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from src.config.settings import get_settings
from src.lexreview.extraction.models import LegalEntities
from src.utils.logger import get_logger

if TYPE_CHECKING:
    pass  # spaCy types imported lazily below

log = get_logger(__name__)
_settings = get_settings()

# Thread-safe singleton cache: model_name → nlp pipeline
_nlp_lock = threading.Lock()
_nlp_cache: dict[str, Any] = {}

# ── Legal domain EntityRuler patterns ─────────────────────────────────────────

_PARTY_PATTERNS: list[dict[str, Any]] = [
    {"label": "PARTY", "pattern": [{"LOWER": "party"}, {"IS_ALPHA": True}]},
    {"label": "PARTY", "pattern": [{"ENT_TYPE": "ORG"}]},
    {"label": "PARTY", "pattern": [{"ENT_TYPE": "PERSON"}]},
]

_JURISDICTION_PATTERNS: list[dict[str, Any]] = [
    {
        "label": "JURISDICTION",
        "pattern": [
            {"LOWER": "state"},
            {"LOWER": "of"},
            {"IS_TITLE": True},
        ],
    },
    {
        "label": "JURISDICTION",
        "pattern": [{"LOWER": "laws"}, {"LOWER": "of"}, {"IS_TITLE": True}],
    },
    {
        "label": "JURISDICTION",
        "pattern": [{"LOWER": "jurisdiction"}, {"LOWER": "of"}, {"IS_TITLE": True}],
    },
    {
        "label": "JURISDICTION",
        "pattern": [{"ENT_TYPE": "GPE"}],
    },
]

_CLAUSE_TYPE_PATTERNS: list[dict[str, Any]] = [
    {"label": "CLAUSE_TYPE", "pattern": [{"LOWER": "indemnif"}, {"LOWER": "ication"}]},
    {"label": "CLAUSE_TYPE", "pattern": [{"LOWER": "indemnification"}]},
    {"label": "CLAUSE_TYPE", "pattern": [{"LOWER": "termination"}]},
    {"label": "CLAUSE_TYPE", "pattern": [{"LOWER": "confidentiality"}]},
    {"label": "CLAUSE_TYPE", "pattern": [{"LOWER": "arbitration"}]},
    {"label": "CLAUSE_TYPE", "pattern": [{"LOWER": "force"}, {"LOWER": "majeure"}]},
    {"label": "CLAUSE_TYPE", "pattern": [{"LOWER": "limitation"}, {"LOWER": "of"}, {"LOWER": "liability"}]},
    {"label": "CLAUSE_TYPE", "pattern": [{"LOWER": "governing"}, {"LOWER": "law"}]},
]

_ALL_PATTERNS = _PARTY_PATTERNS + _JURISDICTION_PATTERNS + _CLAUSE_TYPE_PATTERNS


def _load_nlp(model_name: str) -> Any:
    """Return a cached, ruler-enhanced spaCy pipeline.

    Thread-safe via double-checked locking.

    Args:
        model_name: spaCy pipeline identifier (e.g. ``"en_core_web_sm"``).

    Returns:
        A spaCy ``Language`` object with the legal EntityRuler added.

    Raises:
        RuntimeError: When the spaCy model is not installed.
    """
    if model_name in _nlp_cache:
        return _nlp_cache[model_name]

    with _nlp_lock:
        if model_name in _nlp_cache:
            return _nlp_cache[model_name]

        try:
            import spacy  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "spaCy is not installed. Run: pip install spacy"
            ) from exc

        try:
            nlp = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' not found. "
                f"Run: python -m spacy download {model_name}"
            ) from exc

        # Add EntityRuler *before* the standard NER to avoid conflicts.
        if "entity_ruler" not in nlp.pipe_names:
            ruler = nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(_ALL_PATTERNS)  # type: ignore[union-attr]

        log.info("spaCy pipeline loaded", extra={"model": model_name})
        _nlp_cache[model_name] = nlp
        return nlp


class LegalNER:
    """Named Entity Recognizer for legal documents.

    Uses a spaCy pipeline extended with domain-specific ``EntityRuler``
    patterns to extract parties, dates, monetary amounts, and jurisdictions.

    Args:
        model_name: spaCy pipeline name (defaults to ``settings.spacy_model``).

    Example::

        ner = LegalNER()
        entities = ner.extract("Acme Corp. and Beta LLC agree under Delaware law.")
        # LegalEntities(parties=["Acme Corp.", "Beta LLC"], jurisdictions=["Delaware"], ...)
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or _settings.spacy_model

    def extract(self, text: str) -> LegalEntities:
        """Extract all legal named entities from *text*.

        Args:
            text: Raw document or clause text.

        Returns:
            :class:`~src.lexreview.extraction.models.LegalEntities` with
            deduplicated, normalised entity lists.

        Raises:
            RuntimeError: When the spaCy model cannot be loaded.
        """
        nlp = _load_nlp(self._model_name)
        doc = nlp(text)

        parties: list[str] = []
        dates: list[str] = []
        amounts: list[str] = []
        jurisdictions: list[str] = []

        seen: set[str] = set()

        for ent in doc.ents:
            text_norm = ent.text.strip()
            key = (ent.label_, text_norm)
            if key in seen or not text_norm:
                continue
            seen.add(key)

            if ent.label_ in {"PARTY", "ORG", "PERSON"}:
                parties.append(text_norm)
            elif ent.label_ == "DATE":
                dates.append(text_norm)
            elif ent.label_ in {"MONEY", "CARDINAL"} and any(
                c.isdigit() for c in text_norm
            ):
                amounts.append(text_norm)
            elif ent.label_ in {"JURISDICTION", "GPE", "LOC"}:
                jurisdictions.append(text_norm)

        log.debug(
            "LegalNER extraction complete",
            extra={
                "parties": len(parties),
                "dates": len(dates),
                "amounts": len(amounts),
                "jurisdictions": len(jurisdictions),
            },
        )
        return LegalEntities(
            parties=parties,
            dates=dates,
            amounts=amounts,
            jurisdictions=jurisdictions,
        )

    def extract_batch(self, texts: list[str]) -> list[LegalEntities]:
        """Extract entities from multiple documents using ``nlp.pipe()``.

        Args:
            texts: List of raw document strings.

        Returns:
            List of :class:`LegalEntities` in the same order as *texts*.
        """
        nlp = _load_nlp(self._model_name)
        results: list[LegalEntities] = []
        for doc in nlp.pipe(texts, batch_size=16):
            # Re-use single-doc logic by wrapping
            entities = self._doc_to_entities(doc)
            results.append(entities)
        return results

    def _doc_to_entities(self, doc: Any) -> LegalEntities:
        """Convert a processed spaCy Doc to LegalEntities."""
        parties, dates, amounts, jurisdictions = [], [], [], []
        seen: set[tuple[str, str]] = set()
        for ent in doc.ents:
            text_norm = ent.text.strip()
            key = (ent.label_, text_norm)
            if key in seen or not text_norm:
                continue
            seen.add(key)
            if ent.label_ in {"PARTY", "ORG", "PERSON"}:
                parties.append(text_norm)
            elif ent.label_ == "DATE":
                dates.append(text_norm)
            elif ent.label_ in {"MONEY", "CARDINAL"} and any(
                c.isdigit() for c in text_norm
            ):
                amounts.append(text_norm)
            elif ent.label_ in {"JURISDICTION", "GPE", "LOC"}:
                jurisdictions.append(text_norm)
        return LegalEntities(
            parties=parties, dates=dates, amounts=amounts, jurisdictions=jurisdictions
        )
