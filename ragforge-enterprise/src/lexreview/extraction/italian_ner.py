"""Italian Legal Named Entity Recognition — RAGForge Italia Phase 2.2.

:class:`ItalianLegalNER` wraps ``it_core_news_lg`` extended with a
domain-specific ``EntityRuler`` and regex-backed ``Matcher`` that surfaces
seven Italian legal entity types:

+---------------------+--------------------------------------------------+
| Label               | Examples                                         |
+=====================+==================================================+
| NORMA               | "Art. 2043 c.c.", "D.Lgs. 231/2001"             |
+---------------------+--------------------------------------------------+
| SENTENZA            | "Cass. Civ. Sez. III, n. 12345/2024"            |
+---------------------+--------------------------------------------------+
| SOGGETTO_GIURIDICO  | "società a responsabilità limitata", "p.m."      |
+---------------------+--------------------------------------------------+
| ISTITUZIONE         | "Corte di Cassazione", "AGCM"                   |
+---------------------+--------------------------------------------------+
| TERMINE_LEGALE      | "inadempimento", "mora debendi"                  |
+---------------------+--------------------------------------------------+
| IMPORTO             | "€ 50.000", "duemila euro"                       |
+---------------------+--------------------------------------------------+
| DATA_GIURIDICA      | "entro 30 giorni dalla notifica"                 |
+---------------------+--------------------------------------------------+

Architecture
------------
1. A spaCy ``EntityRuler`` with explicit token-pattern rules is added *before*
   the statistical NER so that rule-based matches always win.
2. A ``Matcher`` (regex-flag rules) captures citation-style entities that are
   better expressed as regular expressions (NORMA citations, SENTENZA refs,
   IMPORTO).
3. The statistical ``ner`` component runs last to pick up unseen proper nouns
   (institutions, parties) that slipped through the rule layer.

Typical usage::

    from src.lexreview.extraction.italian_ner import ItalianLegalNER

    ner = ItalianLegalNER()
    result = ner.extract("L'art. 2043 c.c. impone all'AGCM di ...")
    print(result.norme)          # ["Art. 2043 c.c."]
    print(result.istituzioni)    # ["AGCM"]
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Thread-safe singleton ─────────────────────────────────────────────────────

_nlp_lock = threading.Lock()
_nlp_cache: dict[str, Any] = {}  # model_name → nlp pipeline

# ── Compiled regex patterns (used both in Matcher and stand-alone helpers) ────

# NORMA — article/decree/law citations
_RE_NORMA = re.compile(
    r"""
    (?:
        [Aa]rt(?:icolo)?\.?\s*\d+(?:\s*-?\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?
        (?:\s*,\s*(?:co\.|comma)\s*\d+)?
        (?:\s+(?:c\.c\.|c\.p\.|c\.p\.c\.|c\.p\.p\.|c\.p\.a\.|t\.u\.|T\.U\.|Cost\.))?
    |
        D\.?Lgs\.?\s*(?:n\.\s*)?\d+/\d{4}
    |
        D\.?L\.?\s*(?:n\.\s*)?\d+/\d{4}
    |
        [Ll](?:egge|\.)\s*(?:n\.\s*)?\d+/\d{4}
    |
        D\.?P\.?R\.?\s*(?:n\.\s*)?\d+/\d{4}
    |
        D\.?P\.?C\.?M\.?\s*(?:del\s*)?\d{1,2}[/.]\d{1,2}[/.]\d{4}
    |
        [Rr]eg(?:olamento)?\.\s*(?:\(UE\)|CE|UE)\s+\d+/\d+
    |
        [Dd]irettiva\s+\d{4}/\d+/(?:UE|CE|CEE|EURATOM)
    )
    """,
    re.VERBOSE,
)

# SENTENZA — Italian court decision references
_RE_SENTENZA = re.compile(
    r"""
    (?:
        # Cassazione: handles both "Cass. Civ. Sez. III, n. 12345/2024"
        # and "Cassazione Penale, n. 9876/2023" (section word before comma)
        Cass(?:azione)?\.?\s+
        (?:(?:Civ(?:ile)?|Pen(?:ale)?)\.?\s*)?
        (?:Sez(?:ione)?\.?\s*[IVX\d]+\s*[,—]?\s*)?
        (?:(?:ord(?:inanza)?|sent(?:enza)?|dep(?:osito)?)\.?\s*)?
        [,]?\s*(?:n\.?\s*)?\d+/\d{4}
    |
        (?:TAR|Cons(?:iglio)?\.\s*[Ss]tato|T\.A\.R\.)\s+
        [A-ZÀÁÈÉÌÍÒÓÙÚ][a-zàáèéìíòóùú]+(?:\s+[A-ZÀÁÈÉÌÍÒÓÙÚ][a-zàáèéìíòóùú]+)?
        [,\s]+(?:sez(?:ione)?\.?\s*[IVX\d]+,?\s*)?
        (?:sent(?:enza)?\.?\s*|ord(?:inanza)?\.?\s*)?
        (?:n\.?\s*)?\d+/\d{4}
    |
        Corte\s+Cost(?:ituzionale)?[.,][,\s]+(?:sent(?:enza)?\.?\s*)?
        (?:n\.?\s*)?\d+/\d{4}
    |
        CGUE[,\s]+(?:sent(?:enza)?\.?\s*)?(?:causa\s+)?
        [A-Z]-\d+/\d{2,4}
    |
        CGE[,\s]+\d+/\d{4}
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# IMPORTO — monetary amounts in Italian format.
# The word-form branch uses a single greedy word-sequence pattern followed by
# the word "euro" — this correctly handles compound numerals like "duemila",
# "cinquecentomila", "unmilione" which cannot be matched by a simple alternation
# of root words alone.
_RE_IMPORTO = re.compile(
    r"""
    (?:
        # Numeric form: € 50.000 / 1.234 euro / EUR 500
        €\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?
    |
        \d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*(?:euro|EUR)
    |
        # Word-form: "duemila euro", "cinquantamila euro", "un milione di euro"
        # Match one or more Italian number words (possibly compound) before "euro"
        (?:
            (?:un(?:o|a)?|due|tre|quattro|cinque|sei|sette|otto|nove|dieci|
               undici|dodici|tredici|quattordici|quindici|sedici|diciassette|
               diciotto|diciannove|venti|trenta|quaranta|cinquanta|sessanta|
               settanta|ottanta|novanta|cento|mille|duemila|tremila|quattromi|
               cinquemila|diecimi|ventimi|centomi|unmilione|duemilioni|
               milione|milioni|miliardo|miliardi)
            (?:[a-z]+)?                # absorb compound suffixes (e.g. "mila" in "duemila")
            (?:\s+(?:e\s+)?           # optional linking words
               (?:un(?:o|a)?|due|tre|quattro|cinque|sei|sette|otto|nove|dieci|
                  undici|dodici|mille|mila|cento|milione|miliardo)[a-z]*
            )*
            (?:\s+di)?               # "un milione di euro"
        )\s+euro
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# DATA_GIURIDICA — legal time / deadline expressions
_RE_DATA_GIURIDICA = re.compile(
    r"""
    (?:
        entro\s+(?:il\s+termine\s+(?:perentorio\s+)?(?:di\s+)?)?
        \d+\s+(?:giorn[io]|mes[ei]|ann[io]|ore|settiman[ae])
        (?:\s+(?:dalla?\s+(?:notifica|pubblicazione|scadenza|data)|
                  dal\s+\w+|
                  dall['']?entrata\s+in\s+vigore|
                  dall['']?emanazione))?
    |
        termine\s+(?:perentorio|ordinatorio|decadenziale|prescrizionale)\s+
        (?:di\s+)?\d+\s+(?:giorn[io]|mes[ei]|ann[io])
    |
        (?:entro\s+e\s+non\s+oltre\s+(?:il\s+)?)?
        \d{1,2}\s+
        (?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)
        \s+\d{4}
    |
        (?:dalla?\s+(?:data\s+di\s+)?notifica(?:zione)?|
           dalla?\s+pubblicazione\s+in\s+Gazzetta\s+Ufficiale|
           dall['']?entrata\s+in\s+vigore)\s+
        (?:della?\s+presente\s+(?:legge|sentenza|ordinanza|circolare))?
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ── Output model ──────────────────────────────────────────────────────────────


@dataclass
class ItalianLegalEntities:
    """Structured container for Italian legal named entities.

    All lists contain deduplicated, whitespace-normalised strings preserving
    original casing.

    Attributes:
        norme:              Statutory and regulatory citations.
        sentenze:           Court decision references.
        soggetti_giuridici: Legal subjects and roles.
        istituzioni:        Courts, authorities, and public bodies.
        termini_legali:     Legal terms and Latin maxims.
        importi:            Monetary amounts.
        date_giuridiche:    Legal deadline and date expressions.
    """

    norme: list[str] = field(default_factory=list)
    sentenze: list[str] = field(default_factory=list)
    soggetti_giuridici: list[str] = field(default_factory=list)
    istituzioni: list[str] = field(default_factory=list)
    termini_legali: list[str] = field(default_factory=list)
    importi: list[str] = field(default_factory=list)
    date_giuridiche: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        """Serialise to a plain dictionary.

        Returns:
            Dict mapping label strings to entity lists.
        """
        return {
            "NORMA": self.norme,
            "SENTENZA": self.sentenze,
            "SOGGETTO_GIURIDICO": self.soggetti_giuridici,
            "ISTITUZIONE": self.istituzioni,
            "TERMINE_LEGALE": self.termini_legali,
            "IMPORTO": self.importi,
            "DATA_GIURIDICA": self.date_giuridiche,
        }


# ── EntityRuler token patterns ────────────────────────────────────────────────

_SOGGETTO_PATTERNS: list[dict[str, Any]] = [
    # Società
    {"label": "SOGGETTO_GIURIDICO", "pattern": "società a responsabilità limitata"},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "società per azioni"},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "società in nome collettivo"},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "società in accomandita semplice"},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "società in accomandita per azioni"},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "società cooperativa"},
    # Abbreviazioni
    {
        "label": "SOGGETTO_GIURIDICO",
        "pattern": [{"TEXT": {"REGEX": r"^[Ss]\.?[Rr]\.?[Ll]\.?$"}}],
    },
    {
        "label": "SOGGETTO_GIURIDICO",
        "pattern": [{"TEXT": {"REGEX": r"^[Ss]\.?[Pp]\.?[Aa]\.?$"}}],
    },
    # Ruoli processuali
    {"label": "SOGGETTO_GIURIDICO", "pattern": "pubblico ministero"},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "p.m."},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "P.M."},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "imputato"},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "indagato"},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "ricorrente"},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "resistente"},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "appellante"},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "appellato"},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "controricorrente"},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "debitore"},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "creditore"},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "fideiussore"},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "cedente"},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "cessionario"},
    # Enti pubblici generici
    {"label": "SOGGETTO_GIURIDICO", "pattern": "ente pubblico"},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "amministrazione pubblica"},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "pubblica amministrazione"},
    {"label": "SOGGETTO_GIURIDICO", "pattern": "stazione appaltante"},
]

_ISTITUZIONE_PATTERNS: list[dict[str, Any]] = [
    # Corti supreme
    {"label": "ISTITUZIONE", "pattern": "Corte di Cassazione"},
    {"label": "ISTITUZIONE", "pattern": "Corte Suprema di Cassazione"},
    {"label": "ISTITUZIONE", "pattern": "Corte Costituzionale"},
    {"label": "ISTITUZIONE", "pattern": "Consiglio di Stato"},
    {"label": "ISTITUZIONE", "pattern": "Corte dei Conti"},
    {"label": "ISTITUZIONE", "pattern": "Corte di Giustizia dell'Unione Europea"},
    {"label": "ISTITUZIONE", "pattern": "CGUE"},
    # TAR
    {
        "label": "ISTITUZIONE",
        "pattern": [
            {"TEXT": {"REGEX": r"^T\.?A\.?R\.?$"}},
        ],
    },
    {
        "label": "ISTITUZIONE",
        "pattern": [
            {"LOWER": "tar"},
            {"IS_TITLE": True, "OP": "?"},
        ],
    },
    # Autorità indipendenti
    {"label": "ISTITUZIONE", "pattern": "AGCM"},
    {"label": "ISTITUZIONE", "pattern": "Autorità Garante della Concorrenza e del Mercato"},
    {"label": "ISTITUZIONE", "pattern": "Banca d'Italia"},
    {"label": "ISTITUZIONE", "pattern": "Banca d'Italia"},
    {"label": "ISTITUZIONE", "pattern": "CONSOB"},
    {"label": "ISTITUZIONE", "pattern": "IVASS"},
    {"label": "ISTITUZIONE", "pattern": "ANAC"},
    {"label": "ISTITUZIONE", "pattern": "Garante Privacy"},
    {"label": "ISTITUZIONE", "pattern": "Garante per la protezione dei dati personali"},
    {"label": "ISTITUZIONE", "pattern": "ARERA"},
    {"label": "ISTITUZIONE", "pattern": "AGCOM"},
    # Istituzioni statali
    {"label": "ISTITUZIONE", "pattern": "Ministero della Giustizia"},
    {"label": "ISTITUZIONE", "pattern": "Ministero dell'Economia e delle Finanze"},
    {"label": "ISTITUZIONE", "pattern": "MEF"},
    {"label": "ISTITUZIONE", "pattern": "Agenzia delle Entrate"},
    {"label": "ISTITUZIONE", "pattern": "Agenzia delle Dogane"},
    {"label": "ISTITUZIONE", "pattern": "Guardia di Finanza"},
    {"label": "ISTITUZIONE", "pattern": "Procura della Repubblica"},
    {"label": "ISTITUZIONE", "pattern": "Consiglio Superiore della Magistratura"},
    {"label": "ISTITUZIONE", "pattern": "CSM"},
    {"label": "ISTITUZIONE", "pattern": "Parlamento europeo"},
    {"label": "ISTITUZIONE", "pattern": "Commissione europea"},
]

_TERMINE_LEGALE_PATTERNS: list[dict[str, Any]] = [
    # Responsabilità
    {"label": "TERMINE_LEGALE", "pattern": "inadempimento"},
    {"label": "TERMINE_LEGALE", "pattern": "adempimento"},
    {"label": "TERMINE_LEGALE", "pattern": "mora debendi"},
    {"label": "TERMINE_LEGALE", "pattern": "mora accipiendi"},
    {"label": "TERMINE_LEGALE", "pattern": "exceptio inadimpleti contractus"},
    {"label": "TERMINE_LEGALE", "pattern": "responsabilità aquiliana"},
    {"label": "TERMINE_LEGALE", "pattern": "responsabilità extracontrattuale"},
    {"label": "TERMINE_LEGALE", "pattern": "responsabilità contrattuale"},
    {"label": "TERMINE_LEGALE", "pattern": "responsabilità precontrattuale"},
    {"label": "TERMINE_LEGALE", "pattern": "nesso causale"},
    {"label": "TERMINE_LEGALE", "pattern": "nesso di causalità"},
    {"label": "TERMINE_LEGALE", "pattern": "danno emergente"},
    {"label": "TERMINE_LEGALE", "pattern": "lucro cessante"},
    {"label": "TERMINE_LEGALE", "pattern": "danno non patrimoniale"},
    {"label": "TERMINE_LEGALE", "pattern": "danno biologico"},
    {"label": "TERMINE_LEGALE", "pattern": "danno morale"},
    # Contratti
    {"label": "TERMINE_LEGALE", "pattern": "nullità"},
    {"label": "TERMINE_LEGALE", "pattern": "annullabilità"},
    {"label": "TERMINE_LEGALE", "pattern": "rescissione"},
    {"label": "TERMINE_LEGALE", "pattern": "risoluzione"},
    {"label": "TERMINE_LEGALE", "pattern": "eccezione di inadempimento"},
    {"label": "TERMINE_LEGALE", "pattern": "clausola penale"},
    {"label": "TERMINE_LEGALE", "pattern": "caparra confirmatoria"},
    {"label": "TERMINE_LEGALE", "pattern": "caparra penitenziale"},
    {"label": "TERMINE_LEGALE", "pattern": "mutuo dissenso"},
    # Processo
    {"label": "TERMINE_LEGALE", "pattern": "litispendenza"},
    {"label": "TERMINE_LEGALE", "pattern": "cosa giudicata"},
    {"label": "TERMINE_LEGALE", "pattern": "giudicato"},
    {"label": "TERMINE_LEGALE", "pattern": "onere della prova"},
    {"label": "TERMINE_LEGALE", "pattern": "inversione dell'onere della prova"},
    {"label": "TERMINE_LEGALE", "pattern": "presunzione legale"},
    {"label": "TERMINE_LEGALE", "pattern": "legittimazione attiva"},
    {"label": "TERMINE_LEGALE", "pattern": "legittimazione passiva"},
    {"label": "TERMINE_LEGALE", "pattern": "interesse ad agire"},
    {"label": "TERMINE_LEGALE", "pattern": "difetto di giurisdizione"},
    {"label": "TERMINE_LEGALE", "pattern": "incompetenza per materia"},
    # Diritto penale
    {"label": "TERMINE_LEGALE", "pattern": "dolo"},
    {"label": "TERMINE_LEGALE", "pattern": "colpa grave"},
    {"label": "TERMINE_LEGALE", "pattern": "colpa lieve"},
    {"label": "TERMINE_LEGALE", "pattern": "preterintenzione"},
    {"label": "TERMINE_LEGALE", "pattern": "elemento soggettivo"},
    {"label": "TERMINE_LEGALE", "pattern": "elemento oggettivo"},
    # Prescrizione
    {"label": "TERMINE_LEGALE", "pattern": "prescrizione"},
    {"label": "TERMINE_LEGALE", "pattern": "decadenza"},
    {"label": "TERMINE_LEGALE", "pattern": "interruzione della prescrizione"},
    {"label": "TERMINE_LEGALE", "pattern": "sospensione della prescrizione"},
    # Diritto amministrativo
    {"label": "TERMINE_LEGALE", "pattern": "silenzio amministrativo"},
    {"label": "TERMINE_LEGALE", "pattern": "eccesso di potere"},
    {"label": "TERMINE_LEGALE", "pattern": "violazione di legge"},
    {"label": "TERMINE_LEGALE", "pattern": "incompetenza"},
    {"label": "TERMINE_LEGALE", "pattern": "discrezionalità amministrativa"},
    {"label": "TERMINE_LEGALE", "pattern": "autotutela"},
    # Latini
    {"label": "TERMINE_LEGALE", "pattern": "ultra petita"},
    {"label": "TERMINE_LEGALE", "pattern": "fumus boni iuris"},
    {"label": "TERMINE_LEGALE", "pattern": "periculum in mora"},
    {"label": "TERMINE_LEGALE", "pattern": "in dubio pro reo"},
    {"label": "TERMINE_LEGALE", "pattern": "pacta sunt servanda"},
    {"label": "TERMINE_LEGALE", "pattern": "rebus sic stantibus"},
    {"label": "TERMINE_LEGALE", "pattern": "nemo plus iuris"},
]

_ALL_RULER_PATTERNS: list[dict[str, Any]] = (
    _SOGGETTO_PATTERNS + _ISTITUZIONE_PATTERNS + _TERMINE_LEGALE_PATTERNS
)


# ── spaCy pipeline loader ─────────────────────────────────────────────────────


def _load_italian_nlp(model_name: str) -> Any:
    """Return a cached, ruler-enhanced Italian spaCy pipeline.

    Thread-safe via double-checked locking.

    Args:
        model_name: spaCy pipeline identifier (e.g. ``"it_core_news_lg"``).

    Returns:
        A ``Language`` object with the Italian legal ``EntityRuler`` injected.

    Raises:
        RuntimeError: When spaCy or the model is not installed.
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
                f"Italian spaCy model '{model_name}' not found. "
                f"Run: python -m spacy download {model_name}"
            ) from exc

        # Inject Italian legal EntityRuler *before* the statistical NER.
        if "italian_legal_ruler" not in nlp.pipe_names:
            ruler = nlp.add_pipe(
                "entity_ruler",
                name="italian_legal_ruler",
                before="ner" if "ner" in nlp.pipe_names else None,
            )
            ruler.add_patterns(_ALL_RULER_PATTERNS)  # type: ignore[union-attr]
            log.info(
                "Italian legal EntityRuler injected",
                extra={
                    "model": model_name,
                    "pattern_count": len(_ALL_RULER_PATTERNS),
                },
            )

        log.info("Italian spaCy pipeline loaded", extra={"model": model_name})
        _nlp_cache[model_name] = nlp
        return nlp


# ── ItalianLegalNER ───────────────────────────────────────────────────────────


class ItalianLegalNER:
    """Named Entity Recogniser for Italian legal documents.

    Layers three complementary recognition strategies:

    1. **Regex extraction** — high-precision patterns for NORMA, SENTENZA,
       IMPORTO, and DATA_GIURIDICA citations (impossible to capture reliably
       with token-level rules alone).
    2. **EntityRuler** — token-level rules for SOGGETTO_GIURIDICO, ISTITUZIONE,
       and TERMINE_LEGALE (injected into the spaCy pipeline at load-time).
    3. **Statistical NER** (``it_core_news_lg``) — catches unseen proper nouns,
       additional institution names, and dates recognised by the base model.

    Args:
        model_name: spaCy pipeline to use (default ``"it_core_news_lg"``).

    Example::

        ner = ItalianLegalNER()
        result = ner.extract(
            "Ai sensi dell'art. 2043 c.c. l'AGCM ha irrogato una sanzione "
            "di € 50.000 con provvedimento del 15 marzo 2024."
        )
        print(result.norme)       # ["art. 2043 c.c."]
        print(result.istituzioni) # ["AGCM"]
        print(result.importi)     # ["€ 50.000"]
    """

    DEFAULT_MODEL: str = "it_core_news_lg"

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name: str = model_name or self.DEFAULT_MODEL

    def extract(self, text: str) -> ItalianLegalEntities:
        """Extract all Italian legal entities from *text*.

        Args:
            text: Raw Italian legal text (statute, judgment, contract, etc.).

        Returns:
            :class:`ItalianLegalEntities` with deduplicated entity lists.

        Raises:
            RuntimeError: When the spaCy model cannot be loaded.
        """
        result = ItalianLegalEntities()

        # ── Layer 1: Regex extraction ──────────────────────────────────────
        result.norme = _dedupe(_RE_NORMA.findall(text))
        result.sentenze = _dedupe(_RE_SENTENZA.findall(text))
        result.importi = _dedupe(_RE_IMPORTO.findall(text))
        result.date_giuridiche = _dedupe(_RE_DATA_GIURIDICA.findall(text))

        # ── Layer 2 & 3: spaCy EntityRuler + statistical NER ──────────────
        nlp = _load_italian_nlp(self._model_name)
        doc = nlp(text)

        seen: set[tuple[str, str]] = set()

        for ent in doc.ents:
            norm = ent.text.strip()
            key = (ent.label_, norm)
            if key in seen or not norm:
                continue
            seen.add(key)

            label = ent.label_
            if label == "SOGGETTO_GIURIDICO":
                result.soggetti_giuridici.append(norm)
            elif label == "ISTITUZIONE":
                result.istituzioni.append(norm)
            elif label == "TERMINE_LEGALE":
                result.termini_legali.append(norm)
            # Capture any additional institutions/orgs the statistical model found.
            elif label == "ORG":
                result.istituzioni.append(norm)

        # Dedupe again after spaCy pass.
        result.soggetti_giuridici = _dedupe(result.soggetti_giuridici)
        result.istituzioni = _dedupe(result.istituzioni)
        result.termini_legali = _dedupe(result.termini_legali)

        log.debug(
            "ItalianLegalNER extraction complete",
            extra={
                "norme": len(result.norme),
                "sentenze": len(result.sentenze),
                "soggetti": len(result.soggetti_giuridici),
                "istituzioni": len(result.istituzioni),
                "termini": len(result.termini_legali),
                "importi": len(result.importi),
                "date": len(result.date_giuridiche),
            },
        )
        return result

    def extract_batch(self, texts: list[str]) -> list[ItalianLegalEntities]:
        """Extract entities from multiple Italian legal texts.

        Uses ``nlp.pipe()`` for efficient batch processing of the spaCy layers
        while running regex layers per-text.

        Args:
            texts: List of raw Italian legal text strings.

        Returns:
            List of :class:`ItalianLegalEntities` in the same order as *texts*.
        """
        if not texts:
            return []

        nlp = _load_italian_nlp(self._model_name)
        results: list[ItalianLegalEntities] = []

        docs = list(nlp.pipe(texts, batch_size=16))

        for text, doc in zip(texts, docs):
            result = ItalianLegalEntities(
                norme=_dedupe(_RE_NORMA.findall(text)),
                sentenze=_dedupe(_RE_SENTENZA.findall(text)),
                importi=_dedupe(_RE_IMPORTO.findall(text)),
                date_giuridiche=_dedupe(_RE_DATA_GIURIDICA.findall(text)),
            )

            seen: set[tuple[str, str]] = set()
            for ent in doc.ents:
                norm = ent.text.strip()
                key = (ent.label_, norm)
                if key in seen or not norm:
                    continue
                seen.add(key)

                label = ent.label_
                if label == "SOGGETTO_GIURIDICO":
                    result.soggetti_giuridici.append(norm)
                elif label == "ISTITUZIONE":
                    result.istituzioni.append(norm)
                elif label == "TERMINE_LEGALE":
                    result.termini_legali.append(norm)
                elif label == "ORG":
                    result.istituzioni.append(norm)

            result.soggetti_giuridici = _dedupe(result.soggetti_giuridici)
            result.istituzioni = _dedupe(result.istituzioni)
            result.termini_legali = _dedupe(result.termini_legali)
            results.append(result)

        return results


# ── Internal helpers ──────────────────────────────────────────────────────────


def _dedupe(items: list[str]) -> list[str]:
    """Return a deduplicated, whitespace-normalised list preserving order.

    Args:
        items: Raw string list that may contain duplicates or excess whitespace.

    Returns:
        Deduplicated list of stripped strings.
    """
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        norm = " ".join(item.split())
        if norm and norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result
