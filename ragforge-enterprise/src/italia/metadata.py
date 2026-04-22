"""Italian legal metadata schema for RAGForge Italia.

Defines :class:`TipoDocumento` (document-type enum) and
:class:`ItalianLegalMetadata` (typed provenance dataclass).

These types act as the *typing layer* for Italian legal documents: all
connectors stamp their output :class:`~src.ingestion.loader.Document`
objects with Italian-specific fields via :meth:`ItalianLegalMetadata.to_extra_dict`,
which serialises the dataclass to a flat ``dict[str, Any]`` compatible with
the existing ``IndexedChunk.metadata`` payload field stored in Qdrant.

Round-trip::

    meta = ItalianLegalMetadata(
        fonte="normattiva",
        tipo_documento=TipoDocumento.LEGGE,
        urn_nir="urn:nir:stato:legge:2003-01-09;63",
        numero_atto="L. 63/2003",
        anno=2003,
    )
    d = meta.to_extra_dict()
    assert ItalianLegalMetadata.from_extra_dict(d) == meta
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import date
from enum import Enum
from typing import Any


# ── Document-type taxonomy ────────────────────────────────────────────────────


class TipoDocumento(str, Enum):
    """Controlled vocabulary for Italian legal document types.

    Using ``str`` as the mixin makes enum values JSON-serialisable without
    a custom encoder and compatible with Qdrant payload filters.
    """

    CODICE = "codice"
    LEGGE = "legge"
    DECRETO_LEGISLATIVO = "decreto_legislativo"
    DECRETO_LEGGE = "decreto_legge"
    REGOLAMENTO = "regolamento"
    DIRETTIVA_EU = "direttiva_eu"
    SENTENZA_CASSAZIONE = "sentenza_cassazione"
    SENTENZA_TAR = "sentenza_tar"
    SENTENZA_COSTITUZIONALE = "sentenza_costituzionale"
    CIRCOLARE = "circolare"
    DOTTRINA = "dottrina"
    PROVVEDIMENTO = "provvedimento"
    # Phase 6 additions
    SENTENZA = "sentenza"           # Generic sentenza (for SIECIC/SICID)
    MASSIMA = "massima"             # Headnote / massima (LexisNexis)
    ATTO_NOTARILE = "atto_notarile" # Notarial deed (Notartel)
    ALTRO = "altro"                 # Catch-all for FileNet/CMIS artefacts
    UNKNOWN = "unknown"


# ── Metadata dataclass ────────────────────────────────────────────────────────

_DATE_FMT = "%Y-%m-%d"

# Sentinel used to distinguish "not provided" from None so that optional
# fields can still be round-tripped through dict serialisation.
_MISSING = object()


@dataclass
class ItalianLegalMetadata:
    """Typed provenance metadata for Italian legal documents.

    All fields are optional (``None``) except ``fonte`` and ``tipo_documento``
    which every connector must supply.

    Attributes:
        fonte:              Source identifier, e.g. ``"normattiva"``,
                            ``"cassazione"``, ``"tar"``, ``"eurlex"``.
        tipo_documento:     Document-type classification (:class:`TipoDocumento`).

        urn_nir:            NIR uniform resource name –
                            ``"urn:nir:stato:legge:2003-01-09;63"``.
        numero_atto:        Human-readable act identifier –
                            ``"D.Lgs. 231/2001"``, ``"L. 241/1990"``.
        anno:               Year of the act/sentenza.
        data_vigenza:       Date from which the norm is in force.
        data_abrogazione:   Date on which the norm was repealed, ``None``
                            when still vigente.
        articolo:           Article identifier – ``"Art. 2043"``.
        comma:              Paragraph identifier – ``"comma 1"``.

        numero_sentenza:    Full citation, e.g. ``"Cass. Civ. n. 12345/2024"``.
        sezione:            Court section, e.g. ``"Sezione Lavoro"``.
        data_deposito:      Date the judgment was filed.
        massima:            Official headnote / massima ufficiale.
        ric_numero:         Ricorso (appeal) number.

        materia:            Subject-matter tags –
                            ``["diritto civile", "responsabilità contrattuale"]``.
        parole_chiave:      Free-text keyword tags.
    """

    # Required
    fonte: str
    tipo_documento: TipoDocumento
    url_fonte: str | None = None  # Source URL / URI for the document

    # Legislative identity
    urn_nir: str | None = None
    numero_atto: str | None = None
    anno: int | None = None
    data_vigenza: date | None = None
    data_abrogazione: date | None = None
    articolo: str | None = None
    comma: str | None = None

    # Judicial identity
    numero_sentenza: str | None = None
    sezione: str | None = None
    data_deposito: date | None = None
    massima: str | None = None
    ric_numero: str | None = None

    # Classification
    materia: list[str] = field(default_factory=list)
    parole_chiave: list[str] = field(default_factory=list)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_extra_dict(self) -> dict[str, Any]:
        """Serialise to a flat ``dict`` compatible with ``IndexedChunk.metadata``.

        - :class:`TipoDocumento` values are stored as their string value.
        - :class:`~datetime.date` values are stored as ``"YYYY-MM-DD"`` strings.
        - ``None`` values are stored as ``None`` (not omitted) so that Qdrant
          payload filters can match on ``IS NULL``.
        - List fields are stored as JSON arrays.

        Returns:
            Flat dictionary prefixed with ``"it_"`` namespace to avoid
            collisions with the base ``IndexedChunk.metadata`` fields
            (``source_path``, ``filename``, etc.).
        """
        return {
            "it_fonte": self.fonte,
            "it_tipo_documento": self.tipo_documento.value,
            "it_url_fonte": self.url_fonte,
            "it_urn_nir": self.urn_nir,
            "it_numero_atto": self.numero_atto,
            "it_anno": self.anno,
            "it_data_vigenza": (
                self.data_vigenza.strftime(_DATE_FMT) if self.data_vigenza else None
            ),
            "it_data_abrogazione": (
                self.data_abrogazione.strftime(_DATE_FMT)
                if self.data_abrogazione
                else None
            ),
            "it_articolo": self.articolo,
            "it_comma": self.comma,
            "it_numero_sentenza": self.numero_sentenza,
            "it_sezione": self.sezione,
            "it_data_deposito": (
                self.data_deposito.strftime(_DATE_FMT) if self.data_deposito else None
            ),
            "it_massima": self.massima,
            "it_ric_numero": self.ric_numero,
            "it_materia": self.materia,
            "it_parole_chiave": self.parole_chiave,
        }

    @classmethod
    def from_extra_dict(cls, d: dict[str, Any]) -> "ItalianLegalMetadata":
        """Deserialise from a flat metadata dictionary (inverse of :meth:`to_extra_dict`).

        Args:
            d: Dictionary as stored in ``IndexedChunk.metadata`` or a Qdrant
               payload. Unknown keys are silently ignored.

        Returns:
            :class:`ItalianLegalMetadata` instance.

        Raises:
            KeyError:   When ``it_fonte`` or ``it_tipo_documento`` are absent.
            ValueError: When ``it_tipo_documento`` is not a valid enum member.
        """

        def _parse_date(v: str | None) -> date | None:
            if not v:
                return None
            try:
                return date.fromisoformat(v)
            except (ValueError, TypeError):
                return None

        return cls(
            fonte=d["it_fonte"],
            tipo_documento=TipoDocumento(d["it_tipo_documento"]),
            url_fonte=d.get("it_url_fonte"),
            urn_nir=d.get("it_urn_nir"),
            numero_atto=d.get("it_numero_atto"),
            anno=d.get("it_anno"),
            data_vigenza=_parse_date(d.get("it_data_vigenza")),
            data_abrogazione=_parse_date(d.get("it_data_abrogazione")),
            articolo=d.get("it_articolo"),
            comma=d.get("it_comma"),
            numero_sentenza=d.get("it_numero_sentenza"),
            sezione=d.get("it_sezione"),
            data_deposito=_parse_date(d.get("it_data_deposito")),
            massima=d.get("it_massima"),
            ric_numero=d.get("it_ric_numero"),
            materia=d.get("it_materia") or [],
            parole_chiave=d.get("it_parole_chiave") or [],
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def is_vigente(self, as_of: date | None = None) -> bool:
        """Return ``True`` when the norm has not been repealed.

        Args:
            as_of: Reference date (defaults to today).

        Returns:
            ``True`` if vigente (no abrogation date, or abrogation in the
            future).
        """
        ref = as_of or date.today()
        if self.data_abrogazione is None:
            return True
        return self.data_abrogazione > ref

    def citation(self) -> str:
        """Return a compact human-readable citation string.

        Returns a sentenza citation when :attr:`numero_sentenza` is set, an
        act citation when :attr:`numero_atto` is set, or the ``urn_nir``
        as a last resort.

        Examples:
            ``"Cass. Civ. n. 12345/2024"``
            ``"D.Lgs. 231/2001, Art. 25-ter"``
            ``"urn:nir:stato:legge:2003-01-09;63"``
        """
        if self.numero_sentenza:
            return self.numero_sentenza
        parts: list[str] = []
        if self.numero_atto:
            parts.append(self.numero_atto)
        if self.articolo:
            parts.append(self.articolo)
        if parts:
            return ", ".join(parts)
        return self.urn_nir or f"{self.fonte}/{self.tipo_documento.value}"
