"""Normattiva connector — NIR/URN API (normattiva.it).

Fetches vigente Italian legislation from the Normattiva REST API, which
exposes every act in the Italian legal system via NIR URNs.

Public API (no authentication required):
    https://www.normattiva.it/uri-res/N2Ls?urn=<NIR_URN>&parse=...

Key design decisions
--------------------
- Each ``<articolo>`` element becomes one :class:`~src.ingestion.loader.Document`
  so that retrieval can target article-level granularity (e.g.
  "Art. 2043 c.c." as a single indexable unit).
- The connector uses the NIR XML export endpoint.  HTML exports exist but
  contain significantly more noise (navigation chrome, JavaScript refs).
- Rate limit: ≤ 1 req/s, as specified by normattiva.it robots.txt.

Supported codici (pre-built URN shortcuts):
    - ``codice_civile``
    - ``codice_penale``
    - ``codice_procedura_civile``
    - ``codice_procedura_penale``
    - ``codice_consumo``
    - ``tuir``          (T.U.I.R. — Testo Unico Imposte sui Redditi)
    - ``dlgs_231``      (D.Lgs. 231/2001)
    - ``gdpr``          (Reg. UE 2016/679 — italiano)

Arbitrary NIR URNs can be passed via ``urn=`` parameter.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from src.ingestion.loader import Document
from src.italia.connectors.base import BaseConnector, ConnectorParseError
from src.italia.metadata import ItalianLegalMetadata, TipoDocumento
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── NIR URN shortcuts ─────────────────────────────────────────────────────────
_CODICI_URN: dict[str, str] = {
    "codice_civile": "urn:nir:stato:regio.decreto:1942-03-16;262",
    "codice_penale": "urn:nir:stato:regio.decreto:1930-10-19;1398",
    "codice_procedura_civile": "urn:nir:stato:regio.decreto:1940-10-28;1443",
    "codice_procedura_penale": "urn:nir:stato:decreto.del.presidente.della.repubblica:1988-09-22;477",
    "codice_consumo": "urn:nir:stato:decreto.legislativo:2005-09-06;206",
    "tuir": "urn:nir:stato:decreto.del.presidente.della.repubblica:1986-12-22;917",
    "dlgs_231": "urn:nir:stato:decreto.legislativo:2001-06-08;231",
    "gdpr": "urn:nir:unione.europea:regolamento:2016-04-27;679",
}

_BASE_URL = "https://www.normattiva.it/uri-res/N2Ls"

# Map NIR act-type prefixes to TipoDocumento
_TIPO_MAP: dict[str, TipoDocumento] = {
    "legge": TipoDocumento.LEGGE,
    "decreto.legislativo": TipoDocumento.DECRETO_LEGISLATIVO,
    "decreto-legge": TipoDocumento.DECRETO_LEGGE,
    "decreto.legge": TipoDocumento.DECRETO_LEGGE,
    "regio.decreto": TipoDocumento.CODICE,
    "decreto.del.presidente.della.repubblica": TipoDocumento.REGOLAMENTO,
    "regolamento": TipoDocumento.REGOLAMENTO,
}


def _tipo_from_urn(urn: str) -> TipoDocumento:
    for key, tipo in _TIPO_MAP.items():
        if key in urn:
            return tipo
    return TipoDocumento.LEGGE


def _anno_from_urn(urn: str) -> int | None:
    m = re.search(r":(\d{4})-\d{2}-\d{2};", urn)
    return int(m.group(1)) if m else None


def _numero_from_urn(urn: str) -> str | None:
    m = re.search(r";(\d+)$", urn)
    return m.group(1) if m else None


def _date_from_urn(urn: str) -> date | None:
    m = re.search(r":(\d{4}-\d{2}-\d{2});", urn)
    if m:
        try:
            return date.fromisoformat(m.group(1))
        except ValueError:
            pass
    return None


class NormativaConnector(BaseConnector):
    """Fetches vigente legislation from normattiva.it NIR/URN API.

    Args:
        rate_limit_rps: Max requests per second (default 0.8 — conservative
                        for normattiva.it's strict rate limiting).
        timeout:        HTTP timeout in seconds.

    Example::

        conn = NormativaConnector()
        docs = conn.fetch(codice="codice_civile")
        # or
        docs = conn.fetch(urn="urn:nir:stato:legge:2003-01-09;63")
    """

    source_name = "normattiva"

    def __init__(
        self,
        rate_limit_rps: float = 0.8,
        timeout: int = 45,
    ) -> None:
        super().__init__(rate_limit_rps=rate_limit_rps, timeout=timeout)

    def fetch(  # type: ignore[override]
        self,
        codice: str | None = None,
        urn: str | None = None,
        limit: int | None = None,
        **_: Any,
    ) -> list[Document]:
        """Fetch articles from a codice or arbitrary NIR URN.

        Args:
            codice: Shortcut name from ``_CODICI_URN`` (e.g. ``"codice_civile"``).
            urn:    Explicit NIR URN (takes precedence over ``codice``).
            limit:  Maximum number of articles to return (``None`` = all).

        Returns:
            One :class:`~src.ingestion.loader.Document` per ``<articolo>`` in
            the act.

        Raises:
            :class:`~src.italia.connectors.base.ConnectorError`: On HTTP or
                parse failure.
            ValueError: When neither ``codice`` nor ``urn`` is provided.
        """
        if urn is None and codice is None:
            raise ValueError("Provide either 'codice' or 'urn'.")

        resolved_urn = urn or _CODICI_URN.get(codice or "")
        if not resolved_urn:
            raise ValueError(
                f"Unknown codice '{codice}'. "
                f"Valid values: {sorted(_CODICI_URN.keys())}"
            )

        log.info(
            "Fetching from Normattiva",
            extra={"urn": resolved_urn, "source": self.source_name},
        )

        xml_text = self._fetch_xml(resolved_urn)
        documents = self._parse_xml(xml_text, resolved_urn)

        if limit is not None:
            documents = documents[:limit]

        log.info(
            "Normattiva fetch complete",
            extra={"urn": resolved_urn, "articles": len(documents)},
        )
        return documents

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fetch_xml(self, urn: str) -> str:
        """Retrieve XML export from normattiva.it.

        Args:
            urn: NIR URN to resolve.

        Returns:
            Raw XML string.
        """
        resp = self._get(_BASE_URL, params={"urn": urn, "notNIR": "true"})
        return resp.text

    def _parse_xml(self, xml_text: str, urn: str) -> list[Document]:
        """Parse XML response into per-article Documents.

        Falls back to treating the entire text as one document when no
        ``<articolo>`` elements are found (e.g. preamble-only acts).

        Args:
            xml_text: Raw XML string from the API.
            urn:      NIR URN for metadata population.

        Returns:
            List of :class:`~src.ingestion.loader.Document` objects.
        """
        try:
            from lxml import etree  # noqa: PLC0415
        except ImportError as exc:
            raise ConnectorParseError(
                "lxml is required for XML parsing. Run: pip install lxml"
            ) from exc

        try:
            root = etree.fromstring(xml_text.encode())
        except etree.XMLSyntaxError:
            # Normattiva sometimes returns HTML for error pages.
            log.warning(
                "Normattiva returned invalid XML; treating as single document",
                extra={"urn": urn, "preview": xml_text[:200]},
            )
            return self._fallback_document(xml_text, urn)

        # Extract all namespaces for XPath
        ns = root.nsmap
        # Try to find articolo elements — handle both NIR and generic XML
        articles = root.xpath("//*[local-name()='articolo']") or root.xpath(
            "//*[local-name()='art']"
        )

        tipo = _tipo_from_urn(urn)
        anno = _anno_from_urn(urn)
        numero = _numero_from_urn(urn)
        data_vigenza = _date_from_urn(urn)

        if not articles:
            # Act has no article structure — index as single document.
            text = " ".join(root.itertext()).strip()
            text = re.sub(r"\s+", " ", text)
            if not text:
                return []
            meta = ItalianLegalMetadata(
                fonte=self.source_name,
                tipo_documento=tipo,
                urn_nir=urn,
                numero_atto=self._format_numero_atto(tipo, numero, anno),
                anno=anno,
                data_vigenza=data_vigenza,
            )
            return [self._make_document(text, meta, urn)]

        documents: list[Document] = []
        for art in articles:
            # Extract article number from attributes or child elements.
            art_num: str | None = (
                art.get("id")
                or art.get("num")
                or art.get("numero")
                or self._find_text(art, "num")
                or self._find_text(art, "numero")
            )
            rubrica = self._find_text(art, "rubrica") or ""
            body_parts = [
                " ".join(el.itertext()).strip()
                for el in art.xpath(".//*[local-name()='comma'] | .//*[local-name()='body']")
            ]
            if not body_parts:
                body_parts = [" ".join(art.itertext()).strip()]

            full_text = f"{rubrica}\n\n" + "\n\n".join(body_parts) if rubrica else "\n\n".join(body_parts)
            full_text = re.sub(r"\s+", " ", full_text).strip()

            if not full_text:
                continue

            meta = ItalianLegalMetadata(
                fonte=self.source_name,
                tipo_documento=tipo,
                urn_nir=urn,
                numero_atto=self._format_numero_atto(tipo, numero, anno),
                anno=anno,
                data_vigenza=data_vigenza,
                articolo=f"Art. {art_num}" if art_num else None,
            )
            fname = f"{urn.split(';')[-1]}_art{art_num or len(documents)}.xml"
            doc_uri = f"{urn}~art{art_num}" if art_num else urn
            documents.append(self._make_document(full_text, meta, doc_uri, filename=fname))

        return documents

    @staticmethod
    def _find_text(element: Any, tag: str) -> str | None:
        """Find an immediate child element by local-name and return its text."""
        found = element.xpath(f".//*[local-name()='{tag}']")
        if found:
            return " ".join(found[0].itertext()).strip() or None
        return None

    @staticmethod
    def _format_numero_atto(tipo: TipoDocumento, numero: str | None, anno: int | None) -> str | None:
        if not numero:
            return None
        prefix_map = {
            TipoDocumento.LEGGE: "L.",
            TipoDocumento.DECRETO_LEGISLATIVO: "D.Lgs.",
            TipoDocumento.DECRETO_LEGGE: "D.L.",
            TipoDocumento.CODICE: "R.D.",
            TipoDocumento.REGOLAMENTO: "D.P.R.",
        }
        prefix = prefix_map.get(tipo, "")
        return f"{prefix} {numero}/{anno}".strip() if anno else f"{prefix} {numero}".strip()

    def _fallback_document(self, text: str, urn: str) -> list[Document]:
        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return []
        meta = ItalianLegalMetadata(
            fonte=self.source_name,
            tipo_documento=_tipo_from_urn(urn),
            urn_nir=urn,
            anno=_anno_from_urn(urn),
        )
        return [self._make_document(cleaned, meta, urn)]
