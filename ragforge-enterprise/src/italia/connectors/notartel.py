"""Notartel connector — Italian notary network document integration.

Notartel S.p.A. manages the digital infrastructure of the Consiglio Nazionale
del Notariato (CNN).  This connector reads and exports atti notarili (notarial
deeds) using the Notartel XML schema standardised under art. 62-bis L. 340/2000
and D.Lgs. 110/2010.

Capabilities
~~~~~~~~~~~~
- **Read**: Fetches atti notarili from the Notartel REST/SOAP gateway.
- **Export**: Serialises RAGForge search results into Notartel-compatible XML.
- **Stub mode**: When ``NOTARTEL_STUB=true`` (or ``stub=True`` in the
  constructor), fixture responses are returned without making HTTP calls.  This
  enables full test coverage without active Notartel credentials.

Authentication
~~~~~~~~~~~~~~
Bearer token from the Notartel Identity Provider (IdP)::

    Authorization: Bearer <NOTARTEL_TOKEN>

Tokens are provisioned by the Consiglio Nazionale del Notariato and
expire every 8 hours.  The connector caches the token and never refreshes
it automatically — the caller must pass a fresh token when the current one
expires.

XML schema
~~~~~~~~~~
The Notartel XML schema (``atti-notarili-v3``) wraps each atto in::

    <AttoNotarile xmlns="...">
        <Repertorio>...</Repertorio>
        <Data>YYYY-MM-DD</Data>
        <Notaio>...</Notaio>
        <Oggetto>...</Oggetto>
        <Testo>...</Testo>
        <Parti>
            <Parte ruolo="rogante|disponente|beneficiario">...</Parte>
        </Parti>
    </AttoNotarile>

References
~~~~~~~~~~
- Notartel S.p.A.: https://www.notartel.it
- D.Lgs. 110/2010 — Atti pubblici notarili informatici
- L. 340/2000 — Norme per la semplificazione di alcune tipologie di atti
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import httpx

from src.ingestion.loader import Document, DocumentMetadata
from src.italia.connectors.base import BaseConnector, ConnectorError, ConnectorHTTPError
from src.italia.metadata import ItalianLegalMetadata, TipoDocumento
from src.utils.logger import get_logger

log = get_logger(__name__)

# XML namespace for the Notartel atti-notarili schema.
NOTARTEL_NS = "https://www.notartel.it/schema/atti-notarili-v3"


# ── Fixture data for stub mode ─────────────────────────────────────────────────

_STUB_XML_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<ListaAttiNotarili xmlns="{ns}" generato="{ts}" fonte="stub">
  <AttoNotarile>
    <Repertorio>12345/2024</Repertorio>
    <Data>2024-03-15</Data>
    <Notaio>Mario Rossi</Notaio>
    <Sede>Milano</Sede>
    <Oggetto>Atto di compravendita immobiliare ex art. 1470 c.c.</Oggetto>
    <Testo>Con il presente atto, il signor Giovanni Bianchi, nella qualità di
venditore, trasferisce al signor Luca Verdi, nella qualità di acquirente,
la piena proprietà dell&#39;immobile sito in Milano, via Roma n. 42, foglio 10 mappale 200.
Il prezzo concordato è di Euro 450.000,00 (quattrocentocinquantamila/00),
già integralmente versato.</Testo>
    <Parti>
      <Parte ruolo="disponente">Giovanni Bianchi, CF: BNCGNN70A01F205K</Parte>
      <Parte ruolo="beneficiario">Luca Verdi, CF: VRDLCU85B15G273M</Parte>
    </Parti>
  </AttoNotarile>
</ListaAttiNotarili>
"""


class NotartelConnector(BaseConnector):
    """Read/export connector for the Notartel Italian notary network.

    Args:
        base_url:       Notartel REST gateway base URL.
                        Example: ``"https://api.notartel.it/v3"``
        token:          Active Bearer token from Notartel IdP.
        stub:           If ``True`` (or env ``NOTARTEL_STUB=true``), use
                        fixture data instead of real HTTP calls.
        rate_limit_rps: Max requests per second (default: 1.0).
        timeout:        HTTP timeout in seconds (default: 60).
    """

    source_name = "notartel"

    def __init__(
        self,
        base_url: str = "https://api.notartel.it/v3",
        token: str = "",
        stub: bool | None = None,
        rate_limit_rps: float = 1.0,
        timeout: int = 60,
    ) -> None:
        super().__init__(rate_limit_rps=rate_limit_rps, timeout=timeout)
        self._base_url = base_url.rstrip("/")
        self._token = token
        # Check env var if stub not explicitly set.
        if stub is None:
            stub = os.getenv("NOTARTEL_STUB", "false").lower() in ("true", "1", "yes")
        self._stub = stub

        if self._stub:
            log.info("NotartelConnector running in stub mode — no HTTP calls will be made.")

    @property
    def _auth_headers(self) -> dict[str, str]:
        """Return Bearer auth headers."""
        return {"Authorization": f"Bearer {self._token}"}

    # ── Fetch ─────────────────────────────────────────────────────────────────

    def fetch(  # type: ignore[override]
        self,
        notaio: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        repertorio: str | None = None,
        max_results: int = 50,
    ) -> list[Document]:
        """Fetch atti notarili from the Notartel gateway.

        Args:
            notaio:      Filter by notary name or fiscal code.
            from_date:   Start date (``YYYY-MM-DD``).
            to_date:     End date (``YYYY-MM-DD``).
            repertorio:  Exact repertorio number (e.g. ``"12345/2024"``).
            max_results: Maximum number of atti to retrieve.

        Returns:
            List of :class:`~src.ingestion.loader.Document` objects.
        """
        if self._stub:
            return self._stub_fetch()

        params: dict[str, Any] = {"size": min(max_results, 200)}
        if notaio:
            params["notaio"] = notaio
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if repertorio:
            params["repertorio"] = repertorio

        url = f"{self._base_url}/atti"
        self._bucket.acquire()
        try:
            resp = self._client.get(url, params=params, headers=self._auth_headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ConnectorHTTPError(
                status_code=exc.response.status_code,
                url=url,
                message=str(exc),
            ) from exc

        # Notartel returns XML by default; also accept JSON if negotiated.
        content_type = resp.headers.get("content-type", "")
        if "xml" in content_type:
            return self._parse_xml_response(resp.text)
        else:
            return self._parse_json_response(resp.json())

    def fetch_atto(self, repertorio: str) -> Document | None:
        """Fetch a single atto by repertorio number.

        Args:
            repertorio: Repertorio number (e.g. ``"12345/2024"``).

        Returns:
            :class:`~src.ingestion.loader.Document` or ``None`` if not found.
        """
        results = self.fetch(repertorio=repertorio, max_results=1)
        return results[0] if results else None

    # ── Stub mode ─────────────────────────────────────────────────────────────

    def _stub_fetch(self) -> list[Document]:
        """Return synthetic fixture documents for testing without credentials."""
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        xml_text = _STUB_XML_TEMPLATE.format(ns=NOTARTEL_NS, ts=ts)
        return self._parse_xml_response(xml_text)

    # ── XML parsing ───────────────────────────────────────────────────────────

    def _parse_xml_response(self, xml_text: str) -> list[Document]:
        """Parse a Notartel XML response into Documents.

        Args:
            xml_text: Raw XML string.

        Returns:
            List of :class:`~src.ingestion.loader.Document` objects.
        """
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise ConnectorError(f"Failed to parse Notartel XML: {exc}") from exc

        ns = {"n": NOTARTEL_NS}
        atto_els = root.findall("n:AttoNotarile", ns) or root.findall("AttoNotarile")

        documents: list[Document] = []
        for el in atto_els:
            doc = self._xml_element_to_document(el, ns)
            if doc:
                documents.append(doc)

        log.info("Notartel XML parsed", extra={"atti_found": len(documents)})
        return documents

    def _xml_element_to_document(
        self, el: ET.Element, ns: dict[str, str]
    ) -> Document | None:
        """Convert a single ``<AttoNotarile>`` element to a Document.

        Args:
            el: ``<AttoNotarile>`` XML element.
            ns: Namespace map for XPath.

        Returns:
            :class:`~src.ingestion.loader.Document` or ``None`` on parse error.
        """

        def _text(tag: str) -> str:
            child = el.find(f"n:{tag}", ns)
            if child is None:
                child = el.find(tag)
            return child.text.strip() if child is not None and child.text else ""

        repertorio = _text("Repertorio")
        data_str = _text("Data")
        notaio = _text("Notaio")
        sede = _text("Sede")
        oggetto = _text("Oggetto")
        testo = _text("Testo")

        # Collect parties.
        parti_els_ns = el.findall("n:Parti/n:Parte", ns)
        parti_els = parti_els_ns if parti_els_ns is not None else el.findall("Parti/Parte")
        parti = [
            f"{p.get('ruolo', 'parte')}: {p.text.strip()}"
            for p in parti_els
            if p.text
        ]

        content_parts = [
            f"Repertorio: {repertorio}",
            f"Data: {data_str}",
            f"Notaio: {notaio}",
            f"Sede: {sede}",
            f"Oggetto: {oggetto}",
        ]
        if parti:
            content_parts.append("Parti: " + "; ".join(parti))
        content_parts.append(testo)
        content = "\n".join(content_parts)

        created = _parse_notartel_date(data_str)
        italian_meta = ItalianLegalMetadata(
            tipo_documento=TipoDocumento.ATTO_NOTARILE,
            fonte="notartel",
            url_fonte=f"notartel://atti/{repertorio.replace('/', '_')}",
        )
        extra = italian_meta.to_extra_dict()
        extra.update(
            {
                "notartel_repertorio": repertorio,
                "notartel_notaio": notaio,
                "notartel_sede": sede,
                "notartel_oggetto": oggetto,
            }
        )
        meta = DocumentMetadata(
            filename=f"notartel_{repertorio.replace('/', '_')}.xml",
            page_count=1,
            page_number=1,
            creation_date=created,
            loader_backend="notartel",
            extra={k: str(v) if not isinstance(v, (list, type(None))) else v for k, v in extra.items()},
        )
        return Document(
            content=content,
            metadata=meta,
            source_path=Path(f"notartel://atti/{repertorio.replace('/', '_')}"),
            page_number=1,
        )

    def _parse_json_response(self, data: dict[str, Any]) -> list[Document]:
        """Parse a Notartel JSON response (alternative API format).

        Args:
            data: Parsed JSON dict.

        Returns:
            List of Documents.
        """
        items: list[dict[str, Any]] = data.get("atti", data.get("results", []))
        documents: list[Document] = []
        for item in items:
            repertorio = str(item.get("repertorio", "unknown"))
            content = item.get("testo", "")
            data_str = str(item.get("data", ""))
            created = _parse_notartel_date(data_str)

            italian_meta = ItalianLegalMetadata(
                tipo_documento=TipoDocumento.ATTO_NOTARILE,
                fonte="notartel",
                url_fonte=f"notartel://atti/{repertorio.replace('/', '_')}",
            )
            extra = italian_meta.to_extra_dict()
            extra.update({"notartel_repertorio": repertorio})
            meta = DocumentMetadata(
                filename=f"notartel_{repertorio.replace('/', '_')}.json",
                page_count=1,
                page_number=1,
                creation_date=created,
                loader_backend="notartel",
                extra={k: str(v) if not isinstance(v, (list, type(None))) else v for k, v in extra.items()},
            )
            documents.append(
                Document(
                    content=content,
                    metadata=meta,
                    source_path=Path(f"notartel://atti/{repertorio.replace('/', '_')}"),
                    page_number=1,
                )
            )
        return documents

    # ── Export ────────────────────────────────────────────────────────────────

    def export_to_notartel_xml(
        self,
        documents: list[Document],
        export_date: datetime | None = None,
    ) -> str:
        """Serialise Documents to Notartel-compatible XML.

        The output conforms to the ``atti-notarili-v3`` schema and can be
        imported directly into Notartel's DMS.

        Args:
            documents:   Documents to serialise.
            export_date: Export timestamp (defaults to now UTC).

        Returns:
            UTF-8 XML string.
        """
        ts = (export_date or datetime.now(tz=timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
        root = ET.Element(
            "ListaAttiNotarili",
            attrib={
                "xmlns": NOTARTEL_NS,
                "generato": ts,
                "fonte": "RAGForge-Italia",
            },
        )
        for doc in documents:
            extra = doc.metadata.extra or {}
            atto_el = ET.SubElement(root, "AttoNotarile")
            ET.SubElement(atto_el, "Repertorio").text = str(
                extra.get("notartel_repertorio", doc.metadata.filename)
            )
            ET.SubElement(atto_el, "Data").text = (
                doc.metadata.creation_date.strftime("%Y-%m-%d")
                if doc.metadata.creation_date
                else ""
            )
            ET.SubElement(atto_el, "Notaio").text = str(extra.get("notartel_notaio", ""))
            ET.SubElement(atto_el, "Sede").text = str(extra.get("notartel_sede", ""))
            ET.SubElement(atto_el, "Oggetto").text = str(extra.get("notartel_oggetto", ""))
            ET.SubElement(atto_el, "Testo").text = doc.content

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        import io
        buffer = io.BytesIO()
        tree.write(buffer, encoding="utf-8", xml_declaration=True)
        return buffer.getvalue().decode("utf-8")


# ── Utilities ──────────────────────────────────────────────────────────────────


def _parse_notartel_date(raw: str) -> datetime:
    """Parse Notartel date strings to UTC datetime.

    Args:
        raw: Raw date string.

    Returns:
        Timezone-aware UTC :class:`datetime`.
    """
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return datetime.now(tz=timezone.utc)
