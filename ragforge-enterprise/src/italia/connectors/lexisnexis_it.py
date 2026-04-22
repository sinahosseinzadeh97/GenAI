"""LexisNexis Italia export connector.

Fetches massime (headnotes) and normative articles from the LexisNexis Italia
REST API and can export RAGForge search results back in the LexisNexis XML
export format, enabling round-trip interoperability for Italian law firms.

Authentication
~~~~~~~~~~~~~~
OAuth 2.0 **client-credentials** flow::

    POST /oauth/token
    grant_type=client_credentials
    &client_id=<LEXISNEXIS_CLIENT_ID>
    &client_secret=<LEXISNEXIS_CLIENT_SECRET>
    &scope=lexisnexis.italia.full

The access token is cached in memory and refreshed automatically when it
expires (typically every 3600 s).

LexisNexis XML export format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The connector can serialise a list of retrieved chunks to the standard
LexisNexis Italia XML schema (``massime-export-v2``).  This schema is
used by Italian law firms to import results into their DMS (e.g. Lex24,
Juris Data).

Rate limits
~~~~~~~~~~~
Default: 2 rps (standard LexisNexis Italia subscription).

References
~~~~~~~~~~
- LexisNexis Italia API (internal): https://api.lexisnexis.it/v2/
- LexisNexis XML massime schema: https://api.lexisnexis.it/schema/massime-v2.xsd
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
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


class LexisNexisItaliaConnector(BaseConnector):
    """REST connector for LexisNexis Italia (massime + norme).

    Args:
        base_url:        Base URL for the LexisNexis Italia REST API.
                         Default: ``"https://api.lexisnexis.it/v2"``.
        client_id:       OAuth 2.0 client ID.
        client_secret:   OAuth 2.0 client secret.
        rate_limit_rps:  Max requests per second (default: 2.0 per contract).
        timeout:         HTTP timeout in seconds (default: 45).
    """

    source_name = "lexisnexis_italia"

    def __init__(
        self,
        base_url: str = "https://api.lexisnexis.it/v2",
        client_id: str = "",
        client_secret: str = "",
        rate_limit_rps: float = 2.0,
        timeout: int = 45,
    ) -> None:
        super().__init__(rate_limit_rps=rate_limit_rps, timeout=timeout)
        self._base_url = base_url.rstrip("/")
        self._client_id = client_id
        self._client_secret = client_secret
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0

    # ── OAuth 2.0 ─────────────────────────────────────────────────────────────

    def _ensure_token(self) -> None:
        """Obtain or refresh the OAuth 2.0 access token if expired.

        Raises:
            ConnectorError: When the token endpoint returns an error.
        """
        if self._access_token and time.monotonic() < self._token_expires_at - 60:
            return  # Token still valid (60 s safety margin).

        token_url = f"{self._base_url}/oauth/token"
        self._bucket.acquire()
        try:
            resp = self._client.post(
                token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "scope": "lexisnexis.italia.full",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ConnectorError(
                f"LexisNexis OAuth token request failed ({exc.response.status_code}): {exc}"
            ) from exc

        payload: dict[str, Any] = resp.json()
        self._access_token = payload.get("access_token", "")
        expires_in: int = int(payload.get("expires_in", 3600))
        self._token_expires_at = time.monotonic() + expires_in
        log.info("LexisNexis OAuth token obtained", extra={"expires_in": expires_in})

    @property
    def _auth_headers(self) -> dict[str, str]:
        """Return Bearer auth headers for the current token."""
        self._ensure_token()
        return {"Authorization": f"Bearer {self._access_token}"}

    # ── Fetch API ─────────────────────────────────────────────────────────────

    def fetch(  # type: ignore[override]
        self,
        query: str = "",
        tipo: str = "massima",
        max_results: int = 50,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[Document]:
        """Fetch massime or norme from LexisNexis Italia.

        Args:
            query:       Full-text search query in Italian.
            tipo:        Document type: ``"massima"`` (default) or ``"norma"``.
            max_results: Maximum number of results.
            from_date:   Start date filter (``YYYY-MM-DD``).
            to_date:     End date filter (``YYYY-MM-DD``).

        Returns:
            List of :class:`~src.ingestion.loader.Document` objects.
        """
        endpoint = f"{self._base_url}/search"
        params: dict[str, Any] = {
            "q": query,
            "tipo": tipo,
            "size": min(max_results, 200),
        }
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        self._bucket.acquire()
        try:
            resp = self._client.get(endpoint, params=params, headers=self._auth_headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ConnectorHTTPError(
                status_code=exc.response.status_code,
                url=endpoint,
                message=str(exc),
            ) from exc

        data: dict[str, Any] = resp.json()
        items: list[dict[str, Any]] = data.get("results", [])

        documents = [self._item_to_document(item, tipo) for item in items]
        log.info(
            "LexisNexis Italia fetch complete",
            extra={"tipo": tipo, "fetched": len(documents), "query": query},
        )
        return documents

    def _item_to_document(self, item: dict[str, Any], tipo: str) -> Document:
        """Convert a LexisNexis result item to a RAGForge Document.

        Args:
            item: Raw JSON result object.
            tipo: Document type (``"massima"`` or ``"norma"``).

        Returns:
            :class:`~src.ingestion.loader.Document`.
        """
        tipo_doc = TipoDocumento.MASSIMA if tipo == "massima" else TipoDocumento.LEGGE
        source_id: str = item.get("id", "unknown")
        title: str = item.get("titolo", item.get("title", source_id))
        content: str = item.get("testo", item.get("text", ""))
        fonte: str = item.get("fonte", "lexisnexis_italia")
        url: str = item.get("url", f"{self._base_url}/document/{source_id}")

        date_str: str = item.get("data", item.get("date", ""))
        created = _parse_date(date_str)

        italian_meta = ItalianLegalMetadata(
            tipo_documento=tipo_doc,
            fonte=fonte,
            url_fonte=url,
        )
        extra = italian_meta.to_extra_dict()
        extra.update(
            {
                "lexisnexis_id": source_id,
                "lexisnexis_tipo": tipo,
                "lexisnexis_title": title,
            }
        )
        meta = DocumentMetadata(
            filename=f"lexisnexis_{source_id}.txt",
            page_count=1,
            page_number=1,
            creation_date=created,
            loader_backend="lexisnexis_italia",
            extra={k: str(v) if not isinstance(v, (list, type(None))) else v for k, v in extra.items()},
        )
        return Document(
            content=content,
            metadata=meta,
            source_path=Path(f"lexisnexis://italia/{source_id}"),
            page_number=1,
        )

    # ── XML Export ────────────────────────────────────────────────────────────

    def export_results_xml(
        self,
        documents: list[Document],
        query: str = "",
        export_date: datetime | None = None,
    ) -> str:
        """Serialise a list of Documents to LexisNexis Italia XML export format.

        The output conforms to the ``massime-export-v2`` schema used by Italian
        law firms for import into their DMS. It is valid UTF-8 XML.

        Args:
            documents:   RAGForge Documents to serialise.
            query:       Original search query (embedded in the XML header).
            export_date: Export timestamp (defaults to now UTC).

        Returns:
            UTF-8 XML string.
        """
        ts = (export_date or datetime.now(tz=timezone.utc)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        root = ET.Element(
            "LexisNexisExport",
            attrib={
                "xmlns": "https://api.lexisnexis.it/schema/massime-v2",
                "version": "2.0",
                "generated": ts,
                "source": "RAGForge-Italia",
            },
        )

        header = ET.SubElement(root, "Header")
        ET.SubElement(header, "Query").text = query
        ET.SubElement(header, "ResultCount").text = str(len(documents))
        ET.SubElement(header, "ExportDate").text = ts

        results_el = ET.SubElement(root, "Results")
        for doc in documents:
            extra = doc.metadata.extra or {}
            item_el = ET.SubElement(results_el, "Documento")
            ET.SubElement(item_el, "ID").text = str(
                extra.get("lexisnexis_id", doc.metadata.filename)
            )
            ET.SubElement(item_el, "Tipo").text = str(
                extra.get("lexisnexis_tipo", "massima")
            )
            ET.SubElement(item_el, "Titolo").text = str(
                extra.get("lexisnexis_title", doc.metadata.filename)
            )
            ET.SubElement(item_el, "Fonte").text = str(
                extra.get("it_fonte", "lexisnexis_italia")
            )
            ET.SubElement(item_el, "URL").text = str(
                extra.get("it_url_fonte", extra.get("source_uri", ""))
            )
            ET.SubElement(item_el, "Data").text = (
                doc.metadata.creation_date.strftime("%Y-%m-%d")
                if doc.metadata.creation_date
                else ""
            )
            ET.SubElement(item_el, "Testo").text = doc.content

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        import io
        buffer = io.BytesIO()
        tree.write(buffer, encoding="utf-8", xml_declaration=True)
        return buffer.getvalue().decode("utf-8")

    # ── Webhook verification ──────────────────────────────────────────────────

    @staticmethod
    def verify_webhook_signature(
        payload: bytes,
        signature: str | None,
        secret: str,
    ) -> bool:
        """Verify the HMAC-SHA256 signature from a LexisNexis push notification.

        Args:
            payload:   Raw request body bytes.
            signature: ``X-LexisNexis-Signature`` header value.
            secret:    Shared HMAC secret.

        Returns:
            ``True`` if the signature matches; ``False`` otherwise.
        """
        if not signature:
            return False
        expected = hmac.new(
            key=secret.encode("utf-8"),
            msg=payload,
            digestmod=hashlib.sha256,
        ).hexdigest()
        received = signature.removeprefix("sha256=")
        return hmac.compare_digest(expected, received)


# ── Utilities ──────────────────────────────────────────────────────────────────


def _parse_date(raw: str) -> datetime:
    """Parse a date string (``YYYY-MM-DD`` or ISO 8601) to UTC datetime.

    Args:
        raw: Raw date string.

    Returns:
        Timezone-aware UTC :class:`datetime`, or ``datetime.now(UTC)`` on failure.
    """
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return datetime.now(tz=timezone.utc)
