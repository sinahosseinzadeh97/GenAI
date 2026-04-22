"""IBM FileNet P8 / OpenText Documentum integration connector.

Supports two modes of operation:

**Pull mode** — periodic polling of the Content Engine REST API (CMIS 1.1):
    The connector authenticates with Basic Auth, lists new/updated objects in
    a configured repository, converts them to :class:`~src.ingestion.loader.Document`,
    and feeds them into the RAGForge indexing pipeline.

**Push mode** — webhook receiver:
    FileNet publishes ``objectCreated`` / ``objectUpdated`` / ``objectDeleted``
    events over HTTP POST.  The corresponding FastAPI endpoint lives in
    :mod:`src.italia.connectors.webhook_router` and calls this module's
    :func:`handle_filenet_event`.

Authentication
~~~~~~~~~~~~~~
- Basic Auth: ``FILENET_USERNAME`` / ``FILENET_PASSWORD``
- Session cookie: the connector reuses the LTPA/LtpaToken2 cookie returned
  after the first successful auth call.

References
~~~~~~~~~~
- IBM Content Navigator REST API: https://www.ibm.com/docs/en/content-navigator
- OASIS CMIS 1.1 spec: https://docs.oasis-open.org/cmis/CMIS/v1.1/
- OpenText Documentum REST SDK: https://developer.opentext.com/
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from src.ingestion.loader import Document, DocumentMetadata
from src.italia.connectors.base import BaseConnector, ConnectorError, ConnectorHTTPError
from src.italia.metadata import ItalianLegalMetadata, TipoDocumento
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Data models ────────────────────────────────────────────────────────────────


class FilenetDocument:
    """Lightweight wrapper for a FileNet/Documentum CMIS object.

    Attributes:
        object_id:    CMIS object identifier.
        name:         Document name / filename.
        content_type: MIME type of the primary content stream.
        content:      Extracted text content (after CMIS stream fetch).
        created:      Creation timestamp (UTC).
        modified:     Last-modified timestamp (UTC).
        repository:   Source repository name.
        properties:   Raw CMIS property bag.
    """

    def __init__(
        self,
        object_id: str,
        name: str,
        content_type: str,
        content: str,
        created: datetime,
        modified: datetime,
        repository: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        self.object_id = object_id
        self.name = name
        self.content_type = content_type
        self.content = content
        self.created = created
        self.modified = modified
        self.repository = repository
        self.properties = properties or {}


class FilenetWebhookEvent:
    """Parsed FileNet push event.

    Attributes:
        event_type:  ``"objectCreated"`` | ``"objectUpdated"`` | ``"objectDeleted"``.
        object_id:   CMIS object identifier of the affected document.
        repository:  Repository name.
        timestamp:   Event timestamp (UTC).
        raw:         Original raw event payload dict.
    """

    def __init__(
        self,
        event_type: str,
        object_id: str,
        repository: str,
        timestamp: datetime,
        raw: dict[str, Any],
    ) -> None:
        self.event_type = event_type
        self.object_id = object_id
        self.repository = repository
        self.timestamp = timestamp
        self.raw = raw


# ── Connector ──────────────────────────────────────────────────────────────────


class FilenetDocumentumConnector(BaseConnector):
    """Pull connector for IBM FileNet P8 / OpenText Documentum.

    Connects to the Content Engine (CE) REST gateway via the CMIS 1.1
    browser binding, authenticates with Basic Auth, and fetches documents
    from a configured repository folder.

    Args:
        base_url:    CE REST gateway base URL.
                     Example: ``"https://filenet.example.it/fncmis/resources"``
        repository:  Repository name / ID (typically ``"FPOS"`` or ``"DCTM"``).
        username:    Service-account username.
        password:    Service-account password.
        folder_path: CMIS folder path to poll (default: ``"/"``).
        rate_limit_rps: Requests per second (default: 0.5 — conservative).
        timeout:     HTTP timeout in seconds.
    """

    source_name = "filenet_documentum"

    def __init__(
        self,
        base_url: str,
        repository: str,
        username: str,
        password: str,
        folder_path: str = "/",
        rate_limit_rps: float = 0.5,
        timeout: int = 60,
    ) -> None:
        super().__init__(rate_limit_rps=rate_limit_rps, timeout=timeout)
        self._base_url = base_url.rstrip("/")
        self._repository = repository
        self._username = username
        self._password = password
        self._folder_path = folder_path
        self._session_cookie: str | None = None

    # ── Authentication ────────────────────────────────────────────────────────

    def _authenticate(self) -> None:
        """Obtain an LTPA session cookie from the FileNet CE gateway.

        Raises:
            ConnectorError: When authentication fails.
        """
        url = f"{self._base_url}/{self._repository}"
        try:
            self._bucket.acquire()
            resp = self._client.get(
                url,
                auth=(self._username, self._password),
            )
            resp.raise_for_status()
            # The server sets LtpaToken2 (WebSphere) or a JSESSIONID cookie.
            cookie_header = resp.headers.get("set-cookie", "")
            self._session_cookie = cookie_header.split(";")[0] if cookie_header else None
            log.info(
                "FileNet authentication successful",
                extra={"repository": self._repository, "url": url},
            )
        except httpx.HTTPStatusError as exc:
            raise ConnectorError(
                f"FileNet authentication failed ({exc.response.status_code}): {exc}"
            ) from exc

    # ── Pull mode ─────────────────────────────────────────────────────────────

    def fetch(  # type: ignore[override]
        self,
        max_documents: int = 50,
        since: datetime | None = None,
    ) -> list[Document]:
        """Fetch documents from the configured FileNet repository folder.

        Args:
            max_documents: Maximum number of documents to retrieve.
            since:         Only fetch documents modified after this datetime.

        Returns:
            List of :class:`~src.ingestion.loader.Document` objects.
        """
        if not self._session_cookie:
            self._authenticate()

        documents: list[Document] = []
        query = self._build_cmis_query(max_documents=max_documents, since=since)
        results = self._run_cmis_query(query)

        for item in results:
            try:
                doc = self._cmis_item_to_document(item)
                documents.append(doc)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Failed to convert FileNet item to Document",
                    extra={"object_id": item.get("objectId", "?"), "error": str(exc)},
                )

        log.info(
            "FileNet fetch complete",
            extra={
                "repository": self._repository,
                "fetched": len(documents),
                "max_documents": max_documents,
            },
        )
        return documents

    def _build_cmis_query(
        self, max_documents: int, since: datetime | None
    ) -> str:
        """Build a CMIS 1.1 SQL-domain query string.

        Args:
            max_documents: Maximum results.
            since:         Modification date filter.

        Returns:
            CMIS SQL query string.
        """
        base_q = (
            "SELECT cmis:objectId, cmis:name, cmis:contentStreamMimeType, "
            "cmis:creationDate, cmis:lastModificationDate "
            f"FROM cmis:document WHERE IN_FOLDER('{self._folder_path}')"
        )
        if since:
            ts = since.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            base_q += f" AND cmis:lastModificationDate > TIMESTAMP '{ts}'"
        base_q += f" ORDER BY cmis:lastModificationDate DESC MAXITEMS {max_documents}"
        return base_q

    def _run_cmis_query(self, query: str) -> list[dict[str, Any]]:
        """Execute a CMIS SQL query and return raw result objects.

        Args:
            query: CMIS SQL query string.

        Returns:
            List of raw CMIS object dicts from the ``results`` array.
        """
        url = f"{self._base_url}/{self._repository}/query"
        headers: dict[str, str] = {}
        if self._session_cookie:
            headers["Cookie"] = self._session_cookie

        self._bucket.acquire()
        try:
            resp = self._client.post(
                url,
                json={"statement": query, "maxItems": 200},
                headers=headers,
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            return data.get("results", [])
        except httpx.HTTPStatusError as exc:
            raise ConnectorHTTPError(
                status_code=exc.response.status_code,
                url=url,
                message=str(exc),
            ) from exc

    def _cmis_item_to_document(self, item: dict[str, Any]) -> Document:
        """Convert a CMIS result item to a RAGForge Document.

        Args:
            item: Raw CMIS JSON object from the query results.

        Returns:
            :class:`~src.ingestion.loader.Document` with Italian metadata.
        """
        props = item.get("properties", item)
        object_id: str = str(props.get("cmis:objectId", {}).get("value", "unknown"))
        name: str = str(props.get("cmis:name", {}).get("value", object_id))
        mime: str = str(
            props.get("cmis:contentStreamMimeType", {}).get("value", "application/octet-stream")
        )

        # Attempt to fetch the content stream.
        content = self._fetch_content_stream(object_id)

        created_raw = props.get("cmis:creationDate", {}).get("value", "")
        modified_raw = props.get("cmis:lastModificationDate", {}).get("value", "")
        created = _parse_cmis_date(created_raw)
        modified = _parse_cmis_date(modified_raw)

        italian_meta = ItalianLegalMetadata(
            tipo_documento=TipoDocumento.ALTRO,
            fonte=f"filenet/{self._repository}",
            url_fonte=f"{self._base_url}/{self._repository}/object/{object_id}",
        )
        extra = italian_meta.to_extra_dict()
        extra.update(
            {
                "filenet_object_id": object_id,
                "filenet_repository": self._repository,
                "filenet_mime_type": mime,
                "filenet_modified": modified.isoformat(),
            }
        )

        meta = DocumentMetadata(
            filename=name,
            page_count=1,
            page_number=1,
            creation_date=created,
            loader_backend="filenet_documentum",
            extra={k: str(v) if not isinstance(v, (list, type(None))) else v for k, v in extra.items()},
        )
        return Document(
            content=content,
            metadata=meta,
            source_path=Path(f"filenet://{self._repository}/{object_id}"),
            page_number=1,
        )

    def _fetch_content_stream(self, object_id: str) -> str:
        """Fetch and return the text content stream for a CMIS object.

        Args:
            object_id: CMIS object identifier.

        Returns:
            Decoded text content, or an empty string on failure.
        """
        url = f"{self._base_url}/{self._repository}/object/{object_id}/content"
        headers: dict[str, str] = {}
        if self._session_cookie:
            headers["Cookie"] = self._session_cookie

        self._bucket.acquire()
        try:
            resp = self._client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "Could not fetch FileNet content stream",
                extra={"object_id": object_id, "error": str(exc)},
            )
            return ""

    # ── Export mode ───────────────────────────────────────────────────────────

    def export_to_filenet(
        self,
        content: str,
        document_name: str,
        folder_path: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Upload a document to FileNet via CMIS.

        Args:
            content:       Text content to upload.
            document_name: Target filename in FileNet.
            folder_path:   Target folder (defaults to the configured ``folder_path``).
            properties:    Additional CMIS properties.

        Returns:
            The new CMIS ``objectId`` of the created document.

        Raises:
            ConnectorError: On upload failure.
        """
        if not self._session_cookie:
            self._authenticate()

        target_folder = folder_path or self._folder_path
        url = f"{self._base_url}/{self._repository}/object"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._session_cookie:
            headers["Cookie"] = self._session_cookie

        payload = {
            "properties": {
                "cmis:objectTypeId": {"value": "cmis:document"},
                "cmis:name": {"value": document_name},
                **(properties or {}),
            },
            "folderId": target_folder,
            "content": {"data": content, "mimeType": "text/plain"},
        }

        self._bucket.acquire()
        try:
            resp = self._client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()
            new_id: str = result.get("objectId", "unknown")
            log.info(
                "Document exported to FileNet",
                extra={
                    "object_id": new_id,
                    "name": document_name,
                    "folder": target_folder,
                },
            )
            return new_id
        except httpx.HTTPStatusError as exc:
            raise ConnectorError(
                f"FileNet export failed ({exc.response.status_code}): {exc}"
            ) from exc

    # ── Webhook / push mode ───────────────────────────────────────────────────

    @staticmethod
    def parse_webhook_event(
        payload: bytes,
        signature: str | None,
        secret: str,
    ) -> FilenetWebhookEvent:
        """Parse and verify a FileNet push event.

        Verifies the HMAC-SHA256 signature before parsing, preventing
        unauthenticated event injection.

        Args:
            payload:   Raw request body bytes.
            signature: Value of the ``X-FileNet-Signature`` header
                       (``sha256=<hex>``).
            secret:    Shared HMAC secret configured on the FileNet server.

        Returns:
            Parsed :class:`FilenetWebhookEvent`.

        Raises:
            ConnectorError: When the signature is absent or invalid.
        """
        _verify_hmac(payload=payload, signature=signature, secret=secret)
        data: dict[str, Any] = json.loads(payload)
        return FilenetWebhookEvent(
            event_type=data.get("eventType", "unknown"),
            object_id=data.get("objectId", ""),
            repository=data.get("repositoryId", ""),
            timestamp=_parse_cmis_date(data.get("timestamp", "")),
            raw=data,
        )

    def handle_filenet_event(
        self, event: FilenetWebhookEvent
    ) -> Document | None:
        """Process a parsed FileNet event and return a Document if applicable.

        Args:
            event: Parsed FileNet push event.

        Returns:
            A :class:`~src.ingestion.loader.Document` for ``objectCreated`` and
            ``objectUpdated`` events; ``None`` for ``objectDeleted``.
        """
        if event.event_type == "objectDeleted":
            log.info(
                "FileNet document deleted",
                extra={"object_id": event.object_id},
            )
            return None

        content = self._fetch_content_stream(event.object_id)
        italian_meta = ItalianLegalMetadata(
            tipo_documento=TipoDocumento.ALTRO,
            fonte=f"filenet/{event.repository}",
            url_fonte=f"{self._base_url}/{event.repository}/object/{event.object_id}",
        )
        extra = italian_meta.to_extra_dict()
        extra.update(
            {
                "filenet_object_id": event.object_id,
                "filenet_repository": event.repository,
                "filenet_event_type": event.event_type,
            }
        )
        meta = DocumentMetadata(
            filename=event.object_id,
            page_count=1,
            page_number=1,
            creation_date=event.timestamp,
            loader_backend="filenet_documentum_webhook",
            extra={k: str(v) if not isinstance(v, (list, type(None))) else v for k, v in extra.items()},
        )
        return Document(
            content=content,
            metadata=meta,
            source_path=Path(f"filenet://{event.repository}/{event.object_id}"),
            page_number=1,
        )


# ── Utilities ──────────────────────────────────────────────────────────────────


def _parse_cmis_date(raw: str) -> datetime:
    """Parse a CMIS date string to a UTC datetime.

    Tries ISO 8601 formats commonly returned by FileNet / Documentum.

    Args:
        raw: Raw date string.

    Returns:
        Timezone-aware UTC :class:`datetime`, or ``datetime.now(UTC)`` on failure.
    """
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return datetime.now(tz=timezone.utc)


def _verify_hmac(payload: bytes, signature: str | None, secret: str) -> None:
    """Verify an HMAC-SHA256 webhook signature.

    Args:
        payload:   Raw request body bytes.
        signature: Signature header value, expected format ``sha256=<hex>``.
        secret:    Shared secret string.

    Raises:
        ConnectorError: When the signature is missing or does not match.
    """
    if not signature:
        raise ConnectorError("Missing webhook signature header (X-FileNet-Signature).")
    prefix = "sha256="
    if not signature.startswith(prefix):
        raise ConnectorError(f"Unexpected signature format: {signature!r}")
    received_hex = signature[len(prefix):]
    expected = hmac.new(
        key=secret.encode("utf-8"),
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, received_hex):
        raise ConnectorError("Webhook HMAC-SHA256 signature verification failed.")
