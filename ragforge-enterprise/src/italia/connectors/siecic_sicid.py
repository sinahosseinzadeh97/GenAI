"""SIECIC / SICID Italian court management system connector (READ-ONLY).

SIECIC (Sistema Informatico per il Contenzioso Civile) and SICID (Sistema
Informatico per il Contenzioso Penale) are the court management systems of the
Italian Ministry of Justice.  They hold the Ruolo Generale (civil docket) and
the criminal register respectively.

Access policy
~~~~~~~~~~~~~
These systems are **strictly read-only** for external consumers.  Only
authorised access is permitted under:
  - D.Lgs. 82/2005 (Codice dell'Amministrazione Digitale — CAD), art. 2
  - D.M. 21 febbraio 2011 — Specifiche tecniche Processo Civile Telematico
  - D.M. 27 aprile 2009 — Regole tecniche SIECIC/SICID

This connector therefore raises :class:`NotImplementedError` on any attempt
to call write or export methods.  The read-side is API-compliant with the
Ministry of Justice REST schema (Portale dei Servizi Telematici).

Authentication
~~~~~~~~~~~~~~
API key issued by the Ministry of Justice (Direzione Generale per i Sistemi
Informatici Automatizzati — DGSIA)::

    X-SIECIC-API-Key: <SIECIC_API_KEY>

Access requires:
1. Formal request to DGSIA (Ministero della Giustizia).
2. Technical agreement and PKI certificate provisioning.

Stub mode
~~~~~~~~~
When ``SIECIC_STUB=true`` (or ``stub=True`` in the constructor), fixture
fascicoli are returned so tests pass without credentials.

References
~~~~~~~~~~
- Portale Servizi Telematici: https://pst.giustizia.it
- D.Lgs. 82/2005 (CAD)
- Processo Civile Telematico (PCT) specs: https://pst.giustizia.it/PST/it/pst_2.wp
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from src.ingestion.loader import Document, DocumentMetadata
from src.italia.connectors.base import BaseConnector, ConnectorHTTPError
from src.italia.metadata import ItalianLegalMetadata, TipoDocumento
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Stub fixtures ──────────────────────────────────────────────────────────────

_STUB_FASCICOLI: list[dict[str, Any]] = [
    {
        "numero": "1234",
        "anno": "2024",
        "tribunale": "Tribunale di Milano",
        "sezione": "Prima Sezione Civile",
        "stato": "pendente",
        "oggetto": "Risarcimento danni da responsabilità extracontrattuale (art. 2043 c.c.)",
        "data_iscrizione": "2024-01-15",
        "parti": [
            {"ruolo": "attore", "nome": "Mario Bianchi"},
            {"ruolo": "convenuto", "nome": "Alfa S.r.l."},
        ],
        "udienza_prossima": "2024-06-20",
        "sistema": "SIECIC",
    },
    {
        "numero": "5678",
        "anno": "2023",
        "tribunale": "Tribunale di Roma",
        "sezione": "Seconda Sezione Civile",
        "stato": "definito",
        "oggetto": "Inadempimento contrattuale ex art. 1453 c.c.",
        "data_iscrizione": "2023-03-10",
        "parti": [
            {"ruolo": "attore", "nome": "Beta S.p.A."},
            {"ruolo": "convenuto", "nome": "Gamma S.r.l."},
        ],
        "udienza_prossima": None,
        "sistema": "SIECIC",
    },
]

_STUB_UDIENZE: list[dict[str, Any]] = [
    {
        "data": "2024-06-20",
        "ora": "09:30",
        "tribunale": "Tribunale di Milano",
        "aula": "Aula 3",
        "numero_fascicolo": "1234/2024",
        "oggetto": "Udienza di trattazione",
        "stato": "programmata",
        "sistema": "SIECIC",
    }
]


# ── Connector ──────────────────────────────────────────────────────────────────


class SiecicSicidConnector(BaseConnector):
    """Read-only connector for SIECIC (civil) and SICID (criminal) court systems.

    Args:
        base_url:        Portale dei Servizi Telematici REST API base URL.
                         Example: ``"https://pst.giustizia.it/api/v1"``
        api_key:         DGSIA-issued API key.
        sistema:         ``"SIECIC"`` (civil, default) or ``"SICID"`` (criminal).
        stub:            If ``True`` (or ``SIECIC_STUB=true``), return fixture data.
        rate_limit_rps:  Max requests per second (default: 0.5 — conservative).
        timeout:         HTTP timeout in seconds (default: 60).

    Raises:
        :class:`NotImplementedError`: On any write/export method call.
    """

    source_name = "siecic_sicid"

    #: Methods that must never be called on this connector (enforced read-only).
    _WRITE_METHODS = frozenset({"export", "write", "create", "update", "delete"})

    def __init__(
        self,
        base_url: str = "https://pst.giustizia.it/api/v1",
        api_key: str = "",
        sistema: str = "SIECIC",
        stub: bool | None = None,
        rate_limit_rps: float = 0.5,
        timeout: int = 60,
    ) -> None:
        super().__init__(rate_limit_rps=rate_limit_rps, timeout=timeout)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._sistema = sistema.upper()
        if self._sistema not in ("SIECIC", "SICID"):
            raise ValueError(f"sistema must be 'SIECIC' or 'SICID', got: {sistema!r}")
        if stub is None:
            stub = os.getenv("SIECIC_STUB", "false").lower() in ("true", "1", "yes")
        self._stub = stub

        if self._stub:
            log.info(
                "SiecicSicidConnector running in stub mode — no HTTP calls will be made.",
                extra={"sistema": self._sistema},
            )

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {
            "X-SIECIC-API-Key": self._api_key,
            "X-Sistema": self._sistema,
        }

    # ── Fetch fascicolo ───────────────────────────────────────────────────────

    def fetch_fascicolo(
        self,
        numero: str,
        anno: str,
        tribunale: str,
    ) -> Document | None:
        """Fetch a single fascicolo (court file) by docket number.

        Args:
            numero:     Fascicolo number (e.g. ``"1234"``).
            anno:       Year (e.g. ``"2024"``).
            tribunale:  Court name (e.g. ``"Tribunale di Milano"``).

        Returns:
            :class:`~src.ingestion.loader.Document` or ``None`` if not found.
        """
        if self._stub:
            matching = [
                f
                for f in _STUB_FASCICOLI
                if f["numero"] == numero and f["anno"] == anno
            ]
            return self._fascicolo_to_document(matching[0]) if matching else None

        url = f"{self._base_url}/{self._sistema.lower()}/fascicoli/{anno}/{numero}"
        params = {"tribunale": tribunale}
        self._bucket.acquire()
        try:
            resp = self._client.get(url, params=params, headers=self._auth_headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise ConnectorHTTPError(
                status_code=exc.response.status_code,
                url=url,
                message=str(exc),
            ) from exc

        return self._fascicolo_to_document(resp.json())

    def fetch(  # type: ignore[override]
        self,
        tribunale: str | None = None,
        stato: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        max_results: int = 50,
    ) -> list[Document]:
        """Fetch multiple fascicoli matching the given filters.

        Args:
            tribunale:   Filter by court name.
            stato:       Fascicolo status (``"pendente"`` | ``"definito"``).
            from_date:   Iscrizione start date (``YYYY-MM-DD``).
            to_date:     Iscrizione end date (``YYYY-MM-DD``).
            max_results: Maximum number of fascicoli to return.

        Returns:
            List of :class:`~src.ingestion.loader.Document` objects.
        """
        if self._stub:
            items = _STUB_FASCICOLI[:max_results]
            return [self._fascicolo_to_document(f) for f in items]

        params: dict[str, Any] = {"size": min(max_results, 200)}
        if tribunale:
            params["tribunale"] = tribunale
        if stato:
            params["stato"] = stato
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        url = f"{self._base_url}/{self._sistema.lower()}/fascicoli"
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

        data: dict[str, Any] = resp.json()
        fascicoli: list[dict[str, Any]] = data.get("fascicoli", data.get("results", []))
        return [self._fascicolo_to_document(f) for f in fascicoli]

    def fetch_udienze(
        self,
        data: str,
        tribunale: str | None = None,
    ) -> list[Document]:
        """Fetch hearings (udienze) scheduled on a given date.

        Args:
            data:       Date in ``YYYY-MM-DD`` format.
            tribunale:  Optional court filter.

        Returns:
            List of :class:`~src.ingestion.loader.Document` objects.
        """
        if self._stub:
            return [self._udienza_to_document(u) for u in _STUB_UDIENZE]

        params: dict[str, Any] = {"data": data}
        if tribunale:
            params["tribunale"] = tribunale

        url = f"{self._base_url}/{self._sistema.lower()}/udienze"
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

        data_response: dict[str, Any] = resp.json()
        udienze: list[dict[str, Any]] = data_response.get("udienze", [])
        return [self._udienza_to_document(u) for u in udienze]

    # ── Document factories ────────────────────────────────────────────────────

    def _fascicolo_to_document(self, fascicolo: dict[str, Any]) -> Document:
        """Convert a fascicolo JSON dict to a RAGForge Document.

        Args:
            fascicolo: Raw fascicolo dict.

        Returns:
            :class:`~src.ingestion.loader.Document`.
        """
        numero = str(fascicolo.get("numero", "unknown"))
        anno = str(fascicolo.get("anno", "unknown"))
        tribunale = str(fascicolo.get("tribunale", ""))
        sezione = str(fascicolo.get("sezione", ""))
        stato = str(fascicolo.get("stato", ""))
        oggetto = str(fascicolo.get("oggetto", ""))
        date_str = str(fascicolo.get("data_iscrizione", ""))
        sistema = str(fascicolo.get("sistema", self._sistema))

        # Format parties.
        parti: list[dict[str, str]] = fascicolo.get("parti", [])
        parti_text = "; ".join(
            f"{p.get('ruolo', 'parte')}: {p.get('nome', '')}" for p in parti
        )

        content_parts = [
            f"Sistema: {sistema}",
            f"Fascicolo: {numero}/{anno}",
            f"Tribunale: {tribunale}",
            f"Sezione: {sezione}",
            f"Stato: {stato}",
            f"Oggetto: {oggetto}",
            f"Data iscrizione: {date_str}",
        ]
        if parti_text:
            content_parts.append(f"Parti: {parti_text}")
        next_hearing = fascicolo.get("udienza_prossima")
        if next_hearing:
            content_parts.append(f"Prossima udienza: {next_hearing}")

        content = "\n".join(content_parts)
        created = _parse_court_date(date_str)
        source_id = f"{sistema.lower()}_{anno}_{numero}"

        italian_meta = ItalianLegalMetadata(
            tipo_documento=TipoDocumento.SENTENZA,
            fonte=sistema.lower(),
            url_fonte=f"siecic_sicid://{sistema.lower()}/{anno}/{numero}",
        )
        extra = italian_meta.to_extra_dict()
        extra.update(
            {
                "siecic_numero": numero,
                "siecic_anno": anno,
                "siecic_tribunale": tribunale,
                "siecic_stato": stato,
                "siecic_sistema": sistema,
            }
        )
        meta = DocumentMetadata(
            filename=f"{source_id}.txt",
            page_count=1,
            page_number=1,
            creation_date=created,
            loader_backend=f"{sistema.lower()}_readonly",
            extra={k: str(v) if not isinstance(v, (list, type(None))) else v for k, v in extra.items()},
        )
        return Document(
            content=content,
            metadata=meta,
            source_path=Path(f"siecic_sicid://{sistema.lower()}/{anno}/{numero}"),
            page_number=1,
        )

    def _udienza_to_document(self, udienza: dict[str, Any]) -> Document:
        """Convert an udienza dict to a RAGForge Document.

        Args:
            udienza: Raw udienza dict.

        Returns:
            :class:`~src.ingestion.loader.Document`.
        """
        data_str = str(udienza.get("data", ""))
        ora = str(udienza.get("ora", ""))
        tribunale = str(udienza.get("tribunale", ""))
        aula = str(udienza.get("aula", ""))
        fascicolo = str(udienza.get("numero_fascicolo", ""))
        oggetto = str(udienza.get("oggetto", ""))
        stato = str(udienza.get("stato", ""))
        sistema = str(udienza.get("sistema", self._sistema))

        content = "\n".join(
            [
                f"Sistema: {sistema}",
                f"Udienza: {data_str} ore {ora}",
                f"Tribunale: {tribunale} — {aula}",
                f"Fascicolo: {fascicolo}",
                f"Oggetto: {oggetto}",
                f"Stato: {stato}",
            ]
        )
        created = _parse_court_date(data_str)
        source_id = f"udienza_{sistema.lower()}_{data_str}_{fascicolo.replace('/', '_')}"

        italian_meta = ItalianLegalMetadata(
            tipo_documento=TipoDocumento.ALTRO,
            fonte=sistema.lower(),
            url_fonte=f"siecic_sicid://{sistema.lower()}/udienze/{data_str}",
        )
        extra = italian_meta.to_extra_dict()
        extra.update(
            {
                "siecic_data_udienza": data_str,
                "siecic_tribunale": tribunale,
                "siecic_fascicolo": fascicolo,
                "siecic_sistema": sistema,
            }
        )
        meta = DocumentMetadata(
            filename=f"{source_id}.txt",
            page_count=1,
            page_number=1,
            creation_date=created,
            loader_backend=f"{sistema.lower()}_readonly",
            extra={k: str(v) if not isinstance(v, (list, type(None))) else v for k, v in extra.items()},
        )
        return Document(
            content=content,
            metadata=meta,
            source_path=Path(f"siecic_sicid://{sistema.lower()}/udienze/{data_str}"),
            page_number=1,
        )

    # ── Write methods — strictly prohibited ───────────────────────────────────

    def export(self, *_: Any, **__: Any) -> Any:
        """Write operations are not permitted on SIECIC/SICID (arts. 2 CAD).

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "SIECIC/SICID connector is read-only. "
            "Write operations are not permitted under D.Lgs. 82/2005 (CAD), art. 2."
        )

    def write(self, *_: Any, **__: Any) -> Any:
        """Raises :class:`NotImplementedError`. SIECIC/SICID is read-only."""
        raise NotImplementedError(
            "SIECIC/SICID connector is read-only. "
            "Write operations are not permitted under D.Lgs. 82/2005 (CAD), art. 2."
        )

    def create(self, *_: Any, **__: Any) -> Any:
        """Raises :class:`NotImplementedError`. SIECIC/SICID is read-only."""
        raise NotImplementedError(
            "SIECIC/SICID connector is read-only. "
            "Write operations are not permitted under D.Lgs. 82/2005 (CAD), art. 2."
        )

    def delete(self, *_: Any, **__: Any) -> Any:
        """Raises :class:`NotImplementedError`. SIECIC/SICID is read-only."""
        raise NotImplementedError(
            "SIECIC/SICID connector is read-only. "
            "Write operations are not permitted under D.Lgs. 82/2005 (CAD), art. 2."
        )


# ── Utilities ──────────────────────────────────────────────────────────────────


def _parse_court_date(raw: str) -> datetime:
    """Parse court date strings to UTC datetime.

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
