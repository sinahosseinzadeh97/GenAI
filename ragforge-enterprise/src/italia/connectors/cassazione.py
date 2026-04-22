"""Cassazione connector — italgiure.giustizia.it.

Fetches civil and criminal Italian Supreme Court judgments (sentenze)
from the ItalGiurè portal via its public search API.

Authentication: None required for public access.
API format:     JSON REST.
Pagination:     Offset-based (``start`` / ``rows`` params).

Metadata extracted per sentenza:
  - ``numero_sentenza``: e.g. "Cass. Civ. n. 12345/2024"
  - ``sezione``        : e.g. "Sezione Lavoro", "Sezioni Unite"
  - ``data_deposito``  : date the judgment was filed
  - ``massima``        : official headnote (massima ufficiale)
  - ``materia``        : subject-matter tags from ItalGiurè taxonomy

The full text of the sentenza is stored as ``content``; the massima is
additionally stored in ``it_massima`` for efficient headline lookup.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from src.ingestion.loader import Document
from src.italia.connectors.base import BaseConnector
from src.italia.metadata import ItalianLegalMetadata, TipoDocumento
from src.utils.logger import get_logger

log = get_logger(__name__)

_SEARCH_URL = "https://italgiure.giustizia.it/sncass/rest/ricerca"
_DETAIL_URL = "https://italgiure.giustizia.it/sncass/rest/documento"

_FALLBACK_SEARCH_URL = "https://www.italgiure.giustizia.it/xway/application/nif/isapi/hc.dll"


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            from datetime import datetime  # noqa: PLC0415

            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


class CassazioneConnector(BaseConnector):
    """Fetches Corte di Cassazione sentenze from ItalGiurè.

    Args:
        rate_limit_rps: Max req/s (default 0.8).
        timeout:        HTTP timeout in seconds.

    Example::

        conn = CassazioneConnector()
        docs = conn.fetch(
            materia="diritto del lavoro",
            from_date=date(2023, 1, 1),
            limit=30,
        )
    """

    source_name = "cassazione"

    def __init__(self, rate_limit_rps: float = 0.8, timeout: int = 45) -> None:
        super().__init__(rate_limit_rps=rate_limit_rps, timeout=timeout)

    def fetch(  # type: ignore[override]
        self,
        query: str = "",
        materia: str | None = None,
        sezione: str | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
        limit: int = 50,
        **_: Any,
    ) -> list[Document]:
        """Fetch sentenze from the Corte di Cassazione.

        Args:
            query:     Full-text search query (Italian).
            materia:   Subject matter filter (e.g. ``"diritto del lavoro"``).
            sezione:   Court section filter (e.g. ``"Sezioni Unite"``).
            from_date: Earliest deposition date (inclusive).
            to_date:   Latest deposition date (inclusive).
            limit:     Maximum number of sentenze to return.

        Returns:
            List of :class:`~src.ingestion.loader.Document` objects, one per
            sentenza.  Full text is in ``content``; massima in
            ``metadata.extra["it_massima"]``.
        """
        log.info(
            "Fetching from Corte di Cassazione",
            extra={
                "query": query,
                "materia": materia,
                "sezione": sezione,
                "from_date": str(from_date),
                "limit": limit,
            },
        )

        documents: list[Document] = []
        params = self._build_params(query, materia, sezione, from_date, to_date)

        for raw in self._paginate(
            url=_SEARCH_URL,
            params=params,
            page_key="start",
            size_key="rows",
            page_size=min(limit, 20),
            max_pages=(limit // 20) + 2,
            results_key="documenti",
        ):
            if len(documents) >= limit:
                break
            doc = self._build_document(raw)
            if doc:
                documents.append(doc)

        # If the primary API returned nothing, try the HTML fallback.
        if not documents:
            log.info(
                "Primary ItalGiurè API returned no results; trying HTML fallback",
                extra={"query": query},
            )
            documents = self._html_fallback(query, materia, from_date, to_date, limit)

        log.info(
            "Cassazione fetch complete",
            extra={"source": self.source_name, "count": len(documents)},
        )
        return documents

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_params(
        query: str,
        materia: str | None,
        sezione: str | None,
        from_date: date | None,
        to_date: date | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "query": query or "*",
            "tipoProvvedimento": "sentenza",
        }
        if materia:
            params["materia"] = materia
        if sezione:
            params["sezione"] = sezione
        if from_date:
            params["dataDeposizioneDa"] = from_date.strftime("%d/%m/%Y")
        if to_date:
            params["dataDeposizioneA"] = to_date.strftime("%d/%m/%Y")
        return params

    def _build_document(self, raw: dict[str, Any]) -> Document | None:
        """Convert a raw ItalGiurè JSON record to a Document."""
        # Extract fields — ItalGiurè uses inconsistent key names.
        numero = (
            raw.get("numeroSentenza")
            or raw.get("numero")
            or raw.get("numProvvedimento")
            or ""
        )
        anno_str = raw.get("annoSentenza") or raw.get("anno") or ""
        sezione = raw.get("sezione") or raw.get("sez") or None
        massima = raw.get("massima") or raw.get("testomassima") or None
        full_text = raw.get("testo") or raw.get("testoProvvedimento") or massima or ""
        data_dep_str = raw.get("dataDeposizione") or raw.get("data") or None

        if not full_text:
            return None

        try:
            anno = int(anno_str) if anno_str else None
        except ValueError:
            anno = None

        dep_date = _parse_date(data_dep_str)
        materia_vals: list[str] = []
        raw_materia = raw.get("materia") or raw.get("materie") or []
        if isinstance(raw_materia, list):
            materia_vals = [str(m) for m in raw_materia]
        elif raw_materia:
            materia_vals = [str(raw_materia)]

        numero_sentenza: str | None = None
        if numero:
            sentenza_type = "Civ." if sezione and "CIVIL" in str(sezione).upper() else "Pen."
            anno_suffix = f"/{anno}" if anno else ""
            numero_sentenza = f"Cass. {sentenza_type} n. {numero}{anno_suffix}"

        source_uri = (
            raw.get("url") or raw.get("uri") or f"italgiure://cassazione/{numero}/{anno}"
        )

        meta = ItalianLegalMetadata(
            fonte=self.source_name,
            tipo_documento=TipoDocumento.SENTENZA_CASSAZIONE,
            anno=anno,
            numero_sentenza=numero_sentenza,
            sezione=sezione,
            data_deposito=dep_date,
            massima=massima[:1000] if massima else None,
            materia=materia_vals,
        )
        return self._make_document(
            content=full_text,
            italian_meta=meta,
            source_uri=source_uri,
            filename=f"cassazione_{numero}_{anno}.txt",
        )

    def _html_fallback(
        self,
        query: str,
        materia: str | None,
        from_date: date | None,
        to_date: date | None,
        limit: int,
    ) -> list[Document]:
        """HTML scraping fallback for when the primary JSON API is unavailable."""
        try:
            from bs4 import BeautifulSoup  # noqa: PLC0415
        except ImportError:
            log.warning("beautifulsoup4 not installed; skipping HTML fallback")
            return []

        params: dict[str, Any] = {
            "db": "NCASS",
            "testo": query or "responsabilità",
        }
        if from_date:
            params["dataDa"] = from_date.strftime("%d/%m/%Y")
        if to_date:
            params["dataA"] = to_date.strftime("%d/%m/%Y")

        documents: list[Document] = []
        try:
            resp = self._get(_FALLBACK_SEARCH_URL, params=params)
            soup = BeautifulSoup(resp.text, "html.parser")
            for row in soup.select("tr.risultato")[:limit]:
                cells = row.find_all("td")
                if len(cells) < 3:
                    continue
                text = row.get_text(separator=" ", strip=True)
                if not text:
                    continue
                meta = ItalianLegalMetadata(
                    fonte=self.source_name,
                    tipo_documento=TipoDocumento.SENTENZA_CASSAZIONE,
                )
                doc_url = _FALLBACK_SEARCH_URL
                documents.append(self._make_document(text, meta, doc_url))
        except Exception as exc:  # noqa: BLE001
            log.warning("HTML fallback failed", extra={"error": str(exc)})

        return documents
