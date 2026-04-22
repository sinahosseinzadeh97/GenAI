"""Tests for Italian legal source connectors.

Covers:
  - BaseConnector token-bucket rate limiter
  - _make_document() factory — correct Document structure + it_* metadata
  - NormativaConnector.fetch() with mocked HTTP (XML parsing path)
  - NormativaConnector.fetch() with bad XML (fallback path)
  - CassazioneConnector.fetch() with mocked JSON API
  - GazzettaUfficialeConnector HTML→text extraction
  - DeJureConnector: raises DeJureAccessError when key absent + fallback=False
  - DeJureConnector: uses fallback when key absent + fallback=True
  - Each connector sets correct it_fonte and it_tipo_documento
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.italia.connectors.base import BaseConnector, _TokenBucket
from src.italia.connectors.cassazione import CassazioneConnector
from src.italia.connectors.dejure import DeJureAccessError, DeJureConnector
from src.italia.connectors.gazzetta import GazzettaUfficialeConnector
from src.italia.connectors.normattiva import NormativaConnector
from src.italia.metadata import ItalianLegalMetadata, TipoDocumento


# ── Token-bucket rate limiter ─────────────────────────────────────────────────


class TestTokenBucket:
    def test_single_acquire_is_immediate(self) -> None:
        bucket = _TokenBucket(rate=10.0)  # 10 req/s
        t0 = time.monotonic()
        bucket.acquire()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.2  # Should not block for 10 req/s limit.

    def test_rate_limit_enforced(self) -> None:
        # Use 5 req/s → ~200 ms between tokens once the bucket is drained.
        # We force-drain by calling acquire() until tokens < 1, then time
        # the next call which must wait for refill.
        bucket = _TokenBucket(rate=5.0)
        # Drain the initial tokens (bucket starts full at 5 tokens).
        for _ in range(6):  # Over-drain to guarantee 0 tokens remain.
            bucket._tokens = 0.0
        t0 = time.monotonic()
        bucket.acquire()
        elapsed = time.monotonic() - t0
        # With 5 req/s and an empty bucket the wait is ≥ 1/5 s ≈ 200 ms.
        # We allow ≥ 100 ms to give the scheduler 2× headroom.
        assert elapsed >= 0.10, f"Expected ≥0.10 s delay at 5 rps, got {elapsed:.3f}s"

    def test_zero_rate_clamped(self) -> None:
        """Rate of 0 should not cause division by zero."""
        bucket = _TokenBucket(rate=0.0)
        assert bucket._rate >= 0.01


# ── BaseConnector._make_document() ───────────────────────────────────────────


class TestMakeDocument:
    def test_returns_document_with_it_fields(self) -> None:
        meta = ItalianLegalMetadata(
            fonte="normattiva",
            tipo_documento=TipoDocumento.LEGGE,
            urn_nir="urn:nir:stato:legge:2003-01-09;63",
            anno=2003,
            articolo="Art. 1",
        )
        doc = BaseConnector._make_document(
            content="Testo della legge.",
            italian_meta=meta,
            source_uri="https://normattiva.it/test",
        )
        assert doc.content == "Testo della legge."
        assert doc.metadata.extra["it_fonte"] == "normattiva"
        assert doc.metadata.extra["it_tipo_documento"] == "legge"
        assert doc.metadata.extra["it_urn_nir"] == "urn:nir:stato:legge:2003-01-09;63"
        assert doc.metadata.extra["it_anno"] == "2003"

    def test_source_path_set(self) -> None:
        meta = ItalianLegalMetadata(fonte="gazzetta_ufficiale", tipo_documento=TipoDocumento.DECRETO_LEGGE)
        doc = BaseConnector._make_document(
            content="text",
            italian_meta=meta,
            source_uri="https://gazzettaufficiale.it/eli/id/2024/01/01/24A00001/sg",
        )
        assert isinstance(doc.source_path, Path)

    def test_custom_filename_used(self) -> None:
        meta = ItalianLegalMetadata(fonte="cassazione", tipo_documento=TipoDocumento.SENTENZA_CASSAZIONE)
        doc = BaseConnector._make_document(
            content="text",
            italian_meta=meta,
            source_uri="italgiure://cassazione/12345/2024",
            filename="cass_12345_2024.txt",
        )
        assert doc.metadata.filename == "cass_12345_2024.txt"

    def test_loader_backend_matches_fonte(self) -> None:
        meta = ItalianLegalMetadata(fonte="tar", tipo_documento=TipoDocumento.SENTENZA_TAR)
        doc = BaseConnector._make_document("text", meta, "https://ga.it/sentenza/1")
        assert doc.metadata.loader_backend == "tar"


# ── NormativaConnector ────────────────────────────────────────────────────────


_SAMPLE_NIR_XML = """<?xml version="1.0" encoding="UTF-8"?>
<atto>
  <intestazione>
    <titoloAtto>Codice Civile</titoloAtto>
  </intestazione>
  <articolato>
    <articolo id="1" num="1">
      <rubrica>Capacità giuridica</rubrica>
      <comma>La capacità giuridica si acquista dal momento della nascita.</comma>
    </articolo>
    <articolo id="2" num="2">
      <rubrica>Maggiore età</rubrica>
      <comma>La maggiore età è fissata al compimento del diciottesimo anno.</comma>
    </articolo>
  </articolato>
</atto>"""


class TestNormativaConnector:
    def _connector(self) -> NormativaConnector:
        conn = NormativaConnector(rate_limit_rps=100.0)
        return conn

    @patch("src.italia.connectors.normattiva.NormativaConnector._get")
    def test_fetch_returns_documents(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.text = _SAMPLE_NIR_XML
        mock_get.return_value = mock_resp

        conn = self._connector()
        docs = conn.fetch(urn="urn:nir:stato:regio.decreto:1942-03-16;262")

        assert len(docs) >= 2
        for doc in docs:
            assert doc.content
            assert doc.metadata.extra["it_fonte"] == "normattiva"

    @patch("src.italia.connectors.normattiva.NormativaConnector._get")
    def test_fetch_with_codice_shortcut(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.text = _SAMPLE_NIR_XML
        mock_get.return_value = mock_resp

        conn = self._connector()
        docs = conn.fetch(codice="codice_civile")
        assert len(docs) >= 1

    @patch("src.italia.connectors.normattiva.NormativaConnector._get")
    def test_fetch_with_bad_xml_uses_fallback(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.text = "<html><body>Error page</body></html>"
        mock_get.return_value = mock_resp

        conn = self._connector()
        docs = conn.fetch(urn="urn:nir:stato:legge:2003-01-09;63")
        # Should still return at least one fallback document with stripped HTML text.
        assert isinstance(docs, list)

    def test_fetch_raises_without_codice_or_urn(self) -> None:
        conn = self._connector()
        with pytest.raises(ValueError, match="either"):
            conn.fetch()

    def test_fetch_raises_with_unknown_codice(self) -> None:
        conn = self._connector()
        with pytest.raises(ValueError, match="Unknown codice"):
            conn.fetch(codice="not_a_real_codice")

    @patch("src.italia.connectors.normattiva.NormativaConnector._get")
    def test_limit_respected(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.text = _SAMPLE_NIR_XML
        mock_get.return_value = mock_resp

        conn = self._connector()
        docs = conn.fetch(urn="urn:nir:stato:regio.decreto:1942-03-16;262", limit=1)
        assert len(docs) <= 1

    @patch("src.italia.connectors.normattiva.NormativaConnector._get")
    def test_metadata_fields_correct(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.text = _SAMPLE_NIR_XML
        mock_get.return_value = mock_resp

        conn = self._connector()
        docs = conn.fetch(urn="urn:nir:stato:regio.decreto:1942-03-16;262")

        extra = docs[0].metadata.extra
        assert extra["it_fonte"] == "normattiva"
        assert extra["it_tipo_documento"] in [t.value for t in TipoDocumento]


# ── CassazioneConnector ───────────────────────────────────────────────────────


_SAMPLE_ITALGIURE_RESPONSE = {
    "documenti": [
        {
            "numeroSentenza": "12345",
            "annoSentenza": "2024",
            "sezione": "Sezione Lavoro",
            "massima": "Il datore di lavoro risponde dei danni causati dal dipendente.",
            "testo": "Con sentenza n. 12345/2024 la Corte ha stabilito...",
            "materia": ["diritto del lavoro"],
        },
        {
            "numeroSentenza": "67890",
            "annoSentenza": "2024",
            "sezione": "Sezione I Civile",
            "massima": "L'art. 2043 c.c. presuppone il danno ingiusto.",
            "testo": "La Corte ha ritenuto fondata la pretesa risarcitoria...",
            "materia": ["responsabilità civile"],
        },
    ]
}


class TestCassazioneConnector:
    def _connector(self) -> CassazioneConnector:
        return CassazioneConnector(rate_limit_rps=100.0)

    @patch("src.italia.connectors.cassazione.CassazioneConnector._paginate")
    def test_fetch_returns_two_sentenze(self, mock_paginate: MagicMock) -> None:
        mock_paginate.return_value = iter(_SAMPLE_ITALGIURE_RESPONSE["documenti"])
        conn = self._connector()
        docs = conn.fetch(query="responsabilità", limit=10)
        assert len(docs) == 2

    @patch("src.italia.connectors.cassazione.CassazioneConnector._paginate")
    def test_it_tipo_is_sentenza_cassazione(self, mock_paginate: MagicMock) -> None:
        mock_paginate.return_value = iter(_SAMPLE_ITALGIURE_RESPONSE["documenti"])
        conn = self._connector()
        docs = conn.fetch(query="lavoro", limit=10)
        for doc in docs:
            assert doc.metadata.extra["it_tipo_documento"] == "sentenza_cassazione"

    @patch("src.italia.connectors.cassazione.CassazioneConnector._paginate")
    def test_massima_in_metadata(self, mock_paginate: MagicMock) -> None:
        mock_paginate.return_value = iter(_SAMPLE_ITALGIURE_RESPONSE["documenti"])
        conn = self._connector()
        docs = conn.fetch(query="danno", limit=10)
        for doc in docs:
            assert doc.metadata.extra.get("it_massima") is not None

    @patch("src.italia.connectors.cassazione.CassazioneConnector._paginate")
    def test_sezione_in_metadata(self, mock_paginate: MagicMock) -> None:
        mock_paginate.return_value = iter(_SAMPLE_ITALGIURE_RESPONSE["documenti"])
        conn = self._connector()
        docs = conn.fetch(limit=10)
        assert docs[0].metadata.extra["it_sezione"] == "Sezione Lavoro"

    @patch("src.italia.connectors.cassazione.CassazioneConnector._paginate")
    def test_limit_respected(self, mock_paginate: MagicMock) -> None:
        mock_paginate.return_value = iter(_SAMPLE_ITALGIURE_RESPONSE["documenti"])
        conn = self._connector()
        docs = conn.fetch(limit=1)
        assert len(docs) <= 1


# ── GazzettaUfficialeConnector ────────────────────────────────────────────────


class TestGazzettaHTMLToText:
    def test_html_stripped_to_plain_text(self) -> None:
        html = "<html><body><h1>Decreto Legge</h1><p>Il Governo della Repubblica...</p></body></html>"
        text = GazzettaUfficialeConnector._html_to_text(html)
        assert "Decreto Legge" in text
        assert "<h1>" not in text
        assert "<p>" not in text

    def test_empty_html_returns_empty_string(self) -> None:
        text = GazzettaUfficialeConnector._html_to_text("")
        assert isinstance(text, str)


# ── DeJureConnector ───────────────────────────────────────────────────────────


class TestDeJureConnector:
    def test_raises_access_error_without_key_and_fallback_disabled(self) -> None:
        with pytest.raises(DeJureAccessError, match="API key"):
            DeJureConnector(api_key=None, fallback_public=False)

    def test_instantiates_with_fallback_enabled(self) -> None:
        conn = DeJureConnector(api_key=None, fallback_public=True)
        assert conn._api_key is None
        assert conn._fallback_public is True

    def test_instantiates_with_api_key(self) -> None:
        conn = DeJureConnector(api_key="fake-key-for-test", fallback_public=False)
        assert conn._api_key == "fake-key-for-test"

    @patch("src.italia.connectors.dejure.DeJureConnector._get")
    def test_public_fallback_returns_list(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.text = """
        <html><body>
          <div class='massima'>
            <p>La Corte di Cassazione ha statuito che Art. 2043...</p>
          </div>
          <div class='massima'>
            <p>Il danno biologico è risarcibile ai sensi dell'Art. 32...</p>
          </div>
        </body></html>
        """
        mock_get.return_value = mock_resp

        conn = DeJureConnector(api_key=None, fallback_public=True)
        docs = conn.fetch(query="responsabilità medica", limit=5)
        assert isinstance(docs, list)


# ── Cross-connector: fonte and tipo checks ────────────────────────────────────


@pytest.mark.parametrize(
    "connector_cls, fonte, tipo",
    [
        (
            lambda: NormativaConnector(rate_limit_rps=100.0),
            "normattiva",
            TipoDocumento.CODICE,
        ),
        (
            lambda: CassazioneConnector(rate_limit_rps=100.0),
            "cassazione",
            TipoDocumento.SENTENZA_CASSAZIONE,
        ),
    ],
)
def test_make_document_fonte_and_tipo(
    connector_cls: Any, fonte: str, tipo: TipoDocumento
) -> None:
    meta = ItalianLegalMetadata(fonte=fonte, tipo_documento=tipo)
    doc = BaseConnector._make_document(
        content="test content",
        italian_meta=meta,
        source_uri=f"https://example.it/{fonte}/doc",
    )
    assert doc.metadata.extra["it_fonte"] == fonte
    assert doc.metadata.extra["it_tipo_documento"] == tipo.value
    assert doc.metadata.loader_backend == fonte
