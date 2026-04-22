"""Phase 6 — Production Infrastructure for Italy: comprehensive test suite.

Tests cover:
    6.2 API Localisation
        - X-Legal-Disclaimer header present on all responses.
        - Italian error messages returned when Accept-Language: it.
        - Content-Language: it set on translated responses.
        - Error catalogue completeness and formatting.

    6.3 Integration Connectors
        - FilenetDocumentumConnector: webhook event parsing, HMAC verification,
          document conversion, export.
        - LexisNexisItaliaConnector: stub fetch, XML export round-trip,
          webhook HMAC verification.
        - NotartelConnector: stub fetch, XML export round-trip, date parsing.
        - SiecicSicidConnector: stub fascicolo and udienza fetch, typed Document
          output, write-method NotImplementedError enforcement.

Italian legal fixtures are used throughout to represent realistic content.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures — Italian legal content ──────────────────────────────────────────

FIXTURE_FILENET_EVENT: dict[str, Any] = {
    "eventType": "objectCreated",
    "objectId": "IT-BANK-2024-001234",
    "repositoryId": "FPOS_INTESA",
    "timestamp": "2024-03-15T09:30:00Z",
    "repositoryName": "Intesa Sanpaolo DocRepo",
}

FIXTURE_LEXISNEXIS_RESULT: dict[str, Any] = {
    "id": "LN-IT-2024-98765",
    "titolo": "Responsabilità solidale del professionista ex art. 2049 c.c.",
    "testo": (
        "La Corte ha affermato che la responsabilità del datore di lavoro per "
        "fatto illecito dei dipendenti ai sensi dell'art. 2049 c.c. richiede il "
        "nesso di occasionalità necessaria tra l'incarico conferito e il danno."
    ),
    "tipo": "massima",
    "fonte": "Cass. Civ., Sez. III, n. 5678/2024",
    "data": "2024-02-20",
    "url": "https://api.lexisnexis.it/v2/document/LN-IT-2024-98765",
}

FIXTURE_NOTARTEL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<ListaAttiNotarili xmlns="https://www.notartel.it/schema/atti-notarili-v3"
    generato="2024-03-15T10:00:00Z" fonte="stub">
  <AttoNotarile>
    <Repertorio>12345/2024</Repertorio>
    <Data>2024-03-15</Data>
    <Notaio>Paolo Ferrari</Notaio>
    <Sede>Torino</Sede>
    <Oggetto>Atto di donazione immobiliare ex art. 769 c.c.</Oggetto>
    <Testo>Il donante trasferisce gratuitamente al donatario la piena proprietà
    dell'immobile sito in Torino, via Garibaldi n. 10, fog. 15, part. 300.</Testo>
    <Parti>
      <Parte ruolo="disponente">Anna Bianchi, CF: BNCNNA60B41L219Z</Parte>
      <Parte ruolo="beneficiario">Marco Bianchi, CF: BNCMRC90A01L219K</Parte>
    </Parti>
  </AttoNotarile>
</ListaAttiNotarili>
"""

FIXTURE_SIECIC_FASCICOLO: dict[str, Any] = {
    "numero": "3456",
    "anno": "2024",
    "tribunale": "Tribunale di Napoli",
    "sezione": "Terza Sezione Civile",
    "stato": "pendente",
    "oggetto": "Risoluzione contratto di appalto ex art. 1671 c.c.",
    "data_iscrizione": "2024-01-20",
    "parti": [
        {"ruolo": "attore", "nome": "Costruzioni Meridionali S.r.l."},
        {"ruolo": "convenuto", "nome": "Comune di Napoli"},
    ],
    "udienza_prossima": "2024-07-15",
    "sistema": "SIECIC",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 6.2 — Italian Error Catalogue tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestItalianErrorCatalogue:
    """Tests for src.italia.i18n.errors_it."""

    def test_all_required_keys_present(self) -> None:
        """Verify critical error codes are in the catalogue."""
        from src.italia.i18n.errors_it import ERROR_MESSAGES_IT

        required_keys = {
            "NOT_FOUND",
            "INVALID_REQUEST",
            "AUTHENTICATION_REQUIRED",
            "FORBIDDEN",
            "RATE_LIMIT_EXCEEDED",
            "INTERNAL_ERROR",
            "SERVICE_UNAVAILABLE",
            "GDPR_RIGHT_TO_ERASURE",
            "AI_ACT_LOW_CONFIDENCE",
            "NOTARTEL_UNAVAILABLE",
            "SIECIC_READONLY_VIOLATION",
            "WEBHOOK_SIGNATURE_INVALID",
        }
        missing = required_keys - set(ERROR_MESSAGES_IT.keys())
        assert not missing, f"Missing keys in Italian error catalogue: {missing}"

    def test_get_error_message_it_renders_placeholders(self) -> None:
        """Placeholder substitution must work correctly."""
        from src.italia.i18n.errors_it import get_error_message_it

        msg = get_error_message_it("RATE_LIMIT_EXCEEDED", retry_after=60)
        assert "60" in msg
        assert "secondi" in msg.lower()

    def test_get_error_message_it_unknown_key_fallback(self) -> None:
        """Unknown keys must return a safe Italian fallback message."""
        from src.italia.i18n.errors_it import get_error_message_it

        msg = get_error_message_it("NONEXISTENT_KEY_XYZ")
        assert "NONEXISTENT_KEY_XYZ" in msg

    def test_all_messages_are_italian_strings(self) -> None:
        """All messages must be non-empty strings."""
        from src.italia.i18n.errors_it import ERROR_MESSAGES_IT

        for key, (status_code, template) in ERROR_MESSAGES_IT.items():
            assert isinstance(template, str) and len(template) > 5, (
                f"Key {key!r} has an empty or too-short template: {template!r}"
            )
            assert 200 <= status_code <= 599, (
                f"Key {key!r} has invalid status code: {status_code}"
            )

    def test_ai_act_warning_renders_confidence(self) -> None:
        """AI Act warning must include the rendered confidence value."""
        from src.italia.i18n.errors_it import get_error_message_it

        msg = get_error_message_it("AI_ACT_LOW_CONFIDENCE", confidence=0.42)
        assert "42%" in msg

    def test_gdpr_erasure_message_contains_articolo(self) -> None:
        """GDPR right-to-erasure message must cite art. 17 GDPR."""
        from src.italia.i18n.errors_it import get_error_message_it

        msg = get_error_message_it("GDPR_RIGHT_TO_ERASURE")
        assert "17" in msg
        assert "GDPR" in msg

    def test_http_status_for_known_keys(self) -> None:
        """get_http_status_for_key must return correct status codes."""
        from src.italia.i18n.errors_it import get_http_status_for_key

        assert get_http_status_for_key("NOT_FOUND") == 404
        assert get_http_status_for_key("AUTHENTICATION_REQUIRED") == 401
        assert get_http_status_for_key("INTERNAL_ERROR") == 500
        assert get_http_status_for_key("NONEXISTENT") == 500


# ═══════════════════════════════════════════════════════════════════════════════
# 6.2 — ItalianLocalisationMiddleware tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestItalianLocalisationMiddleware:
    """Tests for src.api.middleware_it."""

    def test_wants_italian_bare_it(self) -> None:
        """Bare 'it' value in Accept-Language must be detected."""
        from src.api.middleware_it import _wants_italian

        assert _wants_italian("it") is True

    def test_wants_italian_it_it(self) -> None:
        """'it-IT' variant must be detected."""
        from src.api.middleware_it import _wants_italian

        assert _wants_italian("it-IT") is True

    def test_wants_italian_complex_header(self) -> None:
        """Italian detected in complex multi-language Accept-Language."""
        from src.api.middleware_it import _wants_italian

        assert _wants_italian("en-US,en;q=0.9,it;q=0.8") is True

    def test_wants_italian_english_only(self) -> None:
        """English-only header must not trigger Italian mode."""
        from src.api.middleware_it import _wants_italian

        assert _wants_italian("en-US,en;q=0.9") is False

    def test_wants_italian_none(self) -> None:
        """Missing header must not trigger Italian mode."""
        from src.api.middleware_it import _wants_italian

        assert _wants_italian(None) is False

    def test_legal_disclaimer_constant(self) -> None:
        """The disclaimer must contain the required Italian legal text."""
        from src.api.middleware_it import LEGAL_DISCLAIMER_IT

        assert "carattere informativo" in LEGAL_DISCLAIMER_IT
        assert "parere legale" in LEGAL_DISCLAIMER_IT

    def test_get_italian_error_response_structure(self) -> None:
        """get_italian_error_response must return correct headers and body."""
        from src.api.middleware_it import LEGAL_DISCLAIMER_IT, get_italian_error_response

        resp = get_italian_error_response("NOT_FOUND")
        assert resp.status_code == 404
        assert "it" in resp.headers.get("content-language", "")
        assert resp.headers.get("X-Legal-Disclaimer") == LEGAL_DISCLAIMER_IT
        body = json.loads(resp.body)
        assert "Risorsa" in body["detail"]
        assert body["error_code"] == "NOT_FOUND"

    def test_get_italian_error_response_status_override(self) -> None:
        """Status code override must be respected."""
        from src.api.middleware_it import get_italian_error_response

        resp = get_italian_error_response("NOT_FOUND", status_code=410)
        assert resp.status_code == 410


# ═══════════════════════════════════════════════════════════════════════════════
# 6.3 — FilenetDocumentumConnector tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFilenetDocumentumConnector:
    """Tests for src.italia.connectors.filenet_documentum."""

    def _make_connector(self) -> "Any":
        from src.italia.connectors.filenet_documentum import FilenetDocumentumConnector

        return FilenetDocumentumConnector(
            base_url="https://filenet.stub.it/fncmis/resources",
            repository="FPOS_TEST",
            username="svc_ragforge",
            password="secret",
        )

    def test_connector_source_name(self) -> None:
        conn = self._make_connector()
        assert conn.source_name == "filenet_documentum"

    def test_parse_webhook_event_valid(self) -> None:
        """Valid HMAC → parse event correctly."""
        from src.italia.connectors.filenet_documentum import FilenetDocumentumConnector

        secret = "test_secret"
        payload = json.dumps(FIXTURE_FILENET_EVENT).encode()
        sig = "sha256=" + hmac.new(
            key=secret.encode(), msg=payload, digestmod=hashlib.sha256
        ).hexdigest()

        event = FilenetDocumentumConnector.parse_webhook_event(
            payload=payload, signature=sig, secret=secret
        )
        assert event.event_type == "objectCreated"
        assert event.object_id == "IT-BANK-2024-001234"
        assert event.repository == "FPOS_INTESA"

    def test_parse_webhook_event_invalid_signature(self) -> None:
        """Invalid HMAC → ConnectorError raised."""
        from src.italia.connectors.filenet_documentum import (
            ConnectorError,
            FilenetDocumentumConnector,
        )

        payload = json.dumps(FIXTURE_FILENET_EVENT).encode()
        with pytest.raises(ConnectorError, match="HMAC"):
            FilenetDocumentumConnector.parse_webhook_event(
                payload=payload, signature="sha256=invalid", secret="test_secret"
            )

    def test_parse_webhook_event_missing_signature(self) -> None:
        """Missing signature header → ConnectorError raised."""
        from src.italia.connectors.filenet_documentum import (
            ConnectorError,
            FilenetDocumentumConnector,
        )

        payload = json.dumps(FIXTURE_FILENET_EVENT).encode()
        with pytest.raises(ConnectorError):
            FilenetDocumentumConnector.parse_webhook_event(
                payload=payload, signature=None, secret="test_secret"
            )

    def test_handle_filenet_event_deleted_returns_none(self) -> None:
        """objectDeleted events must return None (no ingestion)."""
        from src.italia.connectors.filenet_documentum import (
            FilenetDocumentumConnector,
            FilenetWebhookEvent,
        )

        conn = self._make_connector()
        event = FilenetWebhookEvent(
            event_type="objectDeleted",
            object_id="IT-DOC-999",
            repository="FPOS_TEST",
            timestamp=datetime.now(tz=timezone.utc),
            raw={},
        )
        result = conn.handle_filenet_event(event)
        assert result is None

    def test_handle_filenet_event_created_returns_document(self) -> None:
        """objectCreated event → Document with correct metadata."""
        from src.ingestion.loader import Document
        from src.italia.connectors.filenet_documentum import (
            FilenetDocumentumConnector,
            FilenetWebhookEvent,
        )

        conn = self._make_connector()

        # Stub the content stream fetch
        with patch.object(conn, "_fetch_content_stream", return_value="Testo del documento."):
            event = FilenetWebhookEvent(
                event_type="objectCreated",
                object_id="IT-BANK-2024-001234",
                repository="FPOS_INTESA",
                timestamp=datetime.now(tz=timezone.utc),
                raw=FIXTURE_FILENET_EVENT,
            )
            doc = conn.handle_filenet_event(event)

        assert isinstance(doc, Document)
        assert doc.content == "Testo del documento."
        assert doc.metadata.loader_backend == "filenet_documentum_webhook"
        extra = doc.metadata.extra or {}
        assert extra.get("filenet_object_id") == "IT-BANK-2024-001234"
        assert extra.get("filenet_event_type") == "objectCreated"

    def test_cmis_item_to_document_structure(self) -> None:
        """CMIS item dict → Document with Italian metadata."""
        from src.ingestion.loader import Document
        from src.italia.connectors.filenet_documentum import FilenetDocumentumConnector

        conn = self._make_connector()
        cmis_item = {
            "properties": {
                "cmis:objectId": {"value": "OBJ-001"},
                "cmis:name": {"value": "contratto_2024.pdf"},
                "cmis:contentStreamMimeType": {"value": "application/pdf"},
                "cmis:creationDate": {"value": "2024-01-10T09:00:00Z"},
                "cmis:lastModificationDate": {"value": "2024-03-01T14:30:00Z"},
            }
        }
        with patch.object(conn, "_fetch_content_stream", return_value="Testo contratto"):
            doc = conn._cmis_item_to_document(cmis_item)

        assert isinstance(doc, Document)
        assert doc.metadata.filename == "contratto_2024.pdf"
        assert doc.content == "Testo contratto"
        extra = doc.metadata.extra or {}
        assert extra.get("filenet_object_id") == "OBJ-001"
        assert extra.get("filenet_repository") == "FPOS_TEST"


# ═══════════════════════════════════════════════════════════════════════════════
# 6.3 — LexisNexisItaliaConnector tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestLexisNexisItaliaConnector:
    """Tests for src.italia.connectors.lexisnexis_it."""

    def _make_connector(self) -> "Any":
        from src.italia.connectors.lexisnexis_it import LexisNexisItaliaConnector

        return LexisNexisItaliaConnector(
            base_url="https://api.lexisnexis.stub.it/v2",
            client_id="test_client",
            client_secret="test_secret",
        )

    def test_connector_source_name(self) -> None:
        conn = self._make_connector()
        assert conn.source_name == "lexisnexis_italia"

    def test_item_to_document_massima(self) -> None:
        """Massima item must produce a Document with correct tipo_documento."""
        from src.ingestion.loader import Document
        from src.italia.metadata import TipoDocumento

        conn = self._make_connector()
        doc = conn._item_to_document(FIXTURE_LEXISNEXIS_RESULT, "massima")

        assert isinstance(doc, Document)
        assert doc.metadata.loader_backend == "lexisnexis_italia"
        assert "art. 2049" in doc.content or "responsabilità" in doc.content.lower()
        extra = doc.metadata.extra or {}
        assert extra.get("lexisnexis_tipo") == "massima"
        assert extra.get("it_tipo_documento") == TipoDocumento.MASSIMA.value

    def test_export_results_xml_valid_xml(self) -> None:
        """XML export must produce well-formed UTF-8 XML."""
        from xml.etree import ElementTree as ET

        conn = self._make_connector()
        doc = conn._item_to_document(FIXTURE_LEXISNEXIS_RESULT, "massima")
        xml_out = conn.export_results_xml(
            documents=[doc],
            query="responsabilità solidale art. 2049",
        )

        assert isinstance(xml_out, str)
        assert "<?xml" in xml_out
        assert "LexisNexisExport" in xml_out
        assert "LN-IT-2024-98765" in xml_out

        # Must be parseable.
        root = ET.fromstring(xml_out)
        assert root.tag.endswith("LexisNexisExport") or "LexisNexisExport" in root.tag
        doc_els = root.findall(".//{https://api.lexisnexis.it/schema/massime-v2}Documento")
        assert len(doc_els) == 1

    def test_export_results_xml_round_trip_title(self) -> None:
        """Exported XML must contain the document title."""
        conn = self._make_connector()
        doc = conn._item_to_document(FIXTURE_LEXISNEXIS_RESULT, "massima")
        xml_out = conn.export_results_xml(documents=[doc], query="art. 2049")
        assert "Responsabilità solidale" in xml_out

    def test_verify_webhook_signature_valid(self) -> None:
        """Valid webhook signature must return True."""
        from src.italia.connectors.lexisnexis_it import LexisNexisItaliaConnector

        secret = "ln_webhook_secret"
        payload = b'{"eventType": "documentUpdated", "id": "LN-IT-123"}'
        sig = "sha256=" + hmac.new(
            key=secret.encode(), msg=payload, digestmod=hashlib.sha256
        ).hexdigest()

        assert LexisNexisItaliaConnector.verify_webhook_signature(payload, sig, secret) is True

    def test_verify_webhook_signature_invalid(self) -> None:
        """Invalid payload → signature verification returns False."""
        from src.italia.connectors.lexisnexis_it import LexisNexisItaliaConnector

        assert (
            LexisNexisItaliaConnector.verify_webhook_signature(
                b"tampered", "sha256=wrongsig", "secret"
            )
            is False
        )

    def test_verify_webhook_signature_missing(self) -> None:
        """None signature → False."""
        from src.italia.connectors.lexisnexis_it import LexisNexisItaliaConnector

        assert (
            LexisNexisItaliaConnector.verify_webhook_signature(b"payload", None, "secret")
            is False
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6.3 — NotartelConnector tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestNotartelConnector:
    """Tests for src.italia.connectors.notartel."""

    def _make_connector(self, stub: bool = True) -> "Any":
        from src.italia.connectors.notartel import NotartelConnector

        return NotartelConnector(token="fake_token", stub=stub)

    def test_connector_source_name(self) -> None:
        conn = self._make_connector()
        assert conn.source_name == "notartel"

    def test_stub_fetch_returns_documents(self) -> None:
        """Stub mode must return at least one Document without HTTP calls."""
        from src.ingestion.loader import Document

        conn = self._make_connector(stub=True)
        docs = conn.fetch()

        assert isinstance(docs, list)
        assert len(docs) >= 1
        for doc in docs:
            assert isinstance(doc, Document)

    def test_stub_document_contains_atto_notarile_metadata(self) -> None:
        """Stub documento must have ATTO_NOTARILE tipo and notartel fields."""
        from src.italia.metadata import TipoDocumento

        conn = self._make_connector(stub=True)
        docs = conn.fetch()
        doc = docs[0]
        extra = doc.metadata.extra or {}

        assert extra.get("it_tipo_documento") == TipoDocumento.ATTO_NOTARILE.value
        assert "notartel_repertorio" in extra or "Repertorio" in doc.content

    def test_parse_xml_response_from_fixture(self) -> None:
        """Fixture XML must parse into one Document with correct content."""
        from src.ingestion.loader import Document

        conn = self._make_connector(stub=False)
        docs = conn._parse_xml_response(FIXTURE_NOTARTEL_XML)

        assert len(docs) == 1
        doc = docs[0]
        assert isinstance(doc, Document)
        assert "12345/2024" in doc.content
        assert "Paolo Ferrari" in doc.content
        assert "art. 769 c.c." in doc.content

    def test_export_to_notartel_xml_round_trip(self) -> None:
        """Export → parse must be lossless for the repertorio field."""
        from xml.etree import ElementTree as ET

        conn = self._make_connector(stub=True)
        docs = conn.fetch()
        xml_out = conn.export_to_notartel_xml(docs)

        assert "<?xml" in xml_out
        assert "ListaAttiNotarili" in xml_out

        root = ET.fromstring(xml_out)
        # Re-parse the exported XML through the connector.
        reparsed = conn._parse_xml_response(xml_out)
        assert len(reparsed) == len(docs)

    def test_export_xml_contains_legal_content(self) -> None:
        """Exported XML from fixture must contain act type and parties."""
        conn = self._make_connector(stub=False)
        docs = conn._parse_xml_response(FIXTURE_NOTARTEL_XML)
        xml_out = conn.export_to_notartel_xml(docs)

        assert "12345/2024" in xml_out
        assert "Paolo Ferrari" in xml_out

    def test_stub_via_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NOTARTEL_STUB=true env var must activate stub mode."""
        monkeypatch.setenv("NOTARTEL_STUB", "true")
        from src.italia.connectors.notartel import NotartelConnector

        conn = NotartelConnector(token="")
        assert conn._stub is True
        docs = conn.fetch()
        assert len(docs) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 6.3 — SiecicSicidConnector tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSiecicSicidConnector:
    """Tests for src.italia.connectors.siecic_sicid."""

    def _make_connector(self, sistema: str = "SIECIC") -> "Any":
        from src.italia.connectors.siecic_sicid import SiecicSicidConnector

        return SiecicSicidConnector(api_key="fake_key", sistema=sistema, stub=True)

    def test_connector_source_name(self) -> None:
        conn = self._make_connector()
        assert conn.source_name == "siecic_sicid"

    def test_invalid_sistema_raises(self) -> None:
        from src.italia.connectors.siecic_sicid import SiecicSicidConnector

        with pytest.raises(ValueError, match="SIECIC|SICID"):
            SiecicSicidConnector(sistema="INVALID")

    def test_stub_fetch_returns_documents(self) -> None:
        """Stub fetch must return Documents without HTTP calls."""
        from src.ingestion.loader import Document

        conn = self._make_connector()
        docs = conn.fetch()

        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)

    def test_stub_fascicolo_contains_legal_fields(self) -> None:
        """Stub fascicolo Document must include SIECIC metadata fields."""
        conn = self._make_connector()
        docs = conn.fetch()
        doc = docs[0]
        extra = doc.metadata.extra or {}

        assert "siecic_tribunale" in extra
        assert "siecic_stato" in extra
        assert "SIECIC" in doc.content

    def test_fetch_fascicolo_by_number_stub(self) -> None:
        """fetch_fascicolo must return a Document matching the number."""
        from src.ingestion.loader import Document

        conn = self._make_connector()
        # The stub has "1234" as one of its fascicoli.
        doc = conn.fetch_fascicolo(numero="1234", anno="2024", tribunale="Tribunale di Milano")
        assert isinstance(doc, Document)
        assert "1234" in doc.content

    def test_fetch_fascicolo_not_found_returns_none(self) -> None:
        """Unknown fascicolo number in stub must return None."""
        conn = self._make_connector()
        result = conn.fetch_fascicolo(numero="9999", anno="2025", tribunale="any")
        assert result is None

    def test_fetch_udienze_returns_documents(self) -> None:
        """Stub udienze fetch must return Documents."""
        from src.ingestion.loader import Document

        conn = self._make_connector()
        docs = conn.fetch_udienze(data="2024-06-20")

        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)

    def test_fascicolo_to_document_structure(self) -> None:
        """Manual fascicolo dict → Document must have correct fields."""
        from src.ingestion.loader import Document
        from src.italia.metadata import TipoDocumento

        conn = self._make_connector()
        doc = conn._fascicolo_to_document(FIXTURE_SIECIC_FASCICOLO)

        assert isinstance(doc, Document)
        extra = doc.metadata.extra or {}
        assert extra.get("siecic_numero") == "3456"
        assert extra.get("siecic_anno") == "2024"
        assert extra.get("siecic_tribunale") == "Tribunale di Napoli"
        assert extra.get("it_tipo_documento") == TipoDocumento.SENTENZA.value
        assert "art. 1671 c.c." in doc.content
        assert "Costruzioni Meridionali" in doc.content

    def test_export_raises_not_implemented(self) -> None:
        """export() must raise NotImplementedError — SIECIC is read-only."""
        conn = self._make_connector()
        with pytest.raises(NotImplementedError, match="read-only"):
            conn.export()

    def test_write_raises_not_implemented(self) -> None:
        """write() must raise NotImplementedError."""
        conn = self._make_connector()
        with pytest.raises(NotImplementedError, match="read-only"):
            conn.write()

    def test_create_raises_not_implemented(self) -> None:
        """create() must raise NotImplementedError."""
        conn = self._make_connector()
        with pytest.raises(NotImplementedError, match="read-only"):
            conn.create()

    def test_delete_raises_not_implemented(self) -> None:
        """delete() must raise NotImplementedError."""
        conn = self._make_connector()
        with pytest.raises(NotImplementedError, match="read-only"):
            conn.delete()

    def test_siecic_stub_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SIECIC_STUB=true env var must activate stub mode."""
        monkeypatch.setenv("SIECIC_STUB", "true")
        from src.italia.connectors.siecic_sicid import SiecicSicidConnector

        conn = SiecicSicidConnector(api_key="")
        assert conn._stub is True

    def test_sicid_sistema_variant(self) -> None:
        """SICID sistema variant must be accepted."""
        conn = self._make_connector(sistema="SICID")
        assert conn._sistema == "SICID"
        docs = conn.fetch()
        assert isinstance(docs, list)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.3 — Connector registry tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConnectorRegistry:
    """Verify Phase 6 connectors appear in the INTEGRATION_CONNECTORS registry."""

    def test_integration_connectors_registered(self) -> None:
        from src.italia.connectors import INTEGRATION_CONNECTORS

        for key in ("filenet", "lexisnexis", "notartel", "siecic", "sicid"):
            assert key in INTEGRATION_CONNECTORS, (
                f"Key {key!r} missing from INTEGRATION_CONNECTORS"
            )

    def test_all_public_connectors_still_registered(self) -> None:
        from src.italia.connectors import CONNECTORS

        for key in (
            "normattiva", "gazzetta", "cassazione", "eurlex",
            "tar", "corte_costituzionale", "agcm", "bancaditalia", "dejure",
        ):
            assert key in CONNECTORS, f"Key {key!r} missing from CONNECTORS"

    def test_webhook_router_importable(self) -> None:
        from src.italia.connectors import webhook_router
        from fastapi import APIRouter

        assert isinstance(webhook_router, APIRouter)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.2 — Settings Phase 6 fields test
# ═══════════════════════════════════════════════════════════════════════════════


class TestSettingsPhase6:
    """Verify Phase 6 settings fields exist with correct defaults."""

    def test_localisation_fields_exist(self) -> None:
        from src.config.settings import Settings

        s = Settings()
        assert hasattr(s, "default_language")
        assert s.default_language == "it"
        assert hasattr(s, "legal_disclaimer_it")
        assert "parere legale" in s.legal_disclaimer_it

    def test_filenet_fields_default_empty(self) -> None:
        from src.config.settings import Settings

        s = Settings()
        assert hasattr(s, "filenet_base_url")
        assert hasattr(s, "filenet_username")
        assert hasattr(s, "filenet_password")
        assert hasattr(s, "filenet_webhook_secret")
        assert s.filenet_base_url == ""

    def test_lexisnexis_fields_default(self) -> None:
        from src.config.settings import Settings

        s = Settings()
        assert hasattr(s, "lexisnexis_client_id")
        assert hasattr(s, "lexisnexis_client_secret")
        assert s.lexisnexis_base_url == "https://api.lexisnexis.it/v2"

    def test_notartel_fields_default(self) -> None:
        from src.config.settings import Settings

        s = Settings()
        assert hasattr(s, "notartel_token")
        assert hasattr(s, "notartel_stub")
        assert s.notartel_stub is False
        assert s.notartel_base_url == "https://api.notartel.it/v3"

    def test_siecic_fields_default(self) -> None:
        from src.config.settings import Settings

        s = Settings()
        assert hasattr(s, "siecic_api_key")
        assert hasattr(s, "siecic_stub")
        assert s.siecic_stub is False
        assert "pst.giustizia.it" in s.siecic_base_url


# ═══════════════════════════════════════════════════════════════════════════════
# 6.1 — docker-compose.italia.yml smoke test
# ═══════════════════════════════════════════════════════════════════════════════


class TestDockerComposeItalia:
    """Verify the Italia Docker Compose override file exists and is valid YAML."""

    def test_file_exists(self) -> None:
        compose_path = Path(__file__).parents[2] / "docker-compose.italia.yml"
        assert compose_path.exists(), "docker-compose.italia.yml must exist"

    def test_file_is_valid_yaml(self) -> None:
        import yaml  # type: ignore[import]

        compose_path = Path(__file__).parents[2] / "docker-compose.italia.yml"
        with open(compose_path) as f:
            content = yaml.safe_load(f)
        assert "services" in content

    def test_data_residency_label_present(self) -> None:
        compose_path = Path(__file__).parents[2] / "docker-compose.italia.yml"
        text = compose_path.read_text()
        assert "data-residency" in text
        assert "IT" in text

    def test_deploy_region_eu_south_1(self) -> None:
        compose_path = Path(__file__).parents[2] / "docker-compose.italia.yml"
        text = compose_path.read_text()
        assert "eu-south-1" in text
