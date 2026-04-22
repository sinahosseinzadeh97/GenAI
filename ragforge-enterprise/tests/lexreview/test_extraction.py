"""Tests for src/lexreview/extraction/ — Clause, LegalEntities, NER, regex, detector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.lexreview.extraction.clause_detector import ClauseDetector
from src.lexreview.extraction.models import Clause, LegalEntities
from src.lexreview.extraction.regex_extractor import RegexExtractor

# ── Model tests ───────────────────────────────────────────────────────────────


class TestClauseModel:
    def test_clause_valid(self) -> None:
        c = Clause(
            type="indemnification",
            text="Party A shall indemnify Party B...",
            span=(0, 35),
            confidence=0.9,
        )
        assert c.type == "indemnification"
        assert c.confidence == 0.9

    def test_clause_confidence_bounds(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Clause(type="x", text="t", span=(0, 1), confidence=1.5)

    def test_legal_entities_defaults(self) -> None:
        le = LegalEntities()
        assert le.parties == []
        assert le.dates == []
        assert le.amounts == []
        assert le.jurisdictions == []

    def test_legal_entities_populated(self) -> None:
        le = LegalEntities(
            parties=["Acme Corp."],
            dates=["2024-01-01"],
            amounts=["$50,000"],
            jurisdictions=["Delaware"],
        )
        assert le.parties == ["Acme Corp."]
        assert le.amounts == ["$50,000"]


# ── RegexExtractor tests ──────────────────────────────────────────────────────


class TestRegexExtractor:
    @pytest.fixture()
    def extractor(self) -> RegexExtractor:
        return RegexExtractor()

    def test_extract_monetary_amounts_dollar(self, extractor: RegexExtractor) -> None:
        text = "The fee is $50,000.00 payable in 30 days."
        result = extractor.extract(text)
        assert any("50,000" in a for a in result.amounts)

    def test_extract_monetary_amounts_usd_prefix(self, extractor: RegexExtractor) -> None:
        text = "Total consideration: USD 1,000,000 to be paid on signing."
        result = extractor.extract(text)
        assert any("1,000,000" in a for a in result.amounts)

    def test_extract_iso_date(self, extractor: RegexExtractor) -> None:
        text = "Effective Date: 2024-06-30."
        result = extractor.extract(text)
        assert "2024-06-30" in result.dates

    def test_extract_long_date(self, extractor: RegexExtractor) -> None:
        text = "Signed on January 15, 2024 in New York."
        result = extractor.extract(text)
        assert any("January" in d for d in result.dates)

    def test_extract_party_corp(self, extractor: RegexExtractor) -> None:
        text = "Acme Corporation and Beta LLC hereby agree."
        result = extractor.extract(text)
        # at least one party extracted
        assert len(result.parties) >= 1

    def test_extract_jurisdiction_state_of(self, extractor: RegexExtractor) -> None:
        text = "This Agreement is governed by the laws of the State of Delaware."
        result = extractor.extract(text)
        assert any("Delaware" in j for j in result.jurisdictions)

    def test_extract_jurisdiction_governed_by(self, extractor: RegexExtractor) -> None:
        text = "governed by New York law."
        result = extractor.extract(text)
        assert any("New York" in j for j in result.jurisdictions)

    def test_merge_dedupes(self, extractor: RegexExtractor) -> None:
        a = LegalEntities(parties=["Acme Corp."], dates=["2024-01-01"])
        b = LegalEntities(parties=["Acme Corp."], amounts=["$500"])
        merged = extractor.merge(a, b)
        # Acme Corp. should appear only once
        assert merged.parties.count("Acme Corp.") == 1
        assert "$500" in merged.amounts

    def test_empty_text_returns_empty_entities(self, extractor: RegexExtractor) -> None:
        result = extractor.extract("   ")
        assert result.parties == []
        assert result.amounts == []


# ── ClauseDetector tests ──────────────────────────────────────────────────────


class TestClauseDetector:
    @pytest.fixture()
    def detector(self) -> ClauseDetector:
        return ClauseDetector(min_confidence=0.3)

    def test_detect_indemnification(self, detector: ClauseDetector) -> None:
        text = (
            "Party A shall indemnify, defend, and hold harmless Party B from any "
            "losses, damages, costs and expenses arising from Party A's breach."
        )
        clauses = detector.detect(text)
        types = [c.type for c in clauses]
        assert "indemnification" in types

    def test_detect_termination(self, detector: ClauseDetector) -> None:
        text = "Either party may terminate this Agreement upon 30 days written notice of termination."
        clauses = detector.detect(text)
        types = [c.type for c in clauses]
        assert "termination" in types

    def test_detect_confidentiality(self, detector: ClauseDetector) -> None:
        text = "The receiving party shall maintain all confidential information in strict confidence and shall not disclose it to third parties."
        clauses = detector.detect(text)
        types = [c.type for c in clauses]
        assert "confidentiality" in types

    def test_detect_force_majeure(self, detector: ClauseDetector) -> None:
        text = "No party shall be liable for delays caused by force majeure events including acts of God, natural disaster, or pandemic."
        clauses = detector.detect(text)
        types = [c.type for c in clauses]
        assert "force_majeure" in types

    def test_detect_payment(self, detector: ClauseDetector) -> None:
        text = "Payment is due within net 30 days of invoice receipt. Overdue amounts accrue interest."
        clauses = detector.detect(text)
        types = [c.type for c in clauses]
        assert "payment" in types

    def test_clause_confidence_in_bounds(self, detector: ClauseDetector) -> None:
        text = "Party A shall indemnify Party B against all losses."
        clauses = detector.detect(text)
        for c in clauses:
            assert 0.0 <= c.confidence <= 1.0

    def test_clause_span_consistent(self, detector: ClauseDetector) -> None:
        text = "Party A shall indemnify Party B.\nEither party may terminate with notice."
        clauses = detector.detect(text)
        for c in clauses:
            start, end = c.span
            assert 0 <= start < end <= len(text) + 1  # +1 for split edge

    def test_empty_text_returns_no_clauses(self, detector: ClauseDetector) -> None:
        clauses = detector.detect("")
        assert clauses == []

    def test_high_threshold_filters_low_confidence(self) -> None:
        strict = ClauseDetector(min_confidence=0.99)
        text = "Payments shall be made. Indemnification applies."
        clauses = strict.detect(text)
        # Should filter most/all out at 0.99
        assert all(c.confidence >= 0.99 for c in clauses)


# ── LegalNER tests (mocked spaCy) ─────────────────────────────────────────────


class TestLegalNER:
    def _make_mock_ent(self, text: str, label: str) -> MagicMock:
        ent = MagicMock()
        ent.text = text
        ent.label_ = label
        return ent

    def _make_mock_doc(self, ents: list[MagicMock]) -> MagicMock:
        doc = MagicMock()
        doc.ents = ents
        return doc

    @patch("src.lexreview.extraction.ner._load_nlp")
    def test_extract_party_org(self, mock_load: MagicMock) -> None:
        from src.lexreview.extraction.ner import LegalNER

        mock_nlp = MagicMock()
        mock_nlp.return_value = self._make_mock_doc(
            [self._make_mock_ent("Acme Corp.", "ORG")]
        )
        mock_load.return_value = mock_nlp

        ner = LegalNER(model_name="en_core_web_sm")
        result = ner.extract("Acme Corp. agrees to the terms.")
        assert "Acme Corp." in result.parties

    @patch("src.lexreview.extraction.ner._load_nlp")
    def test_extract_date(self, mock_load: MagicMock) -> None:
        from src.lexreview.extraction.ner import LegalNER

        mock_nlp = MagicMock()
        mock_nlp.return_value = self._make_mock_doc(
            [self._make_mock_ent("January 1, 2024", "DATE")]
        )
        mock_load.return_value = mock_nlp

        ner = LegalNER(model_name="en_core_web_sm")
        result = ner.extract("Effective January 1, 2024.")
        assert "January 1, 2024" in result.dates

    @patch("src.lexreview.extraction.ner._load_nlp")
    def test_extract_jurisdiction_gpe(self, mock_load: MagicMock) -> None:
        from src.lexreview.extraction.ner import LegalNER

        mock_nlp = MagicMock()
        mock_nlp.return_value = self._make_mock_doc(
            [self._make_mock_ent("Delaware", "GPE")]
        )
        mock_load.return_value = mock_nlp

        ner = LegalNER(model_name="en_core_web_sm")
        result = ner.extract("Governed by the laws of Delaware.")
        assert "Delaware" in result.jurisdictions

    @patch("src.lexreview.extraction.ner._load_nlp")
    def test_deduplication(self, mock_load: MagicMock) -> None:
        from src.lexreview.extraction.ner import LegalNER

        mock_nlp = MagicMock()
        mock_nlp.return_value = self._make_mock_doc(
            [
                self._make_mock_ent("Acme Corp.", "ORG"),
                self._make_mock_ent("Acme Corp.", "ORG"),  # duplicate
            ]
        )
        mock_load.return_value = mock_nlp

        ner = LegalNER(model_name="en_core_web_sm")
        result = ner.extract("Acme Corp. and Acme Corp. agree.")
        assert result.parties.count("Acme Corp.") == 1

    @patch("src.lexreview.extraction.ner._load_nlp")
    def test_spacy_not_installed_raises(self, mock_load: MagicMock) -> None:
        from src.lexreview.extraction.ner import LegalNER

        mock_load.side_effect = RuntimeError("spaCy is not installed.")
        ner = LegalNER(model_name="en_core_web_sm")
        with pytest.raises(RuntimeError, match="spaCy"):
            ner.extract("Some text.")
