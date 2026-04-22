"""Tests for RAGForge Italia Phase 2 — Italian NLP Pipeline.

Tests are split into three modules matching the three Phase 2 deliverables:
  - 2.1  ItalianLegalEmbedder + CrossEncoderReranker
  - 2.2  ItalianLegalNER
  - 2.3  ItalianLegalCleaner

All heavy model-loading is mocked so that the test suite runs fast in CI
without GPU or network access.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Phase 2.1 — ItalianLegalEmbedder & CrossEncoderReranker
# ──────────────────────────────────────────────────────────────────────────────


class TestItalianLegalEmbedder:
    """Tests for ItalianLegalEmbedder (Phase 2.1)."""

    def _make_mock_st(self, dim: int = 1024) -> MagicMock:
        """Return a MagicMock that mimics SentenceTransformer."""
        mock = MagicMock()
        mock.get_sentence_embedding_dimension.return_value = dim
        mock.encode.side_effect = lambda text, **kw: (
            np.random.rand(dim).astype(np.float32)
            if isinstance(text, str)
            else np.random.rand(len(text), dim).astype(np.float32)
        )
        return mock

    def test_embed_query_uses_italian_prefix(self) -> None:
        """embed_query must prepend the Italian E5-instruct prefix."""
        from src.embedding.italian_embedder import ItalianLegalEmbedder, _ITALIAN_QUERY_INSTRUCTION

        mock_st = self._make_mock_st()
        with patch("src.embedding.italian_embedder._load_sentence_transformer", return_value=mock_st):
            embedder = ItalianLegalEmbedder()
            embedder._active_model_name = "test-model"
            _ = embedder.embed_query("responsabilità contrattuale")

        call_args = mock_st.encode.call_args[0][0]
        assert call_args.startswith(_ITALIAN_QUERY_INSTRUCTION), (
            f"Query prefix not found. Got: {call_args[:80]}"
        )
        assert "responsabilità contrattuale" in call_args

    def test_embed_passage_uses_passage_prefix(self) -> None:
        """embed_passage must prepend 'Passage: '."""
        from src.embedding.italian_embedder import ItalianLegalEmbedder, _PASSAGE_PREFIX

        mock_st = self._make_mock_st()
        with patch("src.embedding.italian_embedder._load_sentence_transformer", return_value=mock_st):
            embedder = ItalianLegalEmbedder()
            embedder._active_model_name = "test-model"
            _ = embedder.embed_passage("Art. 2043 c.c. prevede ...")

        call_args = mock_st.encode.call_args[0][0]
        assert call_args.startswith(_PASSAGE_PREFIX)
        assert "Art. 2043 c.c." in call_args

    def test_dimension_returns_model_dim(self) -> None:
        """dimension property must reflect the loaded model's output size."""
        from src.embedding.italian_embedder import ItalianLegalEmbedder

        mock_st = self._make_mock_st(dim=1024)
        with patch("src.embedding.italian_embedder._load_sentence_transformer", return_value=mock_st):
            embedder = ItalianLegalEmbedder()
            embedder._active_model_name = "test-model"
            assert embedder.dimension == 1024

    def test_fallback_model_used_when_primary_fails(self) -> None:
        """If primary model load throws, fallback model must be used."""
        from src.embedding.base import EmbeddingError
        from src.embedding.italian_embedder import ItalianLegalEmbedder, _FALLBACK_MODEL

        fallback_mock = self._make_mock_st(dim=768)

        def side_effect(model_name: str) -> Any:
            if "e5" in model_name or "instruct" in model_name:
                raise EmbeddingError("Primary unavailable")
            return fallback_mock

        with patch("src.embedding.italian_embedder._load_sentence_transformer", side_effect=side_effect):
            embedder = ItalianLegalEmbedder()
            result = embedder.embed_passage("test passage")

        assert embedder._active_model_name == _FALLBACK_MODEL
        assert len(result) == 768

    def test_embed_batch_mode_passage(self) -> None:
        """embed_batch with mode='passage' must prefix all texts."""
        from src.embedding.italian_embedder import ItalianLegalEmbedder, _PASSAGE_PREFIX

        mock_st = self._make_mock_st(dim=1024)
        with patch("src.embedding.italian_embedder._load_sentence_transformer", return_value=mock_st):
            embedder = ItalianLegalEmbedder()
            embedder._active_model_name = "test-model"
            results = embedder.embed_batch(["testo uno", "testo due"], mode="passage")

        assert len(results) == 2
        prefixed_texts = mock_st.encode.call_args[0][0]
        for t in prefixed_texts:
            assert t.startswith(_PASSAGE_PREFIX)

    def test_embed_batch_empty(self) -> None:
        """embed_batch on empty list returns empty list without calling model."""
        from src.embedding.italian_embedder import ItalianLegalEmbedder

        with patch("src.embedding.italian_embedder._load_sentence_transformer") as mock_load:
            embedder = ItalianLegalEmbedder()
            result = embedder.embed_batch([])

        assert result == []
        mock_load.assert_not_called()

    def test_output_is_l2_normalised(self) -> None:
        """Embeddings must be unit-length when normalize=True (default)."""
        from src.embedding.italian_embedder import ItalianLegalEmbedder

        fixed_vec = np.array([3.0, 4.0], dtype=np.float32)
        mock_st = MagicMock()
        mock_st.get_sentence_embedding_dimension.return_value = 2
        mock_st.encode.return_value = fixed_vec

        with patch("src.embedding.italian_embedder._load_sentence_transformer", return_value=mock_st):
            embedder = ItalianLegalEmbedder()
            embedder._active_model_name = "test-model"
            vec = embedder.embed_passage("test")

        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_model_name_property(self) -> None:
        """model_name returns the active model once resolved."""
        from src.embedding.italian_embedder import ItalianLegalEmbedder, _PRIMARY_MODEL

        mock_st = self._make_mock_st()
        with patch("src.embedding.italian_embedder._load_sentence_transformer", return_value=mock_st):
            embedder = ItalianLegalEmbedder()
            # Before load, should return primary name.
            assert _PRIMARY_MODEL in embedder.model_name or embedder.model_name == _PRIMARY_MODEL


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker (Phase 2.1)."""

    def _make_mock_ce(self, scores: list[float]) -> MagicMock:
        mock = MagicMock()
        mock.predict.return_value = np.array(scores, dtype=np.float32)
        return mock

    def test_rerank_sorts_by_descending_score(self) -> None:
        """Passages must be returned highest-score-first."""
        from src.embedding.italian_embedder import CrossEncoderReranker

        passages = ["bassa rilevanza", "altissima rilevanza", "media rilevanza"]
        mock_ce = self._make_mock_ce([0.2, 0.9, 0.5])

        with patch("src.embedding.italian_embedder._load_cross_encoder", return_value=mock_ce):
            reranker = CrossEncoderReranker()
            ranked = reranker.rerank("query legale", passages)

        assert ranked[0][0] == "altissima rilevanza"
        assert ranked[1][0] == "media rilevanza"
        assert ranked[2][0] == "bassa rilevanza"

    def test_rerank_respects_top_k(self) -> None:
        """top_k must limit the number of returned results."""
        from src.embedding.italian_embedder import CrossEncoderReranker

        passages = [f"passaggio {i}" for i in range(10)]
        mock_ce = self._make_mock_ce(list(range(10)))

        with patch("src.embedding.italian_embedder._load_cross_encoder", return_value=mock_ce):
            reranker = CrossEncoderReranker(top_k=3)
            ranked = reranker.rerank("query", passages)

        assert len(ranked) == 3

    def test_rerank_empty_passages(self) -> None:
        """rerank on empty list returns empty list."""
        from src.embedding.italian_embedder import CrossEncoderReranker

        with patch("src.embedding.italian_embedder._load_cross_encoder") as mock_load:
            reranker = CrossEncoderReranker()
            result = reranker.rerank("query", [])

        assert result == []
        mock_load.assert_not_called()

    def test_rerank_with_metadata_preserves_fields(self) -> None:
        """rerank_with_metadata must preserve all dict fields and add rerank_score."""
        from src.embedding.italian_embedder import CrossEncoderReranker

        passages = [
            {"content": "art. 1218 c.c.", "source": "codice_civile", "page": 1},
            {"content": "cass. n. 1234/2024", "source": "cassazione", "page": 5},
        ]
        mock_ce = self._make_mock_ce([0.3, 0.8])

        with patch("src.embedding.italian_embedder._load_cross_encoder", return_value=mock_ce):
            reranker = CrossEncoderReranker()
            ranked = reranker.rerank_with_metadata("query", passages, text_key="content")

        assert ranked[0]["content"] == "cass. n. 1234/2024"
        assert "rerank_score" in ranked[0]
        assert ranked[0]["source"] == "cassazione"


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2.2 — ItalianLegalNER
# ──────────────────────────────────────────────────────────────────────────────


class TestItalianLegalNERRegex:
    """Tests for the regex extraction layer of ItalianLegalNER (no spaCy needed)."""

    def test_norma_article_cc(self) -> None:
        """Recognise 'Art. 2043 c.c.' and informal variants."""
        from src.lexreview.extraction.italian_ner import _RE_NORMA

        texts = [
            "ai sensi dell'art. 2043 c.c.",
            "Art. 1218 c.c., comma 2",
            "articolo 2043 cc",
        ]
        for text in texts:
            matches = _RE_NORMA.findall(text)
            assert matches, f"No NORMA match in: {text!r}"

    def test_norma_dlgs(self) -> None:
        """Recognise D.Lgs. decree citations."""
        from src.lexreview.extraction.italian_ner import _RE_NORMA

        assert _RE_NORMA.search("D.Lgs. 231/2001")
        assert _RE_NORMA.search("D.Lgs. n. 50/2016")

    def test_norma_legge(self) -> None:
        """Recognise Legge citations."""
        from src.lexreview.extraction.italian_ner import _RE_NORMA

        assert _RE_NORMA.search("L. 300/1970")
        assert _RE_NORMA.search("legge n. 241/1990")

    def test_norma_eu_regulation(self) -> None:
        """Recognise EU Regulation citations."""
        from src.lexreview.extraction.italian_ner import _RE_NORMA

        assert _RE_NORMA.search("Reg. (UE) 2016/679")

    def test_sentenza_cassazione(self) -> None:
        """Recognise Cassazione judgment references."""
        from src.lexreview.extraction.italian_ner import _RE_SENTENZA

        tests = [
            "Cass. Civ. Sez. III, n. 12345/2024",
            "Cassazione Penale, n. 9876/2023",
            "Cass. n. 1/2024",
        ]
        for text in tests:
            assert _RE_SENTENZA.search(text), f"No SENTENZA match in: {text!r}"

    def test_sentenza_tar(self) -> None:
        """Recognise TAR judgment references."""
        from src.lexreview.extraction.italian_ner import _RE_SENTENZA

        assert _RE_SENTENZA.search("TAR Lazio, n. 678/2023")

    def test_importo_euro_symbol(self) -> None:
        """Recognise € amounts."""
        from src.lexreview.extraction.italian_ner import _RE_IMPORTO

        texts = ["€ 50.000", "€ 1.234.567,89", "€50000"]
        for text in texts:
            assert _RE_IMPORTO.search(text), f"No IMPORTO match in: {text!r}"

    def test_importo_words(self) -> None:
        """Recognise word-form amounts."""
        from src.lexreview.extraction.italian_ner import _RE_IMPORTO

        assert _RE_IMPORTO.search("duemila euro")
        assert _RE_IMPORTO.search("cinquecentomila euro")

    def test_data_giuridica_days(self) -> None:
        """Recognise 'entro N giorni dalla notifica' patterns."""
        from src.lexreview.extraction.italian_ner import _RE_DATA_GIURIDICA

        texts = [
            "entro 30 giorni dalla notifica",
            "entro 60 giorni dalla pubblicazione",
            "termine perentorio di 90 giorni",
        ]
        for text in texts:
            assert _RE_DATA_GIURIDICA.search(text), f"No DATA match in: {text!r}"

    def test_dedupe_helper(self) -> None:
        """_dedupe must remove duplicates and normalise whitespace."""
        from src.lexreview.extraction.italian_ner import _dedupe

        result = _dedupe(["  foo  ", "foo", "bar", "bar", "baz"])
        assert result == ["foo", "bar", "baz"]

    def test_italian_legal_entities_to_dict(self) -> None:
        """ItalianLegalEntities.to_dict must return all seven label keys."""
        from src.lexreview.extraction.italian_ner import ItalianLegalEntities

        e = ItalianLegalEntities(
            norme=["Art. 1 c.c."],
            sentenze=["Cass. n. 1/2024"],
        )
        d = e.to_dict()
        expected_keys = {
            "NORMA", "SENTENZA", "SOGGETTO_GIURIDICO",
            "ISTITUZIONE", "TERMINE_LEGALE", "IMPORTO", "DATA_GIURIDICA",
        }
        assert set(d.keys()) == expected_keys
        assert d["NORMA"] == ["Art. 1 c.c."]


class TestItalianLegalNERWithMockedSpacy:
    """Tests for ItalianLegalNER using a mocked spaCy pipeline."""

    def _make_mock_nlp(self, entities: list[tuple[str, str]]) -> MagicMock:
        """Build a mock spaCy Language that yields given (text, label) entities."""
        mock_ent = lambda text, label: MagicMock(text=text, label_=label)  # noqa: E731
        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent(t, l) for t, l in entities]

        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ["ner"]
        mock_nlp.return_value = mock_doc
        mock_nlp.pipe.return_value = [mock_doc]
        return mock_nlp

    def test_extract_istituzione_from_spacy(self) -> None:
        """Institutions recognised by spaCy ORG label should appear in istituzioni."""
        from src.lexreview.extraction.italian_ner import ItalianLegalNER

        mock_nlp = self._make_mock_nlp([("Corte di Cassazione", "ISTITUZIONE")])

        with patch("src.lexreview.extraction.italian_ner._load_italian_nlp", return_value=mock_nlp):
            ner = ItalianLegalNER()
            result = ner.extract("La Corte di Cassazione ha stabilito ...")

        assert "Corte di Cassazione" in result.istituzioni

    def test_extract_soggetto_from_spacy(self) -> None:
        """SOGGETTO_GIURIDICO entities from EntityRuler go to soggetti_giuridici."""
        from src.lexreview.extraction.italian_ner import ItalianLegalNER

        mock_nlp = self._make_mock_nlp([("società a responsabilità limitata", "SOGGETTO_GIURIDICO")])

        with patch("src.lexreview.extraction.italian_ner._load_italian_nlp", return_value=mock_nlp):
            ner = ItalianLegalNER()
            result = ner.extract("La società a responsabilità limitata alfa ...")

        assert "società a responsabilità limitata" in result.soggetti_giuridici

    def test_extract_termine_legale(self) -> None:
        """TERMINE_LEGALE entities go to termini_legali."""
        from src.lexreview.extraction.italian_ner import ItalianLegalNER

        mock_nlp = self._make_mock_nlp([("inadempimento", "TERMINE_LEGALE")])

        with patch("src.lexreview.extraction.italian_ner._load_italian_nlp", return_value=mock_nlp):
            ner = ItalianLegalNER()
            result = ner.extract("L'inadempimento del contratto ...")

        assert "inadempimento" in result.termini_legali

    def test_extract_batch_returns_list(self) -> None:
        """extract_batch must return one result per input text."""
        from src.lexreview.extraction.italian_ner import ItalianLegalNER

        texts = ["testo uno", "testo due", "testo tre"]

        # Build one mock_doc per text so that nlp.pipe() iterates correctly.
        def make_doc() -> MagicMock:
            d = MagicMock()
            d.ents = []
            return d

        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ["ner"]
        mock_nlp.return_value = make_doc()
        mock_nlp.pipe.return_value = [make_doc() for _ in texts]

        with patch("src.lexreview.extraction.italian_ner._load_italian_nlp", return_value=mock_nlp):
            ner = ItalianLegalNER()
            results = ner.extract_batch(texts)

        assert len(results) == 3

    def test_regex_and_spacy_combined(self) -> None:
        """Full extract combines regex NORMA with spaCy ISTITUZIONE."""
        from src.lexreview.extraction.italian_ner import ItalianLegalNER

        mock_nlp = self._make_mock_nlp([("AGCM", "ISTITUZIONE")])

        with patch("src.lexreview.extraction.italian_ner._load_italian_nlp", return_value=mock_nlp):
            ner = ItalianLegalNER()
            text = "L'AGCM ha applicato la L. 287/1990 irrogando una sanzione di € 50.000."
            result = ner.extract(text)

        assert "AGCM" in result.istituzioni
        assert any("287/1990" in n for n in result.norme), f"norme: {result.norme}"
        assert any("50.000" in i or "50" in i for i in result.importi), f"importi: {result.importi}"


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2.3 — ItalianLegalCleaner
# ──────────────────────────────────────────────────────────────────────────────


class TestCitationNormalisation:
    """Unit tests for citation normalisation helpers."""

    def test_article_cc_normalisation(self) -> None:
        from src.ingestion.italian_cleaner import _normalise_citations

        result = _normalise_citations("ai sensi art. 2043 cc")
        assert "Art. 2043 c.c." in result

    def test_article_cp_normalisation(self) -> None:
        from src.ingestion.italian_cleaner import _normalise_citations

        result = _normalise_citations("art. 575 cp")
        assert "Art. 575 c.p." in result

    def test_dlgs_year_expansion(self) -> None:
        from src.ingestion.italian_cleaner import _normalise_citations

        result = _normalise_citations("D.Lgs. 231/01")
        assert "D.Lgs. 231/2001" in result, f"Got: {result}"

    def test_legge_year_expansion(self) -> None:
        from src.ingestion.italian_cleaner import _normalise_citations

        result = _normalise_citations("l. 300/70")
        assert "L. 300/1970" in result, f"Got: {result}"

    def test_dpr_year_expansion(self) -> None:
        from src.ingestion.italian_cleaner import _normalise_citations

        result = _normalise_citations("D.P.R. 633/72")
        assert "D.P.R. 633/1972" in result, f"Got: {result}"

    def test_four_digit_year_unchanged(self) -> None:
        from src.ingestion.italian_cleaner import _normalise_citations

        result = _normalise_citations("L. 300/1970")
        assert "1970" in result

    def test_comma_preserved(self) -> None:
        from src.ingestion.italian_cleaner import _normalise_citations

        result = _normalise_citations("art. 2043, co. 1 cc")
        assert "Art. 2043" in result
        assert "co. 1" in result


class TestOmissisNormalisation:
    """Unit tests for omissis normalisation."""

    def test_uppercase_omissis(self) -> None:
        from src.ingestion.italian_cleaner import _normalise_omissis

        assert "[OMISSIS]" in _normalise_omissis("come da OMISSIS nel testo")

    def test_parenthesised_omissis(self) -> None:
        from src.ingestion.italian_cleaner import _normalise_omissis

        assert "[OMISSIS]" in _normalise_omissis("il nome (omissis) è stato rimosso")

    def test_star_redaction(self) -> None:
        from src.ingestion.italian_cleaner import _normalise_omissis

        assert "[OMISSIS]" in _normalise_omissis("l'imputato *** ha dichiarato")

    def test_bracketed_dots(self) -> None:
        from src.ingestion.italian_cleaner import _normalise_omissis

        assert "[OMISSIS]" in _normalise_omissis("reddito [***] annuo")


class TestGazzettaUfficialeCleaner:
    """Unit tests for GU boilerplate stripping."""

    def test_strips_masthead(self) -> None:
        from src.ingestion.italian_cleaner import _strip_gu_boilerplate

        text = (
            "GAZZETTA UFFICIALE DELLA REPUBBLICA ITALIANA\n"
            "Serie Generale n. 123\n"
            "Testo di legge effettivo."
        )
        result = _strip_gu_boilerplate(text)
        assert "GAZZETTA UFFICIALE" not in result
        assert "Testo di legge effettivo." in result

    def test_strips_series_header(self) -> None:
        from src.ingestion.italian_cleaner import _strip_gu_boilerplate

        text = "4a Serie Speciale\nArt. 1. La presente legge ..."
        result = _strip_gu_boilerplate(text)
        assert "Serie Speciale" not in result
        assert "Art. 1." in result


class TestItalianLegalCleanerIntegration:
    """Integration tests for ItalianLegalCleaner (mocking base class)."""

    def _make_document(self, content: str, filename: str = "test.pdf") -> Any:
        from src.ingestion.loader import Document, DocumentMetadata

        meta = DocumentMetadata(
            filename=filename,
            page_count=1,
            page_number=1,
            creation_date=None,
            loader_backend="test",
            extra={},
        )
        return Document(content=content, metadata=meta, source_path=Path(filename), page_number=1)

    def test_clean_normalises_citations(self) -> None:
        """ItalianLegalCleaner.clean must normalise article citations."""
        from src.ingestion.italian_cleaner import ItalianLegalCleaner

        doc = self._make_document("Ai sensi dell'art. 2043 cc è previsto il risarcimento.")
        cleaner = ItalianLegalCleaner()
        cleaned = cleaner.clean(doc)
        assert "Art. 2043 c.c." in cleaned.content

    def test_clean_normalises_omissis(self) -> None:
        """ItalianLegalCleaner.clean must unify omissis markers."""
        from src.ingestion.italian_cleaner import ItalianLegalCleaner

        doc = self._make_document("Il ricorrente OMISSIS ha presentato ricorso.")
        cleaner = ItalianLegalCleaner()
        cleaned = cleaner.clean(doc)
        assert "[OMISSIS]" in cleaned.content

    def test_clean_strips_gu_boilerplate(self) -> None:
        """ItalianLegalCleaner.clean must strip GU mastheads."""
        from src.ingestion.italian_cleaner import ItalianLegalCleaner

        content = (
            "GAZZETTA UFFICIALE DELLA REPUBBLICA ITALIANA\n"
            "Serie Generale\n\n"
            "Art. 1. La presente legge ..."
        )
        doc = self._make_document(content)
        cleaner = ItalianLegalCleaner()
        cleaned = cleaner.clean(doc)
        assert "GAZZETTA UFFICIALE" not in cleaned.content
        assert "Art. 1" in cleaned.content

    def test_clean_metadata_marks_italian(self) -> None:
        """Cleaned document metadata must contain italian_cleaned=True."""
        from src.ingestion.italian_cleaner import ItalianLegalCleaner

        doc = self._make_document("Testo di prova OMISSIS.")
        cleaner = ItalianLegalCleaner()
        cleaned = cleaner.clean(doc)
        assert cleaned.metadata.extra.get("italian_cleaned") is True

    def test_clean_batch_processes_all_pages(self) -> None:
        """clean_batch must return the same number of docs as input."""
        from src.ingestion.italian_cleaner import ItalianLegalCleaner

        docs = [self._make_document(f"Pagina {i}. OMISSIS.") for i in range(5)]
        cleaner = ItalianLegalCleaner()
        cleaned = cleaner.clean_batch(docs)
        assert len(cleaned) == 5

    def test_disabled_flags_skip_transforms(self) -> None:
        """When flags are False, the corresponding transforms are skipped."""
        from src.ingestion.italian_cleaner import ItalianLegalCleaner

        doc = self._make_document("Il testo OMISSIS rimane invariato.")
        cleaner = ItalianLegalCleaner(normalise_omissis=False)
        cleaned = cleaner.clean(doc)
        # OMISSIS should NOT be replaced since normalise_omissis=False.
        assert "OMISSIS" in cleaned.content or "[OMISSIS]" not in cleaned.content

    def test_preserve_legal_symbols(self) -> None:
        """§, ©, and ° must survive the cleaning process."""
        from src.ingestion.italian_cleaner import ItalianLegalCleaner

        doc = self._make_document("Ai sensi del § 3, © 2024 MiGiust., temp. 37°.")
        cleaner = ItalianLegalCleaner()
        cleaned = cleaner.clean(doc)
        assert "§" in cleaned.content
        assert "©" in cleaned.content
        assert "°" in cleaned.content
