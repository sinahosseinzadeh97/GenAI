"""Tests for ItalianLegalMetadata and TipoDocumento.

Covers:
  - TipoDocumento enum round-trip (value → member → value)
  - ItalianLegalMetadata.to_extra_dict() / from_extra_dict() round-trips
  - Date field ISO serialisation
  - All optional fields None by default
  - is_vigente() logic
  - citation() helper
  - Robustness: unknown it_tipo_documento → ValueError
"""

from __future__ import annotations

from datetime import date

import pytest

from src.italia.metadata import ItalianLegalMetadata, TipoDocumento


# ── TipoDocumento ─────────────────────────────────────────────────────────────


class TestTipoDocumento:
    def test_all_members_are_strings(self) -> None:
        """Every enum member value must be a plain str for Qdrant compatibility."""
        for member in TipoDocumento:
            assert isinstance(member.value, str)

    def test_round_trip_via_value(self) -> None:
        for member in TipoDocumento:
            assert TipoDocumento(member.value) is member

    def test_unknown_value_raises(self) -> None:
        with pytest.raises(ValueError):
            TipoDocumento("totally_fake_value")

    def test_codice_value(self) -> None:
        assert TipoDocumento.CODICE.value == "codice"

    def test_sentenza_cassazione_value(self) -> None:
        assert TipoDocumento.SENTENZA_CASSAZIONE.value == "sentenza_cassazione"


# ── ItalianLegalMetadata dataclass ────────────────────────────────────────────


class TestItalianLegalMetadataDefaults:
    def test_required_fields_only(self) -> None:
        meta = ItalianLegalMetadata(
            fonte="normattiva",
            tipo_documento=TipoDocumento.LEGGE,
        )
        assert meta.fonte == "normattiva"
        assert meta.tipo_documento is TipoDocumento.LEGGE
        assert meta.urn_nir is None
        assert meta.numero_atto is None
        assert meta.anno is None
        assert meta.data_vigenza is None
        assert meta.data_abrogazione is None
        assert meta.articolo is None
        assert meta.comma is None
        assert meta.numero_sentenza is None
        assert meta.sezione is None
        assert meta.data_deposito is None
        assert meta.massima is None
        assert meta.ric_numero is None
        assert meta.materia == []
        assert meta.parole_chiave == []


class TestToExtraDict:
    def test_prefix_applied(self) -> None:
        meta = ItalianLegalMetadata(fonte="cassazione", tipo_documento=TipoDocumento.SENTENZA_CASSAZIONE)
        d = meta.to_extra_dict()
        for key in d:
            assert key.startswith("it_"), f"Key '{key}' is missing 'it_' prefix"

    def test_tipo_documento_serialised_as_string(self) -> None:
        meta = ItalianLegalMetadata(fonte="normattiva", tipo_documento=TipoDocumento.DECRETO_LEGISLATIVO)
        d = meta.to_extra_dict()
        assert d["it_tipo_documento"] == "decreto_legislativo"
        assert isinstance(d["it_tipo_documento"], str)

    def test_dates_serialised_as_iso_strings(self) -> None:
        meta = ItalianLegalMetadata(
            fonte="tar",
            tipo_documento=TipoDocumento.SENTENZA_TAR,
            data_vigenza=date(2024, 3, 15),
            data_deposito=date(2024, 3, 20),
        )
        d = meta.to_extra_dict()
        assert d["it_data_vigenza"] == "2024-03-15"
        assert d["it_data_deposito"] == "2024-03-20"

    def test_none_dates_are_none_not_empty_string(self) -> None:
        meta = ItalianLegalMetadata(fonte="eurlex", tipo_documento=TipoDocumento.REGOLAMENTO)
        d = meta.to_extra_dict()
        assert d["it_data_vigenza"] is None
        assert d["it_data_abrogazione"] is None
        assert d["it_data_deposito"] is None

    def test_list_fields_serialised_correctly(self) -> None:
        meta = ItalianLegalMetadata(
            fonte="cassazione",
            tipo_documento=TipoDocumento.SENTENZA_CASSAZIONE,
            materia=["diritto civile", "responsabilità contrattuale"],
            parole_chiave=["Art. 2043", "danno"],
        )
        d = meta.to_extra_dict()
        assert d["it_materia"] == ["diritto civile", "responsabilità contrattuale"]
        assert d["it_parole_chiave"] == ["Art. 2043", "danno"]


class TestFromExtraDict:
    def _full_meta(self) -> ItalianLegalMetadata:
        return ItalianLegalMetadata(
            fonte="normattiva",
            tipo_documento=TipoDocumento.CODICE,
            urn_nir="urn:nir:stato:regio.decreto:1942-03-16;262",
            numero_atto="R.D. 262/1942",
            anno=1942,
            data_vigenza=date(1942, 7, 21),
            data_abrogazione=None,
            articolo="Art. 2043",
            comma="comma 1",
            numero_sentenza=None,
            sezione=None,
            data_deposito=None,
            massima=None,
            ric_numero=None,
            materia=["diritto civile"],
            parole_chiave=["responsabilità extracontrattuale"],
        )

    def test_round_trip_full(self) -> None:
        original = self._full_meta()
        d = original.to_extra_dict()
        restored = ItalianLegalMetadata.from_extra_dict(d)
        assert restored == original

    def test_round_trip_minimal(self) -> None:
        original = ItalianLegalMetadata(
            fonte="gazzetta_ufficiale",
            tipo_documento=TipoDocumento.DECRETO_LEGGE,
        )
        d = original.to_extra_dict()
        restored = ItalianLegalMetadata.from_extra_dict(d)
        assert restored == original

    def test_missing_required_keys_raise(self) -> None:
        with pytest.raises(KeyError):
            ItalianLegalMetadata.from_extra_dict({"it_fonte": "x"})  # missing tipo
        with pytest.raises(KeyError):
            ItalianLegalMetadata.from_extra_dict({"it_tipo_documento": "legge"})  # missing fonte

    def test_invalid_tipo_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            ItalianLegalMetadata.from_extra_dict({
                "it_fonte": "x",
                "it_tipo_documento": "not_a_real_type",
            })

    def test_malformed_date_returns_none(self) -> None:
        d = {
            "it_fonte": "cassazione",
            "it_tipo_documento": "sentenza_cassazione",
            "it_data_deposito": "not-a-date",
        }
        meta = ItalianLegalMetadata.from_extra_dict(d)
        assert meta.data_deposito is None

    def test_extra_keys_ignored(self) -> None:
        d = {
            "it_fonte": "normattiva",
            "it_tipo_documento": "legge",
            "unknown_key_xyz": "foo",
        }
        meta = ItalianLegalMetadata.from_extra_dict(d)
        assert meta.fonte == "normattiva"


# ── is_vigente() ─────────────────────────────────────────────────────────────


class TestIsVigente:
    def test_vigente_when_no_abrogation(self) -> None:
        meta = ItalianLegalMetadata(fonte="x", tipo_documento=TipoDocumento.LEGGE)
        assert meta.is_vigente() is True

    def test_abrogated_in_past(self) -> None:
        meta = ItalianLegalMetadata(
            fonte="x",
            tipo_documento=TipoDocumento.LEGGE,
            data_abrogazione=date(2020, 1, 1),
        )
        assert meta.is_vigente(as_of=date(2024, 1, 1)) is False

    def test_abrogated_in_future(self) -> None:
        meta = ItalianLegalMetadata(
            fonte="x",
            tipo_documento=TipoDocumento.LEGGE,
            data_abrogazione=date(2099, 12, 31),
        )
        assert meta.is_vigente() is True


# ── citation() ────────────────────────────────────────────────────────────────


class TestCitation:
    def test_sentenza_preferred(self) -> None:
        meta = ItalianLegalMetadata(
            fonte="cassazione",
            tipo_documento=TipoDocumento.SENTENZA_CASSAZIONE,
            numero_sentenza="Cass. Civ. n. 12345/2024",
            numero_atto="L. 99/1999",
        )
        assert meta.citation() == "Cass. Civ. n. 12345/2024"

    def test_numero_atto_with_articolo(self) -> None:
        meta = ItalianLegalMetadata(
            fonte="normattiva",
            tipo_documento=TipoDocumento.CODICE,
            numero_atto="R.D. 262/1942",
            articolo="Art. 2043",
        )
        assert meta.citation() == "R.D. 262/1942, Art. 2043"

    def test_falls_back_to_urn(self) -> None:
        meta = ItalianLegalMetadata(
            fonte="normattiva",
            tipo_documento=TipoDocumento.LEGGE,
            urn_nir="urn:nir:stato:legge:2003-01-09;63",
        )
        assert meta.citation() == "urn:nir:stato:legge:2003-01-09;63"

    def test_last_resort_source_tipo(self) -> None:
        meta = ItalianLegalMetadata(fonte="agcm", tipo_documento=TipoDocumento.PROVVEDIMENTO)
        assert "agcm" in meta.citation()
