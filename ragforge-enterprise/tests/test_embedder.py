"""Tests for the BGE and base embedder implementations.

Covers:
- Output shape correctness (384 dimensions).
- L2 normalisation (unit vector).
- Batch vs single embedding consistency.
- Empty input handling.
- Mode difference (passage vs query prefix).
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embedding.base import BaseEmbedder, EmbeddingError
from src.embedding.bge_embedder import BGEEmbedder


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_fake_model(dim: int = 384) -> MagicMock:
    """Return a MagicMock that mimics a SentenceTransformer model.

    Args:
        dim: Embedding dimension of fake output vectors.

    Returns:
        Configured MagicMock.
    """
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dim

    def fake_encode(texts: list[str] | str, **kwargs: object) -> np.ndarray:
        """Return deterministic fake embeddings."""
        is_single = isinstance(texts, str)
        n = 1 if is_single else len(texts)
        # Produce non-normalised vectors so we can verify normalisation.
        rng = np.random.default_rng(seed=42)
        return rng.random((n, dim)).astype(np.float32)

    model.encode.side_effect = fake_encode
    return model


# ── Tests: BaseEmbedder._l2_normalize ────────────────────────────────────────


class TestL2Normalize:
    """Unit tests for the static _l2_normalize helper."""

    def test_unit_vector_unchanged(self) -> None:
        """A vector already at unit length should be returned approximately unchanged."""
        vec = [1.0, 0.0, 0.0]
        result = BaseEmbedder._l2_normalize(vec)
        assert abs(result[0] - 1.0) < 1e-6
        assert abs(result[1]) < 1e-6
        assert abs(result[2]) < 1e-6

    def test_produces_unit_length(self) -> None:
        """Normalised vectors should have L2 norm ≈ 1."""
        vec = [3.0, 4.0]  # Norm = 5.0
        result = BaseEmbedder._l2_normalize(vec)
        norm = math.sqrt(sum(x**2 for x in result))
        assert abs(norm - 1.0) < 1e-6, f"Norm={norm}, expected 1.0"

    def test_zero_vector_unchanged(self) -> None:
        """Zero vector should be returned without modification (no division by zero)."""
        vec = [0.0, 0.0, 0.0]
        result = BaseEmbedder._l2_normalize(vec)
        assert result == vec


# ── Tests: BGEEmbedder ────────────────────────────────────────────────────────


class TestBGEEmbedder:
    """Integration-style tests for BGEEmbedder using a mocked model."""

    @pytest.fixture(autouse=True)
    def patch_model_loader(self) -> None:
        """Replace the real model loader with a fake for all tests in this class."""
        self._fake_model = _make_fake_model(dim=384)
        with patch(
            "src.embedding.bge_embedder._load_model",
            return_value=self._fake_model,
        ):
            yield

    def test_embed_single_shape(self) -> None:
        """embed_single must return a 384-element list."""
        embedder = BGEEmbedder()
        vec = embedder.embed_single("Hello, RAGForge!")
        assert isinstance(vec, list)
        assert len(vec) == 384

    def test_embed_single_normalised(self) -> None:
        """Vectors must have L2 norm ≈ 1 when normalize=True (default)."""
        embedder = BGEEmbedder(normalize=True)
        vec = embedder.embed_single("Test normalisation")
        norm = math.sqrt(sum(x**2 for x in vec))
        assert abs(norm - 1.0) < 1e-5, f"Norm={norm}"

    def test_embed_single_not_normalised(self) -> None:
        """When normalize=False, vectors need not be unit vectors."""
        embedder = BGEEmbedder(normalize=False)
        vec = embedder.embed_single("Test no normalisation")
        norm = math.sqrt(sum(x**2 for x in vec))
        # Norm from fake model will not be 1 (random values).
        assert norm != 0.0

    def test_embed_batch_shape(self) -> None:
        """embed_batch must return one vector per input text."""
        embedder = BGEEmbedder()
        texts = ["doc one", "doc two", "doc three"]
        vectors = embedder.embed_batch(texts)
        assert len(vectors) == 3
        for vec in vectors:
            assert len(vec) == 384

    def test_embed_batch_empty(self) -> None:
        """embed_batch with empty input should return an empty list."""
        embedder = BGEEmbedder()
        result = embedder.embed_batch([])
        assert result == []

    def test_embed_batch_each_normalised(self) -> None:
        """Every vector in a batch result should be unit-length."""
        embedder = BGEEmbedder(normalize=True)
        texts = [f"sentence {i}" for i in range(10)]
        vectors = embedder.embed_batch(texts, batch_size=4)
        for vec in vectors:
            norm = math.sqrt(sum(x**2 for x in vec))
            assert abs(norm - 1.0) < 1e-5, f"Norm={norm}"

    def test_embed_batch_respects_batch_size(self) -> None:
        """Model.encode should be called ceil(N/batch_size) times."""
        embedder = BGEEmbedder(batch_size=3)
        texts = [f"t{i}" for i in range(7)]
        embedder.embed_batch(texts, batch_size=3)
        # 7 texts / batch_size 3 → calls with 3, 3, 1
        assert self._fake_model.encode.call_count == 3

    def test_dimension_property(self) -> None:
        """dimension property should reflect model output size (384)."""
        embedder = BGEEmbedder()
        assert embedder.dimension == 384

    def test_model_name_property(self) -> None:
        """model_name property must return the configured model string."""
        embedder = BGEEmbedder(model_name="BAAI/bge-small-en-v1.5")
        assert embedder.model_name == "BAAI/bge-small-en-v1.5"

    def test_passage_vs_query_prefix(self) -> None:
        """passage and query modes should prepend different prefixes."""
        embedder = BGEEmbedder()
        embedder.embed_single("test", mode="passage")
        embedder.embed_single("test", mode="query")

        calls = self._fake_model.encode.call_args_list
        passage_input: str = calls[0][0][0]
        query_input: str = calls[1][0][0]

        assert passage_input.startswith("Represent this sentence for searching")
        assert query_input.startswith("Represent this query for searching")

    def test_embed_batch_passage_mode(self) -> None:
        """All texts in a passage-mode batch must get the passage prefix."""
        embedder = BGEEmbedder()
        embedder.embed_batch(["alpha", "beta"], mode="passage")

        call_args = self._fake_model.encode.call_args_list[0]
        batch_input: list[str] = call_args[0][0]
        for text in batch_input:
            assert text.startswith("Represent this sentence for searching")
