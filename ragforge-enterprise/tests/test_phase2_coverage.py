"""Additional coverage tests for OpenAI embedder and Qdrant client factory.

These complement the existing test suite to push total project coverage ≥ 80%.
All external I/O is mocked so no network or Docker is required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embedding.base import EmbeddingError
from src.embedding.openai_embedder import OpenAIEmbedder
from src.vectorstore.base import VectorStoreError
from src.vectorstore.qdrant_store import QdrantStore


# ── Helper: build a fake httpx.Response ──────────────────────────────────────


def _fake_response(
    vectors: list[list[float]],
    prompt_tokens: int = 12,
    status_code: int = 200,
) -> MagicMock:
    """Return a MagicMock that mimics a successful httpx.Response.

    Args:
        vectors:      Embedding vectors to include in the response body.
        prompt_tokens: Token count to include in ``usage``.
        status_code:  HTTP status code.

    Returns:
        Configured MagicMock.
    """
    response = MagicMock()
    response.status_code = status_code
    response.text = "OK"
    response.json.return_value = {
        "data": [
            {"index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": prompt_tokens},
    }
    return response


def _fake_vec(dim: int = 4, val: float = 0.5) -> list[float]:
    """Return a constant vector of length *dim*.

    Args:
        dim: Dimensionality.
        val: Constant fill value.

    Returns:
        List of floats.
    """
    return [val] * dim


# ── Tests: OpenAIEmbedder ─────────────────────────────────────────────────────


class TestOpenAIEmbedder:
    """Tests for :class:`~src.embedding.openai_embedder.OpenAIEmbedder`."""

    @pytest.fixture()
    def embedder(self) -> OpenAIEmbedder:
        """Return an OpenAIEmbedder configured for testing.

        Returns:
            :class:`OpenAIEmbedder` with dim pre-set to avoid the ping call.
        """
        return OpenAIEmbedder(
            api_base="http://fake-api/v1",
            api_key="test-key",
            model_name="test-model",
            dim=4,  # Pre-set dim so dimension property doesn't trigger embed_single.
            normalize=False,
            batch_size=2,
        )

    def _patch_post(self, vecs: list[list[float]]) -> Any:
        """Context manager that patches httpx.Client.post.

        Args:
            vecs: Vectors to include in the mocked response.

        Returns:
            ``unittest.mock.patch`` context manager.
        """
        resp = _fake_response(vecs)
        return patch("httpx.Client.post", return_value=resp)

    # ── property tests ────────────────────────────────────────────────────────

    def test_model_name_property(self, embedder: OpenAIEmbedder) -> None:
        """model_name must return the configured model identifier."""
        assert embedder.model_name == "test-model"

    def test_dimension_property_pre_set(self, embedder: OpenAIEmbedder) -> None:
        """When dim is provided at init, dimension must return it without a call."""
        assert embedder.dimension == 4

    def test_dimension_property_inferred(self) -> None:
        """When dim is None, dimension must be inferred from the first API call."""
        embedder = OpenAIEmbedder(
            api_base="http://fake-api/v1",
            api_key="key",
            model_name="m",
            normalize=False,
        )
        vec = _fake_vec(dim=8, val=0.25)
        with self._patch_post([vec]):
            dim = embedder.dimension
        assert dim == 8

    # ── embed_single ──────────────────────────────────────────────────────────

    def test_embed_single_returns_correct_shape(self, embedder: OpenAIEmbedder) -> None:
        """embed_single must return a list of length *dim*."""
        vec = _fake_vec(4)
        with self._patch_post([vec]):
            result = embedder.embed_single("hello")
        assert isinstance(result, list)
        assert len(result) == 4

    def test_embed_single_normalize_false(self, embedder: OpenAIEmbedder) -> None:
        """When normalize=False, raw API values are returned unchanged."""
        vec = [0.1, 0.2, 0.3, 0.4]
        with self._patch_post([vec]):
            result = embedder.embed_single("text")
        assert all(abs(r - e) < 1e-6 for r, e in zip(result, vec))

    def test_embed_single_normalize_true(self) -> None:
        """When normalize=True, result must have unit L2 norm."""
        import math

        embedder = OpenAIEmbedder(
            api_base="http://fake-api/v1",
            api_key="k",
            model_name="m",
            dim=4,
            normalize=True,
        )
        vec = [3.0, 4.0, 0.0, 0.0]  # Norm = 5.
        with self._patch_post([vec]):
            result = embedder.embed_single("norm test")
        norm = math.sqrt(sum(x**2 for x in result))
        assert abs(norm - 1.0) < 1e-5

    # ── embed_batch ───────────────────────────────────────────────────────────

    def test_embed_batch_empty(self, embedder: OpenAIEmbedder) -> None:
        """embed_batch on empty list must return []."""
        assert embedder.embed_batch([]) == []

    def test_embed_batch_returns_one_per_text(self, embedder: OpenAIEmbedder) -> None:
        """embed_batch must return exactly len(texts) vectors."""
        vecs = [_fake_vec(4, val=float(i)) for i in range(4)]

        call_count = 0

        def _post_side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            # batch_size=2 → two API calls, each returning 2 vectors.
            batch_vecs = vecs[call_count * 2 : call_count * 2 + 2]
            call_count += 1
            return _fake_response(batch_vecs)

        with patch("httpx.Client.post", side_effect=_post_side_effect):
            result = embedder.embed_batch(["a", "b", "c", "d"])

        assert len(result) == 4

    def test_embed_batch_calls_api_ceil_n_over_batch(self, embedder: OpenAIEmbedder) -> None:
        """With batch_size=2 and 3 texts, the API should be hit twice."""
        call_count = 0

        def _post(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            # First call → 2 texts, second call → 1 text.
            n = 2 if call_count == 0 else 1
            call_count += 1
            return _fake_response([_fake_vec(4)] * n)

        with patch("httpx.Client.post", side_effect=_post):
            embedder.embed_batch(["x", "y", "z"])

        assert call_count == 2

    # ── error handling ────────────────────────────────────────────────────────

    def test_non_200_status_raises_embedding_error(self, embedder: OpenAIEmbedder) -> None:
        """HTTP errors from the API must be surfaced as EmbeddingError."""
        resp = MagicMock()
        resp.status_code = 401
        resp.text = "Unauthorized"

        with patch("httpx.Client.post", return_value=resp):
            with pytest.raises(EmbeddingError, match="HTTP 401"):
                embedder.embed_single("fail")

    def test_network_error_raises_embedding_error(self, embedder: OpenAIEmbedder) -> None:
        """Network errors (after retries) must bubble up as EmbeddingError."""
        import httpx

        with patch(
            "httpx.Client.post",
            side_effect=httpx.NetworkError("connection refused"),
        ):
            with pytest.raises(EmbeddingError, match="unreachable"):
                embedder.embed_single("fail")

    def test_import_error_for_missing_httpx(self, embedder: OpenAIEmbedder) -> None:
        """If httpx is not importable, EmbeddingError must be raised."""
        import builtins

        real_import = builtins.__import__

        def _fail_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in ("httpx", "tenacity"):
                raise ImportError("mocked missing pkg")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_fail_import):
            with pytest.raises(EmbeddingError, match="httpx"):
                embedder._call_api(["text"])


# ── Tests: QdrantStore._build_client ─────────────────────────────────────────


class TestQdrantBuildClient:
    """Tests for the QdrantStore client factory."""

    def test_build_client_raises_on_import_error(self) -> None:
        """VectorStoreError must be raised when qdrant-client is not importable."""
        import builtins

        real_import = builtins.__import__

        def _fail(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "qdrant_client":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        # Bypass __init__; call _build_client directly with a minimal stub.
        store = QdrantStore.__new__(QdrantStore)
        store._host = "localhost"
        store._port = 6333
        store._use_grpc = False
        store._collection_name = "x"
        store._hnsw_m = 16
        store._hnsw_ef_construct = 100

        with patch("builtins.__import__", side_effect=_fail):
            with pytest.raises(VectorStoreError, match="not installed"):
                store._build_client()
