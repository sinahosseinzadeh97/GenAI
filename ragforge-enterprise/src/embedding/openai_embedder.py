"""OpenAI-compatible REST API embedder with retry logic.

Supports any OpenAI-compatible endpoint including:

- OpenAI API (``https://api.openai.com/v1``)
- Local vLLM servers
- Ollama with ``/v1`` compatibility mode
- Azure OpenAI

Implements exponential-backoff retry via ``tenacity`` and tracks token usage
per call for cost accounting.

Typical usage::

    from src.embedding.openai_embedder import OpenAIEmbedder

    # Uses settings from .env (OPENAI_API_BASE, OPENAI_API_KEY, etc.)
    embedder = OpenAIEmbedder()
    vector   = embedder.embed_single("Hello world")
    vectors  = embedder.embed_batch(["text 1", "text 2"])
"""

from __future__ import annotations

import time
from typing import Any

from src.config.settings import get_settings
from src.embedding.base import BaseEmbedder, EmbeddingError
from src.utils.logger import get_logger, log_exception

_settings = get_settings()
log = get_logger(__name__, level=_settings.log_level)

# ── Module-level token usage accumulator ─────────────────────────────────────

_total_prompt_tokens: int = 0


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI-compatible REST API embedder.

    Uses ``httpx`` for transport and ``tenacity`` for retry with exponential
    backoff (max 3 attempts).  All configuration is read from
    :func:`~src.config.settings.get_settings` and can be overridden via the
    constructor.

    Args:
        api_base:    Base URL of the OpenAI-compatible API.
        api_key:     Bearer token / API key.
        model_name:  Model identifier to pass in the request body.
        dim:         Embedding dimension.  Optional – if not provided, the
                     dimension is inferred from the first successful API call.
        normalize:   Whether to L2-normalise output embeddings.
        batch_size:  Number of texts per API request.
        timeout:     HTTP request timeout in seconds.
        max_retries: Maximum retry attempts (default 3).

    Raises:
        EmbeddingError: On all non-retryable API failures.

    Example::

        embedder = OpenAIEmbedder(
            api_base="http://localhost:11434/v1",
            api_key="ollama",
            model_name="nomic-embed-text",
        )
    """

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        dim: int | None = None,
        normalize: bool = True,
        batch_size: int = 32,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(normalize=normalize)
        self._api_base: str = (
            api_base or _settings.openai_api_base or "https://api.openai.com/v1"
        ).rstrip("/")
        self._api_key: str = api_key or _settings.openai_api_key or ""
        self._model_name: str = model_name or _settings.openai_embedding_model
        self._dim: int | None = dim
        self._batch_size: int = batch_size
        self._timeout: float = timeout
        self._max_retries: int = max_retries

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """Return the model identifier used in API requests.

        Returns:
            Model name string.
        """
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the embedding dimension.

        Inferred from the first API call if not provided at construction.

        Returns:
            Integer dimension.

        Raises:
            EmbeddingError: If the dimension cannot be resolved.
        """
        if self._dim is not None:
            return self._dim
        # Trigger a single call to infer dimension.
        vec = self.embed_single("ping")
        self._dim = len(vec)
        return self._dim

    # ── Core embedding ────────────────────────────────────────────────────────

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text via the OpenAI-compatible embeddings endpoint.

        Args:
            text: Input text.

        Returns:
            Float vector.

        Raises:
            EmbeddingError: On API failure after exhausting retries.
        """
        vectors = self._call_api([text])
        return vectors[0]

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
        **_kwargs: Any,
    ) -> list[list[float]]:
        """Embed texts in batches against the OpenAI-compatible API.

        Args:
            texts:      List of strings to embed.
            batch_size: Override the default batch size.

        Returns:
            List of float vectors.

        Raises:
            EmbeddingError: On API failure after exhausting retries.
        """
        if not texts:
            return []

        bs = batch_size or self._batch_size
        log.info(
            "OpenAI batch embedding started",
            extra={
                "model": self._model_name,
                "total_texts": len(texts),
                "batch_size": bs,
                "api_base": self._api_base,
            },
        )

        t_start = time.perf_counter()
        results: list[list[float]] = []

        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            batch_vectors = self._call_api(batch)
            results.extend(batch_vectors)

        elapsed = time.perf_counter() - t_start
        throughput = len(texts) / elapsed if elapsed > 0 else float("inf")
        log.info(
            "OpenAI batch embedding complete",
            extra={
                "model": self._model_name,
                "total_texts": len(texts),
                "duration_seconds": round(elapsed, 4),
                "throughput_per_sec": round(throughput, 2),
                "total_tokens_used": _total_prompt_tokens,
            },
        )
        return results

    # ── Private HTTP helper ───────────────────────────────────────────────────

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """POST to the embeddings endpoint with retry logic.

        Args:
            texts: List of texts to embed (one API call per invocation).

        Returns:
            List of float vectors in input order.

        Raises:
            EmbeddingError: When all retries are exhausted or a non-retryable
                error occurs.
        """
        global _total_prompt_tokens  # noqa: PLW0603

        try:
            import httpx  # type: ignore[import-untyped]
            from tenacity import (  # type: ignore[import-untyped]
                retry,
                retry_if_exception_type,
                stop_after_attempt,
                wait_exponential,
            )
        except ImportError as exc:
            raise EmbeddingError(
                "httpx and tenacity are required for OpenAIEmbedder. "
                "Run: pip install httpx tenacity"
            ) from exc

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        payload: dict[str, Any] = {
            "model": self._model_name,
            "input": texts,
        }

        @retry(
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        )
        def _post() -> httpx.Response:
            with httpx.Client(timeout=self._timeout) as client:
                return client.post(
                    f"{self._api_base}/embeddings",
                    headers=headers,
                    json=payload,
                )

        try:
            response = _post()
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise EmbeddingError(
                f"OpenAI API unreachable after {self._max_retries} retries: {exc}"
            ) from exc
        except Exception as exc:
            log_exception(log, "OpenAI API call failed", exc)
            raise EmbeddingError(f"OpenAI API error: {exc}") from exc

        if response.status_code != 200:
            raise EmbeddingError(
                f"OpenAI API returned HTTP {response.status_code}: {response.text[:500]}"
            )

        data: dict[str, Any] = response.json()

        # Track token usage for cost accounting.
        usage = data.get("usage", {})
        prompt_tokens: int = usage.get("prompt_tokens", 0)
        _total_prompt_tokens += prompt_tokens
        log.debug(
            "OpenAI API usage",
            extra={
                "prompt_tokens": prompt_tokens,
                "cumulative_tokens": _total_prompt_tokens,
                "model": self._model_name,
            },
        )

        # Sort by index to guarantee ordering.
        items: list[dict[str, Any]] = sorted(data["data"], key=lambda x: x["index"])
        vectors: list[list[float]] = []
        for item in items:
            vec: list[float] = item["embedding"]
            vectors.append(self._maybe_normalize(vec))

        return vectors
