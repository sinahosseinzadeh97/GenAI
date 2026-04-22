"""Thin async-capable LLM client supporting OpenAI-compatible and Anthropic providers.

:class:`LLMClient` wraps the OpenAI SDK (``openai.OpenAI`` /
``openai.AsyncOpenAI``) **and** the Anthropic SDK
(``anthropic.Anthropic`` / ``anthropic.AsyncAnthropic``) behind a uniform
``complete`` / ``acomplete`` interface with automatic retries via
:mod:`tenacity`.

All settings are read from :func:`~src.config.settings.get_settings` by
default, making the client zero-config in production and fully injectable
in tests.

Typical usage::

    # OpenAI-compatible (default)
    client = LLMClient()
    answer = client.complete([{"role": "user", "content": "Hello"}])

    # Anthropic Claude
    client = LLMClient(provider="anthropic")
    answer = client.complete([{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations

import asyncio
from typing import Any

from openai import AsyncOpenAI, OpenAI  # type: ignore[import-untyped]
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)
_settings = get_settings()


class LLMError(Exception):
    """Raised on unrecoverable LLM call failures."""


class LLMClient:
    """LLM client with sync and async interfaces for OpenAI-compatible and Anthropic providers.

    Args:
        provider:    ``"openai"`` (default) or ``"anthropic"``.
        base_url:    Override ``settings.llm_base_url`` (OpenAI path only).
        model:       Override ``settings.llm_model``.
        api_key:     Override ``settings.openai_api_key`` (OpenAI) or
                     ``settings.anthropic_api_key`` (Anthropic).
        temperature: Override ``settings.llm_temperature``.
        max_tokens:  Override ``settings.llm_max_tokens``.

    Example::

        client = LLMClient(provider="anthropic")
        answer = client.complete([{"role": "user", "content": "Summarise this clause."}])
    """

    def __init__(
        self,
        provider: str = "openai",
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.provider = provider
        self._model = model or _settings.llm_model
        self._temperature = temperature if temperature is not None else _settings.llm_temperature
        self._max_tokens = max_tokens if max_tokens is not None else _settings.llm_max_tokens

        if self.provider == "anthropic":
            try:
                import anthropic  # type: ignore[import-untyped]  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "The 'anthropic' package is required when provider='anthropic'. "
                    "Install it with:  pip install 'anthropic>=0.25.0'"
                ) from exc

            _api_key = api_key or getattr(_settings, "anthropic_api_key", None) or ""
            # Clients are created once so the SDK's HTTP connection pool is reused
            # across every call, eliminating per-request TCP-handshake overhead.
            self._sync_client: Any = anthropic.Anthropic(api_key=_api_key)
            self._async_client: Any = anthropic.AsyncAnthropic(api_key=_api_key)
            self._base_url: str | None = None  # not applicable for Anthropic

        else:  # "openai" or any OpenAI-compatible endpoint
            self._base_url = base_url or _settings.llm_base_url
            _api_key = api_key or _settings.openai_api_key or "sk-no-key"
            # Clients are created once so the SDK's HTTP connection pool is reused
            # across every call, eliminating per-request TCP-handshake overhead.
            self._sync_client = OpenAI(base_url=self._base_url, api_key=_api_key)
            self._async_client = AsyncOpenAI(base_url=self._base_url, api_key=_api_key)

        log.info(
            "LLMClient initialised",
            extra={
                "provider": self.provider,
                "model": self._model,
                "base_url": self._base_url,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            },
        )

    # ── Sync ──────────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def complete(self, messages: list[dict[str, str]]) -> str:
        """Synchronous chat completion with automatic retry.

        Args:
            messages: OpenAI-style message list (role + content dicts).

        Returns:
            Assistant message content string.

        Raises:
            LLMError: On non-retryable or exhausted-retry failures.
        """
        try:
            if self.provider == "anthropic":
                response = self._sync_client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=messages,
                )
                content = response.content[0].text
            else:  # OpenAI-compatible path
                response = self._sync_client.chat.completions.create(
                    messages=messages,  # type: ignore[arg-type]
                    **self._build_kwargs(),
                )
                content = response.choices[0].message.content or ""

            log.debug(
                "LLMClient.complete succeeded",
                extra={
                    "provider": self.provider,
                    "model": self._model,
                    "input_messages": len(messages),
                    "output_tokens": len(content.split()),
                },
            )
            return content
        except Exception as exc:
            log.warning("LLMClient.complete error", extra={"error": str(exc)})
            raise LLMError(f"LLM completion failed: {exc}") from exc

    # ── Async ─────────────────────────────────────────────────────────────────

    async def acomplete(self, messages: list[dict[str, str]]) -> str:
        """Async chat completion with automatic retry.

        Args:
            messages: OpenAI-style message list.

        Returns:
            Assistant message content string.

        Raises:
            LLMError: On non-retryable or exhausted-retry failures.
        """
        for attempt in range(1, 4):
            try:
                if self.provider == "anthropic":
                    response = await self._async_client.messages.create(
                        model=self._model,
                        max_tokens=self._max_tokens,
                        messages=messages,
                    )
                    content = response.content[0].text
                else:  # OpenAI-compatible path
                    response = await self._async_client.chat.completions.create(
                        messages=messages,  # type: ignore[arg-type]
                        **self._build_kwargs(),
                    )
                    content = response.choices[0].message.content or ""

                log.debug(
                    "LLMClient.acomplete succeeded",
                    extra={
                        "provider": self.provider,
                        "model": self._model,
                        "attempt": attempt,
                    },
                )
                return content
            except Exception as exc:
                log.warning(
                    "LLMClient.acomplete error",
                    extra={"error": str(exc), "attempt": attempt},
                )
                if attempt == 3:
                    raise LLMError(
                        f"Async LLM completion failed after 3 attempts: {exc}"
                    ) from exc
                await asyncio.sleep(2**attempt)
        return ""  # unreachable, satisfies mypy

    # ── Convenience helpers ───────────────────────────────────────────────────

    def complete_text(self, prompt: str) -> str:
        """Shortcut: wrap *prompt* as a single user message.

        Args:
            prompt: Plain text prompt string.

        Returns:
            LLM response string.
        """
        return self.complete([{"role": "user", "content": prompt}])

    @property
    def model(self) -> str:
        """The active model identifier."""
        return self._model

    def _build_kwargs(self) -> dict[str, Any]:
        """Build shared keyword arguments for **OpenAI-compatible** completion calls.

        Note:
            This helper is **OpenAI-specific**.  It must not be used on the
            Anthropic path — ``messages.create`` on the Anthropic SDK accepts
            ``model`` and ``max_tokens`` as positional keyword arguments rather
            than a kwargs dict, and does not accept ``temperature`` in the same
            form.
        """
        return {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
