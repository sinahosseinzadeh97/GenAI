from __future__ import annotations
from typing import List, Dict, Any, Optional
import os

class LLMClient:
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        # Normalize base URL for Ollama
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.temperature = float(temperature)

        if self.provider not in {"openai", "ollama"}:
            raise ValueError("provider must be 'openai' or 'ollama'")

    def chat(self, messages: List[Dict[str, str]]) -> str:
        if self.provider == "ollama":
            return self._chat_ollama(messages)
        else:
            return self._chat_openai(messages)

    # ---------- OpenAI ----------
    def _chat_openai(self, messages: List[Dict[str, str]]) -> str:
        try:
            from openai import OpenAI  # requires openai>=1.0
        except Exception as e:
            raise RuntimeError("OpenAI Python package not installed. Please `pip install openai`.") from e

        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""

    # ---------- Ollama ----------
    def _chat_ollama(self, messages: List[Dict[str, str]]) -> str:
        """
        Try Python package `ollama` first; if unavailable or lacks `.chat`,
        fall back to plain REST at {base_url}/api/chat.
        """
        # -- Try the Python package --
        try:
            import ollama  # DO NOT: from ollama import chat
            if not hasattr(ollama, "chat"):
                raise ImportError("Installed `ollama` lacks .chat; using REST fallback.")

            prev = os.environ.get("OLLAMA_HOST")
            try:
                if self.base_url:
                    os.environ["OLLAMA_HOST"] = self.base_url
                resp = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={"temperature": self.temperature},
                )
                # Normalize response shape
                if hasattr(resp, "message"):
                    return getattr(resp.message, "content", "") or ""
                if isinstance(resp, dict):
                    msg = resp.get("message") or {}
                    return msg.get("content", "") or resp.get("content", "") or ""
                return ""
            finally:
                if prev is not None:
                    os.environ["OLLAMA_HOST"] = prev
                else:
                    os.environ.pop("OLLAMA_HOST", None)
        except Exception:
            # -- REST fallback --
            try:
                import requests
            except Exception as e:
                raise RuntimeError("Requests package not installed for REST fallback. `pip install requests`.") from e

            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.temperature},
            }
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return (data.get("message", {}) or {}).get("content", "") or data.get("content", "") or ""
