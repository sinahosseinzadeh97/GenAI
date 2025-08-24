# app/services/embedding.py
from __future__ import annotations
from typing import List
from app.core.config import settings

# مهم: ارث‌بری از Embeddings لانگ‌چین
from langchain_core.embeddings import Embeddings as LCEmbeddings


class AppEmbeddings(LCEmbeddings):
    """
    لایه سازگار با LangChain که بسته به تنظیمات، از OpenAI یا Sentence-Transformers محلی استفاده می‌کند.
    برای سازگاری با FAISSهایی که embedding_function را call می‌کنند، __call__ هم پیاده‌سازی شده.
    """
    def __init__(self) -> None:
        self._use_openai = bool(
            getattr(settings, "USE_OPENAI", False) and getattr(settings, "OPENAI_API_KEY", None)
        )
        if self._use_openai:
            from openai import OpenAI  # بارگذاری تنبل
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self._model_name = getattr(settings, "OPENAI_EMBED_MODEL", "text-embedding-3-small")
            self._provider = "openai"
            self._st = None
        else:
            from sentence_transformers import SentenceTransformer  # بارگذاری تنبل
            self._st = SentenceTransformer("all-MiniLM-L6-v2")
            self._provider = "local"
            self._client = None
            self._model_name = None

    # LangChain API
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        clean = [(t or "").strip() for t in texts]
        if self._provider == "openai":
            resp = self._client.embeddings.create(model=self._model_name, input=clean)
            return [d.embedding for d in resp.data]
        return self._st.encode(clean, normalize_embeddings=True).tolist()

    # LangChain API
    def embed_query(self, text: str) -> List[float]:
        t = (text or "").strip()
        if self._provider == "openai":
            resp = self._client.embeddings.create(model=self._model_name, input=[t])
            return resp.data[0].embedding
        return self._st.encode([t], normalize_embeddings=True).tolist()[0]

    # برای نسخه‌هایی که embedding_function را به‌صورت تابع صدا می‌زنند
    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)


# singleton
embeddings = AppEmbeddings()
