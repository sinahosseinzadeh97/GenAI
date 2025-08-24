# app/services/vectorstore.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
from app.core.config import settings
from .embedding import embeddings

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document as LCDocument

DOCS_STORE_DIR = Path(settings.VECTORSTORE_DIR) / "docs"
LAWS_STORE_DIR = Path(settings.VECTORSTORE_DIR) / "laws"
DOCS_STORE_DIR.mkdir(parents=True, exist_ok=True)
LAWS_STORE_DIR.mkdir(parents=True, exist_ok=True)


class VectorStore:
    def __init__(self, path: Path):
        self.path = path
        self.vs: FAISS | None = None
        self._load()

    def _load(self):
        try:
            # در بعضی نسخه‌ها اسم پارامتر 'embeddings' است (plural)
            self.vs = FAISS.load_local(
                str(self.path),
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            self.vs = None

    def _save(self):
        if self.vs:
            self.vs.save_local(str(self.path))

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        docs = [LCDocument(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        if self.vs is None:
            # در from_documents اسم پارامتر در این نسخه 'embedding' است (singular)
            self.vs = FAISS.from_documents(docs, embedding=embeddings)
        else:
            try:
                self.vs.add_documents(docs)
            except Exception:
                # اگر ابعاد امبدینگِ قبلی با مدل فعلی نمی‌خوانَد، از نو بساز
                self.vs = FAISS.from_documents(docs, embedding=embeddings)
        self._save()

    def search(self, query: str, k: int = 5) -> List[LCDocument]:
        if self.vs is None:
            return []
        return self.vs.similarity_search(query, k=k)


# singleton stores
DOCS = VectorStore(DOCS_STORE_DIR)
LAWS = VectorStore(LAWS_STORE_DIR)
