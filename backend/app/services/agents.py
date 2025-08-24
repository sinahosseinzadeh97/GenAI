# app/services/agents.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from app.core.config import settings
from . import vectorstore
import json

# ----- LLM wrapper -----
class ChatLLM:
    def __init__(self) -> None:
        self._use_openai = bool(getattr(settings, "USE_OPENAI", False) and getattr(settings, "OPENAI_API_KEY", None))
        # می‌توانی در config گزینه OPENAI_MODEL بگذاری؛ در غیر این صورت این پیش‌فرض استفاده می‌شود
        self._model = getattr(settings, "OPENAI_MODEL", None) or "gpt-4o-mini"

        if self._use_openai:
            from openai import OpenAI
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            self._client = None  # حالت محلی/بدون LLM

    def chat(self, system: str, user: str) -> str:
        """برگشت همیشه str"""
        if self._client:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.2,
            )
            # نکته مهم: در SDK جدید باید از .message.content استفاده شود
            return resp.choices[0].message.content or ""
        # fallback محلی
        return (
            "[LOCAL MODE]\n"
            "LLM خاموش است. برای پاسخ دقیق‌تر OPENAI را فعال کنید.\n\n"
            f"System: {system[:200]}\nUser: {user[:500]}"
        )

llm = ChatLLM()

# ----- RAG helpers -----
def retrieve_laws(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """بازیابی از وکتوراستور قوانین"""
    hits = vectorstore.LAWS.search(query, k=k)
    out: List[Dict[str, Any]] = []
    for d in hits:
        md = d.metadata or {}
        out.append({
            "title": md.get("title") or "Legge",
            "text": d.page_content,
            "meta": md,
        })
    return out

def retrieve_docs(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """بازیابی از وکتوراستور اسناد کاربر"""
    hits = vectorstore.DOCS.search(query, k=k)
    out: List[Dict[str, Any]] = []
    for d in hits:
        md = d.metadata or {}
        out.append({
            "title": md.get("title") or md.get("filename") or "Documento",
            "text": d.page_content,
            "meta": md,
        })
    return out

def answer_with_rag(question: str,
                    law_hits: Optional[List[Dict[str, Any]]] = None,
                    doc_hits: Optional[List[Dict[str, Any]]] = None) -> str:
    """ترکیب کانتکست قوانین + اسناد و جواب به سؤال"""
    law_hits = law_hits or []
    doc_hits = doc_hits or []

    ctx_parts: List[str] = []
    if law_hits:
        ctx_parts.append("### ESTRATTI NORMATIVI:\n" + "\n\n".join(h["text"][:800] for h in law_hits))
    if doc_hits:
        ctx_parts.append("### ESTRATTI DOCUMENTO:\n" + "\n\n".join(h["text"][:800] for h in doc_hits))
    context = "\n\n".join(ctx_parts) or "(nessun contesto trovato)"

    system = "Sei un assistente legale. Usa SOLO il contesto fornito. Se l'informazione manca, dillo chiaramente."
    user = f"Domanda: {question}\n\nContesto:\n{context}\n\nRispondi in italiano, breve e puntuale."
    return llm.chat(system, user)

def analyze_document(text: str, title: Optional[str] = None) -> Dict[str, Any]:
    """تحلیل سریع سند: خروجی JSON با summary و tags"""
    system = (
        "Sei un assistente legale. Fornisci un breve sommario (max 5 punti) e 3-6 tag chiave in italiano. "
        "Rispondi in JSON con chiavi: summary (stringa), tags (lista di stringhe)."
    )
    user = f"TITOLO: {title or 'Documento'}\nTESTO:\n{text[:6000]}"
    raw = llm.chat(system, user)
    try:
        data = json.loads(raw)
        return {
            "summary": data.get("summary", ""),
            "tags": data.get("tags", []) or []
        }
    except Exception:
        # اگر LLM خاموش باشد یا JSON ندهد
        return {"summary": raw[:1000], "tags": []}
