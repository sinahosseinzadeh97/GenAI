# app/routers/query.py
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks
import json

from app import models, schemas
from app.core.database import DBSession
from app.services.vectorstore import DOCS
from app.services import agents
from app.services.automation import notify_n8n

router = APIRouter()


@router.post("/query", response_model=schemas.QueryResponse)
async def query(req: schemas.QueryRequest, background: BackgroundTasks):
    # --- 1) Retrieve from docs vectorstore ---
    k_docs = req.top_k or 5
    docs_hits = DOCS.search(req.question, k=k_docs) if DOCS else []

    sources: list[schemas.Source] = []
    context_parts: list[str] = []

    for d in docs_hits:
        meta = d.metadata or {}
        doc_id = int(meta.get("document_id") or 0)
        title = (meta.get("title") or "Documento")
        chunk = d.page_content or ""
        sources.append(schemas.Source(document_id=doc_id, title=title, chunk=chunk))
        context_parts.append(f"[{title}] {chunk}")

    # --- 2) Retrieve laws (defensive: اگر تابع نبود، خالی) ---
    retrieve_laws = getattr(agents, "retrieve_laws", None)
    law_hits = retrieve_laws(req.question, k=3) if callable(retrieve_laws) else []

    laws: list[schemas.Source] = [
        schemas.Source(
            document_id=0,
            title=(h.get("title") or "Legge"),
            # برخی ایمپلیمنت‌ها به‌جای chunk از text استفاده می‌کنند
            chunk=(h.get("text") or h.get("chunk") or "")
        )
        for h in (law_hits or [])
    ]

    # --- 3) Build context (trim برای جلوگیری از توکن زیاد) ---
    law_ctx = [f"[LAW] {l.title}: {l.chunk}" for l in laws]
    context = "\n---\n".join(context_parts + law_ctx)
    context_for_llm = context[:6000]

    # --- 4) Ask the LLM ---
    system = (
        "You are a Legal RAG assistant. You MUST respond in English language ONLY. "
        "CRITICAL: Never use Italian, Spanish, French, or any other language except English. "
        "If the context contains text in other languages, translate the key information to English in your response. "
        "Use ONLY the provided context to answer questions. "
        "Answer concisely in English. Cite sources inline as (Document Title). "
        "If the answer is not in the context, say 'I don't know' in English. "
        "\n"
        "Example of correct English response: "
        "'The employee must comply with data protection regulations including GDPR and CCPA (Document Title).'"
    )

    # Force English by adding explicit instruction in the user message
    english_instruction = "IMPORTANT: You must respond in English language only. Do not use Italian or any other language."
    user = f"{english_instruction}\n\nQuestion: {req.question}\n\nContext:\n{context_for_llm}\n\nPlease provide your answer in English."

    try:
        answer = agents.llm.chat(system, user)
    except Exception as e:
        # اجازه نده کرش کند؛ پیام خطا را هم برگردان
        answer = f"I was unable to generate a response: {e}"

    # --- 5) Optional draft ---
    draft_text = None
    if req.action:
        try:
            draft_text = agents.generate_draft(req.action, context_for_llm, req.question)
        except Exception:
            draft_text = None

    # --- 6) Persist workflow & notify ---
    with DBSession() as db:
        wf = models.Workflow(
            type="query",
            status="completed",
            payload=json.dumps(req.model_dump()),
            result=json.dumps({"answer": answer, "draft": draft_text}),
        )
        db.add(wf)
        db.commit()
        db.refresh(wf)
        workflow_id = wf.id

    notify_n8n(
        "query_answered",
        {"workflow_id": workflow_id, "question": req.question, "answer": answer, "draft": draft_text},
    )

    return schemas.QueryResponse(
        answer=answer,
        sources=sources,
        laws=laws,
        workflow_id=workflow_id,
    )
