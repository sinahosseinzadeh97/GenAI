from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import init_db
from app.core.config import settings
from app.services.vectorstore import LAWS
from app.routers import documents, query, workflows
import json, os

app = FastAPI(title="LegalTech AI Assistant — MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()
    # Load laws dataset once (idempotent)
    data_path = "/app/data/laws_it.json"
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        texts = [it["text"] for it in items]
        metas = [{"title": it.get("title", f"Legge {i}")} for i, it in enumerate(items)]
        if texts:
            LAWS.add_texts(texts, metas)

app.include_router(documents.router)
app.include_router(query.router)
app.include_router(workflows.router)

@app.get("/")
async def root():
    return {"ok": True, "service": app.title}