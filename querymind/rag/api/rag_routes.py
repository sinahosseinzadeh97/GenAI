from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from typing import List

from querymind.rag.ingestion.pdf_parser import parse_pdf
from querymind.rag.ingestion.embedder import get_embedding
from querymind.rag.store.vector_store import save_chunks
from querymind.rag.retrieval.search import search_chunks
from querymind.rag.generation.llm_client import generate_answer
import time
import asyncio
from querymind.logging_config import StructuredLogger

logger = StructuredLogger("querymind.rag")
from querymind.api.middleware.rate_limit import limiter, RAG_LIMIT, INGEST_LIMIT

router = APIRouter(prefix="/rag", tags=["RAG"])

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class ExtractRequest(BaseModel):
    query: str
    field: str

class CompareRequest(BaseModel):
    query: str
    filenames: List[str]

@router.post("/ingest")
@limiter.limit(INGEST_LIMIT)
async def ingest_document(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        return {"error": "Filename missing."}
    
    start_time = time.monotonic()
    content = await file.read()
    
    # 1. Parse PDF
    try:
        parsed_chunks = await asyncio.wait_for(
            asyncio.to_thread(parse_pdf, content, file.filename),
            timeout=60.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="PDF processing timed out. Try a smaller file."
        )
    
    # 2. Embed
    for chunk in parsed_chunks:
        chunk["embedding"] = get_embedding(chunk["text"])
        
    # 3. Store
    await save_chunks(parsed_chunks)
    
    latency_ms = (time.monotonic() - start_time) * 1000
    unique_pages = len(set(c["page_number"] for c in parsed_chunks)) if parsed_chunks else 0
    logger.log_pdf_ingest(filename=file.filename, pages=unique_pages, latency_ms=latency_ms)
    
    return {"message": f"Successfully ingested {len(parsed_chunks)} chunks from {file.filename}"}

@router.post("/search")
@limiter.limit(RAG_LIMIT)
async def search_documents(request: Request, req: SearchRequest):
    start_time = time.time()
    results = await search_chunks(req.query, req.top_k)
    latency_ms = (time.time() - start_time) * 1000
    logger.log_rag_search(query=req.query, results_count=len(results), latency_ms=latency_ms)
    return {"results": results}

@router.post("/extract")
async def extract_field(req: ExtractRequest):
    results = await search_chunks(req.query + f" {req.field}", top_k=5)
    answer = generate_answer(f"Extract the {req.field} based on the query: {req.query}", results)
    
    return {"extracted": answer, "sources": [{"filename": r["filename"], "page": r["page_number"]} for r in results]}

@router.post("/compare")
async def compare_documents(req: CompareRequest):
    results = await search_chunks(req.query, top_k=10)
    
    # filter by filenames if specified
    if req.filenames:
        results = [r for r in results if r["filename"] in req.filenames]
        
    answer = generate_answer(req.query, results)
    
    return {"comparison": answer, "sources": [{"filename": r["filename"], "page": r["page_number"]} for r in results]}
