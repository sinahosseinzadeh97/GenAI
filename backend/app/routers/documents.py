from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from pathlib import Path
from sqlmodel import select
from app.core.config import settings
from app.core.database import DBSession
from app import models, schemas
from app.services import parser, vectorstore, agents
from app.services.automation import notify_n8n
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

router = APIRouter()

FILES_DIR = Path(settings.FILES_DIR)
FILES_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/documents", response_model=schemas.UploadResponse)
async def upload_document(background: BackgroundTasks, file: UploadFile = File(...)):
    # ذخیره‌ی فایل روی دیسک
    data = await file.read()
    dest = FILES_DIR / file.filename
    dest.write_bytes(data)

    # استخراج متن
    text, title = parser.parse_bytes_to_text(file.filename, file.content_type, dest)
    if not (text or "").strip():
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="No text extracted from document")

    # درج رکوردها با یک commit
    with DBSession() as db:
        doc = models.Document(
            filename=file.filename,
            content_type=file.content_type,
            path=str(dest),
            text=text,
            title=title,
        )
        db.add(doc)
        db.flush()                 # <- id می‌گیرد، بدون commit
        doc_id = doc.id

        wf = models.Workflow(
            type="document_analysis",
            status="pending",
            payload=json.dumps({"document_id": doc_id}),
        )
        db.add(wf)
        db.flush()
        wf_id = wf.id

        db.commit()

    # فقط IDها به تسک بک‌گراند
    background.add_task(process_document, doc_id, wf_id)

    return schemas.UploadResponse(document_id=doc_id, workflow_id=wf_id)


def process_document(document_id: int, workflow_id: int):
    # سشن مستقل برای پردازش
    with DBSession() as db:
        doc = db.get(models.Document, document_id)
        wf = db.get(models.Workflow, workflow_id)
        if not doc or not wf:
            return

        wf.status = "running"
        db.add(wf)
        db.commit()

        try:
            # تکه‌تکه کردن متن و ایندکس
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_text(doc.text or "")

            metadatas = [
                {"document_id": doc.id, "title": doc.title or doc.filename, "chunk_id": i}
                for i, _ in enumerate(chunks)
            ]
            vectorstore.DOCS.add_texts(chunks, metadatas)

            # تحلیل سند
            analysis = agents.analyze_document(doc.text or "", doc.title)
            if analysis.get("tags"):
                doc.tags = ",".join(analysis["tags"])

            wf.result = json.dumps({
                "summary": analysis.get("summary"),
                "tags": doc.tags,
            })
            wf.status = "completed"

            db.add(doc)
            db.add(wf)
            db.commit()

            
            notify_n8n("document_analyzed", {
                "document_id": doc.id,
                "summary": analysis.get("summary"),
                "tags": doc.tags,
            })

        except Exception as e:
            # ثبت وضعیت خطا
            wf.status = "error"
            wf.result = json.dumps({"error": str(e)[:1000]})
            db.add(wf)
            db.commit()
            raise
