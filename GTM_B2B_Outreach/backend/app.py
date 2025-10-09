import uuid
import asyncio
import logging
import traceback
from typing import Dict
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from models import RunPayload, PipelineResult
from service import orchestrate
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [os.getenv("CORS_ORIGINS", "http://localhost:5173")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

progress_store: Dict[str, int] = {}
result_store: Dict[str, PipelineResult] = {}
error_store: Dict[str, str] = {}

@app.post("/api/run")
async def run_pipeline(p: RunPayload, bg: BackgroundTasks):
    task_id = str(uuid.uuid4())
    progress_store[task_id] = 0
    
    logger.info(f"Starting pipeline for task {task_id}")

    def _update(x: int):
        progress_store[task_id] = x

    def _job():
        try:
            logger.info(f"Task {task_id}: Starting orchestration")
            res = orchestrate(p, on_progress=_update)
            result_store[task_id] = res
            logger.info(f"Task {task_id}: Completed successfully")
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Task {task_id}: {error_msg}")
            error_store[task_id] = str(e)
            # Store empty result on error
            result_store[task_id] = PipelineResult()
        finally:
            progress_store[task_id] = 100

    bg.add_task(_job)
    return {"task_id": task_id}

@app.get("/api/progress/{task_id}")
async def progress_stream(task_id: str):
    async def gen():
        last = -1
        while True:
            pct = progress_store.get(task_id, 100)
            if pct != last:
                last = pct
                yield {"event": "progress", "data": str(pct)}
            if pct >= 100:
                break
            await asyncio.sleep(0.4)
    return EventSourceResponse(gen())

@app.get("/api/result/{task_id}")
def get_result(task_id: str):
    result = result_store.get(task_id, PipelineResult()).model_dump()
    if task_id in error_store:
        result["error"] = error_store[task_id]
    return result

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "exa_key_set": bool(os.getenv("EXA_API_KEY"))
    }

# run: uvicorn app:app --reload --port 8000
