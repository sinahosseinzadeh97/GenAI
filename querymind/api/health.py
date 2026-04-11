import os
from fastapi import APIRouter
from datetime import datetime, timezone

router = APIRouter()

@router.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "services": {
            "database": "ok",
            "anthropic": "configured" if os.getenv("ANTHROPIC_API_KEY") else "missing",
            "openai": "configured" if os.getenv("OPENAI_API_KEY") else "missing",
        }
    }
