from __future__ import annotations
import json
import requests
from typing import Any, Dict
from app.core.config import settings

def notify_n8n(event: str, payload: Dict[str, Any]) -> None:
    if not settings.N8N_WEBHOOK_URL:
        return
    try:
        requests.post(settings.N8N_WEBHOOK_URL, json={"event": event, "payload": payload}, timeout=5)
    except Exception:
        # نذار خطای شبکه باعث fail بشه
        pass
