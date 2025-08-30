# app.py (stable, no-errors)
# FastAPI + SQLModel (SQLite), bilingual FA/EN replies, SSE streaming.
# Compatible with both new `openai` client (v1.x) and legacy module (v0.x) via Chat Completions API.
# Relationships removed for SQLAlchemy 2.x on Python 3.10.
# Includes validators to avoid 422 on /tasks.
#
# Quick start:
#   pip install fastapi uvicorn pydantic openai python-dotenv sqlmodel sqlalchemy aiosqlite
#   uvicorn --env-file .env app:app --reload
#
# .env example:
#   OPENAI_API_KEY=sk-...
#   OPENAI_MODEL=gpt-4o-mini
#   ACADEMIC_TZ=Europe/Rome
#   DATABASE_URL=sqlite:///./academic.db
#   CORS_ORIGINS=http://127.0.0.1:5500,http://localhost:5500

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Query, Header, Depends
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any, Generator
from datetime import datetime, timezone
from uuid import uuid4
import os
import json
import re

from sqlmodel import SQLModel, Field, create_engine, Session, select

# --- OpenAI client compatibility layer ---
NEW_OPENAI_CLIENT = False
try:
    # New SDK (v1.x)
    from openai import OpenAI
    _client_for_check = OpenAI  # type: ignore
    NEW_OPENAI_CLIENT = True
except Exception:
    NEW_OPENAI_CLIENT = False
    try:
        import openai as openai_legacy  # type: ignore
    except Exception:
        openai_legacy = None  # type: ignore

# ----------------------------------
# FastAPI app
# ----------------------------------
app = FastAPI(title="Academic Assistant API (DB)", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500,http://localhost:5173,http://localhost:3000"
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True, "message": "Academic Assistant is running", "docs": "/docs", "health": "/health"}

@app.get("/favicon.ico")
def favicon():
    return Response(content=b"", media_type="image/x-icon")

# ----------------------------------
# Config
# ----------------------------------
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TZ_HINT = os.getenv("ACADEMIC_TZ", "Europe/Rome")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./academic.db")

# ----------------------------------
# DB Models (no relationships)
# ----------------------------------
class SessionRow(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    preferred_lang: str = Field(default="auto", description="fa | en | bi | auto")

class ProfileRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="sessionrow.id")
    personal_info_json: Optional[str] = None
    learning_prefs_json: Optional[str] = None
    courses_json: Optional[str] = None

class EventRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="sessionrow.id")
    summary: str
    start: datetime
    end: datetime
    location: Optional[str] = None

class TaskRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="sessionrow.id")
    title: str
    course: Optional[str] = None
    status: str
    due: Optional[datetime] = None

class MessageRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="sessionrow.id")
    role: str
    content: str
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# ----------------------------------
# Pydantic I/O Models (+ validators)
# ----------------------------------
from pydantic import BaseModel, field_validator

ALLOWED_TASK_STATUS = {"needsAction", "completed"}

class LearningStyle(BaseModel):
    visual: Optional[bool] = None
    auditory: Optional[bool] = None
    kinesthetic: Optional[bool] = None

class LearningPreferences(BaseModel):
    learning_style: Optional[LearningStyle] = None

class Course(BaseModel):
    name: str
    difficulty: Optional[str] = None

class PersonalInfo(BaseModel):
    name: Optional[str] = None
    major: Optional[str] = None
    academic_year: Optional[int] = None

class Profile(BaseModel):
    personal_info: Optional[PersonalInfo] = None
    learning_preferences: Optional[LearningPreferences] = None
    courses: Optional[List[Course]] = None

class EventIn(BaseModel):
    summary: str
    start: datetime
    end: datetime
    location: Optional[str] = None

class TaskIn(BaseModel):
    title: str
    course: Optional[str] = None
    status: str
    due: Optional[datetime] = None

    @field_validator("status")
    @classmethod
    def _check_status(cls, v: str) -> str:
        if v not in ALLOWED_TASK_STATUS:
            raise ValueError(f"status must be one of {sorted(ALLOWED_TASK_STATUS)}")
        return v

    @field_validator("due", mode="before")
    @classmethod
    def _parse_due(cls, v):
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        s = str(v).strip()
        if s == "" or s.lower() == "null":
            return None
        # Allow "YYYY-MM-DD HH:MM:SS" by normalizing to ISO
        s = s.replace(" ", "T")
        try:
            return datetime.fromisoformat(s)
        except Exception as e:
            raise ValueError("Invalid ISO datetime for 'due' (e.g., 2025-09-03T23:59:00)") from e

class ChatRequest(BaseModel):
    message: str

class SettingsIn(BaseModel):
    preferred_lang: Optional[str] = None  # fa | en | bi | auto

# ----------------------------------
# DB engine
# ----------------------------------
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SQLModel.metadata.create_all(engine)

# ----------------------------------
# Lang + LLM helpers
# ----------------------------------
LANG_FA = "fa"
LANG_EN = "en"
LANG_BI = "bi"
LANG_AUTO = "auto"
FA_CHARS_RE = re.compile(r"[\u0600-\u06FF]")

def detect_lang(text: str) -> str:
    if not text:
        return LANG_AUTO
    return LANG_FA if FA_CHARS_RE.search(text) else LANG_EN


def decide_lang(session_pref: str, user_text: str, accept_language: Optional[str], url_lang: Optional[str]) -> str:
    if url_lang in {LANG_FA, LANG_EN, LANG_BI, LANG_AUTO}:
        pref = url_lang
    elif session_pref in {LANG_FA, LANG_EN, LANG_BI, LANG_AUTO}:
        pref = session_pref
    else:
        pref = LANG_AUTO
    if pref == LANG_AUTO:
        if accept_language:
            al = accept_language.lower()
            if "fa" in al: return LANG_FA
            if "en" in al: return LANG_EN
        return detect_lang(user_text)
    return pref


def build_system_prompt(lang_mode: str) -> str:
    base_rules = f"""
You are a multi-agent academic assistant.
Personas:
- PlannerAgent: builds practical, time-boxed study schedules with priorities and constraints.
- NoteWriterAgent: produces structured, concise study notes.
- AdvisorAgent: gives personalized study strategy, energy/focus management, and a short emergency protocol.
Use timezone hint: {TZ_HINT}.
Structure the reply in three top-level sections:
1) Plan
2) Summary Notes
3) Advice
"""
    fa_rules = """
قوانین پاسخ‌دهی (فارسی):
- پاسخ‌ها را به فارسی بنویس.
- سه بخش واضح بده: «برنامه»، «جزوهٔ خلاصه»، «مشاوره».
- ساعت‌ها و تاریخ‌ها را با درنظرگرفتن منطقهٔ زمانی اشاره‌شده پیشنهاد بده.
"""
    en_rules = """
Response Rules (English):
- Write in English.
- Return three clear sections: "Plan", "Summary Notes", and "Advice".
- Propose times/dates with the configured timezone.
"""
    bi_rules = """
Bilingual mode:
- Provide **both Persian and English** for each of the three sections.
- Put the Persian version first, then the English translation.
- Keep both versions concise and aligned.
"""
    return base_rules + "\n" + ({LANG_FA: fa_rules, LANG_EN: en_rules, LANG_BI: bi_rules}.get(lang_mode, en_rules))


def format_context(db: Session, sid: str) -> str:
    prof = db.exec(select(ProfileRow).where(ProfileRow.session_id == sid)).first()
    parts = []
    if prof:
        try:
            pi = json.loads(prof.personal_info_json) if prof.personal_info_json else None
            lp = json.loads(prof.learning_prefs_json) if prof.learning_prefs_json else None
            cs = json.loads(prof.courses_json) if prof.courses_json else None
        except Exception:
            pi, lp, cs = None, None, None
        if pi:
            parts.append(f"Student: {pi.get('name','N/A')} | Major: {pi.get('major','N/A')} | Year: {pi.get('academic_year','N/A')}")
        if cs:
            parts.append("Courses: " + ", ".join([f"{c.get('name')}({c.get('difficulty','?')})" for c in cs]))
        if lp and lp.get('learning_style'):
            ls = lp['learning_style']
            styles = []
            if ls.get('visual'): styles.append('visual')
            if ls.get('auditory'): styles.append('auditory')
            if ls.get('kinesthetic'): styles.append('kinesthetic')
            if styles:
                parts.append("Learning style: " + ", ".join(styles))
    now = datetime.now(timezone.utc)
    events = db.exec(select(EventRow).where(EventRow.session_id == sid).order_by(EventRow.start)).all()
    upcoming = [e for e in events if (e.end if e.end.tzinfo else e.end.replace(tzinfo=timezone.utc)) >= now]
    ev_lines = [f"- {e.summary} | {e.start.isoformat()} → {e.end.isoformat()} | {e.location or '-'}" for e in upcoming[:20]]
    if ev_lines:
        parts.append("Upcoming events (max 20):\n" + "\n".join(ev_lines))
    tasks = db.exec(select(TaskRow).where(TaskRow.session_id == sid)).all()
    needs = [t for t in tasks if (t.status or '').lower() == 'needsaction']
    needs.sort(key=lambda t: (t.due or datetime.max.replace(tzinfo=timezone.utc)))
    t_lines = [f"- {t.title} | course: {t.course or '-'} | status: {t.status} | due: {(t.due.isoformat() if t.due else '-')}" for t in needs[:50]]
    if t_lines:
        parts.append("Active tasks:\n" + "\n".join(t_lines))
    return "\n".join(parts) if parts else "(no registered context)"


def build_messages(system_prompt: str, context_text: str, user_utterance: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Current student context:\n{context_text}"},
        {"role": "user", "content": user_utterance},
    ]

# ----------------------------------
# API endpoints
# ----------------------------------

def get_db():
    with Session(engine) as db:
        yield db

@app.get("/health")
def health():
    return {"status": "ok", "model": DEFAULT_MODEL, "tz": TZ_HINT}

@app.post("/sessions")
def create_session(db: Session = Depends(get_db)):
    row = SessionRow()
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"session_id": row.id, "preferred_lang": row.preferred_lang}

@app.get("/sessions/{sid}")
def get_session_state(sid: str, db: Session = Depends(get_db)):
    session_row = db.get(SessionRow, sid)
    if not session_row:
        raise HTTPException(404, "session not found")
    prof = db.exec(select(ProfileRow).where(ProfileRow.session_id == sid)).first()
    events = db.exec(select(EventRow).where(EventRow.session_id == sid).order_by(EventRow.start)).all()
    tasks = db.exec(select(TaskRow).where(TaskRow.session_id == sid)).all()
    msgs = db.exec(select(MessageRow).where(MessageRow.session_id == sid)).order_by(MessageRow.ts).all()
    return {
        "session": {"id": session_row.id, "preferred_lang": session_row.preferred_lang, "created_at": session_row.created_at},
        "profile": {
            "personal_info": json.loads(prof.personal_info_json) if prof and prof.personal_info_json else None,
            "learning_preferences": json.loads(prof.learning_prefs_json) if prof and prof.learning_prefs_json else None,
            "courses": json.loads(prof.courses_json) if prof and prof.courses_json else None,
        },
        "events": [
            {"id": e.id, "summary": e.summary, "start": e.start, "end": e.end, "location": e.location}
            for e in events
        ],
        "tasks": [
            {"id": t.id, "title": t.title, "course": t.course, "status": t.status, "due": t.due}
            for t in tasks
        ],
        "messages": [
            {"id": m.id, "role": m.role, "content": m.content, "ts": m.ts}
            for m in msgs
        ],
    }

@app.patch("/sessions/{sid}/settings")
def set_settings(payload: SettingsIn, sid: str, db: Session = Depends(get_db)):
    row = db.get(SessionRow, sid)
    if not row:
        raise HTTPException(404, "session not found")
    if payload.preferred_lang:
        if payload.preferred_lang not in {LANG_FA, LANG_EN, LANG_BI, LANG_AUTO}:
            raise HTTPException(400, "invalid preferred_lang")
        row.preferred_lang = payload.preferred_lang
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"session_id": row.id, "preferred_lang": row.preferred_lang}

@app.patch("/sessions/{sid}/profile")
def set_profile(profile: Profile, sid: str, db: Session = Depends(get_db)):
    row = db.exec(select(ProfileRow).where(ProfileRow.session_id == sid)).first()
    if not db.get(SessionRow, sid):
        raise HTTPException(404, "session not found")
    data = ProfileRow(
        session_id=sid,
        personal_info_json=json.dumps(profile.personal_info.dict() if profile.personal_info else None, ensure_ascii=False),
        learning_prefs_json=json.dumps(profile.learning_preferences.dict() if profile.learning_preferences else None, ensure_ascii=False),
        courses_json=json.dumps([c.dict() for c in (profile.courses or [])], ensure_ascii=False),
    )
    if row:
        data.id = row.id
    db.add(data)
    db.commit()
    db.refresh(data)
    return {"ok": True}

@app.post("/sessions/{sid}/events")
def add_event(event: EventIn, sid: str, db: Session = Depends(get_db)):
    if not db.get(SessionRow, sid):
        raise HTTPException(404, "session not found")
    e = EventRow(session_id=sid, summary=event.summary, start=event.start, end=event.end, location=event.location)
    db.add(e)
    db.commit()
    db.refresh(e)
    return {"id": e.id}

@app.get("/sessions/{sid}/events")
def list_events(sid: str, db: Session = Depends(get_db)):
    if not db.get(SessionRow, sid):
        raise HTTPException(404, "session not found")
    events = db.exec(select(EventRow).where(EventRow.session_id == sid).order_by(EventRow.start)).all()
    return events

@app.post("/sessions/{sid}/tasks")
def add_task(task: TaskIn, sid: str, db: Session = Depends(get_db)):
    if not db.get(SessionRow, sid):
        raise HTTPException(404, "session not found")
    t = TaskRow(session_id=sid, title=task.title, course=task.course, status=task.status, due=task.due)
    db.add(t)
    db.commit()
    db.refresh(t)
    return {"id": t.id}

@app.get("/sessions/{sid}/tasks")
def list_tasks(sid: str, db: Session = Depends(get_db)):
    if not db.get(SessionRow, sid):
        raise HTTPException(404, "session not found")
    tasks = db.exec(select(TaskRow).where(TaskRow.session_id == sid)).all()
    return tasks

# -------- Chat endpoint (SSE + non-stream) with SDK compatibility --------
@app.post("/sessions/{sid}/chat")
async def chat(
    req: ChatRequest,
    sid: str,
    stream: bool = Query(False, description="1 to stream via SSE"),
    lang: Optional[str] = Query(None, description="fa | en | bi | auto"),
    accept_language: Optional[str] = Header(None, alias="Accept-Language"),
    db: Session = Depends(get_db)
):
    # Ensure some OpenAI path exists
    if not (NEW_OPENAI_CLIENT or openai_legacy):  # type: ignore
        raise HTTPException(500, "openai package not installed")

    sess = db.get(SessionRow, sid)
    if not sess:
        raise HTTPException(404, "session not found")

    # persist user message
    m_user = MessageRow(session_id=sid, role="user", content=req.message)
    db.add(m_user)
    db.commit()

    # prompts
    lang_mode = decide_lang(sess.preferred_lang, req.message, accept_language, lang)
    sys_prompt = build_system_prompt(lang_mode)
    ctx = format_context(db, sid)
    messages = build_messages(sys_prompt, ctx, req.message)

    # Helper creators for new/legacy
    def create_completion_stream():
        if NEW_OPENAI_CLIENT:
            client = OpenAI()
            return client.chat.completions.create(model=DEFAULT_MODEL, messages=messages, stream=True)
        else:  # legacy
            openai_legacy.api_key = os.getenv("OPENAI_API_KEY")  # type: ignore
            return openai_legacy.ChatCompletion.create(model=DEFAULT_MODEL, messages=messages, stream=True)  # type: ignore

    def create_completion_once():
        if NEW_OPENAI_CLIENT:
            client = OpenAI()
            return client.chat.completions.create(model=DEFAULT_MODEL, messages=messages, stream=False)
        else:
            openai_legacy.api_key = os.getenv("OPENAI_API_KEY")  # type: ignore
            return openai_legacy.ChatCompletion.create(model=DEFAULT_MODEL, messages=messages, stream=False)  # type: ignore

    if stream:
        def sse_gen() -> Generator[str, None, None]:
            full: List[str] = []
            try:
                stream_obj = create_completion_stream()
                for chunk in stream_obj:  # works for both SDKs
                    piece: Optional[str] = None
                    try:
                        # new SDK object style
                        piece = chunk.choices[0].delta.content  # type: ignore
                    except Exception:
                        try:
                            # legacy dict style
                            piece = chunk["choices"][0]["delta"].get("content")  # type: ignore
                        except Exception:
                            piece = None
                    if piece:
                        full.append(piece)
                        yield f"data: {json.dumps({'chunk': piece}, ensure_ascii=False)}\n\n"
                final_text = "".join(full)
                if final_text:
                    m_bot = MessageRow(session_id=sid, role="assistant", content=final_text)
                    db.add(m_bot)
                    db.commit()
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return StreamingResponse(sse_gen(), media_type="text/event-stream")

    # Non-streaming
    try:
        resp = create_completion_once()
        text: Optional[str] = None
        try:
            text = resp.choices[0].message.content  # type: ignore
        except Exception:
            try:
                text = resp["choices"][0]["message"]["content"]  # type: ignore
            except Exception:
                text = json.dumps(getattr(resp, "to_dict", lambda: resp)(), ensure_ascii=False) if hasattr(resp, "to_dict") else json.dumps(resp, ensure_ascii=False)
        m_bot = MessageRow(session_id=sid, role="assistant", content=text or "")
        db.add(m_bot)
        db.commit()
        return JSONResponse({"reply": text, "lang": lang_mode})
    except Exception as e:
        raise HTTPException(500, str(e))

# EOF
