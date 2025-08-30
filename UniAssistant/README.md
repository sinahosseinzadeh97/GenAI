# UniAssistant
# Uni Assistant — FastAPI + OpenAI + SQLModel (FA/EN, SSE)

A minimal **academic assistant** backend with a real database (SQLModel/SQLite), bilingual replies (**FA/EN** and optional **bi** mode), and **SSE streaming**. Ships with a tiny static **HTML UI** so you can use it immediately—no JSON files and no heavy frontend required.

> This README covers the **single‑user** setup. An optional multi‑user/token variant is provided as `app_multiuser.py`.

---

## ✨ Features

* **Real DB**: `SQLModel` + SQLite (file: `academic.db`).
* **Session state**: persists profile, tasks, events, and chat history.
* **Bilingual output**: `fa` (Persian), `en` (English), or `bi` (both in one response); `auto` detection supported.
* **Streaming**: Server‑Sent Events (SSE) for live tokens; non‑streaming JSON also supported.
* **Zero‑build UI**: a single `index.html` to create sessions, set a profile, add tasks/events, and chat.
* **CORS‑friendly**: configure allowed origins via `.env`.

---

## 🧱 Tech Stack

* **FastAPI** + **Uvicorn**
* **SQLModel** / **SQLAlchemy** / **SQLite**
* **Pydantic v2** validators (resilient input parsing)
* **OpenAI** SDK (uses **Chat Completions** path for compatibility with both new and legacy clients)

---

## 📦 Project Structure

```
.
├── app.py                  # Main API (single‑user), FA/EN, SSE, DB validators
├── app_multiuser.py        # (Optional) multi‑user + token variant
├── dev.sh                  # Run backend + static frontend together
├── index.html              # Minimal UI (static) to use the API
├── academic.db             # SQLite DB (created at first run)
└── README.md               # This file
```

---

## 🔧 Requirements

* Python **3.10+** (tested on macOS 3.10.8)
* An **OpenAI API key** (`OPENAI_API_KEY`)

Install dependencies:

```bash
pip install fastapi uvicorn pydantic openai python-dotenv sqlmodel sqlalchemy aiosqlite
```

---

## ⚙️ Environment (.env)

Create a `.env` next to `app.py`:

```dotenv
OPENAI_API_KEY=sk-...
# optional
OPENAI_MODEL=gpt-4o-mini
ACADEMIC_TZ=Europe/Rome
DATABASE_URL=sqlite:///./academic.db
CORS_ORIGINS=http://127.0.0.1:5500,http://localhost:5500
```

> You can also export these in your shell. The app auto‑loads `.env` via `python-dotenv`.

---

## 🚀 Run

### Option A — One command

```bash
chmod +x dev.sh
./dev.sh
```

* Backend: `http://127.0.0.1:8000` (Docs at `/docs`, Health at `/health`)
* Frontend: `http://127.0.0.1:5500/index.html`
* Stop with **Ctrl+C** (shuts down both).

### Option B — Manual

```bash
uvicorn --env-file .env app:app --reload
python3 -m http.server 5500
```

---

## 🖥️ Using the HTML UI

Go to `http://127.0.0.1:5500/index.html` and:

1. **Create Session** → copy the returned `SID` shown on the page.
2. (Optional) **Language mode** → pick `fa`, `en`, `bi`, or leave `auto`.
3. (Optional) **Profile** → fill Name/Major/Year; learning style (Visual/Auditory/Kinesthetic); courses as `name,difficulty` per line.
4. (Optional) **Add Task/Event** → see formats below.
5. **Chat** → write your prompt; enable **Stream** for live tokens.
6. The **Console** panel shows raw responses/errors for debugging.

**Task format tips**

* `title` is required; `status` must be `needsAction` or `completed`.
* `due` can be `null` or ISO `YYYY-MM-DDTHH:MM:SS`. The API normalizes `YYYY-MM-DD HH:MM:SS` as well.

**Event format tips**

* `summary`, `start`, `end` required; use ISO datetimes; `location` is optional.

---

## 🧪 Quick API Examples

Assuming backend on `http://127.0.0.1:8000`.

### Create session

```bash
curl -s -X POST http://127.0.0.1:8000/sessions | jq
# -> { "session_id": "<SID>", "preferred_lang": "auto" }
```

### Get session state

```bash
curl -s http://127.0.0.1:8000/sessions/<SID> | jq
```

### Set language preference

```bash
curl -s -X PATCH http://127.0.0.1:8000/sessions/<SID>/settings \
  -H 'Content-Type: application/json' \
  -d '{"preferred_lang":"bi"}' | jq
```

### Save profile

```bash
curl -s -X PATCH http://127.0.0.1:8000/sessions/<SID>/profile \
  -H 'Content-Type: application/json' \
  -d '{
    "personal_info": {"name":"Sina","major":"Computer Engineering","academic_year":3},
    "learning_preferences": {"learning_style": {"visual": true, "auditory": false, "kinesthetic": true}},
    "courses": [
      {"name":"Algorithms and Data Structures","difficulty":"high"},
      {"name":"Computer Architecture","difficulty":"medium"}
    ]
  }' | jq
```

### Add task

```bash
curl -s -X POST http://127.0.0.1:8000/sessions/<SID>/tasks \
  -H 'Content-Type: application/json' \
  -d '{"title":"Algorithms – Final Exam","course":"Algorithms and Data Structures","status":"needsAction","due":"2025-09-14T09:00:00"}' | jq
```

### Add event

```bash
curl -s -X POST http://127.0.0.1:8000/sessions/<SID>/events \
  -H 'Content-Type: application/json' \
  -d '{"summary":"Algorithms Exam","start":"2025-09-14T09:00:00","end":"2025-09-14T12:00:00","location":"PoliTo"}' | jq
```

### Chat (non‑stream JSON)

```bash
curl -s -X POST "http://127.0.0.1:8000/sessions/<SID>/chat?lang=bi" \
  -H 'Content-Type: application/json' \
  -d '{"message":"Study plan for the next 3 days; include Persian."}' | jq
```

### Chat (SSE streaming)

```bash
# -N keeps the connection open and prints incoming chunks
curl -N -X POST "http://127.0.0.1:8000/sessions/<SID>/chat?stream=1&lang=bi" \
  -H 'Content-Type: application/json' \
  -d '{"message":"3-day plan + English version"}'
```

---

## 🗄️ Data Model

* `sessionrow(id, created_at, preferred_lang)`
* `profilerow(id, session_id, personal_info_json, learning_prefs_json, courses_json)`
* `eventrow(id, session_id, summary, start, end, location)`
* `taskrow(id, session_id, title, course, status, due)`
* `messagerow(id, session_id, role, content, ts)`

> Relations are *not* declared (to avoid SQLAlchemy 2.x type issues). Queries filter by `session_id`.

---

## 🧰 Troubleshooting

**404 on `/` or favicon** → handled; `/` shows a small JSON banner; `/favicon.ico` returns an empty icon to avoid noise.

**422 on `/tasks`** → usually the `due` format. The API accepts `null`, `""`, or ISO like `2025-09-03T23:59:00`; it also normalizes `YYYY-MM-DD HH:MM:SS`.

**CORS error** → add your frontend origin in `.env`:

```dotenv
CORS_ORIGINS=http://127.0.0.1:5500,http://localhost:5500
```

Restart Uvicorn.

**OpenAI error (`responses` missing)** → app uses Chat Completions for broad compatibility. Ensure `OPENAI_API_KEY` is set and valid.

**DB locked / stale schema** → stop servers (Ctrl+C), delete `academic.db` if needed, restart.

---

## 🔐 Security Notes

This single‑user app isolates data by **session ID (SID)**. Do not share SIDs across people. For per‑user isolation and ownership checks, use `app_multiuser.py` and send `Authorization: Bearer <token>` with each request.

---

## 📝 License

MIT. Feel free to use and modify.

---

## 🙌 Contributing

PRs welcome! Please keep PRs focused (single feature/bugfix) and include a brief description and testing notes.
