# ThinkPath (Python) — Guided LLM Chat

**Bilingual (FA/EN), selectable backend (Ollama or OpenAI).**

## ✨ Features
- 4 **thinking paths**, each with 3 steps (click to run up to any step).
- **Persian / English** UI and answer language.
- Pluggable **providers**: Local **Ollama** or **OpenAI** (Chat Completions).
- Clean Streamlit UI with session-based conversation history.

## 🚀 Quick Start
```bash
# 1) Create venv (optional)
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Install deps
pip install -r requirements.txt

# 3) Run
streamlit run app.py
```

## 🔌 Providers

### Ollama (local)
- Install Ollama: https://ollama.ai (ensure it's running)
- Pull a model, e.g.:
  ```bash
  ollama pull llama3.1:8b
  ```
- In the sidebar, choose **Ollama**, keep base URL (default `http://localhost:11434`), set model (e.g., `llama3.1:8b`).

### OpenAI
- Set your key: `export OPENAI_API_KEY=sk-...` (Windows PowerShell: `$env:OPENAI_API_KEY='sk-...'`)
- In the sidebar, choose **OpenAI** and a model (e.g., `gpt-4o-mini`).

## 🔑 Environment (.env)
Create a `.env` file (or copy `.env.example`) at the project root:
```
OPENAI_API_KEY=sk-...
OLLAMA_BASE_URL=http://localhost:11434
```
The app auto-loads it via `python-dotenv`.

## 🧰 Where this is useful
ThinkPath (Python) shines when you need **structured, multi-angle thinking** with quick iteration:

- **Brainstorming & Ideation:** product features, startup ideas, campaign taglines.
- **Strategy & Roadmapping:** product/tech roadmaps, OKRs, prioritization (RICE/ICE), trade-off discussions.
- **Learning & Study Planning:** break down topics into small steps (bilingual FA/EN).
- **Software Workflows:** decompose tasks, write checklists, outline test plans, migration steps.
- **Research & Discovery:** plan literature reviews, compare approaches, summarize findings.
- **Decision Support:** evaluate options, list pros/cons, define decision criteria.
- **Troubleshooting:** step-by-step diagnostic paths, next-step suggestions.
- **Content Outlines:** article/video/course outlines with ordered action steps.
- **Interview Prep:** structure answers (STAR), create practice paths by role.
- **Offline / Privacy-conscious:** run locally via **Ollama** when cloud is not desired.

## 🪝 Git & Ignore
A `.gitignore` is included to keep virtualenvs, caches, and secrets (like `.env`) out of version control.

## 📁 Structure
- `app.py` — Streamlit UI & logic
- `llm_providers.py` — Provider wrapper for OpenAI / Ollama
- `requirements.txt`
- `.env.example` — sample env file
- `.gitignore`