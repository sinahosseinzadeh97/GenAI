# LegalTech AI Assistant — MVP

An MVP to upload legal documents, build a local FAISS index, and answer questions with citations using a simple multi‑agent flow. Includes optional n8n automation webhooks.

## Quick Start

1. Copy `.env.example` → `.env` and adjust values.
2. `docker-compose up --build`
3. Open the frontend at http://localhost:5173 (backend at http://localhost:8000)

### API Endpoints
- `POST /documents` — Upload a document (PDF/DOCX/TXT). Returns `{document_id, workflow_id}` and runs background analysis + indexing.
- `POST /query` — Body `{ question, top_k?, scope_doc_ids?, action? }` → returns RAG `answer`, `sources`, `laws`, and a `workflow_id`.
- `GET /workflows/{id}` — Retrieve workflow status/result JSON.

### Multi‑Agent (MVP)
- **Agent 1 – Document Analyzer**: Summarizes and tags on upload (background).
- **Agent 2 – Law Retriever**: Retrieves relevant items from a small local Italian laws dataset (FAISS index at startup).
- **Agent 3 – Draft Generator**: Optional `action` to produce a client email, case summary, or contract clause draft.

### Automations
Set `N8N_WEBHOOK_URL` in `.env` to receive webhook events: `document_analyzed`, `query_answered` with payloads.

### Notes
- If `USE_OPENAI=true` and `OPENAI_API_KEY` is set, the app uses OpenAI for both LLM and embeddings. Otherwise it falls back to a local sentence‑transformers model and a lightweight offline draft generator.
- Vector stores persist to `backend/app/storage/vectorstore/`.
- Uploaded files are stored in `backend/app/storage/files/`.

### Extending
- Add authentication (JWT) for lawyer/client users.
- Expand the laws dataset and build a proper ingestion pipeline (per‑source metadata, dates, jurisdictions).
- Add citations with paragraph numbers and confidence scores.
- Replace the offline draft with a local LLM via Ollama when OpenAI isn’t used.