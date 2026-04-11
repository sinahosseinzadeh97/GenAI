#!/usr/bin/env bash
# Run FastAPI backend (Uvicorn) and static frontend (python http.server) together.
# Usage:
#   chmod +x dev.sh
#   ./dev.sh
#
# Requirements:
#   - Python 3 installed
#   - pip install fastapi uvicorn pydantic openai python-dotenv sqlmodel sqlalchemy aiosqlite
#   - app.py in the same folder

set -euo pipefail
cd "$(dirname "$0")"

# Load .env into environment (safe for most simple KEY=VALUE lines)
set -a
[ -f .env ] && . .env || true
set +a

# Ensure CORS allows the frontend origin(s)
export CORS_ORIGINS=${CORS_ORIGINS:-"http://127.0.0.1:5500,http://localhost:5500,http://localhost:5173,http://localhost:3000"}

# Ports (change if needed)
BACK_HOST=${BACK_HOST:-127.0.0.1}
BACK_PORT=${BACK_PORT:-8000}
FRONT_PORT=${FRONT_PORT:-5500}

# Helper: cleanup background jobs on exit
cleanup() {
  echo "\nShutting down..."
  # Kill all background jobs spawned by this script
  jobs -p | xargs -I{} kill {} >/dev/null 2>&1 || true
}
trap cleanup INT TERM EXIT

# Start backend
echo "[dev] Starting backend on http://${BACK_HOST}:${BACK_PORT} ..."
uvicorn app:app --host "$BACK_HOST" --port "$BACK_PORT" --reload &
BACK_PID=$!

# Start frontend static server
echo "[dev] Starting frontend server on http://127.0.0.1:${FRONT_PORT} ..."
python3 -m http.server "$FRONT_PORT" &
FRONT_PID=$!

# macOS: open the frontend in browser (optional)
if command -v open >/dev/null 2>&1; then
  (sleep 1; open "http://127.0.0.1:${FRONT_PORT}/index.html") &
fi

echo "[dev] Backend PID: ${BACK_PID} | Frontend PID: ${FRONT_PID}"
printf "[dev] Press CTRL+C to stop both.\n\n"

# Wait forever (until a child exits or user hits Ctrl+C)
wait
