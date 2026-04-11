import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from .core.config import settings
from .db.mongodb import connect_to_mongo, close_mongo_connection
from .api.chat import router as chat_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown events:
    - Connect to MongoDB when the app starts
    - Close the MongoDB connection when the app shuts down
    """
    await connect_to_mongo()
    yield
    await close_mongo_connection()

# Create the FastAPI app, with docs at /docs, redoc at /redoc, OpenAPI spec at /openapi.json
app = FastAPI(
    title=settings.project_name,
    version=settings.api_version,
    description="Smart Chatbot API with OpenAI integration",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Enable CORS for the front-end origins defined in your settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1) Mount the chat API router under /api/{version}
app.include_router(chat_router, prefix=f"/api/{settings.api_version}")

# 2) Serve static files (JS/CSS/assets) from app/static at the /static path
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 3) Serve index.html at the root URL (/)
@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

# 4) SPA fallback: for any other non‑API path, return index.html
@app.get("/{full_path:path}", include_in_schema=False)
async def serve_spa(full_path: str):
    # If someone tries to hit a real API route that doesn't exist, give a 404
    if full_path.startswith(f"api/{settings.api_version}"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    # Otherwise, serve the front-end app
    return FileResponse(os.path.join(static_dir, "index.html"))

# 5) Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}
