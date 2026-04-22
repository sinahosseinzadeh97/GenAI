# ─────────────────────────────────────────────────────────────────────────────
# RAGForge Enterprise – LexReview Dockerfile
#
# Stages
#   base     – system deps, spaCy model, pip install (runtime deps only)
#   runtime  – non-root uvicorn entrypoint (DEFAULT)
#   finetune – base + GPU/PEFT extras; invoked via --profile finetune
#
# Build:
#   docker build --target runtime  -t lexreview-api:dev .
#   docker build --target finetune -t lexreview-finetune:dev .
# ─────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.11

# ─── Stage 1: base ───────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

# Prevent .pyc files, enable unbuffered stdout for log shipping
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages: curl for healthchecks, build tools for C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python runtime dependencies ──────────────────────────────────────
# Copy the full project metadata so pip can resolve all declared dependencies
COPY pyproject.toml ./
COPY src/ ./src/

# 1. Upgrade toolchain
# 2. Install CPU-only PyTorch (no CUDA bloat; works on Linux/Mac CI and M-series)
# 3. Regular install reads [project.dependencies] from pyproject.toml
# 4. Pull the spaCy English model — merged into one layer to minimise image layers
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install "." && \
    python -m spacy download en_core_web_sm

# (src/ already copied above before install)

# Create directories expected by the application at runtime
RUN mkdir -p /app/logs /app/models /app/data

# ─── Stage 2: runtime ────────────────────────────────────────────────────────
FROM base AS runtime

# Non-root user – never run as root in production
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# Fix ownership – must happen while still root
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Healthcheck is also wired in docker-compose; include here as image-level default
HEALTHCHECK --interval=20s --timeout=8s --start-period=30s --retries=5 \
    CMD curl -sf http://localhost:8000/health || exit 1

# uvicorn entrypoint – workers auto-scales to CPU count inside container
ENTRYPOINT ["uvicorn", "src.api.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info"]

# ─── Stage 3: finetune ───────────────────────────────────────────────────────
FROM base AS finetune

# GPU runtime extras (installed into the same venv as base)
RUN pip install \
        "peft>=0.11.0" \
        "trl>=0.9.0" \
        "bitsandbytes>=0.43.0" \
        "transformers>=4.41.0" \
        "datasets>=2.19.0" \
        "accelerate>=0.30.0"

# Non-root user for the fine-tune job too
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser && \
    chown -R appuser:appgroup /app

USER appuser

# Sentinel file written by entrypoint script on success (healthcheck target)
# The actual training script path is passed at runtime:
#   docker compose --profile finetune run finetune python -m src.finetune.train
ENTRYPOINT ["python", "-m"]
CMD ["src.finetune.train"]
