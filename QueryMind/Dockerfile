FROM python:3.12-slim

WORKDIR /app

# Copy pyproject.toml first (layer caching)
# We also copy README.md because it's referenced in pyproject.toml
COPY pyproject.toml README.md ./

# Create a temporary dummy module so pip install can resolve dependencies layer
RUN mkdir -p querymind && touch querymind/__init__.py && pip install -e . || true

# Copy source code
COPY . .

# Run pip install again to ensure it captures the actual source code
RUN pip install -e .

# Volume mount point: /app/data (for persistent DB)
VOLUME /app/data

# DB_PATH defaults
ENV DB_PATH=/app/data/querymind.db

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Default CMD
CMD ["python", "-m", "querymind.server"]
