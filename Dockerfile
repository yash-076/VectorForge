# ── Stage 1: build / download dependencies ──────────────────────────
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
# so logs appear in real time.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS-level deps required by some torch / numpy wheels.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# ── Pre-download the model so the container starts instantly ─────────
# This bakes the model weights into the image.  Remove these two lines
# if you prefer to mount the model at runtime instead.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

# ── Runtime configuration ────────────────────────────────────────────
# Render injects a PORT env var; fall back to 8100 for local use.
ENV PORT=8100
EXPOSE ${PORT}

# Run with Uvicorn.  Use shell form so $PORT is expanded at runtime.
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
