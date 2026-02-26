"""
Embedding Microservice — FastAPI Application
=============================================

Entry-point for the service.  Responsibilities of this module:

1. Configure logging.
2. Instantiate the shared ``EmbeddingService`` **once** at startup
   via a FastAPI *lifespan* context manager (avoids per-request loading).
3. Wire up thin route handlers that delegate to the service layer.
4. Optionally enforce a simple API-key authentication scheme through
   a dependency.

Architecture overview::

    Request  ──▶  Auth middleware  ──▶  Route handler  ──▶  EmbeddingService
                                                                │
                                                        SentenceTransformer
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv

# Load variables from .env file into the environment.
# This must happen before any os.getenv() calls below.
load_dotenv()

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.schemas import (
    EmbedBatchRequest,
    EmbedBatchResponse,
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
)
from app.services.embedding_service import EmbeddingService

# ── Logging configuration ────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DEVICE = os.getenv("DEVICE", "cpu")

# API key authentication — set the env var to enable it; leave unset to
# disable (useful during local development).
API_KEY: str | None = os.getenv("API_KEY")

# ── API key security scheme ──────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str | None = Security(api_key_header)) -> None:
    """
    Dependency that checks the ``X-API-Key`` header when ``API_KEY`` is
    configured.  If the env var is not set, authentication is skipped so
    the service works out of the box during development.
    """
    if API_KEY is None:
        # Auth disabled — allow all requests.
        return
    if key is None or key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ── Application lifespan (model loading) ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Load the embedding model **once** when the application starts up and
    store it on ``app.state`` so every route handler can access it.
    """
    logger.info("Starting up — loading embedding model …")
    app.state.embedding_service = EmbeddingService(model_name=MODEL_NAME, device=DEVICE)
    logger.info("Embedding model ready.")
    yield
    # Shutdown — nothing to clean up explicitly.
    logger.info("Shutting down embedding service.")


# ── FastAPI application instance ─────────────────────────────────────

app = FastAPI(
    title="Embedding Microservice",
    description="Lightweight AI inference service that generates semantic vector embeddings.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Helper to retrieve the service from app state ────────────────────

def get_embedding_service(request: Request) -> EmbeddingService:
    """FastAPI dependency that pulls the service from ``app.state``."""
    return request.app.state.embedding_service


# ── Route handlers (kept deliberately thin) ──────────────────────────


@app.post(
    "/embed",
    response_model=EmbedResponse,
    summary="Embed a single text",
    dependencies=[Depends(verify_api_key)],
)
async def embed(
    body: EmbedRequest,
    service: EmbeddingService = Depends(get_embedding_service),
) -> EmbedResponse:
    """Generate an embedding vector for a single piece of text."""
    try:
        vector = service.embed(body.text)
        return EmbedResponse(embedding=vector)
    except Exception as exc:
        logger.exception("Error while embedding text.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/embed-batch",
    response_model=EmbedBatchResponse,
    summary="Embed multiple texts",
    dependencies=[Depends(verify_api_key)],
)
async def embed_batch(
    body: EmbedBatchRequest,
    service: EmbeddingService = Depends(get_embedding_service),
) -> EmbedBatchResponse:
    """Generate embedding vectors for a batch of texts."""
    try:
        vectors = service.embed_batch(body.texts)
        return EmbedBatchResponse(embeddings=vectors)
    except Exception as exc:
        logger.exception("Error while embedding batch.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.api_route(
    "/health",
    methods=["GET", "HEAD"],
    response_model=HealthResponse,
    summary="Health check",
)
async def health(
    service: EmbeddingService = Depends(get_embedding_service),
) -> HealthResponse:
    """Return the current service status and loaded model name."""
    return HealthResponse(status="ok", model=MODEL_NAME)
