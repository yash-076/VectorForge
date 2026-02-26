"""
Pydantic Schemas
================
Request and response models used by the API layer.

Keeping schemas separate from route handlers and service logic ensures
a clean separation of concerns and makes them easy to reuse or test
independently.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


# ── Request models ────────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    """Body for ``POST /embed``."""
    text: str = Field(..., min_length=1, description="The text to embed.")


class EmbedBatchRequest(BaseModel):
    """Body for ``POST /embed-batch``."""
    texts: List[str] = Field(..., min_length=1, description="A list of texts to embed.")


# ── Response models ───────────────────────────────────────────────────

class EmbedResponse(BaseModel):
    """Response for ``POST /embed``."""
    embedding: List[float]


class EmbedBatchResponse(BaseModel):
    """Response for ``POST /embed-batch``."""
    embeddings: List[List[float]]


class HealthResponse(BaseModel):
    """Response for ``GET /health``."""
    status: str
    model: str
