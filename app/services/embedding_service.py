"""
Embedding Service Module
========================
Encapsulates all embedding-related business logic.

Architecture notes:
- The SentenceTransformer model is loaded **once** when the service is
  instantiated and reused across every request.
- A threading lock guards the model's `encode()` call so the service
  is safe to use from multiple async workers / threads concurrently.
- All heavy lifting lives here — route handlers stay thin.
"""

from __future__ import annotations

import logging
import threading
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Thread-safe wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu") -> None:
        """
        Load the model into memory.

        Parameters
        ----------
        model_name : str
            HuggingFace model identifier.
        device : str
            Torch device string — kept as ``"cpu"`` for broad compatibility.
        """
        logger.info("Loading SentenceTransformer model '%s' on device '%s' …", model_name, device)
        self._model = SentenceTransformer(model_name, device=device)
        self._lock = threading.Lock()
        logger.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> List[float]:
        """Return the embedding vector for a single piece of text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Return embedding vectors for a list of texts.

        The lock ensures that concurrent requests do not invoke the
        underlying model simultaneously, which could cause data-races in
        non-thread-safe C/C++ extensions.
        """
        if not texts:
            return []

        logger.debug("Encoding %d text(s) …", len(texts))
        with self._lock:
            # .encode() returns a numpy ndarray; convert to plain lists.
            embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
