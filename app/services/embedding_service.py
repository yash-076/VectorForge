"""
Embedding Service Module
========================
Encapsulates all embedding-related business logic.

Architecture notes:
- Uses ``fastembed`` (ONNX Runtime) instead of PyTorch-based
  SentenceTransformers for a dramatically smaller memory footprint
  (~150 MB vs ~500 MB), making it suitable for memory-constrained
  deployments such as Render's starter plan (512 MB).
- The model is loaded **once** when the service is instantiated and
  reused across every request.
- A threading lock guards the model's `embed()` call so the service
  is safe to use from multiple async workers / threads concurrently.
- All heavy lifting lives here — route handlers stay thin.
"""

from __future__ import annotations

import logging
import threading
from typing import List

from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

# fastembed uses its own model name format; map common HuggingFace names.
_MODEL_ALIAS: dict[str, str] = {
    "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}


class EmbeddingService:
    """Thread-safe wrapper around a fastembed TextEmbedding model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu") -> None:
        """
        Load the model into memory.

        Parameters
        ----------
        model_name : str
            HuggingFace / fastembed model identifier.
        device : str
            Kept for API compatibility (fastembed uses ONNX Runtime
            and selects the best available provider automatically).
        """
        resolved = _MODEL_ALIAS.get(model_name, model_name)
        logger.info("Loading fastembed model '%s' …", resolved)
        self._model = TextEmbedding(model_name=resolved)
        self._lock = threading.Lock()
        logger.info("Model loaded successfully (ONNX Runtime backend).")

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
            # fastembed's embed() returns a generator of numpy arrays.
            embeddings = list(self._model.embed(texts))
        return [e.tolist() for e in embeddings]
