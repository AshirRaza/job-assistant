"""Embedding generation using sentence transformers."""

from __future__ import annotations

import hashlib
import json

from sentence_transformers import SentenceTransformer

from utils.config import get_settings
from utils.helpers import setup_logger

logger = setup_logger(__name__)


class EmbeddingService:
    """Generate and cache normalized sentence embeddings."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model_name = settings.embedding_model
        self._model = SentenceTransformer(self.model_name)
        self._cache_path = settings.chroma_persist_dir / "embedding_cache.json"
        self._cache = self._load_cache()
        logger.info("Embedding model loaded: %s", self.model_name)

    def _load_cache(self) -> dict[str, list[float]]:
        cache_dir = self._cache_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not self._cache_path.exists():
            return {}

        try:
            with self._cache_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (json.JSONDecodeError, OSError):
            logger.warning("Embedding cache is unreadable. Rebuilding cache.")
            return {}

        if not isinstance(payload, dict):
            return {}
        return {str(key): value for key, value in payload.items() if isinstance(value, list)}

    def _persist_cache(self) -> None:
        with self._cache_path.open("w", encoding="utf-8") as file:
            json.dump(self._cache, file)

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_text(self, text: str) -> list[float]:
        text_hash = self._hash_text(text)
        cached = self._cache.get(text_hash)
        if cached is not None:
            return cached

        vector = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embedding = vector.tolist()
        self._cache[text_hash] = embedding
        self._persist_cache()
        return embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        missing_hashes: list[str] = []
        missing_texts: list[str] = []

        for text in texts:
            text_hash = self._hash_text(text)
            if text_hash not in self._cache:
                missing_hashes.append(text_hash)
                missing_texts.append(text)

        if missing_texts:
            vectors = self._model.encode(
                missing_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            for text_hash, vector in zip(missing_hashes, vectors, strict=False):
                self._cache[text_hash] = vector.tolist()
            self._persist_cache()

        return [self._cache[self._hash_text(text)] for text in texts]


_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get a singleton embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
