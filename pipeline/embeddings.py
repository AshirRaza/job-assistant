"""Embedding generation using sentence transformers."""

from __future__ import annotations

import hashlib
import json
import math

from sentence_transformers import SentenceTransformer

from utils.config import get_settings
from utils.helpers import setup_logger

logger = setup_logger(__name__)


class EmbeddingService:
    """Generate and cache normalized sentence embeddings."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model_name = settings.embedding_model
        self.offline_mode = settings.offline_mode
        self._vector_size = 384
        self._model: SentenceTransformer | None = None
        if not self.offline_mode:
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded: %s", self.model_name)
        else:
            logger.info("Offline mode enabled: using deterministic local embeddings.")
        self._cache_path = settings.chroma_persist_dir / "embedding_cache.json"
        self._cache = self._load_cache()

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

    def _local_embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self._vector_size
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:2], byteorder="big") % self._vector_size
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[idx] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def embed_text(self, text: str) -> list[float]:
        text_hash = self._hash_text(text)
        cached = self._cache.get(text_hash)
        if cached is not None:
            return cached

        if self.offline_mode:
            embedding = self._local_embed_text(text)
        else:
            assert self._model is not None
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
            if self.offline_mode:
                for text_hash, text in zip(missing_hashes, missing_texts, strict=False):
                    self._cache[text_hash] = self._local_embed_text(text)
            else:
                assert self._model is not None
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
