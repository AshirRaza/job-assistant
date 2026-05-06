"""Cross-encoder re-ranking for retrieved chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sentence_transformers import CrossEncoder

from pipeline.vector_store import RetrievedChunk
from utils.config import get_settings
from utils.helpers import setup_logger

logger = setup_logger(__name__)


@dataclass(frozen=True)
class RerankedChunk:
    """Retrieved chunk with cross-encoder relevance score."""

    id: str
    text: str
    metadata: dict[str, Any]
    distance: float
    rerank_score: float


class CrossEncoderReranker:
    """Re-rank retrieved chunks against JD query text."""

    def __init__(self) -> None:
        settings = get_settings()
        self.enabled = settings.use_reranker
        self.model_name = settings.reranker_model
        self._model: CrossEncoder | None = None

        if self.enabled:
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder re-ranker loaded: %s", self.model_name)
        else:
            logger.info("Cross-encoder re-ranker disabled via USE_RERANKER.")

    def rerank(
        self, *, query_text: str, chunks: list[RetrievedChunk], top_k: int | None = None
    ) -> list[RerankedChunk]:
        if not chunks:
            return []

        if not self.enabled or self._model is None:
            # Preserve retrieval order and map distance to inverse proxy score.
            fallback = [
                RerankedChunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    distance=chunk.distance,
                    rerank_score=1.0 / (1.0 + max(chunk.distance, 0.0)),
                )
                for chunk in chunks
            ]
            return fallback[:top_k] if top_k is not None else fallback

        pairs = [[query_text, chunk.text] for chunk in chunks]
        scores = self._model.predict(pairs)

        reranked = [
            RerankedChunk(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata,
                distance=chunk.distance,
                rerank_score=float(score),
            )
            for chunk, score in zip(chunks, scores, strict=False)
        ]
        reranked.sort(key=lambda item: item.rerank_score, reverse=True)
        return reranked[:top_k] if top_k is not None else reranked


_reranker: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    """Get a singleton reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker
