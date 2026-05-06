"""Vector database indexing and retrieval logic."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import chromadb

from pipeline.embeddings import get_embedding_service
from pipeline.ingestion import DocumentChunk
from utils.config import get_settings
from utils.helpers import setup_logger

logger = setup_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved CV chunk and its retrieval scores."""

    id: str
    text: str
    metadata: dict[str, Any]
    distance: float


class CVVectorStore:
    """Persistent Chroma-backed vector store for CV chunks."""

    def __init__(self, collection_name: str = "cv_chunks") -> None:
        settings = get_settings()
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedding_service = get_embedding_service()

    @staticmethod
    def _build_chunk_id(doc_id: str, chunk: DocumentChunk, index: int) -> str:
        digest = hashlib.sha1(chunk.text.encode("utf-8")).hexdigest()[:12]
        return f"{doc_id}:{index}:{digest}"

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized

    def upsert_cv_chunks(self, chunks: list[DocumentChunk], doc_id: str) -> int:
        """Upsert CV chunks and vectors into Chroma."""
        if not chunks:
            return 0

        documents = [chunk.text for chunk in chunks]
        embeddings = self._embedding_service.embed_texts(documents)
        ids = [self._build_chunk_id(doc_id, chunk, idx) for idx, chunk in enumerate(chunks)]
        metadatas = [self._sanitize_metadata(chunk.metadata) for chunk in chunks]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Upserted %d CV chunks into collection.", len(chunks))
        return len(chunks)

    def query_similar_chunks(self, query_text: str, top_k: int = 8) -> list[RetrievedChunk]:
        """Query CV chunks for one JD text query."""
        query_embedding = self._embedding_service.embed_text(query_text)
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return self._parse_query_results(raw)

    def query_for_jd_chunks(
        self,
        jd_chunks: list[DocumentChunk],
        top_k_per_chunk: int = 8,
        final_top_k: int = 20,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant CV chunks for each JD chunk and deduplicate by best score."""
        if not jd_chunks:
            return []

        query_texts = [chunk.text for chunk in jd_chunks]
        query_embeddings = self._embedding_service.embed_texts(query_texts)

        raw = self._collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k_per_chunk,
            include=["documents", "metadatas", "distances"],
        )
        candidates = self._parse_query_results(raw, flatten_all=True)

        by_id: dict[str, RetrievedChunk] = {}
        for chunk in candidates:
            existing = by_id.get(chunk.id)
            if existing is None or chunk.distance < existing.distance:
                by_id[chunk.id] = chunk

        ranked = sorted(by_id.values(), key=lambda item: item.distance)
        return ranked[:final_top_k]

    @staticmethod
    def _parse_query_results(
        raw: dict[str, Any], *, flatten_all: bool = False
    ) -> list[RetrievedChunk]:
        ids_groups = raw.get("ids") or []
        documents_groups = raw.get("documents") or []
        metadatas_groups = raw.get("metadatas") or []
        distances_groups = raw.get("distances") or []

        if not ids_groups:
            return []

        if flatten_all:
            output: list[RetrievedChunk] = []
            for ids, docs, metas, dists in zip(
                ids_groups, documents_groups, metadatas_groups, distances_groups, strict=False
            ):
                for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
                    output.append(
                        RetrievedChunk(
                            id=str(chunk_id),
                            text=str(doc),
                            metadata=meta or {},
                            distance=float(dist),
                        )
                    )
            return output

        first_ids = ids_groups[0]
        first_docs = documents_groups[0] if documents_groups else []
        first_metas = metadatas_groups[0] if metadatas_groups else []
        first_dists = distances_groups[0] if distances_groups else []

        return [
            RetrievedChunk(
                id=str(chunk_id),
                text=str(doc),
                metadata=meta or {},
                distance=float(dist),
            )
            for chunk_id, doc, meta, dist in zip(
                first_ids, first_docs, first_metas, first_dists, strict=False
            )
        ]


_vector_store: CVVectorStore | None = None


def get_vector_store() -> CVVectorStore:
    """Get a singleton vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = CVVectorStore()
    return _vector_store
