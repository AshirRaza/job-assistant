"""RAG pipeline orchestration layer."""

from __future__ import annotations

import hashlib
import re

from models.schemas import AnalysisRequest, AnalysisResponse
from pipeline.ingestion import chunk_document_pages, resolve_document_text
from pipeline.llm import LLMError, get_claude_analyzer
from pipeline.ner import get_skill_extractor
from pipeline.reranker import get_reranker
from pipeline.vector_store import RetrievedChunk, get_vector_store
from utils.config import get_settings
from utils.helpers import setup_logger, timed_block

logger = setup_logger(__name__)


class RAGPipelineError(Exception):
    """Raised when the RAG orchestration pipeline fails."""


class RAGPipeline:
    """End-to-end orchestration for CV vs JD analysis."""

    def __init__(self) -> None:
        settings = get_settings()
        self._vector_store = get_vector_store()
        self._reranker = get_reranker()
        self._skill_extractor = get_skill_extractor()
        self._llm = get_claude_analyzer()
        self._max_context_tokens = settings.max_context_tokens
        self._summarize_overflow = settings.summarize_overflow
        self._overflow_summary_tokens = settings.overflow_summary_tokens

    @staticmethod
    def _build_doc_id(cv_text: str) -> str:
        digest = hashlib.sha1(cv_text.encode("utf-8")).hexdigest()[:16]
        return f"cv-{digest}"

    @staticmethod
    def _join_page_text(pages: list[dict[str, int | str]]) -> str:
        return "\n\n".join(str(page["text"]) for page in pages)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Lightweight token estimate for budgeting without extra tokenizer deps.
        return len(re.findall(r"\w+|[^\w\s]", text))

    def _truncate_to_tokens(self, text: str, token_limit: int) -> str:
        tokens = re.findall(r"\w+|[^\w\s]", text)
        if len(tokens) <= token_limit:
            return text
        clipped = " ".join(tokens[:token_limit]).strip()
        return clipped

    def _summarize_overflow_chunks(
        self, overflow_chunks: list[RetrievedChunk], token_budget: int
    ) -> str:
        if not overflow_chunks or token_budget <= 0:
            return ""

        snippets: list[str] = []
        for chunk in overflow_chunks[:6]:
            text = chunk.text.strip()
            if not text:
                continue
            # Extractive first-sentence style summary.
            sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
            snippets.append(sentence)

        summary = "Additional relevant CV context (summarized): " + " ".join(snippets)
        return self._truncate_to_tokens(summary, token_budget)

    def _build_context(self, chunks: list[RetrievedChunk], max_tokens: int) -> str:
        sections: list[str] = []
        consumed = 0
        overflow_index = len(chunks)

        for index, chunk in enumerate(chunks):
            snippet = chunk.text.strip()
            if not snippet:
                continue

            page_number = chunk.metadata.get("page_number", "n/a")
            section_type = str(chunk.metadata.get("section_type", "general")).title()
            section = f"[{section_type} | Page {page_number}] {snippet}"
            section_tokens = self._estimate_tokens(section)
            if consumed + section_tokens > max_tokens:
                overflow_index = index
                break
            sections.append(section)
            consumed += section_tokens

        if self._summarize_overflow and overflow_index < len(chunks):
            remaining = chunks[overflow_index:]
            budget = max_tokens - consumed
            if budget > 24:
                summary = self._summarize_overflow_chunks(
                    remaining, min(self._overflow_summary_tokens, budget)
                )
                if summary:
                    sections.append(summary)

        return "\n\n".join(sections)

    def run(self, request: AnalysisRequest) -> AnalysisResponse:
        try:
            with timed_block("ingestion_stage", logger):
                cv_pages = resolve_document_text(
                    source="cv",
                    text=request.cv_text,
                    file_path=request.cv_file_path,
                )
                jd_pages = resolve_document_text(
                    source="jd",
                    text=request.jd_text,
                    file_path=request.jd_file_path,
                )
                cv_chunks = chunk_document_pages(cv_pages, source="cv")
                jd_chunks = chunk_document_pages(jd_pages, source="jd")
                cv_full_text = self._join_page_text(cv_pages)
                jd_full_text = self._join_page_text(jd_pages)

            with timed_block("ner_stage", logger):
                cv_ner = self._skill_extractor.extract(cv_full_text)
                jd_ner = self._skill_extractor.extract(jd_full_text)

            with timed_block("retrieval_stage", logger):
                doc_id = self._build_doc_id(cv_full_text)
                self._vector_store.upsert_cv_chunks(cv_chunks, doc_id=doc_id)
                candidate_top_k = max(request.top_k * 3, 20)
                retrieved = self._vector_store.query_for_jd_chunks(
                    jd_chunks,
                    top_k_per_chunk=candidate_top_k,
                    final_top_k=candidate_top_k,
                    cv_doc_id=doc_id,
                    hybrid_query_text=jd_full_text,
                )

            with timed_block("rerank_stage", logger):
                reranked = self._reranker.rerank(
                    query_text=jd_full_text,
                    chunks=retrieved,
                    top_k=request.top_k,
                )
                selected = [
                    RetrievedChunk(
                        id=item.id,
                        text=item.text,
                        metadata=item.metadata,
                        distance=item.distance,
                    )
                    for item in reranked
                ]

            with timed_block("llm_stage", logger):
                context = self._build_context(selected, max_tokens=self._max_context_tokens)
                response = self._llm.analyze(
                    jd_text=jd_full_text,
                    cv_context=context,
                    cv_skills=cv_ner.normalized_skills,
                    jd_skills=jd_ner.normalized_skills,
                )
            return response
        except LLMError as exc:
            raise RAGPipelineError(str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise RAGPipelineError(f"RAG pipeline failed: {exc}") from exc


_rag_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    """Get a singleton RAG pipeline."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
