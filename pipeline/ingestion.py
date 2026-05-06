"""PDF parsing and document chunking logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.helpers import setup_logger, validate_file_path, validate_text

logger = setup_logger(__name__)


class IngestionError(Exception):
    """Raised when document ingestion fails."""


@dataclass(frozen=True)
class DocumentChunk:
    """A normalized text chunk and its source metadata."""

    text: str
    metadata: dict[str, str | int | None]


SECTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "skills": ("skills", "technical skills", "core competencies", "tech stack"),
    "experience": (
        "experience",
        "work experience",
        "employment history",
        "professional experience",
    ),
    "projects": ("projects", "project experience", "portfolio"),
    "education": ("education", "academic background", "certifications"),
    "summary": ("summary", "profile", "objective", "about me"),
}


def _detect_section_type(text: str) -> str:
    snippet = text.lower()
    for section_type, patterns in SECTION_PATTERNS.items():
        if any(pattern in snippet for pattern in patterns):
            return section_type
    return "general"


def extract_pdf_pages_from_path(file_path: str | Path) -> list[dict[str, int | str]]:
    """Extract non-empty text by page from a PDF file path."""
    pdf_path = validate_file_path(file_path, allowed_suffixes={".pdf"})
    pages: list[dict[str, int | str]] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    pages.append({"page_number": index, "text": page_text})
    except Exception as exc:  # pragma: no cover
        raise IngestionError(f"Failed to parse PDF '{pdf_path}': {exc}") from exc

    if not pages:
        raise IngestionError(f"PDF '{pdf_path}' does not contain extractable text.")
    return pages


def extract_pdf_pages_from_bytes(file_bytes: bytes) -> list[dict[str, int | str]]:
    """Extract non-empty text by page from PDF bytes."""
    if not file_bytes:
        raise IngestionError("PDF bytes are empty.")

    pages: list[dict[str, int | str]] = []
    try:
        from io import BytesIO

        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    pages.append({"page_number": index, "text": page_text})
    except Exception as exc:  # pragma: no cover
        raise IngestionError(f"Failed to parse PDF bytes: {exc}") from exc

    if not pages:
        raise IngestionError("PDF data does not contain extractable text.")
    return pages


def resolve_document_text(
    *,
    source: str,
    text: str | None = None,
    file_path: str | None = None,
) -> list[dict[str, int | str]]:
    """Resolve input into page-like records accepted by chunking.

    For raw text input, a single page record with page number 1 is returned.
    For PDF file input, one record per extractable PDF page is returned.
    """
    if text:
        normalized = validate_text(text, f"{source}_text")
        return [{"page_number": 1, "text": normalized}]

    if file_path:
        return extract_pdf_pages_from_path(file_path)

    raise IngestionError(f"Missing document input for source '{source}'.")


def chunk_document_pages(
    pages: list[dict[str, int | str]],
    *,
    source: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[DocumentChunk]:
    """Split pages into semantic chunks with attached metadata."""
    if not pages:
        raise IngestionError(f"No pages to chunk for source '{source}'.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page in pages:
        page_number = int(page["page_number"])
        page_text = str(page["text"]).strip()
        if not page_text:
            continue

        lines = [line.strip() for line in re.split(r"[\r\n]+", page_text) if line.strip()]
        heading_hint = lines[0][:120] if lines else ""
        page_section_hint = _detect_section_type(heading_hint) if heading_hint else "general"
        split_chunks = splitter.split_text(page_text)
        for chunk_text in split_chunks:
            clean_chunk = chunk_text.strip()
            if not clean_chunk:
                continue
            chunk_section = _detect_section_type(clean_chunk[:220])
            section_type = chunk_section if chunk_section != "general" else page_section_hint
            all_chunks.append(
                DocumentChunk(
                    text=clean_chunk,
                    metadata={
                        "source": source,
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "section_type": section_type,
                    },
                )
            )
            chunk_index += 1

    if not all_chunks:
        raise IngestionError(f"Chunking produced no output for source '{source}'.")

    logger.info("Chunked %s document into %d chunks.", source, len(all_chunks))
    return all_chunks
