"""PDF parsing and document chunking logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.helpers import setup_logger, validate_file_path, validate_text

logger = setup_logger(__name__)


class IngestionError(Exception):
    """Raised when document ingestion fails."""


@dataclass(frozen=True)
class DocumentChunk:
    """A normalized text chunk and its source metadata."""

    text: str
    metadata: dict[str, str | int | None]


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

        split_chunks = splitter.split_text(page_text)
        for chunk_text in split_chunks:
            clean_chunk = chunk_text.strip()
            if not clean_chunk:
                continue
            all_chunks.append(
                DocumentChunk(
                    text=clean_chunk,
                    metadata={
                        "source": source,
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                    },
                )
            )
            chunk_index += 1

    if not all_chunks:
        raise IngestionError(f"Chunking produced no output for source '{source}'.")

    logger.info("Chunked %s document into %d chunks.", source, len(all_chunks))
    return all_chunks
