from __future__ import annotations

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import ValidationError
from starlette.responses import JSONResponse

from models.schemas import AnalysisRequest, AnalysisResponse, ErrorResponse, SkillMatch
from pipeline.ingestion import (
    IngestionError,
    chunk_document_pages,
    extract_pdf_pages_from_bytes,
    resolve_document_text,
)
from utils.config import Settings, get_settings
from utils.helpers import setup_logger

app = FastAPI(title="CV Analyzer API", version="0.1.0")
logger = setup_logger()
settings: Settings = get_settings()


def _error_response(
    *,
    status_code: int,
    error_type: str,
    message: str,
    details: dict[str, str] | None = None,
) -> JSONResponse:
    payload = ErrorResponse(error=error_type, message=message, details=details)
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _build_request_model(
    *,
    cv_text: str | None,
    jd_text: str | None,
    cv_file_path: str | None = None,
    jd_file_path: str | None = None,
    top_k: int = 8,
) -> AnalysisRequest:
    return AnalysisRequest(
        cv_text=cv_text,
        jd_text=jd_text,
        cv_file_path=cv_file_path,
        jd_file_path=jd_file_path,
        top_k=top_k,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    logger.debug("Health check received.")
    return {"status": "ok"}


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def analyze(
    cv_text: str | None = Form(default=None),
    jd_text: str | None = Form(default=None),
    cv_file: UploadFile | None = File(default=None),
    jd_file: UploadFile | None = File(default=None),
    top_k: int = Form(default=8),
) -> AnalysisResponse | JSONResponse:
    try:
        request_model = _build_request_model(cv_text=cv_text, jd_text=jd_text, top_k=top_k)
    except ValidationError as exc:
        return _error_response(
            status_code=422,
            error_type="validation_error",
            message="Invalid analysis request.",
            details={"reason": str(exc)},
        )

    try:
        if cv_file is not None:
            cv_pages = extract_pdf_pages_from_bytes(await cv_file.read())
        else:
            cv_pages = resolve_document_text(source="cv", text=request_model.cv_text)

        if jd_file is not None:
            jd_pages = extract_pdf_pages_from_bytes(await jd_file.read())
        else:
            jd_pages = resolve_document_text(source="jd", text=request_model.jd_text)

        cv_chunks = chunk_document_pages(cv_pages, source="cv")
        jd_chunks = chunk_document_pages(jd_pages, source="jd")
    except IngestionError as exc:
        return _error_response(
            status_code=400,
            error_type="ingestion_error",
            message="Document ingestion failed.",
            details={"reason": str(exc)},
        )
    except ValueError as exc:
        return _error_response(
            status_code=422,
            error_type="validation_error",
            message="Invalid input provided.",
            details={"reason": str(exc)},
        )

    # Placeholder analysis until retrieval and LLM layers are connected.
    return AnalysisResponse(
        match_score=0,
        score_reasoning=(
            f"Ingestion complete. CV chunks: {len(cv_chunks)}, JD chunks: {len(jd_chunks)}. "
            "Full scoring is pending embedding/retrieval/LLM integration."
        ),
        skill_match=SkillMatch(),
        cv_suggestions=[],
        keyword_gaps=[],
        overall_verdict="Ingestion successful; analysis pipeline not fully connected yet.",
        model_used=settings.claude_model,
    )


@app.post(
    "/score",
    responses={200: {"content": {"application/json": {}}}, 400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def score(
    cv_text: str | None = Form(default=None),
    jd_text: str | None = Form(default=None),
    cv_file: UploadFile | None = File(default=None),
    jd_file: UploadFile | None = File(default=None),
) -> dict[str, int] | JSONResponse:
    try:
        request_model = _build_request_model(cv_text=cv_text, jd_text=jd_text, top_k=1)
    except ValidationError as exc:
        return _error_response(
            status_code=422,
            error_type="validation_error",
            message="Invalid score request.",
            details={"reason": str(exc)},
        )

    try:
        if cv_file is not None:
            cv_pages = extract_pdf_pages_from_bytes(await cv_file.read())
        else:
            cv_pages = resolve_document_text(source="cv", text=request_model.cv_text)

        if jd_file is not None:
            jd_pages = extract_pdf_pages_from_bytes(await jd_file.read())
        else:
            jd_pages = resolve_document_text(source="jd", text=request_model.jd_text)

        cv_chunks = chunk_document_pages(cv_pages, source="cv")
        jd_chunks = chunk_document_pages(jd_pages, source="jd")
    except IngestionError as exc:
        return _error_response(
            status_code=400,
            error_type="ingestion_error",
            message="Document ingestion failed.",
            details={"reason": str(exc)},
        )
    except ValueError as exc:
        return _error_response(
            status_code=422,
            error_type="validation_error",
            message="Invalid input provided.",
            details={"reason": str(exc)},
        )

    # Temporary heuristic score until ranking/scoring is implemented.
    heuristic_score = min(100, max(0, int((len(cv_chunks) / max(len(jd_chunks), 1)) * 20)))
    return {"match_score": heuristic_score}
