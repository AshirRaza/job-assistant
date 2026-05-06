from __future__ import annotations

import time
from uuid import uuid4

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.responses import JSONResponse

from models.schemas import AnalysisRequest, AnalysisResponse, ErrorResponse
from pipeline.ingestion import (
    IngestionError,
    extract_pdf_pages_from_bytes,
)
from pipeline.rag import RAGPipelineError, get_rag_pipeline
from utils.config import Settings, get_settings
from utils.helpers import setup_logger

app = FastAPI(title="CV Analyzer API", version="0.1.0")
logger = setup_logger()
settings: Settings = get_settings()
_rag_pipeline = None


def _error_response(
    *,
    status_code: int,
    error_type: str,
    message: str,
    details: dict[str, str] | None = None,
) -> JSONResponse:
    payload = ErrorResponse(error=error_type, message=message, details=details)
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _extract_validation_details(exc: Exception) -> dict[str, str]:
    if hasattr(exc, "errors"):
        try:
            details = getattr(exc, "errors")()
            return {"errors": str(details)}
        except Exception:
            pass
    return {"reason": str(exc)}


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


def _has_uploaded_file(file: UploadFile | None) -> bool:
    return file is not None and bool(file.filename)


def _pages_to_text(pages: list[dict[str, int | str]]) -> str:
    return "\n\n".join(str(page["text"]) for page in pages)


def _get_rag_pipeline():
    global _rag_pipeline
    try:
        if _rag_pipeline is None:
            _rag_pipeline = get_rag_pipeline()
        return _rag_pipeline
    except Exception as exc:  # pragma: no cover
        raise RAGPipelineError(f"Failed to initialize RAG pipeline: {exc}") from exc


async def _resolve_input_text(text: str | None, file: UploadFile | None) -> str | None:
    if _has_uploaded_file(file):
        pages = extract_pdf_pages_from_bytes(await file.read())
        return _pages_to_text(pages)
    if text is not None:
        return text
    return None


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid4()))
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["x-request-id"] = request_id
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    return _error_response(
        status_code=422,
        error_type="validation_error",
        message="Invalid request payload.",
        details=_extract_validation_details(exc),
    )


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(
    _request: Request, exc: ValidationError
) -> JSONResponse:
    return _error_response(
        status_code=422,
        error_type="validation_error",
        message="Validation failed.",
        details=_extract_validation_details(exc),
    )


@app.exception_handler(IngestionError)
async def ingestion_exception_handler(_request: Request, exc: IngestionError) -> JSONResponse:
    return _error_response(
        status_code=400,
        error_type="ingestion_error",
        message="Document ingestion failed.",
        details={"reason": str(exc)},
    )


@app.exception_handler(RAGPipelineError)
async def pipeline_exception_handler(_request: Request, exc: RAGPipelineError) -> JSONResponse:
    return _error_response(
        status_code=400,
        error_type="pipeline_error",
        message="RAG pipeline failed.",
        details={"reason": str(exc)},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)
    return _error_response(
        status_code=500,
        error_type="pipeline_error",
        message="Unexpected server error.",
        details={"reason": str(exc)},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    logger.debug("Health check received.")
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, object]:
    checks: dict[str, dict[str, object]] = {
        "settings_loaded": {"ok": True},
        "anthropic_key_present": {"ok": bool(settings.anthropic_api_key)},
    }
    try:
        _get_rag_pipeline()
        checks["rag_pipeline_init"] = {"ok": True}
    except Exception as exc:
        checks["rag_pipeline_init"] = {"ok": False, "error": str(exc)}

    overall = all(item.get("ok", False) for item in checks.values())
    return {"status": "ready" if overall else "degraded", "checks": checks}


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
    resolved_cv_text = await _resolve_input_text(cv_text, cv_file)
    resolved_jd_text = await _resolve_input_text(jd_text, jd_file)
    request_model = _build_request_model(
        cv_text=resolved_cv_text,
        jd_text=resolved_jd_text,
        top_k=top_k,
    )
    return _get_rag_pipeline().run(request_model)


@app.post(
    "/score",
    response_model=None,
    responses={200: {"content": {"application/json": {}}}, 400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def score(
    cv_text: str | None = Form(default=None),
    jd_text: str | None = Form(default=None),
    cv_file: UploadFile | None = File(default=None),
    jd_file: UploadFile | None = File(default=None),
) -> dict[str, int] | JSONResponse:
    resolved_cv_text = await _resolve_input_text(cv_text, cv_file)
    resolved_jd_text = await _resolve_input_text(jd_text, jd_file)
    request_model = _build_request_model(
        cv_text=resolved_cv_text,
        jd_text=resolved_jd_text,
        top_k=1,
    )
    analysis = _get_rag_pipeline().run(request_model)
    return {"match_score": analysis.match_score}
