"""Pydantic request/response and error contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class CVSuggestion(BaseModel):
    """A targeted rewrite recommendation for a CV section."""

    section: str = Field(..., min_length=1, max_length=100)
    current: str = Field(..., min_length=1)
    suggested: str = Field(..., min_length=1)
    reasoning: str = Field(..., min_length=1)


class SkillMatch(BaseModel):
    """Structured skill matching buckets."""

    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    transferable_skills: list[str] = Field(default_factory=list)


class AnalysisRequest(BaseModel):
    """Input contract for CV vs JD analysis requests.

    API handlers can provide plain text directly, or pass file paths
    (typically to temporary files created from uploaded documents).
    """

    cv_text: str | None = None
    jd_text: str | None = None
    cv_file_path: str | None = None
    jd_file_path: str | None = None
    top_k: int = Field(default=8, ge=1, le=50)
    include_suggestions: bool = True

    @model_validator(mode="after")
    def validate_sources(self) -> "AnalysisRequest":
        if not (self.cv_text or self.cv_file_path):
            raise ValueError("Provide either cv_text or cv_file_path.")
        if not (self.jd_text or self.jd_file_path):
            raise ValueError("Provide either jd_text or jd_file_path.")
        return self


class AnalysisResponse(BaseModel):
    """Structured analysis response returned by the LLM layer."""

    match_score: int = Field(..., ge=0, le=100)
    score_reasoning: str = Field(..., min_length=1)
    skill_match: SkillMatch
    cv_suggestions: list[CVSuggestion] = Field(default_factory=list)
    keyword_gaps: list[str] = Field(default_factory=list)
    overall_verdict: str = Field(..., min_length=1)
    model_used: str = Field(..., min_length=1)


class ErrorResponse(BaseModel):
    """Standard error envelope for API failures."""

    error: Literal["validation_error", "ingestion_error", "pipeline_error", "llm_error"]
    message: str = Field(..., min_length=1)
    details: dict[str, str] | None = None
