"""Claude API calls and prompt handling."""

from __future__ import annotations

from anthropic import Anthropic
from pydantic import ValidationError

from models.schemas import AnalysisResponse, CVSuggestion, SkillMatch
from utils.config import get_settings
from utils.helpers import safe_parse_json, setup_logger

logger = setup_logger(__name__)


class LLMError(Exception):
    """Raised when Claude integration fails."""


class ClaudeAnalyzer:
    """Handles prompt construction, Claude calls, and schema validation."""

    def __init__(self) -> None:
        settings = get_settings()
        self.offline_mode = settings.offline_mode
        self.model_name = settings.claude_model
        self._client: Anthropic | None = None
        if not self.offline_mode:
            if not settings.anthropic_api_key:
                raise LLMError("ANTHROPIC_API_KEY is missing.")
            self._client = Anthropic(api_key=settings.anthropic_api_key)
        else:
            logger.info("Offline mode enabled: using heuristic analyzer without Anthropic API.")

    @staticmethod
    def _build_prompt(
        *,
        jd_text: str,
        cv_context: str,
        cv_skills: list[str],
        jd_skills: list[str],
    ) -> str:
        return (
            "You are an expert technical recruiter and resume optimizer.\n"
            "Analyze the candidate CV against the job description.\n"
            "Return ONLY valid JSON that matches this schema exactly:\n"
            "{\n"
            '  "match_score": integer 0-100,\n'
            '  "score_reasoning": string,\n'
            '  "skill_match": {\n'
            '    "matched_skills": string[],\n'
            '    "missing_skills": string[],\n'
            '    "transferable_skills": string[]\n'
            "  },\n"
            '  "cv_suggestions": [\n'
            "    {\n"
            '      "section": string,\n'
            '      "current": string,\n'
            '      "suggested": string,\n'
            '      "reasoning": string\n'
            "    }\n"
            "  ],\n"
            '  "keyword_gaps": string[],\n'
            '  "overall_verdict": string,\n'
            '  "model_used": string\n'
            "}\n"
            "Rules:\n"
            "1) Output must be raw JSON only (no markdown, no backticks).\n"
            "2) Ensure all required keys are present.\n"
            "3) Keep suggestions concise and actionable.\n\n"
            f"Job Description:\n{jd_text}\n\n"
            f"Retrieved CV Context:\n{cv_context}\n\n"
            f"Extracted CV Skills:\n{cv_skills}\n\n"
            f"Extracted JD Skills:\n{jd_skills}\n"
        )

    @staticmethod
    def _extract_text_content(response: object) -> str:
        content = getattr(response, "content", None)
        if not content:
            raise LLMError("Claude returned an empty response.")

        parts: list[str] = []
        for item in content:
            if getattr(item, "type", None) == "text":
                text = getattr(item, "text", "")
                if text:
                    parts.append(text)
        if not parts:
            raise LLMError("Claude response had no text content.")
        text = "\n".join(parts).strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return text

    def analyze(
        self,
        *,
        jd_text: str,
        cv_context: str,
        cv_skills: list[str],
        jd_skills: list[str],
    ) -> AnalysisResponse:
        if self.offline_mode:
            return self._offline_analyze(
                jd_text=jd_text,
                cv_context=cv_context,
                cv_skills=cv_skills,
                jd_skills=jd_skills,
            )

        prompt = self._build_prompt(
            jd_text=jd_text,
            cv_context=cv_context,
            cv_skills=cv_skills,
            jd_skills=jd_skills,
        )

        try:
            assert self._client is not None
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=2000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = self._extract_text_content(response)
            parsed = safe_parse_json(raw_text)
            parsed["model_used"] = parsed.get("model_used") or self.model_name
            return AnalysisResponse.model_validate(parsed)
        except ValidationError as exc:
            raise LLMError(f"Claude output failed schema validation: {exc}") from exc
        except ValueError as exc:
            raise LLMError(f"Claude output parsing failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover
            raise LLMError(f"Claude API call failed: {exc}") from exc

    def _offline_analyze(
        self,
        *,
        jd_text: str,
        cv_context: str,
        cv_skills: list[str],
        jd_skills: list[str],
    ) -> AnalysisResponse:
        cv_set = set(cv_skills)
        jd_set = set(jd_skills)
        matched = sorted(cv_set & jd_set)
        missing = sorted(jd_set - cv_set)
        transferable = sorted(item for item in cv_set - jd_set if len(item) > 3)[:5]

        denominator = max(len(jd_set), 1)
        match_score = int((len(matched) / denominator) * 100)
        verdict = (
            "Strong alignment for core requirements."
            if match_score >= 70
            else "Moderate fit with several missing requirements."
            if match_score >= 40
            else "Low alignment; substantial skill gaps remain."
        )
        suggestion_text = (
            "Add an explicit skills section and include missing JD keywords naturally in project bullets."
        )
        if missing:
            suggestion_text = (
                "Highlight adjacent experience and add measurable outcomes for: "
                + ", ".join(missing[:5])
                + "."
            )

        return AnalysisResponse(
            match_score=max(0, min(100, match_score)),
            score_reasoning=(
                f"Offline heuristic mode based on skill overlap. "
                f"Matched {len(matched)} of {len(jd_set)} extracted JD skills."
            ),
            skill_match=SkillMatch(
                matched_skills=matched,
                missing_skills=missing,
                transferable_skills=transferable,
            ),
            cv_suggestions=[
                CVSuggestion(
                    section="Skills/Projects",
                    current=cv_context[:200] if cv_context else "N/A",
                    suggested=suggestion_text,
                    reasoning="Improves keyword alignment and ATS discoverability in offline mode.",
                )
            ],
            keyword_gaps=missing[:10],
            overall_verdict=verdict,
            model_used="offline-heuristic-v1",
        )


_claude_analyzer: ClaudeAnalyzer | None = None


def get_claude_analyzer() -> ClaudeAnalyzer:
    """Get a singleton Claude analyzer."""
    global _claude_analyzer
    if _claude_analyzer is None:
        _claude_analyzer = ClaudeAnalyzer()
    return _claude_analyzer
