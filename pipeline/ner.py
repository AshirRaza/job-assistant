"""NER and skill extraction utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass

import spacy
from transformers import pipeline

from utils.config import get_settings
from utils.helpers import setup_logger

logger = setup_logger(__name__)

# Lightweight starter lexicon to supplement spaCy entities for tech terms.
KNOWN_SKILLS = {
    "python",
    "java",
    "javascript",
    "typescript",
    "sql",
    "docker",
    "kubernetes",
    "terraform",
    "aws",
    "gcp",
    "azure",
    "fastapi",
    "django",
    "flask",
    "langchain",
    "llamaindex",
    "pytorch",
    "tensorflow",
    "pandas",
    "numpy",
    "chroma",
    "faiss",
    "rag",
    "graphql",
    "rest",
}

SKILL_ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "nodejs": "node.js",
    "node js": "node.js",
    "postgres": "postgresql",
    "postgre": "postgresql",
    "k8s": "kubernetes",
    "py": "python",
    "tf": "terraform",
    "gcp cloud": "gcp",
    "amazon web services": "aws",
}

NOISE_TERMS = {
    "experience",
    "team",
    "project",
    "development",
    "engineer",
    "company",
    "responsible",
    "requirement",
    "knowledge",
    "ability",
}


@dataclass(frozen=True)
class NERResult:
    """Output of skill/entity extraction."""

    entities: list[str]
    normalized_skills: list[str]
    categories: dict[str, str] | None = None


class SkillExtractor:
    """spaCy-based entity and skill extractor with optional zero-shot categories."""

    def __init__(self) -> None:
        settings = get_settings()
        self._nlp = spacy.load(settings.spacy_model)
        self._zero_shot = None

    @classmethod
    def _normalize_skill(cls, value: str) -> str:
        value = value.strip().lower()
        value = value.replace("&", " and ")
        value = re.sub(r"[/_]", " ", value)
        value = re.sub(r"\s+", " ", value)
        value = value.strip(".:,;()[]{}")
        value = cls._apply_alias(value)
        return value

    @staticmethod
    def _apply_alias(value: str) -> str:
        return SKILL_ALIASES.get(value, value)

    @staticmethod
    def _is_valid_skill_candidate(value: str) -> bool:
        if not value:
            return False
        if value in NOISE_TERMS:
            return False
        if value.isdigit():
            return False
        if len(value) < 2 or len(value) > 40:
            return False
        if not re.search(r"[a-zA-Z]", value):
            return False
        return True

    def _extract_spacy_entities(self, text: str) -> list[str]:
        doc = self._nlp(text)
        skill_like_labels = {"ORG", "PRODUCT", "LANGUAGE"}
        entities = [ent.text.strip() for ent in doc.ents if ent.label_ in skill_like_labels]

        # Add noun chunks that are likely to represent tools/skills.
        for chunk in doc.noun_chunks:
            candidate = chunk.text.strip()
            if chunk.root.pos_ not in {"PROPN", "NOUN"}:
                continue
            if re.fullmatch(r"(the|a|an)\s+.+", candidate.lower()):
                candidate = re.sub(r"^(the|a|an)\s+", "", candidate, flags=re.IGNORECASE)
            if 2 <= len(candidate) <= 40 and re.search(r"[A-Za-z]", candidate):
                entities.append(candidate)
        return entities

    def _extract_known_skills(self, text: str) -> list[str]:
        lower_text = text.lower()
        found: list[str] = []
        for skill in KNOWN_SKILLS:
            if re.search(rf"\b{re.escape(skill)}\b", lower_text):
                found.append(skill)
        return found

    def extract(self, text: str) -> NERResult:
        entities = self._extract_spacy_entities(text)
        combined = entities + self._extract_known_skills(text)
        normalized_set: set[str] = set()
        for item in combined:
            normalized = self._normalize_skill(item)
            if self._is_valid_skill_candidate(normalized):
                normalized_set.add(normalized)

        # Keep deterministic order and prioritize known skill lexicon when present.
        normalized = sorted(normalized_set, key=lambda x: (x not in KNOWN_SKILLS, x))
        raw_entities = sorted({item.strip() for item in entities if item.strip()})
        return NERResult(entities=raw_entities, normalized_skills=normalized, categories=None)

    def categorize_skills_zero_shot(
        self,
        skills: list[str],
        candidate_labels: list[str] | None = None,
    ) -> dict[str, str]:
        """Optional zero-shot categorization for extracted skills."""
        if not skills:
            return {}

        labels = candidate_labels or [
            "machine learning",
            "backend",
            "frontend",
            "devops",
            "data engineering",
            "cloud",
            "database",
        ]
        if self._zero_shot is None:
            self._zero_shot = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
            )

        categories: dict[str, str] = {}
        for skill in skills:
            try:
                result = self._zero_shot(skill, labels)
                top_label = result["labels"][0] if result.get("labels") else "other"
                categories[skill] = top_label
            except Exception as exc:  # pragma: no cover
                logger.warning("Zero-shot classification failed for '%s': %s", skill, exc)
                categories[skill] = "other"
        return categories


_skill_extractor: SkillExtractor | None = None


def get_skill_extractor() -> SkillExtractor:
    """Get a singleton skill extractor instance."""
    global _skill_extractor
    if _skill_extractor is None:
        _skill_extractor = SkillExtractor()
    return _skill_extractor
