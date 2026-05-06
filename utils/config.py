"""Environment-backed runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    anthropic_api_key: str
    claude_model: str
    embedding_model: str
    reranker_model: str
    spacy_model: str
    chroma_persist_dir: Path
    use_reranker: bool
    offline_mode: bool
    max_context_tokens: int
    summarize_overflow: bool
    overflow_summary_tokens: int


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached app settings loaded from environment."""
    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        reranker_model=os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        spacy_model=os.getenv("SPACY_MODEL", "en_core_web_sm"),
        chroma_persist_dir=Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")),
        use_reranker=_parse_bool(os.getenv("USE_RERANKER"), default=True),
        offline_mode=_parse_bool(os.getenv("OFFLINE_MODE"), default=False),
        max_context_tokens=_parse_int(os.getenv("MAX_CONTEXT_TOKENS"), default=2200),
        summarize_overflow=_parse_bool(os.getenv("SUMMARIZE_OVERFLOW"), default=True),
        overflow_summary_tokens=_parse_int(os.getenv("OVERFLOW_SUMMARY_TOKENS"), default=320),
    )
