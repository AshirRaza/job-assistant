"""Shared utility helpers."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generator


def setup_logger(name: str = "cv_analyzer", level: int = logging.INFO) -> logging.Logger:
    """Create or return a configured logger instance."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def timed(label: str | None = None, logger: logging.Logger | None = None) -> Callable:
    """Decorator to measure and log function runtime."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                if logger:
                    logger.info("%s took %.3fs", label or func.__name__, elapsed)

        return wrapper

    return decorator


@contextmanager
def timed_block(
    label: str, logger: logging.Logger | None = None
) -> Generator[None, None, None]:
    """Context manager to measure and log code block runtime."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if logger:
            logger.info("%s took %.3fs", label, elapsed)


def safe_parse_json(value: str) -> dict[str, Any]:
    """Safely parse a JSON object string."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON payload.") from exc

    if not isinstance(parsed, dict):
        raise ValueError("JSON must be an object at the top level.")
    return parsed


def validate_text(
    value: str | None, field_name: str, *, min_length: int = 1, max_length: int = 100_000
) -> str:
    """Validate and normalize text input."""
    if value is None:
        raise ValueError(f"{field_name} is required.")

    normalized = value.strip()
    if len(normalized) < min_length:
        raise ValueError(f"{field_name} must be at least {min_length} characters.")
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters.")
    return normalized


def validate_file_path(path: str | Path, *, allowed_suffixes: set[str] | None = None) -> Path:
    """Validate file path existence and extension."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    if allowed_suffixes is not None:
        suffix = file_path.suffix.lower()
        normalized_allowed = {item.lower() for item in allowed_suffixes}
        if suffix not in normalized_allowed:
            raise ValueError(
                f"Unsupported file type '{suffix}'. Allowed: {sorted(normalized_allowed)}"
            )

    return file_path
