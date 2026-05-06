"""Evaluation runner for CV/JD analysis quality."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.schemas import AnalysisRequest
from pipeline.rag import get_rag_pipeline


def _load_dataset(dataset_path: Path) -> list[dict]:
    with dataset_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("Dataset must be a JSON array.")
    return payload


def _safe_overlap(predicted: list[str], expected: list[str]) -> float:
    p = set(predicted)
    e = set(expected)
    if not e:
        return 1.0
    return len(p & e) / len(e)


def _safe_precision(predicted: list[str], expected: list[str]) -> float:
    p = set(predicted)
    e = set(expected)
    if not p:
        return 1.0
    return len(p & e) / len(p)


def run_eval(dataset_path: str = "evaluation/sample_dataset.json") -> dict[str, float]:
    pipeline = get_rag_pipeline()
    examples = _load_dataset(Path(dataset_path))

    score_errors: list[float] = []
    missing_recall: list[float] = []
    matched_precision: list[float] = []

    for item in examples:
        request = AnalysisRequest(
            cv_text=item["cv_text"],
            jd_text=item["jd_text"],
            top_k=int(item.get("top_k", 6)),
        )
        result = pipeline.run(request)
        expected_score = int(item.get("expected_score", result.match_score))
        expected_missing = item.get("expected_missing_skills", [])
        expected_matched = item.get("expected_matched_skills", [])

        score_errors.append(abs(result.match_score - expected_score))
        missing_recall.append(
            _safe_overlap(result.skill_match.missing_skills, expected_missing)
        )
        matched_precision.append(
            _safe_precision(result.skill_match.matched_skills, expected_matched)
        )

    return {
        "avg_score_absolute_error": round(mean(score_errors), 3),
        "avg_missing_skill_recall": round(mean(missing_recall), 3),
        "avg_matched_skill_precision": round(mean(matched_precision), 3),
    }


if __name__ == "__main__":
    metrics = run_eval()
    print(json.dumps(metrics, indent=2))
