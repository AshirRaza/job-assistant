from pipeline.ner import SkillExtractor


def test_skill_alias_normalization():
    extractor = SkillExtractor()
    result = extractor.extract("Experienced in JS, TS, k8s, Postgres and AWS.")
    normalized = set(result.normalized_skills)
    assert "javascript" in normalized
    assert "typescript" in normalized
    assert "kubernetes" in normalized
    assert "postgresql" in normalized
    assert "aws" in normalized


def test_noise_terms_filtered():
    extractor = SkillExtractor()
    result = extractor.extract(
        "Strong team experience and project development ability with Python and Docker."
    )
    normalized = set(result.normalized_skills)
    assert "python" in normalized
    assert "docker" in normalized
    assert "team" not in normalized
    assert "experience" not in normalized
