from __future__ import annotations

from typing import Any

from src.matching.skill_matcher import compare_skill_sets


def calculate_match_score(
    job: dict[str, Any] | Any,
    candidate_skills: list[str] | dict | None,
    similarity_score: float | None = None,
) -> dict[str, Any]:
    if hasattr(job, "model_dump"):
        job = job.model_dump()
    if not isinstance(job, dict):
        raise TypeError("Job must be a dict-like object or pydantic model")

    required_skills = job.get("required_skills") or []
    preferred_skills = job.get("preferred_skills") or []
    comparison = compare_skill_sets(required_skills, preferred_skills, candidate_skills)

    required_coverage = float(comparison["required_coverage"])
    preferred_coverage = float(comparison["preferred_coverage"])

    if similarity_score is None:
        semantic_component = 0.0
    else:
        semantic_component = max(0.0, min(1.0, float(similarity_score)))

    overall_score = 0.40 * semantic_component + 0.50 * required_coverage + 0.10 * preferred_coverage
    overall_score = round(max(0.0, min(1.0, overall_score)), 4)

    return {
        "overall_score": overall_score,
        "semantic_similarity": round(semantic_component, 4),
        "required_coverage": required_coverage,
        "preferred_coverage": preferred_coverage,
        "matched_required": comparison["matched_required"],
        "matched_preferred": comparison["matched_preferred"],
        "missing_required": comparison["missing_required"],
        "missing_preferred": comparison["missing_preferred"],
        "required_match_count": comparison["required_match_count"],
        "preferred_match_count": comparison["preferred_match_count"],
        "summary": (
            f"Matched {comparison['required_match_count']} required skill(s) and "
            f"{comparison['preferred_match_count']} preferred skill(s)."
        ),
    }
