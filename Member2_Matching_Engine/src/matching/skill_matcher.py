from __future__ import annotations

from typing import Iterable

from src.normalization.skill_normalizer import normalize_skills


def _coerce_iterable(value: Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return list(value.values())
    return list(value)


def extract_resume_skills(resume: object) -> list[str]:
    if resume is None:
        return []
    if isinstance(resume, str):
        return normalize_skills([resume])
    if not hasattr(resume, "get"):
        return normalize_skills(_coerce_iterable(resume))

    skills = list(_coerce_iterable(resume.get("skills")))
    for experience in resume.get("experience") or []:
        if not isinstance(experience, dict):
            continue
        skills.extend(_coerce_iterable(experience.get("tech")))
        skills.extend(_coerce_iterable(experience.get("description")))
        skills.extend(_coerce_iterable(experience.get("role")))
    for project in resume.get("projects") or []:
        if not isinstance(project, dict):
            continue
        skills.extend(_coerce_iterable(project.get("tech")))
        skills.extend(_coerce_iterable(project.get("description")))
        skills.extend(_coerce_iterable(project.get("name")))
    return normalize_skills(skills)


def compare_skill_sets(
    required_skills: Iterable[str] | None,
    preferred_skills: Iterable[str] | None,
    candidate_skills: Iterable[str] | None | dict,
) -> dict[str, object]:
    required = set(normalize_skills(required_skills))
    preferred = set(normalize_skills(preferred_skills))
    if isinstance(candidate_skills, dict):
        candidate = set(extract_resume_skills(candidate_skills))
    else:
        candidate = set(normalize_skills(candidate_skills))

    matched_required = sorted(required & candidate)
    matched_preferred = sorted(preferred & candidate)
    missing_required = sorted(required - candidate)
    missing_preferred = sorted(preferred - candidate)

    required_coverage = 1.0 if not required else len(matched_required) / len(required)
    preferred_coverage = 1.0 if not preferred else len(matched_preferred) / len(preferred)

    return {
        "required_skills": sorted(required),
        "preferred_skills": sorted(preferred),
        "candidate_skills": sorted(candidate),
        "matched_required": matched_required,
        "matched_preferred": matched_preferred,
        "missing_required": missing_required,
        "missing_preferred": missing_preferred,
        "required_match_count": len(matched_required),
        "preferred_match_count": len(matched_preferred),
        "required_coverage": required_coverage,
        "preferred_coverage": preferred_coverage,
    }
