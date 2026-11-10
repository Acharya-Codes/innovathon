from __future__ import annotations

from typing import Any, Iterable

from .jd_schema import InternshipJobDescription


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(";") if part.strip()]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def validate_job_description(raw_data: Any) -> InternshipJobDescription:
    """Validate raw JD-like data and return a structured pydantic model."""
    if isinstance(raw_data, InternshipJobDescription):
        return raw_data

    if isinstance(raw_data, dict):
        normalized = dict(raw_data)
        for field_name in ("responsibilities", "required_skills", "preferred_skills"):
            if field_name in normalized:
                normalized[field_name] = _as_list(normalized[field_name])
        return InternshipJobDescription.model_validate(normalized)

    if hasattr(raw_data, "model_dump"):
        return validate_job_description(raw_data.model_dump())

    raise TypeError("Unsupported job description type for validation")
