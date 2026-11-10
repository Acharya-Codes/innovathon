from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InternshipJobDescription(BaseModel):
    """Structured internship job description extracted from raw text."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    job_id: str = Field(default="unknown-job")
    title: str = ""
    company: str = ""
    location: str = ""
    work_mode: str = ""
    duration: str = ""
    eligibility: str = ""
    responsibilities: List[str] = Field(default_factory=list)
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    description: str = ""

    @field_validator("job_id", mode="before")
    @classmethod
    def _coerce_job_id(cls, value: Any) -> str:
        if value is None:
            return "unknown-job"
        return str(value).strip() or "unknown-job"

    @field_validator("title", "company", "location", "work_mode", "duration", "eligibility", "description", mode="before")
    @classmethod
    def _clean_text_field(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    @field_validator("responsibilities", "required_skills", "preferred_skills", mode="before")
    @classmethod
    def _clean_list_field(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            items = [part.strip() for part in value.split(";")]
            return [item for item in items if item]
        if isinstance(value, (list, tuple, set)):
            cleaned: List[str] = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    cleaned.append(text)
            return cleaned
        return [str(value).strip()] if str(value).strip() else []
