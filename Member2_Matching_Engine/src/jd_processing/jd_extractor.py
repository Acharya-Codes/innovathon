from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from src.jd_processing.jd_schema import InternshipJobDescription
from src.jd_processing.jd_validator import validate_job_description

logger = logging.getLogger(__name__)


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.IGNORECASE)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


def _configure_client(api_key: str | None = None) -> genai.Client:
    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_key:
        raise ValueError("GEMINI_API_KEY is not configured")
    return genai.Client(api_key=resolved_key)


def extract_job_from_text(
    raw_text: str,
    *,
    model_name: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float = 60.0,
    max_retries: int = 3,
) -> InternshipJobDescription:
    """Extract a structured internship record from raw JD text using Gemini."""
    if not raw_text or not raw_text.strip():
        return InternshipJobDescription()

    resolved_model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    try:
        client = _configure_client(api_key=api_key)
        prompt = (
            "You are an extraction engine. Return ONLY valid JSON matching this schema: "
            "{job_id:string, title:string, company:string, location:string, work_mode:string, "
            "duration:string, eligibility:string, responsibilities:list[string], "
            "required_skills:list[string], preferred_skills:list[string], description:string}. "
            "Extract the internship details from the text below. Do not invent missing information. "
            "Do not classify a preferred skill as required unless the JD explicitly indicates it is required. "
            "If a field is missing, output an empty string or empty list as appropriate.\n\n"
            f"{raw_text}"
        )
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                http_options={
                    "timeout": int(timeout_seconds * 1000),
                    "retry_options": {"attempts": max_retries + 1},
                },
            ),
        )
        text = getattr(response, "text", None)
        if not text:
            if response.candidates:
                text = response.candidates[0].content.parts[0].text
        payload = _extract_json_payload(text)
        data = payload if isinstance(payload, dict) else {"description": str(payload)}
        return validate_job_description(data)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("JD extraction failed: %s", exc)
        return InternshipJobDescription(description=raw_text[:500])
    except Exception as exc:  # pragma: no cover - depends on external API state
        logger.warning("Gemini API error while extracting JD: %s", exc)
        return InternshipJobDescription(description=raw_text[:500])


def save_processed_job(job: InternshipJobDescription, output_dir: str | os.PathLike[str] = "data/processed_jobs") -> Path:
    job_dir = Path(output_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    output_path = job_dir / f"{job.job_id}.json"
    output_path.write_text(json.dumps(job.model_dump(), indent=2), encoding="utf-8")
    return output_path
