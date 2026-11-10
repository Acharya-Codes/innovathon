from __future__ import annotations

import json
import os
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


class LearningPathItem(BaseModel):
    topic: str
    resource_type: str
    estimated_hours: int = Field(ge=0)


class GapAnalysis(BaseModel):
    missing_skills: list[str]
    resume_critique: str
    learning_path: list[LearningPathItem]


async def generate_gap_analysis(
    resume_json: dict[str, Any],
    top_job_json: dict[str, Any],
) -> GapAnalysis:
    """Compare a resume with a job and return validated, actionable gap analysis."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=api_key)

    prompt = (
        "Act as a technical recruiter. Compare the student's extracted resume JSON "
        "with the internship job description JSON and identify concrete, evidence-based gaps. "
        "List required or important job skills absent from the resume as missing_skills. "
        "For resume_critique, give actionable feedback on how to rewrite project bullet points "
        "to better demonstrate relevant experience for this job; do not invent experience. "
        "For learning_path, recommend practical learning topics and resource types "
        "(for example, course, documentation, or project), with estimated_hours as a "
        "non-negative integer. Return only data conforming to the requested schema.\n\n"
        f"RESUME JSON:\n{json.dumps(resume_json, ensure_ascii=False)}\n\n"
        f"JOB DESCRIPTION JSON:\n{json.dumps(top_job_json, ensure_ascii=False)}"
    )

    candidate_models = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    response_text = None
    last_error = None

    for model_name in candidate_models:
        try:
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GapAnalysis,
                ),
            )
            response_text = response.text
            if response_text:
                break
        except Exception as err:
            last_error = err

    if not response_text:
        raise RuntimeError(f"Gemini gap analysis failed. Last error: {last_error}")

    return GapAnalysis.model_validate_json(response_text)