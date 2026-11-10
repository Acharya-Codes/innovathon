from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from src.jd_processing.jd_schema import InternshipJobDescription

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
_MODEL: SentenceTransformer | None = None


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(model_name)
    return _MODEL


def _collect_text_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(";") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def build_embedding_text(job: InternshipJobDescription | dict[str, Any] | str) -> str:
    if isinstance(job, str):
        return job.strip()

    if isinstance(job, InternshipJobDescription):
        payload = job.model_dump()
    elif isinstance(job, dict):
        payload = job
    else:
        raise TypeError("Unsupported job type for embedding")

    has_resume_shape = any(key in payload for key in ("skills", "education", "experience", "projects"))
    if has_resume_shape:
        skills = _collect_text_items(payload.get("skills"))

        experience_text: list[str] = []
        for item in payload.get("experience") or []:
            if isinstance(item, dict):
                role = str(item.get("role") or "").strip()
                company = str(item.get("company") or "").strip()
                description = " ".join(_collect_text_items(item.get("description")))
                tech = ", ".join(_collect_text_items(item.get("tech")))
                experience_text.extend(filter(None, [role, company, description, tech]))

        projects_text: list[str] = []
        for item in payload.get("projects") or []:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip()
                description = str(item.get("description") or "").strip()
                tech = ", ".join(_collect_text_items(item.get("tech")))
                projects_text.extend(filter(None, [name, description, tech]))

        parts = [
            "Skills: " + ", ".join(skills),
            "Experience: " + "; ".join(experience_text),
            "Projects: " + "; ".join(projects_text),
        ]
        return " ".join(part.strip() for part in parts if part and str(part).strip())

    title = str(payload.get("title") or "").strip()
    description = str(payload.get("description") or "").strip()
    responsibilities = payload.get("responsibilities") or []
    required = payload.get("required_skills") or []
    preferred = payload.get("preferred_skills") or []

    parts = [
        title,
        description,
        ". ".join(str(item).strip() for item in responsibilities if str(item).strip()),
        "Required skills: " + ", ".join(str(item).strip() for item in required if str(item).strip()),
        "Preferred skills: " + ", ".join(str(item).strip() for item in preferred if str(item).strip()),
    ]
    return " ".join(part.strip() for part in parts if part and str(part).strip())


def embed_job(job: InternshipJobDescription | dict[str, Any] | str) -> list[float]:
    text = build_embedding_text(job)
    if not text:
        return []

    model = get_embedding_model()
    vector = model.encode(text, convert_to_numpy=True, normalize_embeddings=False)
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim == 0:
        vector = vector.reshape(1)
    if vector.ndim != 1:
        raise ValueError(f"Embedding vector has unexpected shape: {vector.shape}")
    if vector.size != EMBEDDING_DIMENSION:
        raise ValueError(f"Expected embedding dimension {EMBEDDING_DIMENSION}, got {vector.size}")
    return vector.astype(float).tolist()


def embed_batch_jobs(jobs: Iterable[InternshipJobDescription | dict[str, Any] | str]) -> list[list[float]]:
    items = list(jobs)
    if not items:
        return []

    texts = [build_embedding_text(item) for item in items]
    texts = [text for text in texts if text.strip()]
    if not texts:
        return []

    model = get_embedding_model()
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(f"Expected embedding dimension {EMBEDDING_DIMENSION}, got {arr.shape}")
    return [row.astype(float).tolist() for row in arr]
