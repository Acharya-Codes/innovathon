from __future__ import annotations

from typing import Sequence

from src.embeddings.embedding_generator import embed_job
from src.vector_db.job_storage import search_jobs


def search_similar_jobs(
    candidate_resume: object,
    collection_name: str = "internships",
    limit: int = 10,
    score_threshold: float | None = None,
) -> list[dict]:
    query_vector = embed_job(candidate_resume)
    if not query_vector:
        return []
    return search_jobs(query_vector=query_vector, collection_name=collection_name, limit=limit, score_threshold=score_threshold)
