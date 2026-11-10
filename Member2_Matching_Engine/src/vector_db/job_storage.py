from __future__ import annotations

import hashlib
from typing import Any, Iterable, Sequence

from qdrant_client.http import models

from src.vector_db.qdrant_connection import COLLECTION_NAME, ensure_collection


def _point_id_from_job_id(job_id: str | int) -> int:
    if isinstance(job_id, int):
        return job_id
    digest = hashlib.sha1(str(job_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _coerce_payload(job: dict[str, Any]) -> dict[str, Any]:
    payload = dict(job)
    payload["job_id"] = str(payload.get("job_id") or payload.get("id") or "unknown-job")
    payload["required_skills"] = list(payload.get("required_skills") or [])
    payload["preferred_skills"] = list(payload.get("preferred_skills") or [])
    payload["responsibilities"] = list(payload.get("responsibilities") or [])
    return payload


def upsert_jobs(jobs: Iterable[dict[str, Any]], collection_name: str = COLLECTION_NAME) -> list[str]:
    """Upsert internship records and vector payloads into Qdrant."""
    jobs_list = list(jobs)
    if not jobs_list:
        return []

    client = ensure_collection(collection_name=collection_name)
    points: list[models.PointStruct] = []
    for job in jobs_list:
        payload = _coerce_payload(job)
        vector = payload.get("embedding") or payload.get("vector")
        if vector is None:
            raise ValueError(f"Job {payload.get('job_id')} is missing an embedding vector")
        point_id = _point_id_from_job_id(payload["job_id"])
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={key: value for key, value in payload.items() if key not in {"embedding", "vector"}},
            )
        )

    if not points:
        return []

    client.upsert(collection_name=collection_name, points=points)
    return [str(point.id) for point in points]


def retrieve_jobs(job_ids: Sequence[str] | None = None, collection_name: str = COLLECTION_NAME, limit: int = 50) -> list[dict[str, Any]]:
    client = ensure_collection(collection_name=collection_name)
    if job_ids:
        point_ids = [_point_id_from_job_id(job_id) for job_id in job_ids]
        results = client.retrieve(collection_name=collection_name, ids=point_ids, with_payload=True, with_vectors=True)
        return [point.model_dump() if hasattr(point, "model_dump") else dict(point) for point in results]

    results = client.scroll(collection_name=collection_name, limit=limit, with_payload=True, with_vectors=True)
    returned = results[0]
    return [point.model_dump() if hasattr(point, "model_dump") else dict(point) for point in returned]


def search_jobs(
    query_vector: Sequence[float],
    collection_name: str = COLLECTION_NAME,
    limit: int = 10,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    client = ensure_collection(collection_name=collection_name)
    response = client.query_points(
        collection_name=collection_name,
        query=list(query_vector),
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
        with_vectors=False,
    )
    matches: list[dict[str, Any]] = []
    for item in response.points:
        payload = dict(item.payload or {})
        payload["score"] = item.score
        payload["id"] = str(item.id)
        matches.append(payload)
    return matches
