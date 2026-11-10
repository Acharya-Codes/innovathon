from __future__ import annotations

import math
import uuid
from typing import Any, Iterable

from src.embeddings.embedding_generator import embed_job
from src.evaluation.synthetic_dataset import generate_synthetic_dataset
from src.matching.skill_matcher import compare_skill_sets
from src.vector_db.job_storage import search_jobs, upsert_jobs
from src.vector_db.qdrant_connection import ensure_collection, get_qdrant_client

CONFIGURATIONS = {
    "Current": {"semantic": 0.40, "required": 0.50, "preferred": 0.10},
    "Skill-heavy": {"semantic": 0.25, "required": 0.65, "preferred": 0.10},
    "Semantic-heavy": {"semantic": 0.60, "required": 0.30, "preferred": 0.10},
}


def _normalized_score(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _weighted_score(job: dict[str, Any], candidate_resume: dict[str, Any], similarity_score: float | None, weights: dict[str, float]) -> float:
    required_skills = job.get("required_skills") or []
    preferred_skills = job.get("preferred_skills") or []
    comparison = compare_skill_sets(required_skills, preferred_skills, candidate_resume)
    required_coverage = float(comparison["required_coverage"])
    preferred_coverage = float(comparison["preferred_coverage"])
    semantic = _normalized_score(similarity_score)
    return (
        weights.get("semantic", 0.0) * semantic
        + weights.get("required", 0.0) * required_coverage
        + weights.get("preferred", 0.0) * preferred_coverage
    )


def _precision_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    if not retrieved_ids:
        return 0.0
    relevant_in_top_k = sum(1 for item in retrieved_ids[:k] if item in relevant_ids)
    return relevant_in_top_k / min(k, len(retrieved_ids[:k])) if retrieved_ids[:k] else 0.0


def _recall_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    relevant_in_top_k = sum(1 for item in retrieved_ids[:k] if item in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def _mrr(relevant_ids: set[str], retrieved_ids: list[str]) -> float:
    for index, job_id in enumerate(retrieved_ids, start=1):
        if job_id in relevant_ids:
            return 1.0 / index
    return 0.0


def _dcg_at_k(labels: list[int], k: int) -> float:
    total = 0.0
    for i, label in enumerate(labels[:k], start=1):
        total += ((2 ** label) - 1) / math.log2(i + 1)
    return total


def _ndcg_at_k(relevance_labels: dict[str, int], retrieved_ids: list[str], k: int) -> float:
    ideal = sorted(relevance_labels.values(), reverse=True)[:k]
    dcg = _dcg_at_k([relevance_labels.get(job_id, 0) for job_id in retrieved_ids], k)
    idcg = _dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else dcg / idcg


def _average_coverage_for_results(results: list[dict[str, Any]], resume: dict[str, Any]) -> tuple[float, float]:
    if not results:
        return 0.0, 0.0

    required_values: list[float] = []
    preferred_values: list[float] = []
    for result in results:
        job = result.get("job_data") or {}
        coverage = compare_skill_sets(job.get("required_skills") or [], job.get("preferred_skills") or [], resume)
        required_values.append(float(coverage["required_coverage"]))
        preferred_values.append(float(coverage["preferred_coverage"]))

    return (
        sum(required_values) / len(required_values),
        sum(preferred_values) / len(preferred_values),
    )


def evaluate_configuration(dataset: dict[str, Any], weights: dict[str, float], collection_prefix: str = "eval_match", top_k: int = 5) -> dict[str, Any]:
    resumes = dataset["resumes"]
    internships = dataset["internships"]
    labels_by_resume = dataset["relevance_labels"]

    collection_name = f"{collection_prefix}_{uuid.uuid4().hex[:8]}"
    ensure_collection(collection_name=collection_name, vector_size=384, distance="Cosine")

    vector_jobs = []
    for job in internships:
        payload = dict(job)
        payload["embedding"] = embed_job(job)
        vector_jobs.append(payload)

    upsert_jobs(vector_jobs, collection_name=collection_name)

    try:
        per_resume_metrics: list[dict[str, Any]] = []
        for resume in resumes:
            resume_id = resume["resume_id"]
            query_vector = embed_job(resume)
            search_results = search_jobs(query_vector=query_vector, collection_name=collection_name, limit=max(top_k * 4, 10))
            scored_candidates = []
            for item in search_results:
                job_id = str(item.get("job_id"))
                job = next(job for job in internships if job["job_id"] == job_id)
                semantic_score = float(item.get("score") or 0.0)
                weighted = _weighted_score(job, resume, semantic_score, weights)
                scored_candidates.append({
                    "job_id": job_id,
                    "score": weighted,
                    "semantic_score": semantic_score,
                    "job_data": job,
                })
            scored_candidates.sort(key=lambda entry: entry["score"], reverse=True)
            retrieved_ids = [str(row["job_id"]) for row in scored_candidates[:top_k]]
            relevant_ids = {job_id for job_id, label in labels_by_resume[resume_id].items() if label >= 1}
            relevant_label_by_id = {job_id: int(label) for job_id, label in labels_by_resume[resume_id].items()}

            required_cov, preferred_cov = _average_coverage_for_results(scored_candidates[:top_k], resume)
            per_resume_metrics.append({
                "resume_id": resume_id,
                "retrieved_ids": retrieved_ids,
                "relevant_ids": sorted(relevant_ids),
                "precision@5": _precision_at_k(relevant_ids, retrieved_ids, top_k),
                "recall@5": _recall_at_k(relevant_ids, retrieved_ids, top_k),
                "mrr": _mrr(relevant_ids, retrieved_ids),
                "ndcg@5": _ndcg_at_k(relevant_label_by_id, retrieved_ids, top_k),
                "avg_required_skill_coverage": required_cov,
                "avg_preferred_skill_coverage": preferred_cov,
            })

        aggregate = {
            "config": weights,
            "precision@5": round(sum(item["precision@5"] for item in per_resume_metrics) / len(per_resume_metrics), 4) if per_resume_metrics else 0.0,
            "recall@5": round(sum(item["recall@5"] for item in per_resume_metrics) / len(per_resume_metrics), 4) if per_resume_metrics else 0.0,
            "mrr": round(sum(item["mrr"] for item in per_resume_metrics) / len(per_resume_metrics), 4) if per_resume_metrics else 0.0,
            "ndcg@5": round(sum(item["ndcg@5"] for item in per_resume_metrics) / len(per_resume_metrics), 4) if per_resume_metrics else 0.0,
            "average_required_skill_coverage": round(sum(item["avg_required_skill_coverage"] for item in per_resume_metrics) / len(per_resume_metrics), 4) if per_resume_metrics else 0.0,
            "average_preferred_skill_coverage": round(sum(item["avg_preferred_skill_coverage"] for item in per_resume_metrics) / len(per_resume_metrics), 4) if per_resume_metrics else 0.0,
            "per_resume": per_resume_metrics,
        }
        return aggregate
    finally:
        try:
            get_qdrant_client().delete_collection(collection_name=collection_name)
        except Exception:
            pass


def compare_configurations(dataset: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    report = {
        "seed": dataset.get("seed", 42),
        "resume_count": dataset.get("resume_count", 0),
        "internship_count": dataset.get("internship_count", 0),
        "configurations": {},
        "selection_note": "Synthetic evaluation only; no automatic formula selection based on these results.",
    }
    for name, config in CONFIGURATIONS.items():
        report["configurations"][name] = evaluate_configuration(dataset, config, top_k=top_k)
    return report


def format_evaluation_report(report: dict[str, Any]) -> str:
    lines = [
        "Synthetic matching quality evaluation",
        f"Resumes: {report.get('resume_count', 0)} | Internships: {report.get('internship_count', 0)} | Seed: {report.get('seed', 42)}",
        "",
    ]
    for name, metrics in report.get("configurations", {}).items():
        lines.append(
            f"{name}: P@5={metrics.get('precision@5', 0.0):.4f} | R@5={metrics.get('recall@5', 0.0):.4f} | "
            f"MRR={metrics.get('mrr', 0.0):.4f} | NDCG@5={metrics.get('ndcg@5', 0.0):.4f} | "
            f"ReqCov={metrics.get('average_required_skill_coverage', 0.0):.4f} | PrefCov={metrics.get('average_preferred_skill_coverage', 0.0):.4f}"
        )
    lines.append("")
    lines.append(report.get("selection_note", "Synthetic evaluation only; no automatic formula selection based on these results."))
    return "\n".join(lines)


def evaluate_match_report(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(results)
    if not rows:
        return {"total_jobs": 0, "average_score": 0.0, "top_score": 0.0, "results": []}

    scores = [float(item.get("overall_score", 0.0)) for item in rows]
    return {
        "total_jobs": len(rows),
        "average_score": round(sum(scores) / len(scores), 2),
        "top_score": round(max(scores), 2),
        "results": rows,
    }
