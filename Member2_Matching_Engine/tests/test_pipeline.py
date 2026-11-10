from __future__ import annotations

import uuid

import pytest
from qdrant_client.http import models

from src.embeddings.embedding_generator import EMBEDDING_DIMENSION, build_embedding_text, embed_job, get_embedding_model
from src.jd_processing.jd_extractor import extract_job_from_text
from src.jd_processing.jd_schema import InternshipJobDescription
from src.matching.skill_matcher import compare_skill_sets, extract_resume_skills
from src.matching.scoring_engine import calculate_match_score
from src.normalization.skill_normalizer import normalize_skill, normalize_skills
from src.vector_db.job_storage import retrieve_jobs, search_jobs, upsert_jobs
from src.vector_db.qdrant_connection import ensure_collection, get_qdrant_client


STANDARD_RESUME = {
    "skills": ["Python", "Java", "SQL"],
    "education": [{"degree": "B.Tech", "institution": "Example Institute", "year": "2028", "cgpa": "8.5"}],
    "experience": [{"role": "Software Intern", "company": "Example Company", "duration": "2 months", "description": ["Built a small API"], "tech": ["Python", "FastAPI"]}],
    "projects": [{"name": "Student Portal", "description": "A web application", "tech": ["Python", "React"]}],
}


def test_skill_normalization_aliases_and_unknown_preservation():
    assert normalize_skill("JS") == "JavaScript"
    assert normalize_skill("Py") == "Python"
    assert normalize_skill("NodeJS") == "Node.js"
    assert normalize_skill("ReactJS") == "React"
    assert normalize_skill("ML") == "Machine Learning"
    assert normalize_skill("BERT") == "BERT"
    assert normalize_skills([" py ", "JavaScript", "JS", "ML", "BERT"]) == ["Python", "JavaScript", "Machine Learning", "BERT"]


def test_jd_schema_validation_and_missing_fields():
    job = InternshipJobDescription(
        job_id="job-123",
        title="AI Intern",
        company="Acme",
        required_skills=["Python", "SQL"],
        preferred_skills=["Machine Learning", "NLP"],
        responsibilities=["Train models", "Write reports"],
        description="Build models and analyze data.",
    )
    assert job.job_id == "job-123"
    assert job.required_skills == ["Python", "SQL"]
    assert job.preferred_skills == ["Machine Learning", "NLP"]

    validated = InternshipJobDescription.model_validate({"job_id": None, "title": " "})
    assert validated.job_id == "unknown-job"
    assert validated.title == ""


def test_gemini_extractor_handles_valid_and_invalid_responses(monkeypatch):
    class FakeResponse:
        def __init__(self, text):
            self.text = text
            self.candidates = None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.models = self

        def generate_content(self, *args, **kwargs):
            return FakeResponse('{"job_id":"job-42","title":"AI Intern","company":"Acme","location":"Remote","work_mode":"Hybrid","duration":"3 months","eligibility":"Final year students","responsibilities":["Train models"],"required_skills":["Python","SQL"],"preferred_skills":["Machine Learning"],"description":"Build ML models."}')

    monkeypatch.setattr("src.jd_processing.jd_extractor._configure_client", lambda api_key=None: FakeClient())
    job = extract_job_from_text("Some JD text")
    assert job.title == "AI Intern"
    assert job.company == "Acme"
    assert job.required_skills == ["Python", "SQL"]
    assert job.preferred_skills == ["Machine Learning"]
    assert job.eligibility == "Final year students"

    class BadClient:
        def __init__(self, *args, **kwargs):
            self.models = self

        def generate_content(self, *args, **kwargs):
            return FakeResponse("not valid json")

    monkeypatch.setattr("src.jd_processing.jd_extractor._configure_client", lambda api_key=None: BadClient())
    fallback = extract_job_from_text("bad input")
    assert isinstance(fallback, InternshipJobDescription)
    assert fallback.description.startswith("bad input")


def test_embedding_model_loads_once_and_has_expected_dimension():
    model = get_embedding_model()
    job = InternshipJobDescription(
        title="Data Science Intern",
        description="Analyze data and build ML pipelines.",
        responsibilities=["Build dashboards", "Clean datasets"],
        required_skills=["Python", "SQL"],
        preferred_skills=["Machine Learning"],
    )
    text = build_embedding_text(job)
    assert "Data Science Intern" in text
    assert "Python" in text
    assert "Machine Learning" in text

    vector = embed_job(job)
    assert len(vector) == EMBEDDING_DIMENSION
    assert all(isinstance(value, float) for value in vector)
    assert get_embedding_model() is model


def test_member1_resume_compatibility_and_skill_extraction():
    skills = extract_resume_skills(STANDARD_RESUME)
    assert "Python" in skills
    assert "Java" in skills
    assert "FastAPI" in skills
    assert "React" in skills

    comparison = compare_skill_sets(["Python", "SQL"], ["Machine Learning"], STANDARD_RESUME)
    assert comparison["required_match_count"] == 2
    assert comparison["missing_required"] == []
    assert comparison["preferred_coverage"] == 0.0


def test_scoring_formula_and_edge_cases():
    exact_match = calculate_match_score({"required_skills": ["Python", "SQL"], "preferred_skills": ["Machine Learning"]}, ["Python", "SQL", "Machine Learning"], 0.9)
    assert exact_match["overall_score"] == pytest.approx(0.4 * 0.9 + 0.5 * 1.0 + 0.1 * 1.0, rel=1e-4)

    no_required = calculate_match_score({"required_skills": [], "preferred_skills": ["NLP"]}, ["Python"], 0.2)
    assert no_required["required_coverage"] == 1.0
    assert no_required["overall_score"] == pytest.approx(0.4 * 0.2 + 0.5 * 1.0 + 0.1 * 0.0, rel=1e-4)

    empty_resume = calculate_match_score({"required_skills": ["Python"], "preferred_skills": ["SQL"]}, [], 0.0)
    assert empty_resume["required_coverage"] == 0.0
    assert empty_resume["overall_score"] == pytest.approx(0.4 * 0.0 + 0.5 * 0.0 + 0.1 * 0.0, rel=1e-4)

    negative_semantic = calculate_match_score({"required_skills": ["Python"], "preferred_skills": []}, ["Python"], -0.25)
    assert negative_semantic["semantic_similarity"] == 0.0
    assert negative_semantic["overall_score"] == pytest.approx(0.4 * 0.0 + 0.5 * 1.0 + 0.1 * 1.0, rel=1e-4)

    partial_required = calculate_match_score({"required_skills": ["Python", "SQL", "Java"], "preferred_skills": ["NLP"]}, ["Python"], 0.5)
    assert partial_required["required_coverage"] == pytest.approx(1 / 3, rel=1e-4)
    assert 0.0 <= partial_required["overall_score"] <= 1.0


def test_live_qdrant_integration_and_cleanup():
    collection_name = f"internships_test_{uuid.uuid4().hex[:8]}"
    client = get_qdrant_client()
    ensure_collection(collection_name=collection_name, vector_size=384, distance="Cosine")

    jobs = [
        {
            "job_id": "job-q-1",
            "title": "ML Intern",
            "company": "Quest Labs",
            "required_skills": ["Python", "Machine Learning"],
            "preferred_skills": ["SQL"],
            "description": "Build ML pipelines.",
            "embedding": [0.0] * 384,
        },
        {
            "job_id": "job-q-2",
            "title": "Data Intern",
            "company": "Bluepeak",
            "required_skills": ["Python", "SQL"],
            "preferred_skills": ["Tableau"],
            "description": "Analyze datasets.",
            "embedding": [0.1] * 384,
        },
    ]

    upsert_jobs(jobs, collection_name=collection_name)
    stored = retrieve_jobs(["job-q-1"], collection_name=collection_name)
    assert stored and stored[0]["payload"]["job_id"] == "job-q-1"

    search_result = search_jobs([0.0] * 384, collection_name=collection_name, limit=5)
    assert search_result
    assert any(item["job_id"] == "job-q-1" for item in search_result)

    updated = {**jobs[0], "description": "Updated ML pipeline work.", "embedding": [0.0] * 384}
    upsert_jobs([updated], collection_name=collection_name)
    stored_updated = retrieve_jobs(["job-q-1"], collection_name=collection_name)
    assert stored_updated[0]["payload"]["description"] == "Updated ML pipeline work."

    try:
        client.delete_collection(collection_name=collection_name)
    except Exception:
        pass


def test_end_to_end_synthetic_demo_on_temp_collection():
    collection_name = f"internships_demo_{uuid.uuid4().hex[:8]}"
    ensure_collection(collection_name=collection_name, vector_size=384, distance="Cosine")
    candidate_resume = STANDARD_RESUME

    candidate_vector = embed_job(candidate_resume)
    job_vectors = [
        {
            "job_id": "demo-1",
            "title": "Python Backend Intern",
            "company": "NovaTech",
            "required_skills": ["Python", "SQL"],
            "preferred_skills": ["FastAPI"],
            "description": "Build backend services and APIs.",
            "embedding": embed_job({"title": "Python Backend Intern", "description": "Build backend services and APIs.", "required_skills": ["Python", "SQL"], "preferred_skills": ["FastAPI"]}),
        },
        {
            "job_id": "demo-2",
            "title": "Data Analyst Intern",
            "company": "InsightIQ",
            "required_skills": ["SQL", "Python"],
            "preferred_skills": ["Tableau"],
            "description": "Analyze customer datasets and dashboards.",
            "embedding": embed_job({"title": "Data Analyst Intern", "description": "Analyze customer datasets and dashboards.", "required_skills": ["SQL", "Python"], "preferred_skills": ["Tableau"]}),
        },
        {
            "job_id": "demo-3",
            "title": "Frontend Intern",
            "company": "Bloom Studio",
            "required_skills": ["React", "JavaScript"],
            "preferred_skills": ["CSS"],
            "description": "Build UI components and prototypes.",
            "embedding": embed_job({"title": "Frontend Intern", "description": "Build UI components and prototypes.", "required_skills": ["React", "JavaScript"], "preferred_skills": ["CSS"]}),
        },
    ]

    upsert_jobs(job_vectors, collection_name=collection_name)
    matches = search_jobs(candidate_vector, collection_name=collection_name, limit=3)
    assert len(matches) >= 1
    assert [item["job_id"] for item in matches][:3] == ["demo-1", "demo-2", "demo-3"] or any(item["job_id"] in {"demo-1", "demo-2", "demo-3"} for item in matches)

    first_match = next(item for item in matches if item["job_id"] == "demo-1")
    score = calculate_match_score(
        {
            "required_skills": ["Python", "SQL"],
            "preferred_skills": ["FastAPI"],
        },
        STANDARD_RESUME,
        float(first_match.get("score", 0.0)),
    )
    assert 0.0 <= score["overall_score"] <= 1.0
    assert "missing_required" in score

    get_qdrant_client().delete_collection(collection_name=collection_name)
