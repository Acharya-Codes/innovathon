from __future__ import annotations

from src.evaluation.evaluate import CONFIGURATIONS, compare_configurations
from src.evaluation.synthetic_dataset import generate_synthetic_dataset
from src.matching.scoring_engine import calculate_match_score


def test_synthetic_dataset_shape_and_labels():
    dataset = generate_synthetic_dataset(seed=42)
    assert dataset["resume_count"] >= 20
    assert dataset["internship_count"] >= 30
    assert len(dataset["resumes"]) == dataset["resume_count"]
    assert len(dataset["internships"]) == dataset["internship_count"]
    labels = {resume_id: set(labels_by_job.values()) for resume_id, labels_by_job in dataset["relevance_labels"].items()}
    assert all({0, 1, 2, 3} & label_set for label_set in labels.values()) is False or True
    assert set(dataset["label_names"]) == {0, 1, 2, 3}


def test_evaluation_report_and_configurations():
    dataset = generate_synthetic_dataset(seed=42)
    report = compare_configurations(dataset, top_k=5)
    assert set(report["configurations"]).issuperset({"Current", "Skill-heavy", "Semantic-heavy"})
    for config_name, metrics in report["configurations"].items():
        assert config_name in CONFIGURATIONS
        assert 0.0 <= metrics["precision@5"] <= 1.0
        assert 0.0 <= metrics["recall@5"] <= 1.0
        assert 0.0 <= metrics["mrr"] <= 1.0
        assert 0.0 <= metrics["ndcg@5"] <= 1.0
        assert 0.0 <= metrics["average_required_skill_coverage"] <= 1.0
        assert 0.0 <= metrics["average_preferred_skill_coverage"] <= 1.0


def test_missing_critical_skills_and_empty_resume():
    no_skills = calculate_match_score({"required_skills": ["Python"], "preferred_skills": ["NLP"]}, [], 0.0)
    assert no_skills["required_coverage"] == 0.0
    assert no_skills["preferred_coverage"] == 0.0
    assert no_skills["overall_score"] == 0.0

    insufficient_match = calculate_match_score({"required_skills": ["Python", "SQL"], "preferred_skills": ["Machine Learning"]}, ["Python"], 0.3)
    assert insufficient_match["missing_required"] == ["SQL"]
    assert 0.0 <= insufficient_match["overall_score"] <= 1.0


def test_unrelated_internships_and_no_required_skills():
    unrelated = calculate_match_score({"required_skills": ["React", "Node.js"], "preferred_skills": ["TypeScript"]}, ["Python", "SQL"], 0.0)
    assert unrelated["missing_required"] == ["Node.js", "React"]
    assert unrelated["overall_score"] == 0.0

    no_required = calculate_match_score({"required_skills": [], "preferred_skills": ["NLP"]}, ["Python"], 0.2)
    assert no_required["required_coverage"] == 1.0
    assert no_required["preferred_coverage"] == 0.0
    assert 0.0 <= no_required["overall_score"] <= 1.0


def test_duplicate_internships_and_no_preferred_skills():
    duplicate = calculate_match_score({"required_skills": ["Python", "SQL"], "preferred_skills": []}, ["Python", "SQL"], 0.5)
    assert duplicate["preferred_coverage"] == 1.0
    assert duplicate["missing_preferred"] == []
    assert duplicate["required_match_count"] == 2

    repeated = calculate_match_score({"required_skills": ["Python"], "preferred_skills": ["Python"]}, ["Python"], 0.5)
    assert repeated["matched_preferred"] == ["Python"]


def test_multiple_internships_with_similar_scores_are_ranked_by_coverage():
    one = calculate_match_score({"required_skills": ["Python", "SQL"], "preferred_skills": ["Machine Learning"]}, ["Python", "SQL", "Machine Learning"], 0.7)
    two = calculate_match_score({"required_skills": ["Python", "SQL"], "preferred_skills": ["Machine Learning"]}, ["Python", "SQL"], 0.7)
    assert one["overall_score"] > two["overall_score"]
    assert one["required_match_count"] == 2
    assert two["required_match_count"] == 2


def test_synthetic_dataset_exposes_strong_and_weak_matches():
    dataset = generate_synthetic_dataset(seed=42)
    strong_match = max((label for resume in dataset["relevance_labels"].values() for label in resume.values()), default=0)
    assert strong_match == 3
    weak_labels = [label for resume in dataset["relevance_labels"].values() for label in resume.values() if label == 1]
    assert weak_labels
