from __future__ import annotations

import random
from typing import Any

RELEVANCE_LABELS = {
    0: "Irrelevant",
    1: "Weak match",
    2: "Partial match",
    3: "Strong match",
}

RESUME_BLUEPRINTS = [
    {"resume_id": "resume_01", "experience_level": "intern", "skills": ["Python", "SQL", "Excel"]},
    {"resume_id": "resume_02", "experience_level": "intern", "skills": ["Python", "FastAPI", "SQL"]},
    {"resume_id": "resume_03", "experience_level": "intern", "skills": ["Java", "Spring", "SQL"]},
    {"resume_id": "resume_04", "experience_level": "junior", "skills": ["Python", "Machine Learning", "NLP", "SQL"]},
    {"resume_id": "resume_05", "experience_level": "junior", "skills": ["Python", "Deep Learning", "PyTorch", "SQL"]},
    {"resume_id": "resume_06", "experience_level": "junior", "skills": ["JavaScript", "React", "CSS", "HTML"]},
    {"resume_id": "resume_07", "experience_level": "junior", "skills": ["JavaScript", "Node.js", "Express", "SQL"]},
    {"resume_id": "resume_08", "experience_level": "junior", "skills": ["AWS", "Docker", "Python", "SQL"]},
    {"resume_id": "resume_09", "experience_level": "junior", "skills": ["Data Analysis", "Tableau", "Excel", "SQL"]},
    {"resume_id": "resume_10", "experience_level": "junior", "skills": ["Python", "Tableau", "Power BI", "SQL"]},
    {"resume_id": "resume_11", "experience_level": "mid", "skills": ["Python", "Machine Learning", "TensorFlow", "SQL"]},
    {"resume_id": "resume_12", "experience_level": "mid", "skills": ["Python", "Computer Vision", "OpenCV", "NLP"]},
    {"resume_id": "resume_13", "experience_level": "mid", "skills": ["JavaScript", "React", "TypeScript", "Node.js"]},
    {"resume_id": "resume_14", "experience_level": "mid", "skills": ["C#", ".NET", "SQL", "Azure"]},
    {"resume_id": "resume_15", "experience_level": "mid", "skills": ["Python", "Kafka", "Docker", "AWS"]},
    {"resume_id": "resume_16", "experience_level": "mid", "skills": ["PostgreSQL", "Python", "ETL", "Data Warehousing"]},
    {"resume_id": "resume_17", "experience_level": "senior", "skills": ["Python", "Machine Learning", "NLP", "AWS"]},
    {"resume_id": "resume_18", "experience_level": "senior", "skills": ["JavaScript", "React", "GraphQL", "Node.js"]},
    {"resume_id": "resume_19", "experience_level": "senior", "skills": ["Python", "SQL", "Machine Learning", "MLOps"]},
    {"resume_id": "resume_20", "experience_level": "senior", "skills": ["C++", "Computer Vision", "PyTorch", "Linux"]},
]

INTERNSHIP_BLUEPRINTS = [
    {"job_id": "job_01", "title": "Python Backend Intern", "required_skills": ["Python", "SQL"], "preferred_skills": ["FastAPI"], "location": "Remote"},
    {"job_id": "job_02", "title": "Machine Learning Intern", "required_skills": ["Python", "Machine Learning"], "preferred_skills": ["NLP"], "location": "Hybrid"},
    {"job_id": "job_03", "title": "Data Analyst Intern", "required_skills": ["SQL", "Python"], "preferred_skills": ["Tableau"], "location": "Remote"},
    {"job_id": "job_04", "title": "Frontend Web Intern", "required_skills": ["JavaScript", "React"], "preferred_skills": ["CSS"], "location": "Remote"},
    {"job_id": "job_05", "title": "Cloud Intern", "required_skills": ["AWS", "Python"], "preferred_skills": ["Docker"], "location": "On-site"},
    {"job_id": "job_06", "title": "NLP Intern", "required_skills": ["Python", "NLP"], "preferred_skills": ["Machine Learning"], "location": "Remote"},
    {"job_id": "job_07", "title": "Java Backend Intern", "required_skills": ["Java", "Spring"], "preferred_skills": ["SQL"], "location": "On-site"},
    {"job_id": "job_08", "title": "Data Engineering Intern", "required_skills": ["SQL", "Python"], "preferred_skills": ["ETL"], "location": "Hybrid"},
    {"job_id": "job_09", "title": "AI Research Intern", "required_skills": ["Python", "Machine Learning"], "preferred_skills": ["PyTorch"], "location": "Remote"},
    {"job_id": "job_10", "title": "Fullstack Intern", "required_skills": ["JavaScript", "Node.js"], "preferred_skills": ["React"], "location": "Remote"},
    {"job_id": "job_11", "title": "Computer Vision Intern", "required_skills": ["Python", "Computer Vision"], "preferred_skills": ["OpenCV"], "location": "Hybrid"},
    {"job_id": "job_12", "title": "Cloud Engineer Intern", "required_skills": ["AWS", "Docker"], "preferred_skills": ["Python"], "location": "Remote"},
    {"job_id": "job_13", "title": "Security Intern", "required_skills": ["C++", "Linux"], "preferred_skills": ["Cybersecurity"], "location": "On-site"},
    {"job_id": "job_14", "title": "Product Analyst Intern", "required_skills": ["SQL"], "preferred_skills": ["Python"], "location": "Remote"},
    {"job_id": "job_15", "title": "React Frontend Intern", "required_skills": ["JavaScript", "React"], "preferred_skills": ["TypeScript"], "location": "Remote"},
    {"job_id": "job_16", "title": "Business Intelligence Intern", "required_skills": ["SQL", "Tableau"], "preferred_skills": ["Excel"], "location": "Hybrid"},
    {"job_id": "job_17", "title": "MLOps Intern", "required_skills": ["Python", "AWS"], "preferred_skills": ["Machine Learning"], "location": "Remote"},
    {"job_id": "job_18", "title": "GraphQL Intern", "required_skills": ["JavaScript", "Node.js"], "preferred_skills": ["GraphQL"], "location": "Remote"},
    {"job_id": "job_19", "title": "Deep Learning Intern", "required_skills": ["Python", "Deep Learning"], "preferred_skills": ["TensorFlow"], "location": "Hybrid"},
    {"job_id": "job_20", "title": "Embedded Systems Intern", "required_skills": ["C++", "Linux"], "preferred_skills": ["Embedded"], "location": "On-site"},
    {"job_id": "job_21", "title": "Data Visualization Intern", "required_skills": ["SQL", "Tableau"], "preferred_skills": ["Python"], "location": "Remote"},
    {"job_id": "job_22", "title": "Document AI Intern", "required_skills": ["Python", "NLP"], "preferred_skills": ["Machine Learning"], "location": "Remote"},
    {"job_id": "job_23", "title": "Platform Intern", "required_skills": ["Docker", "AWS"], "preferred_skills": ["Python"], "location": "Hybrid"},
    {"job_id": "job_24", "title": "DevOps Intern", "required_skills": ["Docker", "Linux"], "preferred_skills": ["AWS"], "location": "On-site"},
    {"job_id": "job_25", "title": "JavaScript Frontend Intern", "required_skills": ["JavaScript", "CSS"], "preferred_skills": ["React"], "location": "Remote"},
    {"job_id": "job_26", "title": "Game Dev Intern", "required_skills": ["C++", "Game Development"], "preferred_skills": ["OpenGL"], "location": "On-site"},
    {"job_id": "job_27", "title": "Database Intern", "required_skills": ["SQL", "PostgreSQL"], "preferred_skills": ["Python"], "location": "Hybrid"},
    {"job_id": "job_28", "title": "API Engineer Intern", "required_skills": ["Python", "FastAPI"], "preferred_skills": ["SQL"], "location": "Remote"},
    {"job_id": "job_29", "title": "Data Science Intern", "required_skills": ["Python", "Machine Learning", "SQL"], "preferred_skills": ["NLP"], "location": "Remote"},
    {"job_id": "job_30", "title": "Generalist Intern", "required_skills": ["Excel"], "preferred_skills": ["Python"], "location": "Remote"},
]


def _build_resume(resume_template: dict[str, Any]) -> dict[str, Any]:
    return {
        "resume_id": resume_template["resume_id"],
        "experience_level": resume_template["experience_level"],
        "skills": list(resume_template["skills"]),
        "education": [{"degree": "B.Tech", "institution": "Example Institute", "year": "2028", "cgpa": "8.5"}],
        "experience": [{"role": "Software Intern", "company": "Example Company", "duration": "2 months", "description": ["Built a small project"], "tech": resume_template["skills"][:3]}],
        "projects": [{"name": "Demo Project", "description": "Built a project using analysis and coding skills.", "tech": resume_template["skills"][:2]}],
    }


def _relevance_label(resume: dict[str, Any], job: dict[str, Any]) -> int:
    candidate = set(resume["skills"])
    required = set(job["required_skills"])
    preferred = set(job["preferred_skills"])
    required_overlap = len(required & candidate)
    preferred_overlap = len(preferred & candidate)
    required_total = len(required)
    preferred_total = len(preferred)

    if required_total and required_overlap == required_total and preferred_overlap >= max(1, preferred_total):
        return 3
    if required_total and required_overlap >= max(1, round(required_total * 0.7)):
        return 2
    if required_overlap > 0 or preferred_overlap > 0:
        return 1
    return 0


def generate_synthetic_dataset(seed: int = 42) -> dict[str, Any]:
    rng = random.Random(seed)
    resumes = [_build_resume(template) for template in RESUME_BLUEPRINTS]
    internships = []
    for template in INTERNSHIP_BLUEPRINTS:
        internships.append({
            "job_id": template["job_id"],
            "title": template["title"],
            "company": f"Company {template['job_id'].split('_')[-1]}",
            "location": template["location"],
            "work_mode": "Remote" if template["location"] == "Remote" else "Hybrid",
            "duration": "3 months",
            "eligibility": "Open to students",
            "responsibilities": ["Work with the team on product tasks.", "Ship small features and documentation."],
            "required_skills": list(template["required_skills"]),
            "preferred_skills": list(template["preferred_skills"]),
            "description": f"Role focused on {template['title'].lower()} and related product work.",
        })

    labels: dict[str, dict[str, int]] = {}
    for resume in resumes:
        labels[resume["resume_id"]] = {}
        for job in internships:
            labels[resume["resume_id"]][job["job_id"]] = _relevance_label(resume, job)

    rng.shuffle(resumes)
    rng.shuffle(internships)
    return {
        "seed": seed,
        "resume_count": len(resumes),
        "internship_count": len(internships),
        "resumes": resumes,
        "internships": internships,
        "relevance_labels": labels,
        "label_names": RELEVANCE_LABELS,
    }
