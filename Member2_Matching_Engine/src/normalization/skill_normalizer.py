from __future__ import annotations

import re
from typing import Iterable

ALIASES = {
    "javascript": "JavaScript",
    "js": "JavaScript",
    "python": "Python",
    "py": "Python",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "natural language processing": "Natural Language Processing",
    "nlp": "Natural Language Processing",
    "reactjs": "React",
    "react js": "React",
    "react": "React",
    "nodejs": "Node.js",
    "node js": "Node.js",
    "node.js": "Node.js",
    "postgresql": "PostgreSQL",
    "sql": "SQL",
    "excel": "Excel",
    "tableau": "Tableau",
    "pandas": "pandas",
    "numpy": "NumPy",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "java": "Java",
    "c sharp": "C#",
    "c#": "C#",
    "c++": "C++",
    "aws": "AWS",
    "gcp": "GCP",
    "azure": "Azure",
    "html": "HTML",
    "css": "CSS",
    "fastapi": "FastAPI",
    "postgres": "PostgreSQL",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "machine-learning": "Machine Learning",
    "natural-language-processing": "Natural Language Processing",
}


def _normalize_key(skill: str) -> str:
    text = str(skill).strip().lower()
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9+#.]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_skill(skill: str | None) -> str:
    if skill is None:
        return ""
    text = str(skill).strip()
    if not text:
        return ""

    key = _normalize_key(text)
    if key in ALIASES:
        return ALIASES[key]

    lowered = text.lower()
    if lowered in ALIASES:
        return ALIASES[lowered]

    alternate_key = re.sub(r"\s+", " ", text).strip()
    if alternate_key.lower() in ALIASES:
        return ALIASES[alternate_key.lower()]

    return re.sub(r"\s+", " ", text).strip()


def normalize_skills(skills: Iterable[str] | None) -> list[str]:
    if skills is None:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_skill in skills:
        if raw_skill is None:
            continue
        skill = normalize_skill(raw_skill)
        if not skill:
            continue
        key = skill.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(skill)
    return normalized
