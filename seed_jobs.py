import os
import sys
import uuid
from pathlib import Path

# Setup sys.path to find Member2 modules
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir / "Member2_Matching_Engine"))

from src.embeddings.embedding_generator import embed_job
from src.vector_db.qdrant_connection import ensure_collection

# Sample Tech Internships to seed into Qdrant
SAMPLE_INTERNSHIPS = [
    {
        "id": str(uuid.uuid4()),
        "title": "Software Engineering Intern - Full Stack",
        "company": "TechInnovate Solutions",
        "location": "Remote / Bengaluru",
        "skills": ["Python", "FastAPI", "React", "PostgreSQL", "Docker", "Git"],
        "description": (
            "We are looking for a Full Stack Software Engineering Intern to help build scalable web applications. "
            "You will work with Python, FastAPI, React, PostgreSQL, Docker containers, and RESTful APIs. "
            "Experience with modern UI frameworks, Git version control, and CI/CD pipelines is highly preferred."
        )
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Backend Engineering Intern",
        "company": "CloudScale Systems",
        "location": "Hybrid / Hyderabad",
        "skills": ["Python", "Flask", "Redis", "Celery", "PostgreSQL", "Microservices"],
        "description": (
            "Join our core backend team to build high-throughput microservices and async job processors. "
            "Requirements include solid mastery of Python, relational databases (PostgreSQL/MySQL), caching with Redis, "
            "asynchronous task queues like Celery, and automated testing with pytest."
        )
    },
    {
        "id": str(uuid.uuid4()),
        "title": "AI / Machine Learning Intern",
        "company": "Cognitive Edge Labs",
        "location": "Remote",
        "skills": ["Python", "PyTorch", "NumPy", "pandas", "Scikit-Learn", "FastAPI"],
        "description": (
            "Seeking an AI/ML Intern passionate about NLP, vector databases, and LLM applications. "
            "Hands-on experience with Python, PyTorch or TensorFlow, NumPy, pandas, data processing, "
            "and serving model inference endpoints via FastAPI or Flask is required."
        )
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Frontend Development Intern",
        "company": "PixelCraft Studio",
        "location": "Pune, India",
        "skills": ["JavaScript", "React", "HTML/CSS", "TypeScript", "Tailwind CSS", "Git"],
        "description": (
            "PixelCraft is looking for a UI/UX-focused Frontend Development Intern. "
            "Must be proficient in React, modern JavaScript/TypeScript, CSS flexbox/grid, Tailwind CSS, "
            "and integrating frontend components with backend REST API endpoints."
        )
    }
]

def seed_qdrant():
    print("Connecting to Qdrant...", flush=True)
    client = ensure_collection()
    
    from qdrant_client.models import PointStruct
    
    points = []
    print(f"Generating vector embeddings for {len(SAMPLE_INTERNSHIPS)} jobs...", flush=True)
    
    for idx, job in enumerate(SAMPLE_INTERNSHIPS):
        # Generate embedding vector for the job description + skills combined
        vector = embed_job(job)
        
        point = PointStruct(
            id=idx + 1,  # Integer ID or string UUID
            vector=vector,
            payload=job
        )
        points.append(point)
    
    # Batch upsert points into Qdrant collection
    collection_name = "internships"
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print(f"--> Successfully seeded {len(points)} jobs into Qdrant collection '{collection_name}'!", flush=True)

if __name__ == "__main__":
    seed_qdrant()