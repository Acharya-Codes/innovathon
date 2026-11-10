# Member 2 Matching Engine

This project processes internship job descriptions, normalizes skills, generates semantic embeddings, stores them in Qdrant, and ranks internship matches for a candidate resume.

## Setup

1. Create a local `.env` file from `.env.example`.
2. Fill in `GEMINI_API_KEY` and optionally `GEMINI_MODEL`.
3. Ensure Qdrant is running locally at `http://localhost:6333`.
4. Install dependencies with `pip install -r requirements.txt` if needed.

## Components

- `src/jd_processing/`: extraction, validation, and schema definitions.
- `src/normalization/`: centralized skill alias and normalization logic.
- `src/embeddings/`: sentence-transformer embedding generation.
- `src/vector_db/`: Qdrant connection and job upsert/retrieval helpers.
- `src/matching/`: semantic search and score computation logic.
- `src/evaluation/`: scoring summary utilities.

## Example

```bash
python main.py
python main.py --evaluate
```

The job-processing pipeline uses Gemini to parse raw JD text into a validated `InternshipJobDescription`, saves the result under `data/processed_jobs/`, and is ready to be integrated with the backend.

## Synthetic evaluation

The project includes a reproducible synthetic benchmark covering 20 resumes and 30 internships with labeled relevance grades from 0 (irrelevant) to 3 (strong match). It compares the current formula against skill-heavy and semantic-heavy alternatives without auto-selecting a winner.
