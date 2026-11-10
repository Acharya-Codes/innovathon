from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware


# 1. Workspace Root
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

# 2. Append module paths to sys.path
sys.path.append(str(WORKSPACE_ROOT / "Member2_Matching_Engine"))
sys.path.append(str(WORKSPACE_ROOT / "Resume_Ingestor" / "universal_resume_ingestor"))
sys.path.append(str(Path(__file__).resolve().parent))

# 3. Load .env files ONLY if GEMINI_API_KEY isn't already set in system environment
if not os.getenv("AQ.Ab8RN6L5hq6zlaV16xuFRQCwIsbEh_2YHKQzwFM8vRhmtlqaJQ"):
    env_locations = [
        WORKSPACE_ROOT / "Member2_Matching_Engine" / ".env",
        WORKSPACE_ROOT / "Resume_Ingestor" / "universal_resume_ingestor" / ".env",
        WORKSPACE_ROOT / "Member3_API" / ".env",
        WORKSPACE_ROOT / ".env",
    ]
    for env_path in env_locations:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)

# Check GOOGLE_API_KEY as an alternative name
api_key = os.getenv("AQ.Ab8RN6L5hq6zlaV16xuFRQCwIsbEh_2YHKQzwFM8vRhmtlqaJQ") or os.getenv("GOOGLE_API_KEY")

# 4. Direct Fallback: Hardcode your key here if .env files keep failing
if not api_key:
    # REPLACE THE STRING BELOW WITH YOUR ACTUAL GEMINI API KEY (e.g. "AIzaSy...")
    api_key = "AQ.Ab8RN6L5hq6zlaV16xuFRQCwIsbEh_2YHKQzwFM8vRhmtlqaJQ"
    os.environ["AQ.Ab8RN6L5hq6zlaV16xuFRQCwIsbEh_2YHKQzwFM8vRhmtlqaJQ"] = api_key

print(f"--> [BOOT LOG] GEMINI_API_KEY status: {'LOADED' if api_key else 'MISSING'}", flush=True)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from gap_analyzer import generate_gap_analysis
from ingestor import ingest_resume_file
from src.embeddings.embedding_generator import embed_job
from src.matching.scoring_engine import calculate_match_score
from src.vector_db.job_storage import search_jobs

logger = logging.getLogger(__name__)

app = FastAPI(title="Resume-to-Internship Matching API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/match")
async def match_resume(file: UploadFile = File(...)) -> dict[str, Any]:
    """Extract a resume and return the five best-scoring internships."""
    temporary_path: Path | None = None
    try:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".webp"}:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Upload a PDF, DOCX, TXT, PNG, JPG, JPEG, or WEBP resume.",
            )

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
            shutil.copyfileobj(file.file, temporary_file)

        if suffix == ".pdf":
            with temporary_path.open("rb") as saved_file:
                if saved_file.read(5) != b"%PDF-":
                    raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")

        print("[1/3] Calling the resume ingestor...", flush=True)
        import time

        # Active production models
        candidate_models = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
        resume_data = None
        last_err = None

        for model_name in candidate_models:
            print(f"--> Attempting resume extraction with model: {model_name}...", flush=True)
            # Try up to 3 retries for transient 503/504 errors
            for attempt in range(1, 4):
                try:
                    resume_data = ingest_resume_file(temporary_path, model=model_name)
                    print(f"--> Ingestor succeeded using {model_name}", flush=True)
                    break
                except Exception as err:
                    last_err = err
                    err_msg = str(err)
                    if ("503" in err_msg or "504" in err_msg) and attempt < 3:
                        print(f"--> Transient error ({err_msg[:40]}...). Retrying attempt {attempt + 1}/3 in 3s...", flush=True)
                        time.sleep(3)
                    else:
                        print(f"--> Model {model_name} failed: {err}", flush=True)
                        break
            if resume_data is not None:
                break

        if resume_data is None:
            raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")

        print("[2/3] Querying Qdrant and scoring matching internships...", flush=True)
        resume_vector = embed_job(resume_data)
        if not resume_vector:
            raise ValueError("Could not create an embedding from the extracted resume.")

        jobs = search_jobs(query_vector=resume_vector, limit=5)
        scored_jobs = []
        for job in jobs:
            semantic_similarity = job.get("score")
            scored_jobs.append(
                {
                    "internship": job,
                    "match_score": calculate_match_score(
                        job,
                        resume_data,
                        similarity_score=semantic_similarity,
                    ),
                }
            )

        scored_jobs.sort(
            key=lambda result: result["match_score"]["overall_score"],
            reverse=True,
        )
        print(f"[2/3] Matching completed; scored {len(scored_jobs)} internship(s).", flush=True)

        top_job = scored_jobs[0]["internship"] if scored_jobs else None
        print("[3/3] Calling the Gemini Gap Analyzer...", flush=True)
        gap_analysis = (
            await generate_gap_analysis(resume_data, top_job)
            if top_job is not None
            else None
        )
        if top_job is None:
            print("[3/3] No internship matches were found; gap analysis was skipped.", flush=True)
        else:
            print("[3/3] Gemini Gap Analyzer completed successfully.", flush=True)

        return {
            "resume": resume_data,
            "top_internships": scored_jobs,
            "gap_analysis": gap_analysis.model_dump() if gap_analysis is not None else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error("Resume-to-internship matching failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {str(e)}",
        ) from e
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                logger.exception("Could not remove temporary resume file %s.", temporary_path)