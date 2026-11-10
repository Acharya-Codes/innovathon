from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.embeddings import embedding_generator
from src.evaluation.evaluate import compare_configurations, format_evaluation_report
from src.evaluation.synthetic_dataset import generate_synthetic_dataset
from src.jd_processing.jd_extractor import extract_job_from_text, save_processed_job
from src.jd_processing.jd_schema import InternshipJobDescription
from src.vector_db import job_storage
from src.vector_db.qdrant_connection import ensure_collection

load_dotenv()

logger = logging.getLogger(__name__)


def process_single_job_description(raw_text: str) -> InternshipJobDescription:
    job = extract_job_from_text(raw_text, model_name=os.getenv("GEMINI_MODEL"))
    save_processed_job(job)
    return job


def embed_and_store_jobs(
    processed_jobs_dir: str | os.PathLike[str] = "data/processed_jobs",
) -> int:
    job_paths = sorted(
        path for path in Path(processed_jobs_dir).glob("*") if path.is_file() and path.suffix.lower() == ".json"
    )
    if not job_paths:
        logger.info("No processed job JSON files found in %s.", processed_jobs_dir)
        return 0

    upserted_count = 0
    failed_count = 0
    for path in job_paths:
        try:
            with path.open(encoding="utf-8") as job_file:
                job = InternshipJobDescription.model_validate(json.load(job_file))

            payload = job.model_dump()
            payload["embedding"] = embedding_generator.embed_job(job)
            if not payload["embedding"]:
                raise ValueError("Generated an empty embedding vector")

            point_ids = job_storage.upsert_jobs([payload])
            if not point_ids:
                raise RuntimeError("Qdrant did not report an upserted point")
            upserted_count += len(point_ids)
            logger.info("Stored %s in Qdrant as point %s.", path.name, ", ".join(point_ids))
        except Exception:
            failed_count += 1
            logger.exception("Failed to embed or store processed job %s.", path)

    logger.info(
        "Qdrant job ingestion complete: %d of %d jobs upserted; %d failed.",
        upserted_count,
        len(job_paths),
        failed_count,
    )
    return upserted_count


def run_synthetic_evaluation() -> str:
    dataset = generate_synthetic_dataset(seed=42)
    report = compare_configurations(dataset, top_k=5)
    print(format_evaluation_report(report))
    return format_evaluation_report(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Member 2 matching engine")
    parser.add_argument("--evaluate", action="store_true", help="Run the synthetic scoring comparison across configurations.")
    args = parser.parse_args()

    if args.evaluate:
        run_synthetic_evaluation()
    else:
        sample_jobs_dir = Path("data/raw_jobs")
        sample_jobs_dir.mkdir(parents=True, exist_ok=True)
        raw_job_paths = sorted(
            path
            for path in sample_jobs_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".json"}
        )
        if not raw_job_paths:
            logger.info("No raw job descriptions were found in data/raw_jobs.")
        else:
            for path in raw_job_paths:
                try:
                    raw_text = path.read_text(encoding="utf-8")
                    job = process_single_job_description(raw_text)
                    logger.info("Processed %s -> %s.", job.title or path.name, job.job_id)
                except Exception:
                    logger.exception("Failed to process raw job description %s.", path)

        ensure_collection()
        embed_and_store_jobs()
