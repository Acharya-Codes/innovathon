from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CORE_PACKAGES = [
    "fastapi",
    "uvicorn",
    "python-multipart",
    "pydantic",
    "Pillow",
    "google-generativeai",
    "pymupdf",
]
GENERATED_DIRS = {".venv", "venv", "__pycache__", ".git", ".pytest_cache"}


def find_requirements_files(root: Path) -> list[Path]:
    requirements_files: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in GENERATED_DIRS]
        if "requirements.txt" in filenames:
            requirements_files.append(Path(current_root) / "requirements.txt")
    return sorted(requirements_files)


def install(description: str, arguments: list[str]) -> bool:
    print(f"[INSTALLING] {description}", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", *arguments],
            check=False,
        )
    except OSError as exc:
        print(f"[ERROR] {description}: could not run pip: {exc}", flush=True)
        return False

    if result.returncode == 0:
        print(f"[SUCCESS] {description}", flush=True)
        return True

    print(f"[ERROR] {description}: pip exited with status {result.returncode}.", flush=True)
    return False


def main() -> int:
    print(f"Using Python executable: {sys.executable}", flush=True)
    print(f"Searching workspace: {WORKSPACE_ROOT}", flush=True)

    requirements_files = find_requirements_files(WORKSPACE_ROOT)
    if not requirements_files:
        print("[INFO] No requirements.txt files found.", flush=True)

    failures = 0
    for requirements_file in requirements_files:
        success = install(
            f"dependencies from {requirements_file.relative_to(WORKSPACE_ROOT)}",
            ["-r", str(requirements_file)],
        )
        failures += not success

    success = install("core orchestrator packages", CORE_PACKAGES)
    failures += not success

    if failures:
        print(f"[ERROR] Dependency synchronization finished with {failures} failed step(s).", flush=True)
        return 1

    print("[SUCCESS] All dependency installation steps completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
