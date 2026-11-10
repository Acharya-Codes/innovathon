@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo   Rebuilding the Python virtual environment
echo   Folder: %CD%
echo ========================================

if exist ".venv\" (
    echo [1/5] Removing the existing .venv folder...
    rmdir /s /q ".venv"
    if exist ".venv\" goto :failed
) else (
    echo [1/5] No existing .venv folder found.
)

echo [2/5] Creating a fresh .venv environment...
python -m venv .venv
if errorlevel 1 goto :failed

echo [3/5] Activating the new environment...
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :failed

echo [4/5] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :failed

if exist "requirements.txt" (
    echo [5/5] Installing dependencies from requirements.txt...
    python -m pip install -r "requirements.txt"
    if errorlevel 1 goto :failed
) else (
    echo [5/5] No requirements.txt found; skipping project dependencies.
)

echo Installing FastAPI, Uvicorn, and multipart support...
python -m pip install fastapi uvicorn python-multipart google-generativeai
if errorlevel 1 goto :failed

echo.
echo Environment setup completed successfully.
echo The .venv environment is active in this command window.
exit /b 0

:failed
echo.
echo ERROR: Environment setup failed. Review the error above and try again.
exit /b 1
