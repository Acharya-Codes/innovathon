# Universal Resume Ingestor (Member 1 -> Member 2)

This replaces the original `ingestor.py` while preserving the expected output keys:

```json
{
  "skills": [],
  "education": {"degree": "", "institution": "", "year": "", "cgpa": ""},
  "projects": [{"name": "", "description": "", "tech": []}]
}
```

## Supported inputs

- Text/selectable PDFs
- Scanned PDFs
- Mixed PDFs (some selectable pages, some scanned pages)
- PNG, JPG/JPEG, WEBP, TIFF/TIF, BMP, and GIF (first frame)

PDFs are read locally with PyMuPDF. Each PDF page is also rendered to an image so Gemini can inspect scanned text, columns, and layout. Image files are normalized with Pillow and sent as visual inputs. Gemini performs the visual OCR and structured extraction. No separate Tesseract installation is needed for this approach.

## 1. Create an environment (Windows PowerShell)

Open PowerShell in this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Set the Gemini API key

Set it in the current PowerShell window (replace the placeholder with your own key):

```powershell
$env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Do not paste the key into the Python file, GitHub, or team chat. The key is needed because the resume's text and page images are sent to Google's Gemini API for extraction.

## 3. Run on a file

```powershell
.\.venv\Scripts\python.exe ingestor.py "C:\path\to\resume.pdf"
```

Image example:

```powershell
.\.venv\Scripts\python.exe ingestor.py "C:\path\to\resume_scan.jpg"
```

Save JSON to a file:

```powershell
.\.venv\Scripts\python.exe ingestor.py "C:\path\to\resume.pdf" --output resume.json
```

## 4. Call from FastAPI / another Python module

```python
from ingestor import ingest_resume

# `file_bytes` is the bytes read from an uploaded file.
resume_data = ingest_resume(file_bytes, filename=upload.filename)
```

The `filename` parameter is important for clear format handling and logging, though PDF detection also checks the file signature.

## 5. Model selection

The default model is `gemini-3.8-flash`. If that model is not enabled for your API key/project, set a model name you can access:

```powershell
$env:GEMINI_MODEL = "YOUR_AVAILABLE_GEMINI_MODEL"
```

Or pass `--model YOUR_AVAILABLE_GEMINI_MODEL` on the command line.

## Limits and notes

- Default maximum upload size: 20 MB.
- Default maximum PDF length: 6 pages.
- Default PDF render resolution: 180 DPI.
- These limits can be changed using the environment variables documented in `ingestor.py`.
- OCR quality depends on image clarity, rotation, blur, handwriting, and language. Poor-quality scans can still produce missing or incorrect fields; review extracted JSON before using it for ranking.
- This accepts PDFs and images, not DOCX/RTF/ZIP files.
- Obtain permission before sending real candidates' resumes to an external API. Use anonymized or synthetic resumes for development where possible.
