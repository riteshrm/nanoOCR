
import cv2
import json
import uuid
import html
import re
import numpy as np

from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import (
    HTMLResponse,
    FileResponse,
    RedirectResponse,
)

from custom_engine_onnx import OCREngine

app = FastAPI(title="Leegality OCR")
engine = OCREngine()

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# 1. Home page: image upload
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>Leegality OCR</h1>
    <form action="/ocr" method="post"
          enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Process Image</button>
    </form>
    """


# 2. Process image
@app.post("/ocr")
async def process_image(file: UploadFile = File(...)):
    content = await file.read()

    image = cv2.imdecode(
        np.frombuffer(content, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(400, "Invalid image")

    text_results, redacted_image = await run_in_threadpool(
        engine.process, image
    )

    job_id = uuid.uuid4()
    job_dir = OUTPUT_DIR / str(job_id)
    job_dir.mkdir()

    stem = Path(file.filename or "image.png").stem
    stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem) or "image"

    json_path = job_dir / f"{stem}_text.json"
    image_path = job_dir / f"{stem}_redacted.png"

    json_path.write_text(
        json.dumps(text_results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    if not cv2.imwrite(str(image_path), redacted_image):
        raise HTTPException(500, "Failed to save image")

    return RedirectResponse(
        url=f"/results/{job_id}",
        status_code=303
    )


def get_output_file(job_id, suffix):
    job_dir = OUTPUT_DIR / str(job_id)

    matches = list(job_dir.glob(f"*{suffix}"))

    if not matches:
        raise HTTPException(404, "Result not found")

    return matches[0]


# 3. Show OCR results in browser
@app.get("/results/{job_id}", response_class=HTMLResponse)
def show_results(job_id: uuid.UUID):
    json_path = get_output_file(job_id, "_text.json")

    parsed_text = json.loads(
        json_path.read_text(encoding="utf-8")
    )

    display_text = html.escape(
        json.dumps(parsed_text, indent=2, ensure_ascii=False)
    )

    return f"""
    <h1>OCR Results</h1>

    <h2>Parsed Text</h2>
    <pre>{display_text}</pre>

    <a href="/download/{job_id}/text">
        Download JSON
    </a>

    <h2>Redacted Image</h2>
    <img src="/results/{job_id}/preview"
         style="max-width:90%; display:block; margin-bottom:20px;">

    <a href="/download/{job_id}/redacted">
        Download Redacted Image
    </a>

    <p><a href="/">Process another image</a></p>
    """


# 4. Display image inside browser
@app.get("/results/{job_id}/preview")
def preview_image(job_id: uuid.UUID):
    path = get_output_file(job_id, "_redacted.png")
    return FileResponse(path, media_type="image/png")


# 5. Download individual files
@app.get("/download/{job_id}/{file_type}")
def download(job_id: uuid.UUID, file_type: str):
    if file_type == "text":
        path = get_output_file(job_id, "_text.json")
        media_type = "application/json"

    elif file_type == "redacted":
        path = get_output_file(job_id, "_redacted.png")
        media_type = "image/png"

    else:
        raise HTTPException(404, "Invalid file type")

    return FileResponse(
        path,
        filename=path.name,
        media_type=media_type,
        content_disposition_type="attachment"
    )
