
import cv2
import json
import uuid
import html
import re
import numpy as np
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import (
    HTMLResponse,
    FileResponse,
    RedirectResponse,
)

from ocr_engine import OCREngine

app = FastAPI(title="OCR")
app.mount("/app", StaticFiles(directory="app"), name="app")
engine = OCREngine()

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@app.get("/")
def home():
    return FileResponse("app/index.html")


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


@app.get("/results/{job_id}", response_class=HTMLResponse)
def show_results(job_id: uuid.UUID):
    json_path = get_output_file(job_id, "_text.json")
    parsed_text = json.loads(json_path.read_text(encoding="utf-8"))

    rows = ""
    for i, item in enumerate(parsed_text, 1):
        word = item["word"]
        if isinstance(word, list):
            word = word[0] if word else ""

        box = item["bounding_box"]

        rows += f"""
        <tr>
            <td>{i}</td>
            <td>{html.escape(str(word))}</td>
            <td>{html.escape(str(box))}</td>
        </tr>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>OCR Results</title>
        <link rel="stylesheet" href="/app/style.css">
    </head>
    <body>
        <main class="container">
            <header>
                <h1>OCR Results</h1>
                <p>Recognition and redaction completed</p>
            </header>

            <section class="card">
                <h2>Recognized Text</h2>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Text</th>
                                <th>Bounding Box</th>
                            </tr>
                        </thead>
                        <tbody>{rows}</tbody>
                    </table>
                </div>
            </section>

            <section class="card">
                <h2>Redacted Image</h2>
                <img class="results-image"
                     src="/results/{job_id}/preview"
                     alt="Redacted image">
            </section>

            <section class="card">
                <div class="actions">
                    <a class="download" href="/download/{job_id}/text">
                        Download JSON
                    </a>
                    <a class="download" href="/download/{job_id}/redacted">
                        Download Redacted PNG
                    </a>
                </div>
            </section>

            <a class="back" href="/">← Process another image</a>
        </main>
    </body>
    </html>
    """

@app.get("/results/{job_id}/preview")
def preview_image(job_id: uuid.UUID):
    path = get_output_file(job_id, "_redacted.png")
    return FileResponse(path, media_type="image/png")


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