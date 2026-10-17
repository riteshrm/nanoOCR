
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

app = FastAPI(title="Leegality OCR")

BASE_DIR = Path(__file__).resolve().parent


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/api/process")
async def process_image(file: UploadFile = File(...)):
    if file.content_type != "image/png":
        raise HTTPException(
            status_code=400,
            detail="Only PNG images are supported",
        )

    content = await file.read()

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Maximum file size is 10 MB",
        )

    image = cv2.imdecode(
        np.frombuffer(content, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image",
        )

    height, width = image.shape[:2]

    return {
        "filename": file.filename,
        "width": width,
        "height": height,
        "size_bytes": len(content),
    }
