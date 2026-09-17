import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from input_agent import AUDIO_EXTS, IMAGE_EXTS, MAX_FILE_MB, run_input_agent

app = FastAPI(title="Video Pipeline — Input Agent")

TEMP_DIR = Path(tempfile.gettempdir()) / "video_pipeline_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)   # ye line original code me missing thi

ALLOWED = {"image": IMAGE_EXTS, "voice": AUDIO_EXTS}


@app.post("/generate")
async def generate(
    input_type: str = Form(...),
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),   # = None likhna galat hai, File(None) chahiye
):
    input_type = input_type.strip().lower()
    if input_type not in ("text", "image", "voice"):
        raise HTTPException(400, "input_type text / image / voice hona chahiye")

    temp_path: Optional[Path] = None
    try:
        if input_type == "text":
            if not text or not text.strip():
                raise HTTPException(400, "text field khali hai")
            raw_input = text
        else:
            if file is None or not file.filename:
                raise HTTPException(400, f"{input_type} ke liye file upload karni padegi")

            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in ALLOWED[input_type]:
                raise HTTPException(400, f"Unsupported format {ext}")

            temp_path = TEMP_DIR / f"{uuid.uuid4().hex}{ext}"
            with open(temp_path, "wb") as buf:
                shutil.copyfileobj(file.file, buf)

            if temp_path.stat().st_size > MAX_FILE_MB * 1024 * 1024:
                raise HTTPException(413, f"File {MAX_FILE_MB} MB se badi hai")

            raw_input = str(temp_path)

        result = run_input_agent(raw_input, input_type)

        if result.get("error"):
            raise HTTPException(422, result["error"])

        return {
            "topic": result["topic"],
            "source_type": result.get("source_type"),
            "source_text": result.get("source_text"),
        }
    finally:
        # temp file har haal me delete — original code ise chhod deta tha
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}