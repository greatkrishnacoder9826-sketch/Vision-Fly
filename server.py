from __future__ import annotations
 
import logging
import os
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
 
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
 
from guardrails import get_approved_script
from image_agent import from_scene_agent
from input_agent import AUDIO_EXTS, IMAGE_EXTS, MAX_FILE_MB, run_input_agent
from scene_agent import from_approved as scenes_from_approved
from video_assembler import DEFAULT_STORAGE_DIR, get_video_library, run_video_assembler
from voice_agent import from_approved as voice_from_approved
 
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("visionfly")
 
APP_NAME = "Vision-Fly X Paradox"
 
app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
 
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STORAGE_DIR = Path(DEFAULT_STORAGE_DIR)
UPLOAD_TMP_DIR = Path(tempfile.gettempdir()) / "visionfly_uploads"
 
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_TMP_DIR.mkdir(parents=True, exist_ok=True)
 
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/media", StaticFiles(directory=STORAGE_DIR), name="media")
 
# ─────────────────────────────────────────────────────────────
# Job store — sirf in-memory. Server restart hone par jobs khatam ho
# jaayenge, par finished videos video_library/ me safe rehti hain.
# ─────────────────────────────────────────────────────────────
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
executor = ThreadPoolExecutor(max_workers=2)   # ek time pe max 2 videos parallel bane
 
 
def _set_job(job_id: str, **kwargs) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)
 
 
def _new_job() -> str:
    job_id = uuid.uuid4().hex[:10]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued", "stage": "queued", "progress": 0,
            "message": "Job queue me hai...", "video_url": None,
            "title": None, "duration_sec": None, "error": None,
        }
    return job_id
 
 
STAGE_PROGRESS = {
    "input": (10, "Input samajh rahe hain..."),
    "script": (30, "Script likh rahe hain aur safety check kar rahe hain..."),
    "voice_scene": (55, "Audio bana rahe hain aur scenes plan kar rahe hain..."),
    "image": (80, "Scenes ki images generate ho rahi hain..."),
    "assemble": (95, "Video assemble ho rahi hai..."),
}
 
 
def _stage(job_id: str, name: str) -> None:
    pct, msg = STAGE_PROGRESS[name]
    _set_job(job_id, status="running", stage=name, progress=pct, message=msg)
 
 
# ─────────────────────────────────────────────────────────────
# Pipeline runner — background thread me chalta hai
# ─────────────────────────────────────────────────────────────
def run_pipeline(job_id: str, raw_input: str, input_type: str, opts: dict) -> None:
    try:
        _stage(job_id, "input")
        inp = run_input_agent(raw_input, input_type)
        if input_type in ("image", "voice"):
            Path(raw_input).unlink(missing_ok=True)   # temp upload consume ho chuki hai
        if inp.get("error"):
            raise RuntimeError(inp["error"])
 
        _stage(job_id, "script")
        approved = get_approved_script(
            inp["topic"], inp.get("source_text", ""),
            duration_sec=opts["duration_sec"], language=opts["language"], tone=opts["tone"],
        )
        if not approved.get("approved"):
            raise RuntimeError(
                f"Script approve nahi hui ({approved.get('verdict')}): {approved.get('reason')}"
            )
        script = approved["script"]
 
        _stage(job_id, "voice_scene")
        # Voice + Scene dono APPROVED SCRIPT se independently branch hote hain -> parallel
        with ThreadPoolExecutor(max_workers=2) as pool:
            voice_fut = pool.submit(voice_from_approved, approved)
            scene_fut = pool.submit(scenes_from_approved, approved, aspect_ratio=opts["aspect_ratio"])
            voice_result = voice_fut.result()
            scene_result = scene_fut.result()
 
        if voice_result.get("error"):
            raise RuntimeError(f"Voice Agent fail: {voice_result['error']}")
        if not scene_result.get("scenes_out"):
            raise RuntimeError(f"Scene Agent fail: {scene_result.get('error')}")
 
        _stage(job_id, "image")
        image_result = from_scene_agent(scene_result)
        if not image_result.get("images") or image_result.get("error"):
            raise RuntimeError(f"Image Agent fail: {image_result.get('error')}")
 
        _stage(job_id, "assemble")
        video = run_video_assembler(
            title=script["title"], voice_result=voice_result, image_result=image_result,
            aspect_ratio_out=opts["aspect_ratio"], storage_dir=str(STORAGE_DIR),
        )
        if video.get("error"):
            raise RuntimeError(video["error"])
 
        video_url = f"/media/{Path(video['video_path']).name}"
        _set_job(
            job_id, status="done", stage="done", progress=100,
            message="Video ban gayi!", video_url=video_url,
            title=script["title"], duration_sec=video["duration_sec"],
        )
        logger.info("[job %s] done -> %s", job_id, video_url)
 
    except Exception as e:                                    # noqa: BLE001
        logger.error("[job %s] fail: %s", job_id, e)
        _set_job(job_id, status="error", stage="error", progress=100, message=str(e), error=str(e))
 
 
# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.post("/api/generate")
async def generate(
    input_type: str = Form(...),
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    duration_sec: int = Form(45),
    language: str = Form("hinglish"),
    tone: str = Form("educational"),
    aspect_ratio: str = Form("9:16"),
):
    input_type = input_type.strip().lower()
    if input_type not in ("text", "image", "voice"):
        raise HTTPException(400, "input_type text / image / voice hona chahiye")
 
    if input_type == "text":
        if not text or not text.strip():
            raise HTTPException(400, "text field khali hai")
        raw_input = text
    else:
        if file is None or not file.filename:
            raise HTTPException(400, f"{input_type} ke liye file upload karni padegi")
        ext = os.path.splitext(file.filename)[1].lower()
        allowed = IMAGE_EXTS if input_type == "image" else AUDIO_EXTS
        if ext not in allowed:
            raise HTTPException(400, f"Unsupported format: {ext}")
 
        tmp_path = UPLOAD_TMP_DIR / f"{uuid.uuid4().hex}{ext}"
        with open(tmp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
 
        if tmp_path.stat().st_size > MAX_FILE_MB * 1024 * 1024:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(413, f"File {MAX_FILE_MB} MB se badi hai")
 
        raw_input = str(tmp_path)
 
    job_id = _new_job()
    opts = {
        "duration_sec": max(15, min(300, duration_sec)),
        "language": language, "tone": tone, "aspect_ratio": aspect_ratio,
    }
    executor.submit(run_pipeline, job_id, raw_input, input_type, opts)
    return {"job_id": job_id}
 
 
@app.get("/api/status/{job_id}")
def status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job nahi mila")
    return job
 
 
@app.get("/api/videos")
def videos():
    lib = get_video_library(str(STORAGE_DIR))
    for rec in lib:
        rec["video_url"] = f"/media/{rec['file']}"
    return list(reversed(lib))   # naye pehle
 
 
@app.get("/health")
def health():
    return {"status": "ok", "app": APP_NAME}
 
 
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
 

