from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("video_assembler")
FPS = int(os.getenv("VIDEO_FPS", "24"))
CODEC = "libx264"
AUDIO_CODEC = "aac"
APPLY_ZOOM = os.getenv("VIDEO_ZOOM", "true").lower() == "true"
ZOOM_AMOUNT = 0.06

DEFAULT_STORAGE_DIR = os.getenv("VIDEO_STORAGE_DIR", "video_library")
INDEX_FILENAME = "library.json"

ASPECT_DIMENSIONS = {
    "9:16": (768, 1344), "16:9": (1344, 768), "1:1": (1024, 1024), "4:5": (896, 1120),
}
DEFAULT_DIMENSIONS = (768, 1344)
class TimelineEntry(TypedDict, total=False):
    segment_id: str
    audio_path: str
    image_path: str
    duration_sec: float


class AssemblerState(TypedDict, total=False):
    # Voice Agent + Image Agent se aata hai
    title: str
    segments: list[dict]     # segment_id, audio_path, duration_sec
    images: list[dict]       # scene_no, image_path
    aspect_ratio_out: str

    # config
    work_dir: str            # temp build directory
    storage_dir: str         # persistent local folder
    apply_zoom: bool

    # output
    video_path: str          # final path inside storage_dir
    duration_sec: float
    n_scenes: int
    error: Optional[str]


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", (text or "video").strip().lower())
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:60] or "video"


def build_timeline(segments: list[dict], images: list[dict]) -> list[TimelineEntry]:
    """Har audio segment ko sahi image ke saath jodta hai (hook/cta fallback samet)."""
    if not segments:
        raise ValueError("Koi audio segment nahi mila")
    if not images:
        raise ValueError("Koi image nahi mili")

    by_scene = {img["scene_no"]: img["image_path"] for img in images if img.get("image_path")}
    if not by_scene:
        raise ValueError("Koi valid image path nahi mili")

    ordered_scene_nos = sorted(by_scene.keys())
    first_image = by_scene[ordered_scene_nos[0]]
    last_image = by_scene[ordered_scene_nos[-1]]

    timeline: list[TimelineEntry] = []
    for seg in segments:
        seg_id = seg.get("segment_id", "")
        audio_path = seg.get("audio_path")
        duration = seg.get("duration_sec")

        if not audio_path or not Path(audio_path).is_file():
            raise ValueError(f"Segment '{seg_id}' ki audio file nahi mili: {audio_path}")
        if not duration or duration <= 0:
            raise ValueError(f"Segment '{seg_id}' ki duration invalid hai: {duration}")

        if seg_id.startswith("scene_"):
            scene_no = int(seg_id.split("_", 1)[1])
            image_path = by_scene.get(scene_no)
            if not image_path:
                raise ValueError(f"Scene {scene_no} ki image nahi mili")
        elif seg_id == "hook":
            image_path = first_image
        elif seg_id == "cta":
            image_path = last_image
        else:
            image_path = first_image   # unknown segment type — safe default

        timeline.append(TimelineEntry(
            segment_id=seg_id, audio_path=audio_path,
            image_path=image_path, duration_sec=float(duration),
        ))

    return timeline

def _ken_burns(clip, duration: float):
    """Halka zoom-in effect. Kuch bhi fail ho (moviepy version mismatch waghera)
    to bas static image clip return kar do — cosmetic feature ke liye pipeline
    todna sahi nahi hai."""
    try:
        from moviepy import vfx

        w, h = clip.size
        zoomed = clip.with_effects([vfx.Resize(lambda t: 1 + ZOOM_AMOUNT * (t / duration))])
        return zoomed.with_effects([vfx.Crop(x_center=zoomed.w / 2, y_center=zoomed.h / 2,
                                              width=w, height=h)])
    except Exception as e:                                    # noqa: BLE001
        logger.warning("Ken Burns zoom fail (%s) -> static image use kar rahe hain", e)
        return clip


def build_clips(timeline: list[TimelineEntry], width: int, height: int, apply_zoom: bool):
    from moviepy import AudioFileClip, ImageClip

    clips = []
    audio_clips = []   # separately track so we can close them in finally
    for entry in timeline:
        img_clip = ImageClip(entry["image_path"]).resized(new_size=(width, height))
        img_clip = img_clip.with_duration(entry["duration_sec"])
        if apply_zoom:
            img_clip = _ken_burns(img_clip, entry["duration_sec"])

        audio_clip = AudioFileClip(entry["audio_path"])
        audio_clips.append(audio_clip)
        img_clip = img_clip.with_audio(audio_clip)
        clips.append(img_clip)

    return clips, audio_clips

def save_to_local_storage(temp_video_path: Path, title: str, storage_dir: Path,
                           duration_sec: float, n_scenes: int) -> Path:
    storage_dir.mkdir(parents=True, exist_ok=True)
    final_name = f"{slugify(title)}_{uuid.uuid4().hex[:8]}.mp4"
    final_path = storage_dir / final_name
    shutil.move(str(temp_video_path), str(final_path))

    index_path = storage_dir / INDEX_FILENAME
    try:
        records = json.loads(index_path.read_text()) if index_path.exists() else []
    except (json.JSONDecodeError, OSError):
        logger.warning("library.json corrupt tha, naya bana rahe hain")
        records = []

    records.append({
        "title": title,
        "file": final_name,
        "path": str(final_path.resolve()),
        "duration_sec": round(duration_sec, 2),
        "n_scenes": n_scenes,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    index_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    return final_path

def assemble_node(state: AssemblerState) -> dict:
    segments = state.get("segments") or []
    images = state.get("images") or []

    try:
        timeline = build_timeline(segments, images)
    except ValueError as e:
        return {"error": f"Timeline nahi ban paayi: {e}", "video_path": ""}

    aspect_ratio = state.get("aspect_ratio_out", "9:16")
    width, height = ASPECT_DIMENSIONS.get(aspect_ratio, DEFAULT_DIMENSIONS)
    apply_zoom = state.get("apply_zoom", APPLY_ZOOM)

    work_dir = Path(state.get("work_dir") or tempfile.mkdtemp(prefix="video_assembler_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_out = work_dir / f"tmp_{uuid.uuid4().hex[:8]}.mp4"

    clips, audio_clips, final_clip = [], [], None
    try:
        from moviepy import concatenate_videoclips

        clips, audio_clips = build_clips(timeline, width, height, apply_zoom)
        final_clip = concatenate_videoclips(clips, method="compose")

        final_clip.write_videofile(
            str(tmp_out), fps=FPS, codec=CODEC, audio_codec=AUDIO_CODEC,
            threads=os.cpu_count() or 2, logger=None,
        )

        total_duration = sum(e["duration_sec"] for e in timeline)
        n_scenes = sum(1 for e in timeline if e["segment_id"].startswith("scene_"))

        storage_dir = Path(state.get("storage_dir") or DEFAULT_STORAGE_DIR)
        final_path = save_to_local_storage(
            tmp_out, state.get("title", "video"), storage_dir, total_duration, n_scenes
        )

        logger.info("[assembler] done | %.1fs | %s", total_duration, final_path)
        return {
            "video_path": str(final_path),
            "duration_sec": round(total_duration, 2),
            "n_scenes": n_scenes,
            "error": None,
        }

    except ImportError as e:
        return {"error": f"moviepy install nahi hai ya ffmpeg missing hai: {e}", "video_path": ""}
    except Exception as e:
        logger.error("[assembler] fail: %s", e)
        return {"error": f"Video assemble nahi hui: {e}", "video_path": ""}
    finally:
        for c in audio_clips:
            try:
                c.close()
            except Exception:
                pass
        for c in clips:
            try:
                c.close()
            except Exception:
                pass
        if final_clip is not None:
            try:
                final_clip.close()
            except Exception:
                pass
        tmp_out.unlink(missing_ok=True)
        shutil.rmtree(work_dir, ignore_errors=True)


def build_video_assembler():
    builder = StateGraph(AssemblerState)
    builder.add_node("assemble_node", assemble_node)
    builder.add_edge(START, "assemble_node")
    builder.add_edge("assemble_node", END)
    return builder.compile()


video_assembler = build_video_assembler()


def run_video_assembler(
    title: str,
    voice_result: dict,
    image_result: dict,
    aspect_ratio_out: str = "9:16",
    storage_dir: Optional[str] = None,
    apply_zoom: Optional[bool] = None,
) -> AssemblerState:
    """
    voice_result = Voice Agent ka output (segments list)
    image_result = Image Agent ka output (images list)
    """
    if voice_result.get("error") and not voice_result.get("segments"):
        return AssemblerState(error=f"Voice Agent fail: {voice_result['error']}", video_path="")
    if image_result.get("error") and not image_result.get("images"):
        return AssemblerState(error=f"Image Agent fail: {image_result['error']}", video_path="")

    return video_assembler.invoke(
        {
            "title": title,
            "segments": voice_result.get("segments", []),
            "images": image_result.get("images", []),
            "aspect_ratio_out": aspect_ratio_out,
            "storage_dir": storage_dir,
            "apply_zoom": apply_zoom,
        }
    )


def get_video_library(storage_dir: str = DEFAULT_STORAGE_DIR) -> list[dict]:
    """Ab tak ki saari local videos ki list — koi DB nahi, seedha JSON index."""
    index_path = Path(storage_dir) / INDEX_FILENAME
    if not index_path.exists():
        return []
    try:
        return json.loads(index_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []


# Test — ab FAKE paths nahi, output_audio/ me jo asli files bani hain unhe glob se
# dhundta hai. Duration bhi hardcode nahi, mutagen se real mp3 length nikalta hai.
# ─────────────────────────────────────────────────────────────
def _latest_segment(prefix: str, folder: str = "output_audio") -> dict:
    from mutagen.mp3 import MP3

    matches = sorted(Path(folder).glob(f"{prefix}_*.mp3"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(
            f"{prefix}_*.mp3 '{folder}/' me nahi mili — pehle voice_agent chala ke audio generate kar"
        )
    path = matches[-1]
    duration = round(MP3(str(path)).info.length, 2)
    return {"audio_path": str(path), "duration_sec": duration}


if __name__ == "__main__":
    dummy_voice = {
        "segments": [
            {"segment_id": "hook", **_latest_segment("hook")},
            {"segment_id": "scene_1", **_latest_segment("scene_1")},
            {"segment_id": "scene_2", **_latest_segment("scene_2")},
            {"segment_id": "cta", **_latest_segment("cta")},
        ]
    }

    # NOTE: apne image_agent ke actual output_images/ folder ke hisab se
    # ye paths update kar — abhi placeholder hain
    dummy_images = {
        "images": [
            {"scene_no": 1, "image_path": "output_images/scene_1.png"},
            {"scene_no": 2, "image_path": "output_images/scene_2.png"},
        ]
    }

    for img in dummy_images["images"]:
        if not Path(img["image_path"]).is_file():
            raise FileNotFoundError(
                f"{img['image_path']} nahi mili — dummy_images me apne actual "
                f"output_images/ ke sahi filenames daal"
            )

    out = run_video_assembler("AI in Agriculture", dummy_voice, dummy_images)
    if out.get("error"):
        print("ERROR:", out["error"])
    else:
        print("VIDEO:", out["video_path"])
        print("DURATION:", out["duration_sec"], "s |", out["n_scenes"], "scenes")


