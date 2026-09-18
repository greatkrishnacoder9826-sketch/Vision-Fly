from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("voice_agent")

DEFAULT_VOICE_ID = "shubh"    # Male, conversational. Female: "anushka"
VOICE_ID = os.getenv("SARVAM_VOICE_ID", DEFAULT_VOICE_ID)

DEFAULT_MODEL_ID = "bulbul:v3"
MODEL_ID = os.getenv("SARVAM_MODEL", DEFAULT_MODEL_ID)

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0        # seconds, exponential
MAX_WORKERS = 4            # concurrent TTS calls
MAX_CHARS_PER_SEGMENT = 2500   # bulbul:v3 ki per-request limit se safe margin
class VoiceSettings(TypedDict, total=False):
    pace: float       # speed, e.g. 1.0 = normal, 1.2 = 20% faster
    pitch: float       # e.g. 0.0 = default, -0.5 to 0.5
    loudness: float    # e.g. 1.0 = default


DEFAULT_VOICE_SETTINGS: VoiceSettings = {
    "pace": 1.0,
    "pitch": 0.0,
    "loudness": 1.0,
}



class AudioSegment(TypedDict, total=False):
    segment_id: str          # "hook" | "scene_1" | "scene_2" | ... | "cta"
    text: str
    audio_path: str
    duration_sec: float
    error: Optional[str]


class VoiceState(TypedDict, total=False):
    # Script Agent (approved) se aata hai
    title: str
    hook: str
    scenes: list[dict]
    cta: str
    language: str

    # config
    voice_id: str              # Sarvam speaker name, e.g. "shubh"
    model_id: str               # "bulbul:v3" ya "bulbul:v2"
    output_dir: str
    voice_settings: VoiceSettings

    # output
    segments: list[AudioSegment]     # hook + scenes + cta, order preserved
    scenes: list[dict]               # duration_sec ab REAL audio length se update
    total_duration_sec: float
    error: Optional[str]



AVAILABLE_SPEAKERS = [
    "anushka", "abhilash", "manisha", "vidya", "arya", "karun", "hitesh",
    "aditya", "ritu", "priya", "neha", "rahul", "pooja", "rohan", "simran",
    "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun", "manan",
    "sumit", "roopa", "kabir", "aayan", "shubh", "ashutosh", "advait",
    "anand", "tanya", "tarun", "sunny", "mani", "gokul", "vijay", "shruti",
    "suhani", "mohit", "kavitha", "rehan", "soham", "rupali",
]


def list_voices() -> list[str]:
    """Apne available Sarvam speakers dekhne ke liye — voice_id choose karne me madad."""
    return AVAILABLE_SPEAKERS



def get_audio_duration(path: Path) -> float:
    from mutagen.mp3 import MP3

    return round(MP3(str(path)).info.length, 2)


async def _synthesize_async(
    text: str, voice_id: str, model_id: str, voice_settings: VoiceSettings, out_path: Path
) -> None:
    from sarvamai import SarvamAI

    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY .env me set nahi hai")

    client = SarvamAI(api_subscription_key=api_key)

    response = client.text_to_speech.convert(
        text=text,
        language_code="hi-IN",             # installed SDK me ye naam hai, target_ nahi
        speaker=voice_id,
        model=model_id,
        pace=voice_settings.get("pace", 1.0),
        pitch=voice_settings.get("pitch", 0.0),
        loudness=voice_settings.get("loudness", 1.0),
        output_audio_codec="mp3",           # .mp3 extension se match rahega
    )

    audio_bytes = base64.b64decode("".join(response.audios))
    out_path.write_bytes(audio_bytes)


def synthesize_segment(
    segment_id: str,
    text: str,
    out_dir: Path,
    voice_id: str,
    model_id: str,
    voice_settings: VoiceSettings,
) -> AudioSegment:
    text = (text or "").strip()
    if not text:
        return AudioSegment(segment_id=segment_id, text=text, error="Text khali hai")

    text = text[:MAX_CHARS_PER_SEGMENT]
    out_path = out_dir / f"{segment_id}_{uuid.uuid4().hex[:8]}.mp3"

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            asyncio.run(_synthesize_async(text, voice_id, model_id, voice_settings, out_path))

            if not out_path.exists() or out_path.stat().st_size == 0:
                raise RuntimeError("Sarvam ne khali/missing audio file di")

            duration = get_audio_duration(out_path)
            logger.info("[%s] ok | %.2fs | attempt %d", segment_id, duration, attempt)
            return AudioSegment(
                segment_id=segment_id, text=text, audio_path=str(out_path),
                duration_sec=duration, error=None,
            )

        except Exception as e:                                # noqa: BLE001
            last_err = e
            out_path.unlink(missing_ok=True)
            msg = str(e).lower()

            # auth galat ya invalid speaker/model — retry karne ka fayda nahi
            if "403" in msg or "401" in msg or "api_key" in msg or "invalid" in msg:
                logger.error("[%s] non-retryable: %s", segment_id, e)
                break

            logger.warning("[%s] attempt %d/%d fail: %s", segment_id, attempt, MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)

    return AudioSegment(segment_id=segment_id, text=text, error=f"TTS fail: {last_err}")


def _build_segment_list(state: VoiceState) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    if (state.get("hook") or "").strip():
        items.append(("hook", state["hook"]))
    for s in state.get("scenes") or []:
        scene_no = s.get("scene_no")
        items.append((f"scene_{scene_no}", s.get("narration", "")))
    if (state.get("cta") or "").strip():
        items.append(("cta", state["cta"]))
    return items


def voice_node(state: VoiceState) -> dict:
    scenes = state.get("scenes") or []
    if not scenes:
        return {"error": "Script me koi scene nahi hai — Voice Agent ko kuch nahi mila", "segments": []}

    voice_id = state.get("voice_id") or VOICE_ID
    model_id = state.get("model_id") or MODEL_ID
    voice_settings = {**DEFAULT_VOICE_SETTINGS, **(state.get("voice_settings") or {})}

    out_dir = Path(state.get("output_dir") or (Path.cwd() / "output_audio"))
    out_dir.mkdir(parents=True, exist_ok=True)

    items = _build_segment_list(state)
    if not items:
        return {"error": "Synthesize karne ke liye koi text nahi mila", "segments": []}

    logger.info("[voice] %d segments, %d workers, voice=%s", len(items), MAX_WORKERS, voice_id)

    results: dict[str, AudioSegment] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(items))) as pool:
        futures = {
            pool.submit(synthesize_segment, seg_id, text, out_dir, voice_id, model_id, voice_settings): seg_id
            for seg_id, text in items
        }
        for fut in as_completed(futures):
            seg_id = futures[fut]
            try:
                results[seg_id] = fut.result()
            except Exception as e:                            # noqa: BLE001
                results[seg_id] = AudioSegment(segment_id=seg_id, error=f"Unexpected: {e}")

    # original order restore karo (as_completed order random hoti hai)
    ordered = [results[seg_id] for seg_id, _ in items]
    failed = [s for s in ordered if s.get("error")]

    if failed:
        # partial success ko bhi disk pe chhod dena taaki caller retry-per-segment kar sake
        details = "; ".join(f"{s['segment_id']}: {s['error']}" for s in failed)
        return {
            "segments": ordered,
            "error": f"{len(failed)}/{len(ordered)} segment(s) fail: {details}",
        }

    # scenes ke duration_sec ko REAL audio length se overwrite —
    # ab Video Assembler audio aur image ko exactly sync kar sakta hai
    duration_by_scene = {
        s["segment_id"]: s["duration_sec"] for s in ordered if s["segment_id"].startswith("scene_")
    }
    updated_scenes = []
    for sc in scenes:
        sc = dict(sc)
        key = f"scene_{sc.get('scene_no')}"
        if key in duration_by_scene:
            sc["duration_sec"] = duration_by_scene[key]
        updated_scenes.append(sc)

    total = round(sum(s["duration_sec"] for s in ordered), 2)
    logger.info("[voice] all %d segments ok | total %.2fs", len(ordered), total)

    return {
        "segments": ordered,
        "scenes": updated_scenes,
        "total_duration_sec": total,
        "error": None,
    }


def build_voice_agent():
    builder = StateGraph(VoiceState)
    builder.add_node("voice_node", voice_node)
    builder.add_edge(START, "voice_node")
    builder.add_edge("voice_node", END)
    return builder.compile()


voice_agent = build_voice_agent()


def run_voice_agent(
    script: dict,
    voice_id: Optional[str] = None,
    model_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    voice_settings: Optional[VoiceSettings] = None,
) -> VoiceState:
    """script = approved Script Agent output (title, hook, scenes, cta, language)."""
    return voice_agent.invoke(  # type: ignore[return-value]
        {
            "title": script.get("title", ""),
            "hook": script.get("hook", ""),
            "scenes": script.get("scenes", []),
            "cta": script.get("cta", ""),
            "language": script.get("language", "hinglish"),
            "voice_id": voice_id,
            "model_id": model_id,
            "output_dir": output_dir,
            "voice_settings": voice_settings,
        }
    )

def from_approved(result: dict, **kwargs) -> VoiceState:
    """result = guardrails.get_approved_script() ka output."""
    if not result.get("approved"):
        verdict = result.get("verdict")
        reason = result.get("reason")
        return VoiceState(error=f"Script approved nahi hai ({verdict}): {reason}", segments=[])
    return run_voice_agent(result["script"], **kwargs)

# Test

if __name__ == "__main__":
    dummy_script = {
        "title": "AI in Devlopment",
        "hook": "Kya AI Web Dev ka future change kar sakta hai",
        "scenes": [
            {"scene_no": 1, "narration": "AI ab kisano ki madad kar raha hai",
             "visual_description": "drone over green field", "duration_sec": 8},
            {"scene_no": 2, "narration": "Sensors mitti ki nami batate hain",
             "visual_description": "soil sensor macro shot", "duration_sec": 6},
        ],
        "cta": "Follow karo aur AI ki duniya explore karo",
        "language": "hinglish",
    }

    out = run_voice_agent(dummy_script)
    if out.get("error"):
        print("ERROR:", out["error"])
    else:
        total = out["total_duration_sec"]
        print(f"\nTotal duration: {total}s")
        for seg in out["segments"]:
            print(f"  [{seg['segment_id']}] {seg['duration_sec']}s -> {seg['audio_path']}")
        print("\nUpdated scene durations (real, not estimated):")
        for sc in out["scenes"]:
            print(f"  Scene {sc['scene_no']}: {sc['duration_sec']}s")