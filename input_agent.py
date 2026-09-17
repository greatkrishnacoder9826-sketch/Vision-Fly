from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("input_agent")

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".webm"}

MAX_TEXT_CHARS = 2000
MAX_TOPIC_CHARS = 200
MAX_FILE_MB = 25

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
CAPTION_MODEL = os.getenv("CAPTION_MODEL", "Salesforce/blip-image-captioning-large")
STT_LANGS = ["hi-IN", "en-IN"]  # pehle Hindi/Hinglish, fail ho to Indian English

InputType = Literal["text", "image", "voice"]


# ─────────────────────────────────────────────────────────────
# State
# total=False zaroori hai: invoke ke time sirf input_type + raw_input
# dete ho, baaki keys baad me fill hoti hain.
# ─────────────────────────────────────────────────────────────
class InputState(TypedDict, total=False):
    input_type: Optional[InputType]  # "text" | "image" | "voice" | None (auto-detect)
    raw_input: str                   # text string / image path / audio path
    topic: str                       # FINAL OUTPUT -> Script Agent ko yahi jaata hai
    source_text: str                 # raw caption ya transcript (debug/audit ke liye)
    source_type: str                 # kis branch se aaya
    error: Optional[str]             # None = sab theek


# ─────────────────────────────────────────────────────────────
# Lazy singletons (module import pe kuch bhi load nahi hota)
# ─────────────────────────────────────────────────────────────
_llm: Any = None
_blip: Any = None


def get_llm():
    global _llm
    if _llm is None:
        from langchain_groq import ChatGroq

        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY .env me set nahi hai")
        _llm = ChatGroq(model=GROQ_MODEL, temperature=0.1, max_retries=2, timeout=30)
        logger.info("Groq LLM loaded: %s", GROQ_MODEL)
    return _llm


def get_blip():
    """Local BLIP — sirf tab load hota hai jab HF API fail ho jaye."""
    global _blip
    if _blip is None:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        name = "Salesforce/blip-image-captioning-base"
        _blip = (
            BlipProcessor.from_pretrained(name),
            BlipForConditionalGeneration.from_pretrained(name),
        )
        logger.info("Local BLIP loaded (fallback)")
    return _blip


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def clean_text(raw: str) -> str:
    if not raw or not raw.strip():
        raise ValueError("Input text khali hai")
    text = re.sub(r"\s+", " ", raw.strip())          # extra spaces/newlines
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)      # invisible control chars
    return text[:MAX_TEXT_CHARS]


_FILLERS = {
    "bhai", "yaar", "please", "plz", "ek", "mujhe", "mereko", "banao", "bana",
    "do", "dijiye", "chahiye", "video", "par", "pe", "ke", "ki", "ka", "baare",
    "me", "mein", "about", "make", "a", "on", "create", "generate", "the",
}


def _fallback_topic(text: str) -> str:
    """LLM down ho to bhi pipeline chalti rahe — rough but usable topic."""
    words = [w for w in re.findall(r"[\w'-]+", text) if w.lower() not in _FILLERS]
    topic = " ".join(words[:12]) or text[:MAX_TOPIC_CHARS]
    return topic.strip()[:MAX_TOPIC_CHARS]


def normalize_topic(text: str, mode: str = "request") -> str:
    """Raw text / caption / transcript -> ek clean one-line video topic."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Normalize karne ke liye text khali hai")

    if mode == "caption":
        instruction = (
            "An image shows the scene below. Turn it into a short, engaging "
            "video topic (max 12 words). Return ONLY the topic, one line, no quotes."
        )
    else:
        instruction = (
            "Convert the user request below into a single clear video topic. "
            "Remove filler words, greetings and conversational phrasing. "
            "Keep the user's original language intent. "
            "Return ONLY the topic, one line, max 12 words, no quotes."
        )

    try:
        resp = get_llm().invoke(f"{instruction}\n\nInput:\n{text}")
        topic = (resp.content or "").strip().strip('"').strip("'")
        topic = topic.splitlines()[0].strip() if topic else ""
        if not topic:
            raise ValueError("LLM ne khali response diya")
        return topic[:MAX_TOPIC_CHARS]
    except Exception as e:                                   # noqa: BLE001
        logger.warning("normalize_topic fail (%s) -> rule-based fallback", e)
        return _fallback_topic(text)


def _check_file(path_str: str, allowed: set[str], kind: str) -> Path:
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"{kind} file nahi mili: {path}")
    if path.suffix.lower() not in allowed:
        raise ValueError(f"Unsupported {kind} format: {path.suffix}")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise ValueError(f"{kind} file bahut badi hai ({size_mb:.1f} MB > {MAX_FILE_MB} MB)")
    return path


# ─────────────────────────────────────────────────────────────
# Node 1 — TEXT
# ─────────────────────────────────────────────────────────────
def text_node(state: InputState) -> dict:
    logger.info("[text] cleaning input")
    try:
        cleaned = clean_text(state.get("raw_input", ""))
        # 8 word se chhota, already-clean input pe LLM call waste hai
        topic = normalize_topic(cleaned) if len(cleaned.split()) > 8 else cleaned
        return {"topic": topic, "source_text": cleaned, "source_type": "text", "error": None}
    except Exception as e:                                   # noqa: BLE001
        logger.error("[text] fail: %s", e)
        return {"topic": "", "source_type": "text", "error": f"Text input invalid: {e}"}


# ─────────────────────────────────────────────────────────────
# Node 2 — IMAGE  (HF Inference API, fallback = local BLIP)
# ─────────────────────────────────────────────────────────────
def _caption_via_hf(image_bytes: bytes) -> str:
    from huggingface_hub import InferenceClient

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN set nahi hai (anonymous calls rate-limited hain)")
    client = InferenceClient(token=token, timeout=60)
    out = client.image_to_text(image_bytes, model=CAPTION_MODEL)
    return getattr(out, "generated_text", None) or str(out)


def _caption_via_local(path: Path) -> str:
    from PIL import Image

    processor, model = get_blip()
    image = Image.open(path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=40)
    return processor.decode(output[0], skip_special_tokens=True)


def image_node(state: InputState) -> dict:
    logger.info("[image] captioning")
    try:
        path = _check_file(state.get("raw_input", ""), IMAGE_EXTS, "Image")
        try:
            caption = _caption_via_hf(path.read_bytes())
        except Exception as api_err:                          # noqa: BLE001
            logger.warning("[image] HF API fail (%s) -> local BLIP", api_err)
            caption = _caption_via_local(path)

        caption = (caption or "").strip()
        if not caption:
            raise RuntimeError("Caption khali aaya")

        return {
            "topic": normalize_topic(caption, mode="caption"),
            "source_text": caption,
            "source_type": "image",
            "error": None,
        }
    except Exception as e:                                    # noqa: BLE001
        logger.error("[image] fail: %s", e)
        return {"topic": "", "source_type": "image", "error": f"Image samajh nahi aayi: {e}"}


# ─────────────────────────────────────────────────────────────
# Node 3 — VOICE  (Google STT; faster-whisper ho to wahi better)
# ─────────────────────────────────────────────────────────────
def _to_wav_16k_mono(src: Path) -> tuple[Path, bool]:
    """Returns (wav_path, is_temp). Google STT sirf WAV/FLAC leta hai."""
    if src.suffix.lower() == ".wav":
        return src, False

    from pydub import AudioSegment  # ffmpeg system pe installed hona chahiye

    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    AudioSegment.from_file(src).set_channels(1).set_frame_rate(16000).export(tmp, format="wav")
    return Path(tmp), True


def _transcribe_whisper(path: Path) -> str:
    """Optional but recommended — offline, Hinglish pe kaafi accurate."""
    from faster_whisper import WhisperModel

    model = WhisperModel(os.getenv("WHISPER_SIZE", "base"), compute_type="int8")
    segments, _ = model.transcribe(str(path))
    return " ".join(seg.text for seg in segments).strip()   


def _transcribe_google(wav_path: Path) -> str:
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    with sr.AudioFile(str(wav_path)) as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.record(source)

    last_err: Exception | None = None
    for lang in STT_LANGS:
        try:
            text = recognizer.recognize_google(audio, language=lang)
            if text.strip():
                return text.strip()
        except sr.UnknownValueError as e:
            last_err = e
        except sr.RequestError as e:
            raise RuntimeError(f"Google STT se connect nahi hua: {e}") from e
    raise ValueError(f"Audio samajh nahi aaya — clear bol kar dobara try karo ({last_err})")


def voice_node(state: InputState) -> dict:
    logger.info("[voice] transcribing")
    wav_path: Optional[Path] = None
    is_temp = False
    try:
        src = _check_file(state.get("raw_input", ""), AUDIO_EXTS, "Audio")

        try:
            transcript = _transcribe_whisper(src)
        except ImportError:
            wav_path, is_temp = _to_wav_16k_mono(src)
            transcript = _transcribe_google(wav_path)

        if not transcript.strip():
            raise ValueError("Transcript khali aaya")

        return {
            "topic": normalize_topic(transcript),
            "source_text": transcript,
            "source_type": "voice",
            "error": None,
        }
    except Exception as e:                                    # noqa: BLE001
        logger.error("[voice] fail: %s", e)
        return {"topic": "", "source_type": "voice", "error": f"Voice process nahi hui: {e}"}
    finally:
        if wav_path and is_temp and wav_path.exists():
            wav_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────
# Router
# IMPORTANT: ye function hamesha in 3 me se ek hi string return kare,
# warna LangGraph "no matching edge" error dega.
# ─────────────────────────────────────────────────────────────
def route_input(state: InputState) -> InputType:
    declared = (state.get("input_type") or "").strip().lower()
    if declared in ("text", "image", "voice"):
        return declared  # type: ignore[return-value]

    raw = (state.get("raw_input") or "").strip()
    ext = Path(raw).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "voice"
    return "text"


# ─────────────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────────────
def build_input_agent():
    builder = StateGraph(InputState)
    builder.add_node("text_node", text_node)
    builder.add_node("image_node", image_node)
    builder.add_node("voice_node", voice_node)

    builder.add_conditional_edges(
        START,
        route_input,
        {"text": "text_node", "image": "image_node", "voice": "voice_node"},
    )
    builder.add_edge("text_node", END)
    builder.add_edge("image_node", END)
    builder.add_edge("voice_node", END)
    return builder.compile()


input_agent = build_input_agent()


def run_input_agent(raw_input: str, input_type: Optional[str] = None) -> InputState:
    """Baaki pipeline ke liye clean entry point."""
    result: InputState = input_agent.invoke(
        {"input_type": input_type, "raw_input": raw_input}  # type: ignore[arg-type]
    )
    if not result.get("error") and not (result.get("topic") or "").strip():
        result["error"] = "Topic generate nahi ho paaya"
    return result


# ─────────────────────────────────────────────────────────────
# Test;
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out = run_input_agent("bhai ek video banao AI ke baare me farming me kaise use hota hai", "text")
    print("\nTOPIC :", out.get("topic"))
    print("SOURCE:", out.get("source_type"))
    print("ERROR :", out.get("error"))

    # out = run_input_agent("temp/farm.jpg", "image")
    # out = run_input_agent("temp/voice.mp3", "voice")
    # out = run_input_agent("temp/voice.mp3")   # auto-detect by extension