"""
Input Agent — Multi-agent Video Generation Pipeline
Text / Image / Voice input ko clean "Topic/Idea" text me convert karta hai,
jo aage Script Agent ko feed hota hai.
"""

import os
import re
from typing import TypedDict, Literal, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

load_dotenv()

# ─────────────────────────────────────────────
# Shared LLM (topic normalization ke liye)
# ─────────────────────────────────────────────
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)


def normalize_topic(text: str) -> str:
    """Raw text/caption/transcript ko clean, video-worthy topic me convert karta hai."""
    prompt = (
        "Convert the following user request into a single clear video topic. "
        "Remove filler words and conversational phrasing. "
        "Return ONLY the topic in one line, nothing else.\n\n"
        f"User input:\n{text}"
    )
    response = llm.invoke(prompt)
    return response.content.strip()


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
class InputState(TypedDict):
    input_type: Optional[Literal["text", "image", "voice"]]
    raw_input: str      # text string, image path, ya audio path
    topic: str          # final output — Script Agent ko yahi milega


# ─────────────────────────────────────────────
# Node: Text
# ─────────────────────────────────────────────
def clean_text(raw: str) -> str:
    if not raw or not raw.strip():
        raise ValueError("Input text khali hai")

    text = raw.strip()
    text = re.sub(r"\s+", " ", text)                  # multiple spaces/newlines -> single space
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)        # invisible control chars hatao

    if len(text) > 2000:
        text = text[:2000]

    return text


def text_node(state: InputState) -> dict:
    print("\n[Text branch] Cleaning text input...")
    cleaned = clean_text(state["raw_input"])

    # chhote clean inputs pe LLM call waste mat karo
    if len(cleaned.split()) > 8:
        topic = normalize_topic(cleaned)
    else:
        topic = cleaned

    return {"topic": topic}


# ─────────────────────────────────────────────
# Node: Image (HuggingFace vision)
# ─────────────────────────────────────────────
def image_node(state: InputState) -> dict:
    print("\n[Image branch] Generating caption via HuggingFace...")
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=os.getenv("HF_TOKEN"))
    image_path = state["raw_input"]

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        caption = client.image_to_text(
            image_bytes,
            model="Salesforce/blip-image-captioning-large",
        )
        text = caption.generated_text if hasattr(caption, "generated_text") else str(caption)

    except Exception as e:
        raise RuntimeError(f"Image captioning fail hua: {e}")

    return {"topic": normalize_topic(text)}


# ─────────────────────────────────────────────
# Node: Voice (Google STT)
# ─────────────────────────────────────────────
def convert_to_wav(audio_path: str) -> str:
    """Google STT sirf WAV/FLAC leta hai — mp3/m4a ko convert karna padega."""
    if audio_path.lower().endswith(".wav"):
        return audio_path

    from pydub import AudioSegment

    wav_path = os.path.splitext(audio_path)[0] + ".wav"
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(wav_path, format="wav")
    return wav_path


def voice_node(state: InputState) -> dict:
    print("\n[Voice branch] Transcribing audio via Google STT...")
    import speech_recognition as sr

    audio_path = state["raw_input"]
    wav_path = convert_to_wav(audio_path)
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(wav_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)

        # hi-IN Hinglish ko sabse acche se handle karta hai
        text = recognizer.recognize_google(audio_data, language="hi-IN")

    except sr.UnknownValueError:
        raise ValueError("Audio samajh nahi aaya — clear bol kar dobara try karo")
    except sr.RequestError as e:
        raise RuntimeError(f"Google STT service se connect nahi hua: {e}")
    finally:
        if wav_path != audio_path and os.path.exists(wav_path):
            os.remove(wav_path)

    return {"topic": normalize_topic(text)}


# ─────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────
def route_input(state: InputState) -> str:
    if state.get("input_type"):
        return state["input_type"]

    # fallback: agar input_type nahi diya, extension se detect karo
    raw = state["raw_input"].lower()
    if raw.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return "image"
    elif raw.endswith((".mp3", ".wav", ".m4a")):
        return "voice"
    return "text"


# ─────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────
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

input_agent = builder.compile()


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Text test
    result = input_agent.invoke({
        "input_type": "text",
        "raw_input": "bhai ek video banao AI ke baare me farming me kaise use hota hai",
    })
    print("\nFinal topic (text):", result["topic"])

    # 2. Image test (uncomment aur path daal)
    # result = input_agent.invoke({"input_type": "image", "raw_input": "temp/farm.jpg"})
    # print("\nFinal topic (image):", result["topic"])

    # 3. Voice test (uncomment aur path daal)
    # result = input_agent.invoke({"input_type": "voice", "raw_input": "temp/voice.mp3"})
    # print("\nFinal topic (voice):", result["topic"])