"""
Script Agent — Multi-Agent Video Generation Pipeline
=====================================================
INPUT AGENT (topic)  ->  SCRIPT AGENT  ->  script  ->  GUARDRAILS
                              ↑                            │
                              └──────── REWRITE ───────────┘

Kaam:
  1. Topic ko ek structured video script me convert karna (hook + scenes + CTA)
  2. Har scene ke saath narration (Voice Agent ke liye) aur visual_description
     (Scene/Image Agent ke liye) dena
  3. Guardrail se REWRITE aane par usi feedback ko le kar script sudharna

Design rules (Input Agent jaise hi):
  * Node kabhi raise nahi karta -> error state me jaata hai
  * LLM strict JSON deta hai, aur parser fenced/dirty JSON bhi handle karta hai
  * Invalid JSON pe apne aap retry (MAX_GEN_ATTEMPTS tak)
  * Rewrite loop pe hard limit (MAX_REWRITES) -> infinite loop se bachne ke liye
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("script_agent")

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

DEFAULT_DURATION = 60        # seconds
MIN_DURATION, MAX_DURATION = 15, 300
MAX_GEN_ATTEMPTS = 3         # JSON parse fail hone par retry
MAX_REWRITES = 2             # guardrail feedback ke baad kitni baar sudharna hai

# narration speed — duration estimate ke liye (words per second)
WPS = {"hindi": 2.4, "hinglish": 2.5, "english": 2.8}

Language = Literal["hindi", "hinglish", "english"]
Tone = Literal["educational", "storytelling", "news", "motivational", "funny"]


# ─────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────
class Scene(TypedDict):
    scene_no: int
    narration: str            # -> Voice Agent
    visual_description: str   # -> Scene Agent -> Image Agent
    duration_sec: int


class ScriptState(TypedDict, total=False):
    # Input Agent se aata hai
    topic: str
    source_text: str
    source_type: str

    # user/config options
    duration_sec: int
    language: Language
    tone: Tone

    # guardrail se wapas aata hai (REWRITE branch)
    feedback: Optional[str]
    rewrite_count: int

    # output
    title: str
    hook: str
    scenes: list[Scene]
    cta: str
    full_narration: str       # saara narration jodkar — Voice Agent ka shortcut
    est_duration_sec: int
    attempts: int
    error: Optional[str]


# ─────────────────────────────────────────────────────────────
# Lazy LLM
# ─────────────────────────────────────────────────────────────
_llm: Any = None


def get_llm(temperature: float = 0.7):
    global _llm
    if _llm is None:
        from langchain_groq import ChatGroq

        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY .env me set nahi hai")
        _llm = ChatGroq(model=GROQ_MODEL, temperature=temperature, max_retries=2, timeout=60)
        logger.info("Groq LLM loaded: %s", GROQ_MODEL)
    return _llm


# ─────────────────────────────────────────────────────────────
# JSON parsing — LLM ka output kabhi saaf nahi aata
# ─────────────────────────────────────────────────────────────
def extract_json(raw: str) -> dict:
    """```json fences, preamble text, trailing commas — sab handle karta hai."""
    if not raw or not raw.strip():
        raise ValueError("LLM ne khali response diya")

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Response me JSON object nahi mila")
    candidate = text[start : end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", candidate)   # trailing comma hatao
        return json.loads(cleaned)


# ─────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────
SCHEMA = """{
  "title": "short catchy video title",
  "hook": "first 3-5 seconds ka attention grabbing line",
  "scenes": [
    {
      "scene_no": 1,
      "narration": "is scene ka bola jaane wala text",
      "visual_description": "English me detailed visual description for an image generator",
      "duration_sec": 8
    }
  ],
  "cta": "closing call to action"
}"""

LANG_RULE = {
    "hindi": "Narration shuddh but simple Hindi me likho (Devanagari script).",
    "hinglish": "Narration natural Hinglish me likho (Roman script, jaise log actually bolte hain).",
    "english": "Write the narration in simple, conversational English.",
}


def build_prompt(state: ScriptState) -> str:
    duration = state.get("duration_sec", DEFAULT_DURATION)
    language = state.get("language", "hinglish")
    tone = state.get("tone", "educational")
    n_scenes = max(3, min(12, round(duration / 8)))

    parts = [
        "You are a professional short-form video scriptwriter.",
        f"\nTOPIC: {state['topic']}",
    ]

    src = (state.get("source_text") or "").strip()
    if src and src.lower() != state["topic"].lower():
        parts.append(f"ORIGINAL USER INPUT (extra context): {src[:500]}")

    parts += [
        f"\nREQUIREMENTS:",
        f"- Total video length: about {duration} seconds",
        f"- Roughly {n_scenes} scenes, har scene 5-10 seconds ka",
        f"- Tone: {tone}",
        f"- {LANG_RULE[language]}",
        "- 'visual_description' ALWAYS in English, image-generation friendly: subject, "
        "setting, lighting, camera angle, style. No text/words inside the image.",
        "- Koi real person, brand logo, copyrighted character ya song lyric mat use karo.",
        "- Sab scenes ke duration_sec ka total roughly total length ke barabar ho.",
    ]

    # ── REWRITE branch: guardrail ka feedback yahan inject hota hai ──
    feedback = (state.get("feedback") or "").strip()
    if feedback:
        parts += [
            "\nIMPORTANT — previous version was REJECTED by the safety review.",
            f"Reviewer feedback: {feedback}",
            "Rewrite the script fixing exactly these issues. Topic wahi rakho, "
            "problem wale hisse ko safe alternative se replace karo.",
        ]

    parts += [
        "\nReturn ONLY valid JSON in exactly this shape, no markdown, no commentary:",
        SCHEMA,
    ]
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────
# Validation / normalization
# ─────────────────────────────────────────────────────────────
def normalize_script(data: dict, language: Language) -> dict:
    if not isinstance(data, dict):
        raise ValueError("JSON object expected")

    title = str(data.get("title") or "").strip()
    hook = str(data.get("hook") or "").strip()
    cta = str(data.get("cta") or "").strip()
    raw_scenes = data.get("scenes")

    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ValueError("'scenes' list missing ya khali hai")

    scenes: list[Scene] = []
    for i, s in enumerate(raw_scenes, start=1):
        if not isinstance(s, dict):
            raise ValueError(f"Scene {i} object nahi hai")
        narration = str(s.get("narration") or "").strip()
        visual = str(s.get("visual_description") or "").strip()
        if not narration or not visual:
            raise ValueError(f"Scene {i} me narration ya visual_description khali hai")

        try:
            dur = int(float(s.get("duration_sec") or 0))
        except (TypeError, ValueError):
            dur = 0
        if dur <= 0:
            dur = max(3, round(len(narration.split()) / WPS[language]))

        scenes.append(
            Scene(scene_no=i, narration=narration, visual_description=visual, duration_sec=dur)
        )

    if not title:
        title = scenes[0]["narration"][:60]

    full = " ".join([hook] + [s["narration"] for s in scenes] + [cta]).strip()
    est = round(len(full.split()) / WPS[language])

    return {
        "title": title,
        "hook": hook,
        "scenes": scenes,
        "cta": cta,
        "full_narration": full,
        "est_duration_sec": est,
    }


# ─────────────────────────────────────────────────────────────
# Node: generate script
# ─────────────────────────────────────────────────────────────
def script_node(state: ScriptState) -> dict:
    topic = (state.get("topic") or "").strip()
    if not topic:
        return {"error": "Topic khali hai — Input Agent se kuch nahi aaya", "scenes": []}

    duration = state.get("duration_sec", DEFAULT_DURATION)
    if not (MIN_DURATION <= duration <= MAX_DURATION):
        duration = DEFAULT_DURATION
    language: Language = state.get("language", "hinglish")

    is_rewrite = bool((state.get("feedback") or "").strip())
    logger.info("[script] %s | topic=%r", "REWRITE" if is_rewrite else "generate", topic[:60])

    prompt = build_prompt({**state, "duration_sec": duration, "language": language})  # type: ignore[arg-type]

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_GEN_ATTEMPTS + 1):
        try:
            resp = get_llm().invoke(prompt)
            parsed = extract_json(resp.content or "")
            script = normalize_script(parsed, language)

            logger.info(
                "[script] ok | %d scenes | ~%ds (target %ds)",
                len(script["scenes"]), script["est_duration_sec"], duration,
            )
            return {
                **script,
                "duration_sec": duration,
                "language": language,
                "attempts": attempt,
                "rewrite_count": state.get("rewrite_count", 0) + (1 if is_rewrite else 0),
                "feedback": None,       # consume kar liya, clear kar do
                "error": None,
            }
        except Exception as e:                                # noqa: BLE001
            last_err = e
            logger.warning("[script] attempt %d/%d fail: %s", attempt, MAX_GEN_ATTEMPTS, e)
            prompt += "\n\nPrevious attempt was invalid JSON. Return ONLY the raw JSON object."

    return {
        "scenes": [],
        "attempts": MAX_GEN_ATTEMPTS,
        "error": f"Script generate nahi hui ({MAX_GEN_ATTEMPTS} attempts): {last_err}",
    }


# ─────────────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────────────
def build_script_agent():
    builder = StateGraph(ScriptState)
    builder.add_node("script_node", script_node)
    builder.add_edge(START, "script_node")
    builder.add_edge("script_node", END)
    return builder.compile()


script_agent = build_script_agent()


def run_script_agent(
    topic: str,
    source_text: str = "",
    duration_sec: int = DEFAULT_DURATION,
    language: Language = "hinglish",
    tone: Tone = "educational",
    feedback: Optional[str] = None,
    rewrite_count: int = 0,
) -> ScriptState:
    """
    Normal call  : run_script_agent(topic)
    Rewrite call : run_script_agent(topic, feedback="reviewer ne kya bola",
                                    rewrite_count=prev+1)
    """
    if rewrite_count > MAX_REWRITES:
        return ScriptState(
            error=f"Rewrite limit ({MAX_REWRITES}) cross ho gayi — manual review chahiye",
            scenes=[],
            rewrite_count=rewrite_count,
        )

    return script_agent.invoke(  # type: ignore[return-value]
        {
            "topic": topic,
            "source_text": source_text,
            "duration_sec": duration_sec,
            "language": language,
            "tone": tone,
            "feedback": feedback,
            "rewrite_count": rewrite_count,
        }
    )


# ─────────────────────────────────────────────────────────────
# Input Agent ke saath jodne ka helper
# ─────────────────────────────────────────────────────────────
def from_input_agent(input_result: dict, **kwargs) -> ScriptState:
    if input_result.get("error"):
        return ScriptState(error=f"Input Agent fail: {input_result['error']}", scenes=[])
    return run_script_agent(
        topic=input_result["topic"],
        source_text=input_result.get("source_text", ""),
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out = run_script_agent("AI in Agriculture", duration_sec=45, language="hinglish")

    if out.get("error"):
        print("ERROR:", out["error"])
    else:
        print("\nTITLE :", out["title"])
        print("HOOK  :", out["hook"])
        print(f"EST   : {out['est_duration_sec']}s / target {out['duration_sec']}s\n")
        for s in out["scenes"]:
            print(f"--- Scene {s['scene_no']} ({s['duration_sec']}s)")
            print("  narration:", s["narration"])
            print("  visual   :", s["visual_description"])
        print("\nCTA   :", out["cta"])

    # Rewrite loop ka example (guardrail ke baad):
    # out2 = run_script_agent(
    #     "AI in Agriculture",
    #     feedback="Scene 3 me ek real brand ka naam hai, usse hatao",
    #     rewrite_count=out.get("rewrite_count", 0) + 1,
    # )