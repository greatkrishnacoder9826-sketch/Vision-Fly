"""
Scene Agent — Image Prompt Generation with Cross-Scene Consistency
====================================================================
APPROVED SCRIPT (scenes with visual_description)
              ↓
         SCENE AGENT
              ↓
   IMAGE PROMPTS (per scene, style-consistent)
              ↓
         IMAGE AGENT

Problem jo ye solve karta hai: har scene ka visual_description Script Agent
ne ALAG-ALAG generate kiya tha, ek dusre ko dekhe bina. Agar wahi seedha
Image Agent ko de diya jaaye to scene 1 me "cartoon style farmer" aur scene 3
me "photorealistic farmer" aa sakta hai — video visually inconsistent lagegi.

Scene Agent isliye:
  1. SAARE scenes ek saath dekhta hai (ek hi LLM call me)
  2. Ek GLOBAL style decide karta hai (art style, lighting, color grade)
  3. Recurring characters/settings ko fix karta hai (jaise "farmer" har scene
     me same description ke saath aaye)
  4. Har scene ka final image_prompt banata hai = global style + fixed
     character description + scene ka visual moment + camera/lighting

Design rules (baaki agents jaise hi):
  * Node kabhi raise nahi karta
  * LLM fail ho to RULE-BASED FALLBACK — raw visual_description + generic
    style suffix use karke prompt bana leta hai (kam consistent, par pipeline
    rukti nahi). Ye safety-critical nahi hai (guardrails jaisa fail-closed
    nahi chahiye) isliye fail-open theek hai.
  * Seed deterministic hota hai (topic+scene se hash) — same topic dobara
    chalao to same seed milega, reproducibility ke liye achha hai.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("scene_agent")

GROQ_MODEL = os.getenv("SCENE_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))

DEFAULT_ASPECT_RATIO = "9:16"     # shorts/reels default
VALID_ASPECT_RATIOS = {"9:16", "16:9", "1:1", "4:5"}

BASE_NEGATIVE_PROMPT = (
    "text, watermark, logo, signature, extra limbs, extra fingers, deformed hands, "
    "blurry, low quality, distorted face, disfigured, out of frame, duplicate, "
    "bad anatomy, cropped, worst quality, jpeg artifacts"
)

AspectRatio = Literal["9:16", "16:9", "1:1", "4:5"]


# ─────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────
class Character(TypedDict, total=False):
    name: str
    description: str          # image-gen friendly, fixed across scenes


class ScenePrompt(TypedDict, total=False):
    scene_no: int
    narration: str             # pass-through, Video Assembler ko chahiye
    duration_sec: float        # pass-through (Voice Agent ne update kiya ho to wahi)
    image_prompt: str
    negative_prompt: str
    seed: int


class SceneAgentState(TypedDict, total=False):
    # Script/Voice Agent se aata hai
    title: str
    scenes: list[dict]         # input: scene_no, narration, visual_description, duration_sec
    language: str

    # config
    aspect_ratio: AspectRatio
    style_hint: Optional[str]  # user optionally force kar sakta hai, e.g. "3D pixar style"
    base_seed: Optional[int]

    # output
    global_style: str
    characters: list[Character]
    aspect_ratio_out: AspectRatio
    scenes_out: list[ScenePrompt]
    used_fallback: bool
    error: Optional[str]


# ─────────────────────────────────────────────────────────────
# Lazy LLM
# ─────────────────────────────────────────────────────────────
_llm: Any = None


def get_llm():
    global _llm
    if _llm is None:
        from langchain_groq import ChatGroq

        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY .env me set nahi hai")
        _llm = ChatGroq(model=GROQ_MODEL, temperature=0.6, max_retries=2, timeout=45)
        logger.info("Scene LLM loaded: %s", GROQ_MODEL)
    return _llm


def extract_json(raw: str) -> dict:
    if not raw or not raw.strip():
        raise ValueError("LLM ne khali response diya")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Response me JSON object nahi mila")
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", candidate))


def deterministic_seed(*parts: str) -> int:
    """Same input -> same seed. Reproducible regeneration ke liye."""
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 2_147_483_647   # int32 range me fit


# ─────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────
SCHEMA = """{
  "global_style": "one detailed sentence describing the shared visual style: art style, "
                   "color palette, lighting mood, rendering quality (e.g. 'cinematic "
                   "photorealistic, warm golden-hour lighting, shallow depth of field, "
                   "shot on 35mm film')",
  "characters": [
    {"name": "farmer", "description": "middle-aged Indian farmer, wearing white kurta and "
                                       "turban, weathered hands, warm friendly expression"}
  ],
  "scenes": [
    {"scene_no": 1, "image_prompt": "full self-contained image generation prompt combining "
                                     "global style + character description + this scene's "
                                     "specific action/setting + camera angle"}
  ]
}"""


def build_prompt(state: SceneAgentState) -> str:
    scenes = state["scenes"]
    scene_lines = "\n".join(
        f"Scene {s.get('scene_no')}: {s.get('visual_description', s.get('narration', ''))}"
        for s in scenes
    )
    style_hint = (state.get("style_hint") or "").strip()

    parts = [
        "You are an art director preparing image-generation prompts for an AI video.",
        f"\nVIDEO TITLE: {state.get('title', '')}",
        f"\nSCENES (raw visual descriptions):\n{scene_lines}",
        "\nTASK:",
        "1. Decide ONE consistent global visual style that fits ALL scenes together.",
        "2. Identify recurring characters, animals, or objects that appear in more than one "
        "scene, and give each a FIXED, detailed, image-gen-friendly description so the same "
        "subject looks identical across scenes. If nothing recurs, return an empty list.",
        "3. For EACH scene, write ONE final self-contained image_prompt: it must combine the "
        "global style + any relevant fixed character description + that scene's specific "
        "action/setting + a camera angle/framing note. The prompt must be usable ALONE, "
        "without needing the other scenes' prompts for context.",
        "\nRULES:",
        "- All prompts in English, regardless of narration language.",
        "- No text, letters, or words should appear inside the generated image.",
        "- No real, named public figures. No brand names or logos. No copyrighted characters.",
        "- Keep each image_prompt under 80 words.",
    ]
    if style_hint:
        parts.append(f"- User requested style: {style_hint} — honor this in global_style.")

    parts += ["\nReturn ONLY valid JSON in exactly this shape, no markdown, no commentary:", SCHEMA]
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────
# Normalize LLM output -> final per-scene prompts
# ─────────────────────────────────────────────────────────────
def normalize_output(
    data: dict, scenes: list[dict], title: str, base_seed: Optional[int]
) -> tuple[str, list[Character], list[ScenePrompt]]:
    if not isinstance(data, dict):
        raise ValueError("JSON object expected")

    global_style = str(data.get("global_style") or "").strip()
    if not global_style:
        raise ValueError("'global_style' missing")

    characters: list[Character] = []
    for c in data.get("characters") or []:
        if isinstance(c, dict) and c.get("name") and c.get("description"):
            characters.append(Character(name=str(c["name"]).strip(),
                                         description=str(c["description"]).strip()))

    prompt_by_scene = {}
    for s in data.get("scenes") or []:
        if isinstance(s, dict) and s.get("scene_no") is not None and s.get("image_prompt"):
            prompt_by_scene[int(s["scene_no"])] = str(s["image_prompt"]).strip()

    out: list[ScenePrompt] = []
    for sc in scenes:
        scene_no = int(sc.get("scene_no", 0))
        prompt = prompt_by_scene.get(scene_no)
        if not prompt:
            # is ek scene ke liye LLM prompt nahi de paaya -> isi scene ke liye fallback
            prompt = _fallback_prompt(sc, global_style)

        seed = (base_seed if base_seed is not None else deterministic_seed(title, str(scene_no))) + scene_no

        out.append(
            ScenePrompt(
                scene_no=scene_no,
                narration=sc.get("narration", ""),
                duration_sec=sc.get("duration_sec", 0),
                image_prompt=prompt[:600],
                negative_prompt=BASE_NEGATIVE_PROMPT,
                seed=seed,
            )
        )

    if not out:
        raise ValueError("Koi scene prompt nahi ban paaya")

    return global_style, characters, out


# ─────────────────────────────────────────────────────────────
# Fallback — LLM poora fail ho ya ek scene miss ho, dono cases ke liye
# ─────────────────────────────────────────────────────────────
def _fallback_prompt(scene: dict, style: str) -> str:
    visual = (scene.get("visual_description") or scene.get("narration") or "a relevant scene").strip()
    style = style or "cinematic, high quality, natural lighting"
    return f"{visual}, {style}"


def full_fallback(state: SceneAgentState) -> dict:
    style = (state.get("style_hint") or "cinematic, high quality, natural lighting, 35mm film look").strip()
    title = state.get("title", "")
    scenes = state["scenes"]

    out: list[ScenePrompt] = []
    for sc in scenes:
        scene_no = int(sc.get("scene_no", 0))
        out.append(
            ScenePrompt(
                scene_no=scene_no,
                narration=sc.get("narration", ""),
                duration_sec=sc.get("duration_sec", 0),
                image_prompt=_fallback_prompt(sc, style)[:600],
                negative_prompt=BASE_NEGATIVE_PROMPT,
                seed=deterministic_seed(title, str(scene_no)),
            )
        )

    logger.warning("[scene] full rule-based fallback used (%d scenes)", len(out))
    return {
        "global_style": style,
        "characters": [],
        "scenes_out": out,
        "used_fallback": True,
        "error": None,
    }


# ─────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────
def scene_node(state: SceneAgentState) -> dict:
    scenes = state.get("scenes") or []
    if not scenes:
        return {"error": "Script me koi scene nahi hai — Scene Agent ko kuch nahi mila",
                "scenes_out": []}

    aspect_ratio = state.get("aspect_ratio") or DEFAULT_ASPECT_RATIO
    if aspect_ratio not in VALID_ASPECT_RATIOS:
        logger.warning("[scene] invalid aspect_ratio %r -> default %s", aspect_ratio, DEFAULT_ASPECT_RATIO)
        aspect_ratio = DEFAULT_ASPECT_RATIO

    try:
        resp = get_llm().invoke(build_prompt(state))
        data = extract_json(resp.content or "")
        global_style, characters, scenes_out = normalize_output(
            data, scenes, state.get("title", ""), state.get("base_seed")
        )
        logger.info("[scene] ok | %d scenes | %d recurring character(s)", len(scenes_out), len(characters))
        return {
            "global_style": global_style,
            "characters": characters,
            "aspect_ratio_out": aspect_ratio,
            "scenes_out": scenes_out,
            "used_fallback": False,
            "error": None,
        }
    except Exception as e:                                    # noqa: BLE001
        logger.warning("[scene] LLM fail (%s) -> rule-based fallback", e)
        result = full_fallback({**state, "aspect_ratio": aspect_ratio})  # type: ignore[dict-item]
        result["aspect_ratio_out"] = aspect_ratio
        return result


# ─────────────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────────────
def build_scene_agent():
    builder = StateGraph(SceneAgentState)
    builder.add_node("scene_node", scene_node)
    builder.add_edge(START, "scene_node")
    builder.add_edge("scene_node", END)
    return builder.compile()


scene_agent = build_scene_agent()


def run_scene_agent(
    script: dict,
    aspect_ratio: AspectRatio = DEFAULT_ASPECT_RATIO,
    style_hint: Optional[str] = None,
    base_seed: Optional[int] = None,
) -> SceneAgentState:
    """script = approved Script Agent output OR Voice Agent output (scenes with real durations)."""
    return scene_agent.invoke(  # type: ignore[return-value]
        {
            "title": script.get("title", ""),
            "scenes": script.get("scenes", []),
            "language": script.get("language", "hinglish"),
            "aspect_ratio": aspect_ratio,
            "style_hint": style_hint,
            "base_seed": base_seed,
        }
    )


def from_approved(result: dict, **kwargs) -> SceneAgentState:
    """result = guardrails.get_approved_script() ka output. (Voice Agent se PARALLEL branch hai,
    isliye approved script seedha yahan aata hai, Voice Agent ka wait nahi karta.)"""
    if not result.get("approved"):
        return SceneAgentState(error=f"Script approved nahi hai ({result.get('verdict')}): "
                                      f"{result.get('reason')}", scenes_out=[])
    return run_scene_agent(result["script"], **kwargs)


# ─────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dummy_script = {
        "title": "AI in Agriculture",
        "scenes": [
            {"scene_no": 1, "narration": "AI ab kisano ki madad kar raha hai",
             "visual_description": "an Indian farmer looking at a drone flying over his field",
             "duration_sec": 8},
            {"scene_no": 2, "narration": "Sensors mitti ki nami batate hain",
             "visual_description": "the same farmer checking a soil sensor reading on his phone",
             "duration_sec": 6},
        ],
        "language": "hinglish",
    }

    out = run_scene_agent(dummy_script, aspect_ratio="9:16")
    if out.get("error"):
        print("ERROR:", out["error"])
    else:
        print("GLOBAL STYLE:", out["global_style"])
        print("CHARACTERS  :", out["characters"])
        print("FALLBACK?   :", out["used_fallback"])
        for s in out["scenes_out"]:
            print(f"\n--- Scene {s['scene_no']} (seed={s['seed']}, {s['duration_sec']}s)")
            print("  prompt   :", s["image_prompt"])
            print("  negative :", s["negative_prompt"][:60], "...")


            