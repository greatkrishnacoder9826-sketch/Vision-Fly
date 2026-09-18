from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("image_agent")


HF_TOKEN = os.getenv("HF_TOKEN")

# Order matters — pehla free/fast, phir fallback. Free tier pe FLUX.1-schnell
# sabse reliable mila hai (fast, kam queue); SDXL fallback ke liye achha hai.
DEFAULT_MODELS = [
    m.strip() for m in os.getenv(
        "HF_IMAGE_MODELS",
        "black-forest-labs/FLUX.1-schnell,stabilityai/stable-diffusion-xl-base-1.0"
    ).split(",") if m.strip()
]

# Models jo negative_prompt / guidance_scale accept NAHI karte (distilled, no-CFG).
# Inke liye wahi params bhej kar TypeError khaane ki zarurat nahi.
NO_CFG_MODELS = {"black-forest-labs/flux.1-schnell", "black-forest-labs/flux.1-dev"}

MAX_RETRIES = 3
MAX_MODEL_LOAD_WAIT = 60      # HF ka estimated_time isse zyada ho to us model ko skip karo
MAX_WORKERS = 2               # free tier rate limit ke andar rakhna — 4+ pe 429 milna shuru ho jaata hai
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 7.0

# aspect_ratio -> (width, height), SD/FLUX ke liye 8 ke multiple me
ASPECT_DIMENSIONS = {
    "9:16": (768, 1344),
    "16:9": (1344, 768),
    "1:1": (1024, 1024),
    "4:5": (896, 1120),
}
DEFAULT_DIMENSIONS = (768, 1344)


class GeneratedImage(TypedDict, total=False):
    scene_no: int
    image_path: str
    model_used: str
    seed: int
    error: Optional[str]


class ImageAgentState(TypedDict, total=False):
    # Scene Agent se aata hai
    scenes_out: list[dict]        # scene_no, image_prompt, negative_prompt, seed, duration_sec
    aspect_ratio_out: str

    # config
    output_dir: str
    models: list[str]

    # output
    images: list[GeneratedImage]
    error: Optional[str]



_client: Any = None


def get_client():
    global _client
    if _client is None:
        from huggingface_hub import InferenceClient

        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN .env me set nahi hai (free account se bhi mil jaata hai)")
        _client = InferenceClient(token=HF_TOKEN, timeout=120)
        logger.info("HF InferenceClient ready")
    return _client


def _parse_wait_seconds(err: Exception) -> Optional[float]:
    """HF ka 503 error message me 'estimated_time' hota hai (seconds). Usse nikalo."""
    match = re.search(r"estimated[_ ]time['\"]?\s*[:=]\s*([\d.]+)", str(err), re.IGNORECASE)
    return float(match.group(1)) if match else None


def _is_no_cfg_model(model: str) -> bool:
    return model.strip().lower() in NO_CFG_MODELS



def _generate_with_model(
    model: str, prompt: str, negative_prompt: str, seed: int, width: int, height: int,
) -> bytes:
    client = get_client()
    kwargs: dict[str, Any] = {"model": model, "width": width, "height": height, "seed": seed}

    if not _is_no_cfg_model(model):
        kwargs["negative_prompt"] = negative_prompt
        kwargs["guidance_scale"] = DEFAULT_GUIDANCE
        kwargs["num_inference_steps"] = DEFAULT_STEPS
    else:
        # schnell distilled hai — kam steps me hi best result, zyada steps waste
        kwargs["num_inference_steps"] = 4

    try:
        image = client.text_to_image(prompt, **kwargs)
    except TypeError:
        # is model version ne shayad koi param accept nahi kiya — bare-minimum se retry
        logger.warning("[%s] extra kwargs rejected -> bare call retry", model)
        image = client.text_to_image(prompt, model=model, width=width, height=height)

    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def synthesize_image(
    scene_no: int, prompt: str, negative_prompt: str, seed: int,
    width: int, height: int, out_dir: Path, models: list[str],
) -> GeneratedImage:
    prompt = (prompt or "").strip()
    if not prompt:
        return GeneratedImage(scene_no=scene_no, error="image_prompt khali hai")

    last_err: Optional[Exception] = None
    for model in models:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                image_bytes = _generate_with_model(model, prompt, negative_prompt, seed, width, height)
                if not image_bytes:
                    raise RuntimeError("Model ne khali image di")

                out_path = out_dir / f"scene_{scene_no}.png"
                out_path.write_bytes(image_bytes)
                logger.info("[scene_%d] ok | model=%s | attempt %d", scene_no, model, attempt)
                return GeneratedImage(
                    scene_no=scene_no, image_path=str(out_path),
                    model_used=model, seed=seed, error=None,
                )

            except Exception as e:                            # noqa: BLE001
                last_err = e
                wait = _parse_wait_seconds(e)

                if wait is not None:
                    if wait > MAX_MODEL_LOAD_WAIT:
                        logger.warning("[scene_%d] %s loading %.0fs — too long, next model", scene_no, model, wait)
                        break                                   # is model ko skip karo, agla try
                    logger.info("[scene_%d] %s cold-starting, waiting %.0fs", scene_no, model, wait)
                    time.sleep(wait + 1)
                    continue                                    # load hone ke baad dobara try (attempt consume nahi)

                msg = str(e).lower()
                if "401" in msg or "unauthorized" in msg or "invalid token" in msg:
                    logger.error("[scene_%d] auth error, saare models skip: %s", scene_no, e)
                    return GeneratedImage(scene_no=scene_no, error=f"HF auth fail: {e}")

                logger.warning("[scene_%d] %s attempt %d/%d fail: %s", scene_no, model, attempt, MAX_RETRIES, e)
                if attempt < MAX_RETRIES:
                    time.sleep(2 * attempt)

        # is model ke saare attempts khatam -> agla model try karo
        logger.warning("[scene_%d] model %s exhausted, trying next", scene_no, model)

    return GeneratedImage(scene_no=scene_no, error=f"Saare models fail ho gaye: {last_err}")


def image_node(state: ImageAgentState) -> dict:
    scenes = state.get("scenes_out") or []
    if not scenes:
        return {"error": "Scene Agent se koi scene prompt nahi mila", "images": []}

    aspect_ratio = state.get("aspect_ratio_out", "9:16")
    width, height = ASPECT_DIMENSIONS.get(aspect_ratio, DEFAULT_DIMENSIONS)
    models = state.get("models") or DEFAULT_MODELS
    if not models:
        return {"error": "Koi image model configure nahi hai (HF_IMAGE_MODELS)", "images": []}

    out_dir = Path(state.get("output_dir") or (Path.cwd() / "output_images"))
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[image] %d scenes | %dx%d | models=%s", len(scenes), width, height, models)

    results: dict[int, GeneratedImage] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(scenes))) as pool:
        futures = {
            pool.submit(
                synthesize_image,
                sc.get("scene_no", i),
                sc.get("image_prompt", ""),
                sc.get("negative_prompt", ""),
                sc.get("seed", 0),
                width, height, out_dir, models,
            ): sc.get("scene_no", i)
            for i, sc in enumerate(scenes, start=1)
        }
        for fut in as_completed(futures):
            scene_no = futures[fut]
            try:
                results[scene_no] = fut.result()
            except Exception as e:                            # noqa: BLE001
                results[scene_no] = GeneratedImage(scene_no=scene_no, error=f"Unexpected: {e}")

    ordered = [results[sc.get("scene_no", i)] for i, sc in enumerate(scenes, start=1)]
    failed = [img for img in ordered if img.get("error")]

    if failed:
        details = "; ".join(f"scene {img['scene_no']}: {img['error']}" for img in failed)
        return {"images": ordered, "error": f"{len(failed)}/{len(ordered)} image(s) fail: {details}"}

    logger.info("[image] all %d images ok", len(ordered))
    return {"images": ordered, "error": None}


def build_image_agent():
    builder = StateGraph(ImageAgentState)
    builder.add_node("image_node", image_node)
    builder.add_edge(START, "image_node")
    builder.add_edge("image_node", END)
    return builder.compile()


image_agent = build_image_agent()


def run_image_agent(
    scene_result: dict,
    output_dir: Optional[str] = None,
    models: Optional[list[str]] = None,
) -> ImageAgentState:
    """scene_result = Scene Agent ka output (scenes_out + aspect_ratio_out)."""
    if scene_result.get("error") and not scene_result.get("scenes_out"):
        return ImageAgentState(error=f"Scene Agent fail: {scene_result['error']}", images=[])

    return image_agent.invoke(  # type: ignore[return-value]
        {
            "scenes_out": scene_result.get("scenes_out", []),
            "aspect_ratio_out": scene_result.get("aspect_ratio_out", "9:16"),
            "output_dir": output_dir,
            "models": models,
        }
    )


def from_scene_agent(scene_result: dict, **kwargs) -> ImageAgentState:
    return run_image_agent(scene_result, **kwargs)



if __name__ == "__main__":
    dummy_scenes = {
        "scenes_out": [
            {"scene_no": 1, "image_prompt": "Indian farmer in white kurta looking at a drone "
                                             "over a green wheat field, cinematic, golden hour",
             "negative_prompt": "text, watermark, blurry", "seed": 12345, "duration_sec": 8},
            {"scene_no": 2, "image_prompt": "same farmer checking a soil sensor reading on his "
                                             "phone, close up, natural lighting",
             "negative_prompt": "text, watermark, blurry", "seed": 12346, "duration_sec": 6},
        ],
        "aspect_ratio_out": "9:16",
    }

    out = run_image_agent(dummy_scenes)
    if out.get("error"):
        print("ERROR:", out["error"])
    for img in out.get("images", []):
        status = img.get("error") or f"ok -> {img.get('image_path')} ({img.get('model_used')})"
        print(f"Scene {img['scene_no']}: {status}")