# Vision-Fly X Paradox

**Turn a thought into a video.** Type an idea, drop a photo, or send a voice note — Vision-Fly writes the script, runs it through safety checks, voices it, generates the visuals, and cuts a finished reel. Fully local, no deployment required.

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-0aa76d)
![LangGraph](https://img.shields.io/badge/agents-LangGraph-4F9D6E)
![Status](https://img.shields.io/badge/status-local%20%2F%20in%20development-orange)

---

## What it does

Give it a topic — as text, a photo, or a voice note — and Vision-Fly runs it through a chain of AI agents that plan, write, check, voice, illustrate, and assemble a short video, entirely on your own machine.

```
                       👤 USER
                        │
          ┌─────────────┼─────────────┐
          │              │              │
        TEXT           IMAGE          VOICE
          │              │              │
          └─────────────┼─────────────┘
                        ↓
                  🧠 INPUT AGENT
                        ↓
                   📝 SCRIPT AGENT
                        ↓
                  GENERATED SCRIPT
                        ↓
          ┌─────────────┼─────────────┐
          ↓              ↓              ↓
       TOXICITY      COPYRIGHT       CULTURE
          │              │              │
          └─────────────┼─────────────┘
                        ↓
                  ⚖️  DECISION
                   /          \
                PASS         REWRITE
                 ↓              │
                 └──────←───────┘
                        ↓
                APPROVED SCRIPT
                  /            \
                 ↓              ↓
           🎙️ VOICE AGENT   🎬 SCENE AGENT
                 ↓              ↓
              AUDIO        IMAGE PROMPTS
                                 ↓
                          🖼️ IMAGE AGENT
                                 ↓
                              IMAGES
                                 ↓
                     🎥 VIDEO ASSEMBLER
                                 ↓
                        FINAL VIDEO.mp4
                                 ↓
                        📁 LOCAL STORAGE
                                 ↓
                           👤 USER
```

Every agent is built to fail gracefully — if one step breaks (a model is down, a rate limit hits), the rest of the pipeline doesn't crash with it.

## Features

- **Three ways in** — type a topic, upload a photo (captioned with BLIP), or drop/record a voice note (transcribed with speech-to-text)
- **Safety net built in** — every script is checked for toxicity, copyright/IP issues, and cultural sensitivity before it's allowed through, with an automatic rewrite loop if something needs fixing
- **Consistent visuals** — one agent plans a shared art style and recurring character descriptions across all scenes before any image is generated, so the video doesn't look like it was made by five different artists
- **Runs on free tiers** — Groq for the LLM calls, HuggingFace's free inference API for images, no paid API required to get started
- **A real UI, not just a script** — a dark, glassmorphic web studio with live progress tracking, a video player, and a library of everything you've generated
- **Nothing leaves your machine** — videos are stored locally in `video_library/`, no cloud upload

## Tech stack

| Layer | Tool |
|---|---|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Script / scene / safety reasoning | [Groq](https://groq.com) (`openai/gpt-oss-120b`) |
| Image captioning (photo input) | HuggingFace BLIP |
| Speech-to-text (voice input) | Google STT / `faster-whisper` |
| Text-to-speech | Sarvam AI |
| Image generation | HuggingFace Inference API (FLUX.1-schnell → SDXL fallback) |
| Video assembly | MoviePy + FFmpeg |
| Backend | FastAPI |
| Frontend | HTML / CSS / vanilla JS |

## Project structure

```
Vision-Fly X Paradox/
├── server.py              # FastAPI orchestrator — runs the full pipeline as a background job
├── input_agent.py         # text / image / voice → topic
├── script_agent.py        # topic → structured script (hook, scenes, CTA)
├── guardrails.py          # toxicity / copyright / culture checks + PASS-REWRITE decision loop
├── voice_agent.py         # script → narration audio (Sarvam TTS)
├── scene_agent.py         # script → consistent, image-generation-ready prompts
├── image_agent.py         # prompts → images (HuggingFace, free tier)
├── video_assembler.py     # images + audio → final .mp4, saved locally
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── video_library/         # generated videos + library.json (auto-created, gitignored)
├── .env                   # API keys (not committed)
└── README.md
```

## Getting started

### 1. Clone and set up a virtual environment

```bash
git clone https://github.com/<your-username>/vision-fly-x-paradox.git
cd vision-fly-x-paradox
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn python-multipart python-dotenv langgraph langchain-groq \
            huggingface_hub transformers pillow torch SpeechRecognition pydub \
            faster-whisper moviepy imageio-ffmpeg mutagen
```

> `imageio-ffmpeg` bundles its own FFmpeg binary, so you don't need to install FFmpeg system-wide.

### 3. Add your API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key
HF_TOKEN=your_huggingface_token
SARVAM_API_KEY=your_sarvam_key
```

All three have free tiers — this project was built to run without a paid API key.

### 4. Run it

```bash
uvicorn server:app --reload --port 8000
```

Open **http://localhost:8000** — the studio UI loads there.

## How a video gets made

1. **Input Agent** turns whatever you gave it (text, a photo, a voice note) into a single clean topic string.
2. **Script Agent** expands that topic into a full script: a hook, several scenes with narration + visual description, and a closing call-to-action.
3. **Guardrails** run three checks in parallel — toxicity, copyright/IP, cultural sensitivity — and either pass the script through or send it back for a rewrite (up to 2 attempts before it's flagged for manual review).
4. **Voice Agent** and **Scene Agent** run in parallel: one turns the script into narration audio, the other turns each scene into a style-consistent image prompt.
5. **Image Agent** generates one image per scene from those prompts.
6. **Video Assembler** stitches images and audio together, matching each scene's image to its exact narration length, and saves the final `.mp4` locally.

## Roadmap

Things planned but not yet built:

- [ ] User accounts + a MySQL layer to track who generated what (files stay on disk; only paths get stored in SQL)
- [ ] Text-cleaning pass on the generated script before it reaches the Voice Agent
- [ ] Configurable voice settings — gender, language, speed, style
- [ ] More video transitions — zoom-out, fade, blur, and a dedicated background/ambience agent

## License

MIT — see [LICENSE](LICENSE) for details.

## Author

Built by **Krishna**.

<img width="1796" height="802" alt="image" src="https://github.com/user-attachments/assets/50014e68-7ae2-41e2-843a-531d273278b9" />
