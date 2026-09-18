"""
Guardrail Agents + Decision Node
=================================
                 GENERATED SCRIPT
                        ↓
        ┌───────────────┼───────────────┐
     TOXICITY       COPYRIGHT        CULTURE      ← teeno PARALLEL chalte hain
        └───────────────┼───────────────┘
                        ↓
                   ⚖️  DECISION
                   /          \
                PASS         REWRITE  → wapas Script Agent ko feedback
                                       (aur BLOCK — jo unfixable ho)

Har check LLM se ek structured verdict leta hai:
    {passed, severity 0-3, issues: [{scene_no, problem, fix}]}

severity:
  0 = clean
  1 = minor  -> PASS ke saath note
  2 = major  -> REWRITE
  3 = severe -> BLOCK (rewrite se theek nahi hoga, ya illegal hai)

Design rules (baaki agents jaise hi):
  * Node kabhi raise nahi karta
  * LLM fail ho to FAIL-CLOSED — yaani REWRITE maano, PASS nahi.
    (Safety check crash hone ka matlab "sab theek hai" nahi hota.)
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
logger = logging.getLogger("guardrails")

GROQ_MODEL = os.getenv("GUARDRAIL_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))

REWRITE_AT = 2   # is severity se REWRITE
BLOCK_AT = 3     # is severity se BLOCK

Verdict = Literal["PASS", "REWRITE", "BLOCK"]


# ─────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────
class Issue(TypedDict, total=False):
    scene_no: Optional[int]
    problem: str
    fix: str


class CheckReport(TypedDict, total=False):
    check: str
    passed: bool
    severity: int
    issues: list[Issue]
    error: Optional[str]


class GuardState(TypedDict, total=False):
    # Script Agent se aata hai
    title: str
    hook: str
    scenes: list[dict]
    cta: str
    full_narration: str
    language: str

    # teeno parallel nodes alag-alag key likhte hain -> koi write conflict nahi
    toxicity_report: CheckReport
    copyright_report: CheckReport
    culture_report: CheckReport

    # decision node ka output
    verdict: Verdict
    feedback: Optional[str]
    max_severity: int
    error: Optional[str]


# ─────────────────────────────────────────────────────────────
# Lazy LLM — temperature 0, kyunki judgement consistent chahiye
# ─────────────────────────────────────────────────────────────
_llm: Any = None


def get_llm():
    global _llm
    if _llm is None:
        from langchain_groq import ChatGroq

        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY .env me set nahi hai")
        _llm = ChatGroq(model=GROQ_MODEL, temperature=0, max_retries=2, timeout=45)
        logger.info("Guardrail LLM loaded: %s", GROQ_MODEL)
    return _llm


def extract_json(raw: str) -> dict:
    if not raw or not raw.strip():
        raise ValueError("Khali response")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("JSON object nahi mila")
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", candidate))


# ─────────────────────────────────────────────────────────────
# Script ko review-friendly text me badalna
# ─────────────────────────────────────────────────────────────
def render_script(state: GuardState) -> str:
    lines = [f"TITLE: {state.get('title','')}", f"HOOK: {state.get('hook','')}", ""]
    for s in state.get("scenes") or []:
        lines.append(f"[Scene {s.get('scene_no')}]")
        lines.append(f"  narration: {s.get('narration','')}")
        lines.append(f"  visual   : {s.get('visual_description','')}")
    lines += ["", f"CTA: {state.get('cta','')}"]
    return "\n".join(lines)


VERDICT_SCHEMA = """{
  "passed": true/false,
  "severity": 0,
  "issues": [
    {"scene_no": 2, "problem": "kya galat hai", "fix": "isse kaise theek karein"}
  ]
}"""

SEVERITY_RULE = """severity scale:
  0 = bilkul clean, koi issue nahi
  1 = minor / borderline — publish ho sakta hai
  2 = major — script rewrite honi chahiye
  3 = severe — illegal ya rewrite se theek na hone layak

Agar severity 0 hai to "issues" khali array rakho.
Har issue me 'fix' ZAROORI hai — writer usi ko follow karke rewrite karega.
NOTE: 'visual' lines image generator ke liye hain, unko bhi check karo."""


def run_check(name: str, instruction: str, script_text: str) -> CheckReport:
    prompt = (
        f"{instruction}\n\n{SEVERITY_RULE}\n\n"
        f"--- SCRIPT ---\n{script_text}\n--- END ---\n\n"
        f"Return ONLY valid JSON, no markdown, no commentary:\n{VERDICT_SCHEMA}"
    )
    try:
        resp = get_llm().invoke(prompt)
        data = extract_json(resp.content or "")

        severity = int(data.get("severity") or 0)
        severity = max(0, min(3, severity))

        issues: list[Issue] = []
        for it in data.get("issues") or []:
            if not isinstance(it, dict):
                continue
            problem = str(it.get("problem") or "").strip()
            if not problem:
                continue
            scene_no = it.get("scene_no")
            issues.append(
                Issue(
                    scene_no=int(scene_no) if isinstance(scene_no, (int, float)) else None,
                    problem=problem,
                    fix=str(it.get("fix") or "").strip() or "is hisse ko safe alternative se badlo",
                )
            )

        # LLM aksar passed=true ke saath severity 2 de deta hai — severity ko final maano
        if issues and severity == 0:
            severity = 1
        passed = severity < REWRITE_AT

        logger.info("[%s] severity=%d issues=%d", name, severity, len(issues))
        return CheckReport(check=name, passed=passed, severity=severity, issues=issues, error=None)

    except Exception as e:  # noqa: BLE001
        # FAIL-CLOSED: check hi nahi chala to safe nahi maan sakte
        logger.error("[%s] fail: %s -> fail-closed (severity 2)", name, e)
        return CheckReport(
            check=name,
            passed=False,
            severity=REWRITE_AT,
            issues=[Issue(scene_no=None, problem=f"{name} check nahi chal paaya: {e}",
                          fix="script dobara generate karo")],
            error=str(e),
        )


# ─────────────────────────────────────────────────────────────
# Check 1 — TOXICITY
# ─────────────────────────────────────────────────────────────
TOXICITY_PROMPT = """You are a content safety reviewer for short-form videos.

Flag: gaali/profanity, hate speech (religion, caste, gender, region, disability),
harassment or personal attacks, violence ya gore, self-harm ya suicide content,
sexual/adult content, dangerous instructions (weapons, drugs, illegal acts),
minors ko involve karne wala unsafe content, aur medical/financial claims jo
nuksaan pahuncha sakte hain.

Severity 3 sirf tab jab content illegal ho ya serious real-world harm kare."""


def toxicity_node(state: GuardState) -> dict:
    return {"toxicity_report": run_check("toxicity", TOXICITY_PROMPT, render_script(state))}


# ─────────────────────────────────────────────────────────────
# Check 2 — COPYRIGHT / IP
# ─────────────────────────────────────────────────────────────
COPYRIGHT_PROMPT = """You are an IP and copyright reviewer for short-form videos.

Flag: song lyrics ya poem lines (ek line bhi), film/TV dialogues, kisi book ya
article se copy kiya gaya text, brand names aur logos (Nike, Coca-Cola, etc.),
copyrighted characters (Mickey Mouse, Marvel, anime characters, game characters),
real identifiable public figures (actors, politicians, cricketers) jinke naam ya
shakal use ho rahi hai, trademarked taglines, aur kisi specific artist ke style
ki nakal ("in the style of <living artist>").

'visual' lines ko dhyan se dekho — image generators ke liye brand/character
mention sabse bada risk wahan hota hai.

Severity 3 sirf tab jab poora script hi kisi copyrighted work ka reproduction ho."""

# saste, deterministic pre-filter — LLM miss kar de to bhi ye pakad le
BRAND_PATTERNS = re.compile(
    r"\b(nike|adidas|puma|coca[- ]?cola|pepsi|mcdonald'?s|starbucks|disney|marvel|"
    r"pixar|pokemon|mickey mouse|spider[- ]?man|batman|superman|iron man|"
    r"apple iphone|samsung galaxy|netflix|youtube logo|instagram logo|"
    r"tata|reliance|jio|amul|maggi)\b",
    re.IGNORECASE,
)


def copyright_node(state: GuardState) -> dict:
    script_text = render_script(state)
    report = run_check("copyright", COPYRIGHT_PROMPT, script_text)

    hits = sorted({m.group(0).lower() for m in BRAND_PATTERNS.finditer(script_text)})
    if hits:
        existing = " ".join(i["problem"].lower() for i in report.get("issues", []))
        new = [h for h in hits if h not in existing]
        if new:
            report["issues"] = list(report.get("issues", [])) + [
                Issue(scene_no=None,
                      problem=f"Brand/IP naam mila: {', '.join(new)}",
                      fix="brand name hatao, generic description use karo "
                          "(jaise 'a sports shoe', 'a cola bottle')")
            ]
            report["severity"] = max(int(report.get("severity", 0)), REWRITE_AT)
            report["passed"] = False
            logger.info("[copyright] regex pre-filter hits: %s", new)

    return {"copyright_report": report}


# ─────────────────────────────────────────────────────────────
# Check 3 — CULTURE / SENSITIVITY  (India-focused)
# ─────────────────────────────────────────────────────────────
CULTURE_PROMPT = """You are a cultural sensitivity reviewer for videos made mainly
for an Indian audience (Hindi/Hinglish/English).

Flag: religious disrespect ya gods/rituals ka casual use, caste references ya
casteist stereotypes, regional/linguistic stereotypes (North vs South, Bihari,
Madrasi jaise terms), gender stereotypes, skin-colour ya body shaming,
communal ya politically inflammatory content, national symbols ka galat use
(flag, anthem, army), aur sensitive historical/political events jinpe
unnecessary opinion diya gaya ho.

Ye bhi dekho ki 'visual' descriptions me clothing, rituals ya festivals
galat tarike se to depict nahi ho rahe.

Severity 3 sirf tab jab content communal hatred bhadka sakta ho."""


def culture_node(state: GuardState) -> dict:
    return {"culture_report": run_check("culture", CULTURE_PROMPT, render_script(state))}


# ─────────────────────────────────────────────────────────────
# DECISION node — fan-in
# ─────────────────────────────────────────────────────────────
def build_feedback(reports: list[CheckReport]) -> str:
    """Script Agent ke liye actionable rewrite instructions."""
    lines: list[str] = []
    for r in reports:
        if not r.get("issues"):
            continue
        lines.append(f"[{r['check'].upper()} — severity {r.get('severity')}]")
        for i in r["issues"]:
            where = f"Scene {i['scene_no']}" if i.get("scene_no") else "Overall"
            lines.append(f"  - {where}: {i['problem']} → FIX: {i.get('fix','')}")
    return "\n".join(lines)


def decision_node(state: GuardState) -> dict:
    reports = [
        state.get("toxicity_report") or CheckReport(check="toxicity", severity=0, issues=[]),
        state.get("copyright_report") or CheckReport(check="copyright", severity=0, issues=[]),
        state.get("culture_report") or CheckReport(check="culture", severity=0, issues=[]),
    ]
    max_sev = max(int(r.get("severity", 0)) for r in reports)

    if max_sev >= BLOCK_AT:
        verdict: Verdict = "BLOCK"
    elif max_sev >= REWRITE_AT:
        verdict = "REWRITE"
    else:
        verdict = "PASS"

    feedback = build_feedback(reports) if verdict != "PASS" else None

    logger.info("[decision] %s (max severity %d)", verdict, max_sev)
    return {"verdict": verdict, "max_severity": max_sev, "feedback": feedback, "error": None}


# ─────────────────────────────────────────────────────────────
# Graph — fan-out / fan-in
# START se teeno edges ek hi superstep me chalte hain (parallel),
# decision_node teeno ke complete hone ka wait karta hai.
# ─────────────────────────────────────────────────────────────
def build_guardrail_agent():
    builder = StateGraph(GuardState)
    builder.add_node("toxicity_node", toxicity_node)
    builder.add_node("copyright_node", copyright_node)
    builder.add_node("culture_node", culture_node)
    builder.add_node("decision_node", decision_node)

    for node in ("toxicity_node", "copyright_node", "culture_node"):
        builder.add_edge(START, node)
        builder.add_edge(node, "decision_node")

    builder.add_edge("decision_node", END)
    return builder.compile()


guardrail_agent = build_guardrail_agent()


def run_guardrails(script: dict) -> GuardState:
    """script = Script Agent ka output dict."""
    if script.get("error"):
        return GuardState(verdict="BLOCK", feedback=f"Script Agent fail: {script['error']}",
                          max_severity=3)
    if not script.get("scenes"):
        return GuardState(verdict="BLOCK", feedback="Script me koi scene nahi hai", max_severity=3)

    return guardrail_agent.invoke(  # type: ignore[return-value]
        {
            "title": script.get("title", ""),
            "hook": script.get("hook", ""),
            "scenes": script.get("scenes", []),
            "cta": script.get("cta", ""),
            "full_narration": script.get("full_narration", ""),
            "language": script.get("language", "hinglish"),
        }
    )


# ─────────────────────────────────────────────────────────────
# ORCHESTRATOR — Script ↔ Guardrail ka poora PASS/REWRITE loop
# ─────────────────────────────────────────────────────────────
def get_approved_script(topic: str, source_text: str = "", **kwargs) -> dict:
    """
    Loop: script banao -> check karo -> REWRITE aaye to feedback ke saath dobara.
    Return: {"approved": bool, "script": {...}, "verdict": ..., "reports": {...}}
    """
    from script_agent_c import MAX_REWRITES, run_script_agent

    feedback: Optional[str] = None
    rewrite_count = 0
    script: dict = {}
    guard: GuardState = GuardState()

    while rewrite_count <= MAX_REWRITES:
        script = run_script_agent(
            topic, source_text=source_text, feedback=feedback,
            rewrite_count=rewrite_count, **kwargs
        )
        if script.get("error"):
            return {"approved": False, "script": script, "verdict": "BLOCK",
                    "reason": script["error"]}

        guard = run_guardrails(script)
        verdict = guard.get("verdict")

        if verdict == "PASS":
            logger.info("APPROVED after %d rewrite(s)", rewrite_count)
            return {"approved": True, "script": script, "verdict": "PASS",
                    "reports": _reports(guard), "rewrites": rewrite_count}

        if verdict == "BLOCK":
            logger.warning("BLOCKED — rewrite se theek nahi hoga")
            return {"approved": False, "script": script, "verdict": "BLOCK",
                    "reason": guard.get("feedback"), "reports": _reports(guard)}

        feedback = guard.get("feedback")
        rewrite_count += 1
        logger.info("REWRITE #%d", rewrite_count)

    return {"approved": False, "script": script, "verdict": "REWRITE",
            "reason": f"{MAX_REWRITES} rewrites ke baad bhi pass nahi hua — manual review",
            "reports": _reports(guard)}


def _reports(guard: GuardState) -> dict:
    return {
        "toxicity": guard.get("toxicity_report"),
        "copyright": guard.get("copyright_report"),
        "culture": guard.get("culture_report"),
    }


# ─────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = get_approved_script("AI in Agriculture", duration_sec=45, language="hinglish")

    print("\nVERDICT :", result["verdict"], "| approved:", result["approved"])
    if result["approved"]:
        print("REWRITES:", result["rewrites"])
        for s in result["script"]["scenes"]:
            print(f"  Scene {s['scene_no']}: {s['narration']}")
    else:
        print("REASON  :", result.get("reason"))