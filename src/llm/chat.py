"""Build grounded chat context and run report Q&A."""

from __future__ import annotations

from typing import Any, Optional

from src.llm.providers import SYSTEM_PROMPT, chat_completion


def build_report_context(
    *,
    markers: list[dict[str, Any]],
    patient_summary: str = "",
    report_date: Optional[str] = None,
    sex: Optional[str] = None,
    health_score: Optional[int] = None,
    filename: str = "",
) -> str:
    lines = [
        "REPORT CONTEXT (extracted programmatically):",
        f"- Filename: {filename or 'n/a'}",
        f"- Report date: {report_date or 'n/a'}",
        f"- Sex context: {sex or 'n/a'}",
        f"- Educational health score: {health_score if health_score is not None else 'n/a'}/100",
        "",
        "Biomarkers:",
    ]
    if not markers:
        lines.append("- None available")
    else:
        for m in markers:
            ref = ""
            if m.get("ref_low") is not None and m.get("ref_high") is not None:
                ref = f" (ref {m['ref_low']}–{m['ref_high']})"
            lines.append(
                f"- {m.get('name')}: {m.get('value')} {m.get('unit') or ''} "
                f"[{m.get('status', 'n/a')}]{ref}"
            )

    if patient_summary:
        lines.extend(["", "Patient summary:", patient_summary[:2500]])

    lines.extend(["", "Remember: educational discussion only; not a diagnosis."])
    return "\n".join(lines)


def ask_about_report(
    question: str,
    *,
    provider_name: str,
    report_context: str,
    history: list[dict[str, str]] | None = None,
    model: str | None = None,
) -> str:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": report_context},
    ]
    for turn in history or []:
        if turn.get("role") in {"user", "assistant"} and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question.strip()})
    return chat_completion(provider_name, messages, model=model).strip()


def offline_answer(question: str, markers: list[dict[str, Any]]) -> str:
    """Deterministic helper when no free LLM provider is configured."""
    q = question.lower().strip()
    if not markers:
        return (
            "No report biomarkers are loaded yet. Analyze a report first, then ask again.\n\n"
            "For free LLM chat:\n"
            "1. Install Ollama (https://ollama.com) then `ollama pull llama3.2`\n"
            "2. Or set `GROQ_API_KEY` / `GEMINI_API_KEY`"
        )

    flagged = [m for m in markers if m.get("status") not in {None, "normal"}]
    if any(k in q for k in ("abnormal", "flag", "concern", "outside", "borderline")):
        if not flagged:
            return (
                "From the extracted values, no biomarkers are currently flagged outside "
                "typical ranges. This is educational only — confirm with your clinician.\n\n"
                "_Offline helper — connect Ollama/Groq/Gemini for richer chat._"
            )
        bullets = "\n".join(
            f"- {m.get('name')}: {m.get('value')} {m.get('unit') or ''} ({m.get('status')})"
            for m in flagged
        )
        return (
            "These extracted markers are outside typical ranges:\n"
            f"{bullets}\n\n"
            "This tool does not diagnose conditions. Please review with a healthcare professional.\n\n"
            "_Offline helper — connect Ollama/Groq/Gemini for richer chat._"
        )

    hits = []
    for m in markers:
        name = (m.get("name") or "").lower()
        key = (m.get("key") or "").lower().replace("_", " ")
        if (name and name in q) or (key and key in q):
            hits.append(m)
    if hits:
        parts = []
        for m in hits:
            parts.append(
                f"**{m.get('name')}**: {m.get('value')} {m.get('unit') or ''} "
                f"(status: {m.get('status', 'n/a')}). {m.get('explanation', '')}"
            )
        return (
            "\n\n".join(parts)
            + "\n\nEducational only — not a diagnosis.\n\n"
            "_Offline helper — connect Ollama/Groq/Gemini for richer chat._"
        )

    names = ", ".join(str(m.get("name", "")) for m in markers[:12])
    return (
        "Offline mode can answer basic questions about extracted markers.\n\n"
        f"Available markers include: {names}.\n\n"
        "Try: “What looks abnormal?” or ask about a specific test like hemoglobin.\n\n"
        "Free LLM setup:\n"
        "1. Ollama → `ollama pull llama3.2`\n"
        "2. Groq → set `GROQ_API_KEY`\n"
        "3. Gemini → set `GEMINI_API_KEY`"
    )
