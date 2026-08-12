"""Free LLM providers for report chat (Ollama / Groq / Gemini)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import httpx

SYSTEM_PROMPT = """You are a careful educational assistant for lab report literacy.
You help users understand extracted biomarker values in plain language.

Hard rules:
- Do NOT diagnose diseases.
- Do NOT prescribe treatments, dosages, or medications.
- Do NOT claim certainty about the user's health.
- Encourage discussing results with a qualified clinician.
- Only use the provided report context; if something is missing, say so.
- Keep answers concise (short paragraphs or bullets).
- Start with educational framing when discussing abnormal values.
"""


@dataclass
class LLMProviderStatus:
    name: str
    available: bool
    detail: str
    model: str = ""


_PROVIDER_CACHE: tuple[float, list["LLMProviderStatus"]] | None = None
_PROVIDER_CACHE_TTL_SEC = 45.0
_OLLAMA_PROBE_TIMEOUT_SEC = 0.35


def clear_provider_cache() -> None:
    """Drop cached provider probe results (e.g. after env/key changes)."""
    global _PROVIDER_CACHE
    _PROVIDER_CACHE = None


def _streamlit_secret(key: str) -> Optional[str]:
    try:
        import streamlit as st

        val = st.secrets.get(key)
        return str(val) if val else None
    except Exception:
        return None


def detect_providers(*, force_refresh: bool = False) -> list[LLMProviderStatus]:
    """Probe LLM backends. Results are cached briefly to keep the UI snappy."""
    global _PROVIDER_CACHE
    now = time.monotonic()
    if (
        not force_refresh
        and _PROVIDER_CACHE is not None
        and now - _PROVIDER_CACHE[0] < _PROVIDER_CACHE_TTL_SEC
    ):
        return _PROVIDER_CACHE[1]

    statuses: list[LLMProviderStatus] = []

    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
    try:
        with httpx.Client(timeout=_OLLAMA_PROBE_TIMEOUT_SEC) as client:
            r = client.get(f"{ollama_host}/api/tags")
            if r.status_code == 200:
                models = [m.get("name", "") for m in r.json().get("models", [])]
                chosen = (
                    ollama_model
                    if any(ollama_model in m for m in models)
                    else (models[0] if models else ollama_model)
                )
                statuses.append(
                    LLMProviderStatus(
                        "Ollama (local, free)",
                        True,
                        f"Connected · {len(models)} model(s)",
                        chosen,
                    )
                )
            else:
                statuses.append(
                    LLMProviderStatus("Ollama (local, free)", False, f"HTTP {r.status_code}", ollama_model)
                )
    except Exception:
        statuses.append(
            LLMProviderStatus(
                "Ollama (local, free)",
                False,
                "Not running — install from https://ollama.com and run `ollama pull llama3.2`",
                ollama_model,
            )
        )

    groq_key = os.getenv("GROQ_API_KEY") or _streamlit_secret("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    if groq_key and groq_key not in {"", "PASTE_YOUR_GROQ_KEY_HERE"}:
        statuses.append(LLMProviderStatus("Groq (free API tier)", True, "API key detected", groq_model))
    else:
        statuses.append(
            LLMProviderStatus(
                "Groq (free API tier)",
                False,
                "Set GROQ_API_KEY in env or .streamlit/secrets.toml",
                groq_model,
            )
        )

    gemini_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or _streamlit_secret("GEMINI_API_KEY")
        or _streamlit_secret("GOOGLE_API_KEY")
    )
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    if gemini_key:
        statuses.append(
            LLMProviderStatus("Google Gemini (free API tier)", True, "API key detected", gemini_model)
        )
    else:
        statuses.append(
            LLMProviderStatus(
                "Google Gemini (free API tier)",
                False,
                "Set GEMINI_API_KEY in env or .streamlit/secrets.toml",
                gemini_model,
            )
        )

    _PROVIDER_CACHE = (now, statuses)
    return statuses


def chat_completion(
    provider_name: str,
    messages: list[dict[str, str]],
    *,
    model: Optional[str] = None,
) -> str:
    providers = {p.name: p for p in detect_providers()}
    status = providers.get(provider_name)
    if not status or not status.available:
        raise RuntimeError(f"Provider unavailable: {provider_name}")

    use_model = model or status.model
    if provider_name.startswith("Ollama"):
        return _chat_ollama(messages, use_model)
    if provider_name.startswith("Groq"):
        return _chat_groq(messages, use_model)
    if provider_name.startswith("Google Gemini"):
        return _chat_gemini(messages, use_model)
    raise RuntimeError(f"Unknown provider: {provider_name}")


def _chat_ollama(messages: list[dict[str, str]], model: str) -> str:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{host}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
    return (data.get("message") or {}).get("content") or data.get("response") or ""


def _chat_groq(messages: list[dict[str, str]], model: str) -> str:
    api_key = os.getenv("GROQ_API_KEY") or _streamlit_secret("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    with httpx.Client(timeout=60.0) as client:
        r = client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"]


def _chat_gemini(messages: list[dict[str, str]], model: str) -> str:
    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or _streamlit_secret("GEMINI_API_KEY")
        or _streamlit_secret("GOOGLE_API_KEY")
    )
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing")

    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload: dict = {
        "contents": contents,
        "generationConfig": {"temperature": 0.2},
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, params={"key": api_key}, json=payload)
        r.raise_for_status()
        data = r.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {json.dumps(data)[:400]}")
    parts = candidates[0].get("content", {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts)
