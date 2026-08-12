"""
AI Medical Test Report Analyzer — Phase 5

Phase 1: Upload · OCR · Extract · Explain · Score
Phase 2: SQLite persistence · History · Compare · Trends
Phase 3: Educational risk estimation
Phase 4: Free LLM chat (Ollama / Groq / Gemini)
Phase 5: Clinician PDF export · FastAPI backend
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.explanations import DISCLAIMER, build_doctor_summary, build_patient_summary
from src.analysis.health_score import compute_health_score
from src.analysis.status import analyze_biomarkers
from src.db.repository import (
    compare_reports,
    count_reports,
    delete_all_reports,
    delete_report,
    delete_reports,
    ensure_db,
    export_report_json,
    get_report,
    list_reports,
    save_report,
    trend_rows,
)
from src.export.pdf_report import build_doctor_pdf
from src.llm.chat import ask_about_report, build_report_context, offline_answer
from src.llm.providers import detect_providers
from src.ml.features import RISK_FEATURE_SPECS
from src.ml.predict import models_available, predict_all
from src.ocr.extractor import extract_text, ocr_status
from src.parsing.biomarkers import parse_report_text
from src.utils.file_handlers import is_supported, read_upload_bytes
from src.utils.history import analyzed_to_dict_list

st.set_page_config(
    page_title="MediParse · Lab Report Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

STATUS_EMOJI = {
    "normal": "🟢",
    "borderline": "🟡",
    "low": "🔴",
    "high": "🔴",
}

CATEGORY_ICON = {
    "ok": "✔",
    "warn": "⚠",
    "alert": "✖",
}

DIRECTION_LABEL = {
    "up": "↑ higher",
    "down": "↓ lower",
    "unchanged": "→ same",
    "new": "+ new",
    "missing": "– missing",
}


PAGES = [
    "Home",
    "Analyze",
    "History",
    "Compare",
    "Trends",
    "Risk",
    "Chat",
]


def init_state() -> None:
    ensure_db()
    if "last_text" not in st.session_state:
        st.session_state.last_text = ""
    if "sex_override" not in st.session_state:
        st.session_state.sex_override = "auto"
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Home"
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "light"
    if "_reports_cache_v" not in st.session_state:
        st.session_state._reports_cache_v = 0


def bump_reports_cache() -> None:
    st.session_state._reports_cache_v = int(st.session_state.get("_reports_cache_v", 0)) + 1


@st.cache_data(ttl=20, show_spinner=False)
def cached_report_count(_version: int) -> int:
    return count_reports()


@st.cache_data(ttl=20, show_spinner=False)
def cached_report_summaries(_version: int) -> list[dict]:
    return [
        {
            "id": r.id,
            "filename": r.filename,
            "report_date": r.report_date,
            "sex": r.sex,
            "health_score": r.health_score,
            "created_at": r.created_at,
            "marker_count": len(r.markers),
        }
        for r in list_reports()
    ]


@st.cache_data(ttl=30, show_spinner=False)
def cached_workspace_status(_version: int) -> dict:
    llm = detect_providers()
    online = [p for p in llm if p.available]
    return {
        "report_count": count_reports(),
        "risk_ready": models_available(),
        "llm_ready": bool(online),
        "llm_label": online[0].name.split("(")[0].strip() if online else "",
        "ocr": ocr_status(),
    }


def go_to(page: str) -> None:
    """Navigate from CTA buttons; keep pills in sync for the next run."""
    if page in PAGES:
        st.session_state.active_page = page
        st.session_state.top_nav_pills = page
        st.rerun()


def render_navbar() -> str:
    """Compact top navbar: brand + settings, then full-width page pills."""
    current = st.session_state.get("active_page", "Home")
    if "nav_dark_toggle" not in st.session_state:
        st.session_state.nav_dark_toggle = st.session_state.get("theme_mode", "light") == "dark"
    if current not in PAGES:
        current = "Home"
        st.session_state.active_page = current

    # Initialize pills once; never overwrite after user clicks (that blocked navigation).
    if "top_nav_pills" not in st.session_state:
        st.session_state.top_nav_pills = current

    ws = cached_workspace_status(st.session_state.get("_reports_cache_v", 0))
    llm_chip = (ws["llm_label"] or "LLM")[:12] if ws["llm_ready"] else "Offline"
    chips = [
        ("ok" if ws["ocr"].get("rapidocr") else "warn", "OCR"),
        ("info", f"{ws['report_count']} saved"),
        ("ok" if ws["risk_ready"] else "muted", "Risk" if ws["risk_ready"] else "Risk off"),
        ("ok" if ws["llm_ready"] else "muted", llm_chip),
    ]
    chips_html = "".join(
        f'<span class="mp-chip mp-chip-{tone}">{label}</span>' for tone, label in chips
    )

    selected = st.session_state.get("top_nav_pills") or current
    brand_col, sex_col, theme_col = st.columns([5.4, 1.6, 1.7], gap="small")
    with brand_col:
        st.markdown(
            f"""
            <div class="mp-nav-card">
              <div class="mp-nav-card-row">
                <div class="mp-nav-brand">
                  <div class="mp-nav-brand-name">MediParse</div>
                  <div class="mp-nav-brand-page">{selected if selected in PAGES else current}</div>
                </div>
                <div class="mp-nav-chips">{chips_html}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sex_col:
        sex = st.selectbox(
            "Sex for reference ranges",
            options=["auto", "female", "male"],
            format_func=lambda x: {
                "auto": "Sex · Auto",
                "female": "Sex · Female",
                "male": "Sex · Male",
            }[x],
            key="sex_override",
            label_visibility="collapsed",
            help="Biological sex used for reference ranges",
        )
    with theme_col:
        st.toggle(
            "Dark",
            key="nav_dark_toggle",
            help="Switch light / dark theme",
        )
    st.session_state.theme_toggle = bool(st.session_state.get("nav_dark_toggle", False))
    st.session_state.theme_mode = "dark" if st.session_state.theme_toggle else "light"

    selected = st.pills(
        "Navigate",
        options=PAGES,
        selection_mode="single",
        key="top_nav_pills",
        label_visibility="collapsed",
    )
    # Pills drive navigation. Do not overwrite top_nav_pills here — that blocked clicks.
    if selected in PAGES:
        st.session_state.active_page = selected

    return sex


def inject_styles() -> None:
    if "theme_toggle" in st.session_state:
        st.session_state.theme_mode = "dark" if st.session_state.theme_toggle else "light"
    dark = st.session_state.get("theme_mode", "light") == "dark"
    scheme = "dark" if dark else "light"
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,560;9..144,700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

        :root {{
          --mp-bg: {"#0b1220" if dark else "#f4f7f6"};
          --mp-surface: {"#111827" if dark else "#ffffff"};
          --mp-surface-2: {"#1e293b" if dark else "#f8fafc"};
          --mp-text: {"#e2e8f0" if dark else "#0f172a"};
          --mp-muted: {"#cbd5e1" if dark else "#64748b"};
          --mp-body: {"#cbd5e1" if dark else "#475569"};
          --mp-border: {"#475569" if dark else "#dbe5e2"};
          --mp-border-soft: {"#334155" if dark else "#e2e8f0"};
          --mp-accent: {"#2dd4bf" if dark else "#0f766e"};
          --mp-accent-strong: #0f766e;
          --mp-shadow: {"0 1px 2px rgba(0,0,0,0.35)" if dark else "0 1px 2px rgba(15, 23, 42, 0.04)"};
          --mp-secondary-btn-bg: {"#334155" if dark else "#ffffff"};
          --mp-secondary-btn-border: {"#94a3b8" if dark else "#cbd5e1"};
          --mp-meter-track: {"#334155" if dark else "#f1f5f9"};
          --mp-input-bg: {"#0f172a" if dark else "#ffffff"};
          --mp-alert-success-bg: {"#064e3b" if dark else "#ecfdf5"};
          --mp-alert-success-text: {"#a7f3d0" if dark else "#065f46"};
          --mp-alert-info-bg: {"#0c4a6e" if dark else "#eff6ff"};
          --mp-alert-info-text: {"#bae6fd" if dark else "#1e40af"};
          --mp-alert-warn-bg: {"#78350f" if dark else "#fffbeb"};
          --mp-alert-warn-text: {"#fde68a" if dark else "#92400e"};
          --mp-alert-error-bg: {"#7f1d1d" if dark else "#fef2f2"};
          --mp-alert-error-text: {"#fecaca" if dark else "#991b1b"};
        }}

        :root, html, body, .stApp {{
          color-scheme: {scheme} !important;
          font-family: 'Source Sans 3', sans-serif;
          {"--background-color: #0b1220; --secondary-background-color: #111827; --text-color: #e2e8f0;" if dark else ""}
        }}
        .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main,
        .main .block-container {{
          background: var(--mp-bg) !important;
          color: var(--mp-text) !important;
        }}
        .block-container {{
          padding-top: 3.25rem !important;
          padding-bottom: 2.5rem;
          max-width: 1280px;
        }}
        header[data-testid="stHeader"] {{
          background: transparent !important;
          border-bottom: none !important;
          z-index: 999;
        }}
        [data-testid="stToolbar"] {{ z-index: 1000; }}
        [data-testid="stDecoration"] {{ display: none; }}
        div[data-testid="stStatusWidget"] {{ display: none; }}

        /* Hide Streamlit sidebar — navigation lives in the top navbar */
        section[data-testid="stSidebar"],
        div[data-testid="stSidebarCollapsedControl"],
        button[kind="headerNoPadding"],
        [data-testid="collapsedControl"] {{
          display: none !important;
        }}
        [data-testid="stAppViewContainer"] {{
          margin-left: 0 !important;
        }}

        h1, h2, h3, h4, h5, h6,
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
        label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
        [data-testid="stMetricDelta"],
        .stRadio label, .stCheckbox label, .stToggle label,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] span {{
          color: var(--mp-text) !important;
        }}
        /* Keep branded hero / risk banners readable */
        .home-hero .home-brand {{ color: #ffffff !important; }}
        .home-hero .home-tag {{ color: #99f6e4 !important; }}
        .home-hero .home-lead {{ color: rgba(248,250,252,0.92) !important; }}
        .home-hero h1 {{ color: #ffffff !important; }}
        .risk-hero h3 {{ color: #ffffff !important; }}
        .risk-hero p {{ color: #e2e8f0 !important; }}

        .stCaption, [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p {{
          color: var(--mp-muted) !important;
          opacity: 1 !important;
        }}
        hr {{
          border-color: var(--mp-border) !important;
          opacity: 1 !important;
        }}

        /* Top navbar card */
        .mp-nav-card {{
          background: var(--mp-surface);
          border: 1px solid var(--mp-border);
          border-radius: 14px;
          padding: 0.85rem 1.1rem;
          margin: 0 0 0.35rem 0;
          box-shadow: var(--mp-shadow);
          position: relative;
          z-index: 2;
          overflow: visible;
        }}
        .mp-nav-card-row {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.85rem;
          flex-wrap: wrap;
        }}
        .mp-nav-brand {{
          min-width: 0;
        }}
        .mp-nav-brand-name {{
          font-family: 'Fraunces', Georgia, serif;
          font-size: 1.45rem;
          font-weight: 700;
          color: var(--mp-text) !important;
          letter-spacing: -0.02em;
          line-height: 1.15;
        }}
        .mp-nav-brand-page {{
          font-size: 0.78rem;
          font-weight: 600;
          color: var(--mp-accent) !important;
          margin-top: 0.1rem;
        }}
        .mp-nav-chips {{
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 0.4rem;
          justify-content: flex-end;
        }}
        .mp-chip {{
          display: inline-flex;
          align-items: center;
          border-radius: 999px;
          padding: 0.22rem 0.65rem;
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.01em;
          white-space: nowrap;
          border: 1px solid transparent;
        }}
        .mp-chip-ok {{
          background: var(--mp-alert-success-bg);
          color: var(--mp-alert-success-text) !important;
          border-color: {"#065f46" if dark else "#a7f3d0"};
        }}
        .mp-chip-info {{
          background: var(--mp-alert-info-bg);
          color: var(--mp-alert-info-text) !important;
          border-color: {"#075985" if dark else "#bfdbfe"};
        }}
        .mp-chip-warn {{
          background: var(--mp-alert-warn-bg);
          color: var(--mp-alert-warn-text) !important;
          border-color: {"#92400e" if dark else "#fde68a"};
        }}
        .mp-chip-muted {{
          background: var(--mp-surface-2);
          color: var(--mp-muted) !important;
          border-color: var(--mp-border-soft);
        }}
        /* Align sex/theme widgets with brand card */
        div[data-testid="stHorizontalBlock"]:has(.mp-nav-card) {{
          align-items: center !important;
          margin-bottom: 0.25rem !important;
        }}
        .mp-nav-card {{
          margin-bottom: 0 !important;
        }}
        div[data-testid="stHorizontalBlock"]:has(.mp-nav-card) [data-testid="stSelectbox"],
        div[data-testid="stHorizontalBlock"]:has(.mp-nav-card) [data-testid="stToggle"] {{
          margin-top: 0.35rem;
        }}
        div[data-testid="stSelectbox"] {{
          min-width: 8.5rem;
        }}
        [data-testid="stToggle"] {{
          min-width: auto;
          white-space: nowrap;
          display: flex;
          align-items: center;
        }}
        [data-testid="stToggle"] label,
        [data-testid="stToggle"] p {{
          white-space: nowrap !important;
        }}
        /* Keep nav pills on one line */
        [data-testid="stPills"] {{
          flex-wrap: nowrap !important;
        }}
        [data-testid="stPills"] > div {{
          flex-wrap: nowrap !important;
          gap: 0.4rem !important;
          overflow-x: auto;
          scrollbar-width: none;
        }}
        [data-testid="stPills"] > div::-webkit-scrollbar {{
          display: none;
        }}
        [data-testid="stPills"] button {{
          padding: 0.32rem 0.85rem !important;
          min-height: 2.1rem !important;
          white-space: nowrap !important;
          flex: 0 0 auto !important;
        }}
        /* Active pill contrast */
        [data-testid="stPills"] button[aria-checked="true"],
        [data-testid="stBaseButton-pillsActive"],
        button[kind="pillsActive"] {{
          background: var(--mp-accent-strong) !important;
          color: #ffffff !important;
          border-color: var(--mp-accent-strong) !important;
        }}
        [data-testid="stPills"] button[aria-checked="true"] p,
        [data-testid="stPills"] button[aria-checked="true"] span {{
          color: #ffffff !important;
        }}

        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"],
        .stButton > button[data-testid="stBaseButton-primary"],
        button[kind="primary"],
        button[data-testid="stBaseButton-primary"] {{
          background-color: var(--mp-accent-strong) !important;
          border-color: var(--mp-accent-strong) !important;
          color: #ffffff !important;
        }}
        .stButton > button[kind="primary"] p,
        .stButton > button[kind="primary"] span,
        button[kind="primary"] p,
        button[kind="primary"] span,
        button[data-testid="stBaseButton-primary"] p,
        button[data-testid="stBaseButton-primary"] span {{
          color: #ffffff !important;
        }}
        .stButton > button[kind="secondary"],
        .stButton > button[data-testid="baseButton-secondary"],
        .stButton > button[data-testid="stBaseButton-secondary"],
        button[kind="secondary"],
        button[data-testid="stBaseButton-secondary"],
        .main .stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]) {{
          background-color: var(--mp-secondary-btn-bg) !important;
          border: 1px solid var(--mp-secondary-btn-border) !important;
          color: var(--mp-text) !important;
          box-shadow: {"0 1px 2px rgba(0,0,0,0.25)" if dark else "none"} !important;
        }}
        .stButton > button[kind="secondary"] p,
        .stButton > button[kind="secondary"] span,
        button[kind="secondary"] p,
        button[kind="secondary"] span,
        button[data-testid="stBaseButton-secondary"] p,
        button[data-testid="stBaseButton-secondary"] span,
        .main .stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]) p,
        .main .stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]) span {{
          color: var(--mp-text) !important;
        }}
        .main .stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]):hover {{
          border-color: var(--mp-accent) !important;
          background-color: {"#3d4f66" if dark else "#f8fafc"} !important;
        }}

        /* Theme toggle visibility (navbar + anywhere in main) */
        .main [data-testid="stToggle"] label,
        .main [data-testid="stToggle"] [data-testid="stWidgetLabel"] p {{
          color: var(--mp-text) !important;
        }}
        .main [data-testid="stToggle"] [role="switch"] {{
          background-color: {"#475569" if dark else "#cbd5e1"} !important;
          border: 1px solid var(--mp-secondary-btn-border) !important;
        }}
        .main [data-testid="stToggle"] [role="switch"][aria-checked="true"] {{
          background-color: var(--mp-accent-strong) !important;
          border-color: var(--mp-accent-strong) !important;
        }}

        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input,
        [data-baseweb="select"] > div,
        [data-baseweb="base-input"],
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stMultiSelect"] > div > div,
        [data-testid="stSelectbox"] input,
        [data-testid="stMultiSelect"] input,
        .react-aria-ComboBox input,
        input[role="combobox"] {{
          background-color: var(--mp-input-bg) !important;
          color: var(--mp-text) !important;
          -webkit-text-fill-color: var(--mp-text) !important;
          border-color: var(--mp-border) !important;
          caret-color: var(--mp-text) !important;
        }}
        [data-testid="stSelectbox"] button,
        .react-aria-ComboBox button {{
          color: var(--mp-text) !important;
          background: transparent !important;
        }}
        [data-testid="stSelectbox"] [role="group"],
        .react-aria-ComboBox [role="group"] {{
          background-color: var(--mp-input-bg) !important;
          border: 1px solid var(--mp-border) !important;
          border-radius: 8px !important;
        }}
        [data-baseweb="popover"] ul,
        [data-baseweb="menu"],
        [role="listbox"],
        [data-testid="stSelectboxVirtualDropdown"],
        div[data-radix-popper-content-wrapper] {{
          background-color: var(--mp-surface-2) !important;
          color: var(--mp-text) !important;
        }}
        [data-baseweb="popover"] li,
        [role="option"] {{
          color: var(--mp-text) !important;
          background-color: var(--mp-surface-2) !important;
        }}
        [role="option"][data-focused="true"],
        [role="option"][aria-selected="true"] {{
          background-color: var(--mp-accent-strong) !important;
          color: #ffffff !important;
        }}
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] {{
          background-color: var(--mp-surface-2) !important;
          border-color: var(--mp-border) !important;
          color: var(--mp-text) !important;
        }}
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] span {{
          color: var(--mp-muted) !important;
        }}
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary {{
          background: var(--mp-surface) !important;
          border-color: var(--mp-border) !important;
          color: var(--mp-text) !important;
        }}
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"],
        [data-testid="stDataFrameResizable"] {{
          background: var(--mp-surface) !important;
          color: var(--mp-text) !important;
        }}
        [data-testid="stChatMessage"],
        [data-testid="stChatInput"] {{
          background: var(--mp-surface) !important;
          color: var(--mp-text) !important;
          border-color: var(--mp-border) !important;
        }}
        [data-testid="stTabs"] button {{
          color: var(--mp-muted) !important;
        }}
        [data-testid="stTabs"] button[aria-selected="true"] {{
          color: var(--mp-accent) !important;
        }}

        /* Alert / status boxes (sidebar + main) — Streamlit uses stAlertContainer */
        [data-testid="stAlertContainer"] {{
          border: 1px solid var(--mp-border) !important;
          border-radius: 8px !important;
        }}
        [data-testid="stAlertContentSuccess"],
        [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) [data-testid="stAlertContainer"] {{
          background-color: var(--mp-alert-success-bg) !important;
          color: var(--mp-alert-success-text) !important;
        }}
        [data-testid="stAlertContentInfo"],
        [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) [data-testid="stAlertContainer"] {{
          background-color: var(--mp-alert-info-bg) !important;
          color: var(--mp-alert-info-text) !important;
        }}
        [data-testid="stAlertContentWarning"],
        [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) [data-testid="stAlertContainer"] {{
          background-color: var(--mp-alert-warn-bg) !important;
          color: var(--mp-alert-warn-text) !important;
        }}
        [data-testid="stAlertContentError"],
        [data-testid="stAlert"]:has([data-testid="stAlertContentError"]) [data-testid="stAlertContainer"] {{
          background-color: var(--mp-alert-error-bg) !important;
          color: var(--mp-alert-error-text) !important;
        }}
        [data-testid="stAlertContentSuccess"] p,
        [data-testid="stAlertContentSuccess"] span,
        [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) p,
        [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) span {{
          color: var(--mp-alert-success-text) !important;
        }}
        [data-testid="stAlertContentInfo"] p,
        [data-testid="stAlertContentInfo"] span,
        [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) p,
        [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) span {{
          color: var(--mp-alert-info-text) !important;
        }}
        [data-testid="stAlertContentWarning"] p,
        [data-testid="stAlertContentWarning"] span,
        [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) p,
        [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) span {{
          color: var(--mp-alert-warn-text) !important;
        }}
        [data-testid="stAlertContentError"] p,
        [data-testid="stAlertContentError"] span,
        [data-testid="stAlert"]:has([data-testid="stAlertContentError"]) p,
        [data-testid="stAlert"]:has([data-testid="stAlertContentError"]) span {{
          color: var(--mp-alert-error-text) !important;
        }}
        /* Fallback: boost faint Streamlit alpha success/info fills in dark mode */
        section[data-testid="stSidebar"] [data-testid="stAlertContainer"] {{
          background-color: var(--mp-surface-2) !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) [data-testid="stAlertContainer"] {{
          background-color: var(--mp-alert-success-bg) !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) [data-testid="stAlertContainer"] {{
          background-color: var(--mp-alert-info-bg) !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) [data-testid="stAlertContainer"] {{
          background-color: var(--mp-alert-warn-bg) !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stAlert"]:has([data-testid="stAlertContentError"]) [data-testid="stAlertContainer"] {{
          background-color: var(--mp-alert-error-bg) !important;
        }}

        .disclaimer {{
            background: var(--mp-surface);
            border: 1px solid var(--mp-border-soft);
            border-left: 4px solid var(--mp-accent);
            padding: 0.85rem 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            color: var(--mp-body);
            line-height: 1.45;
        }}
        .score-box {{
            background: var(--mp-surface);
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            text-align: center;
            border: 1px solid var(--mp-border-soft);
            color: var(--mp-text);
        }}
        .score-value {{
            font-size: 2.6rem;
            font-weight: 700;
            color: var(--mp-accent);
            line-height: 1.1;
        }}

        .home-hero {{
          position: relative;
          z-index: 1;
          overflow: hidden;
          border-radius: 20px;
          padding: 2.1rem 2rem 1.9rem;
          margin: 0 0 1.25rem 0;
          color: #f8fafc;
          background: linear-gradient(135deg, #0f172a 0%, #115e59 58%, #0e7490 100%);
          box-shadow: 0 10px 28px rgba(15, 23, 42, 0.12);
        }}
        .home-brand {{
          font-family: 'Fraunces', Georgia, serif;
          font-size: clamp(2.2rem, 3.6vw, 3rem);
          font-weight: 700;
          letter-spacing: -0.02em;
          line-height: 1.05;
          margin: 0 0 0.55rem 0;
          color: #ffffff;
        }}
        .home-tag {{
          display: inline-block;
          font-size: 0.75rem;
          font-weight: 600;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: #99f6e4;
          margin-bottom: 0.7rem;
        }}
        .home-lead {{
          max-width: 38rem;
          margin: 0;
          font-size: 1.05rem;
          line-height: 1.55;
          color: rgba(248,250,252,0.9);
        }}
        .home-section-title {{
          font-family: 'Fraunces', Georgia, serif;
          font-size: 1.35rem;
          color: var(--mp-text);
          margin: 0.4rem 0 0.85rem 0;
          font-weight: 560;
        }}
        .home-step, .home-stat, .home-note, .risk-card, .risk-insight {{
          background: var(--mp-surface);
          border: 1px solid var(--mp-border);
          border-radius: 14px;
        }}
        .home-step {{ padding: 1.05rem 1.1rem; height: 100%; }}
        .home-step-num {{
          font-family: 'Fraunces', Georgia, serif;
          font-size: 1.5rem;
          color: var(--mp-accent);
          margin-bottom: 0.25rem;
          font-weight: 700;
        }}
        .home-step h4 {{ margin: 0 0 0.35rem 0; font-size: 1.02rem; color: var(--mp-text); }}
        .home-step p {{ margin: 0; color: var(--mp-body); font-size: 0.92rem; line-height: 1.45; }}
        .home-stat {{ padding: 0.95rem 1rem; text-align: center; }}
        .home-stat-value {{
          font-family: 'Fraunces', Georgia, serif;
          font-size: 1.7rem;
          color: var(--mp-accent);
          font-weight: 700;
          line-height: 1.1;
        }}
        .home-stat-label {{ margin-top: 0.25rem; font-size: 0.8rem; color: var(--mp-muted); font-weight: 600; }}
        .home-note {{ margin-top: 1rem; padding: 0.85rem 1rem; color: var(--mp-body); font-size: 0.9rem; line-height: 1.45; }}

        .risk-hero {{
            background: linear-gradient(135deg, #0f172a 0%, #134e4a 100%);
            color: #f8fafc;
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }}
        .risk-hero h3 {{
            margin: 0 0 0.35rem 0;
            font-family: 'Fraunces', Georgia, serif;
            font-weight: 600;
            font-size: 1.25rem;
            color: #ffffff;
        }}
        .risk-hero p {{ margin: 0; opacity: 0.85; font-size: 0.92rem; color: #e2e8f0; }}
        .risk-card {{ padding: 1rem 1.1rem 1.15rem; height: 100%; }}
        .risk-card-label {{
            font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase;
            color: var(--mp-muted); font-weight: 600; margin-bottom: 0.35rem;
        }}
        .risk-card-pct {{
            font-family: 'Fraunces', Georgia, serif; font-size: 2.1rem; font-weight: 700;
            line-height: 1.1; margin: 0.15rem 0; color: var(--mp-text);
        }}
        .risk-badge {{
            display: inline-block; font-size: 0.75rem; font-weight: 600;
            padding: 0.2rem 0.55rem; border-radius: 999px; margin-top: 0.35rem;
        }}
        .risk-meter {{
            width: 100%; height: 8px; background: var(--mp-meter-track); border-radius: 999px;
            margin-top: 0.85rem; overflow: hidden;
        }}
        .risk-meter > span {{ display: block; height: 100%; border-radius: 999px; }}
        .risk-meta {{ margin-top: 0.65rem; font-size: 0.75rem; color: var(--mp-muted); }}
        .risk-section-title {{
            font-family: 'Fraunces', Georgia, serif; font-size: 1.15rem; font-weight: 600;
            color: var(--mp-text); margin: 1.2rem 0 0.6rem 0; padding-bottom: 0.35rem;
            border-bottom: 1px solid var(--mp-border-soft);
        }}
        .risk-insight {{
            padding: 0.85rem 1rem; font-size: 0.9rem; color: var(--mp-body);
            line-height: 1.5; margin-bottom: 0.75rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_theme(fig) -> None:
    """Match Plotly charts to the active light/dark UI theme."""
    dark = st.session_state.get("theme_mode", "light") == "dark"
    if dark:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.55)",
            font=dict(color="#e2e8f0"),
            title_font_color="#e2e8f0",
        )
    else:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.6)",
            font=dict(color="#0f172a"),
            title_font_color="#0f172a",
        )


def _risk_plotly_layout(fig, title: str = "") -> None:
    apply_plotly_theme(fig)
    updates = dict(margin=dict(l=10, r=10, t=40 if title else 10, b=10))
    if title:
        updates["title"] = title
    fig.update_layout(**updates)


def load_sample_text(name: str = "sample_blood_report.txt") -> str:
    return (ROOT / "samples" / name).read_text(encoding="utf-8")


def run_pipeline(text: str, sex_override: str, filename: str = "pasted_text.txt"):
    parsed = parse_report_text(text)
    sex = None if sex_override == "auto" else sex_override
    if sex is None:
        sex = parsed.patient_sex

    analyzed = analyze_biomarkers(parsed.biomarkers, sex=sex)
    health = compute_health_score(analyzed)
    patient_summary = build_patient_summary(analyzed, health, parsed.report_date)
    doctor_summary = build_doctor_summary(analyzed, health, parsed.report_date)
    markers = analyzed_to_dict_list(analyzed)
    return parsed, analyzed, health, patient_summary, doctor_summary, markers, sex


def clear_extracted_state(*, clear_results: bool = False) -> None:
    """Reset OCR/sample text held in session (and optionally analysis results)."""
    st.session_state.last_text = ""
    st.session_state.last_source = ""
    st.session_state.text_fingerprint = None
    st.session_state.extraction_meta = None
    st.session_state.upload_fingerprint = None
    # Widget keys must be updated before the widget renders
    st.session_state.editable_report_text = ""
    if clear_results and "current" in st.session_state:
        del st.session_state["current"]


def render_upload_panel():
    st.subheader("1. Upload or paste a report")

    mode = st.radio(
        "Input method",
        options=["Upload file", "Paste text", "Try sample"],
        horizontal=True,
        key="input_mode_radio",
    )

    # Switching input method should not keep the previous sample/upload text around
    prev_mode = st.session_state.get("input_mode_active")
    if prev_mode is not None and prev_mode != mode:
        clear_extracted_state(clear_results=True)
    st.session_state.input_mode_active = mode

    extracted_text = None
    source_name = None
    extraction_meta = None

    if mode == "Upload file":
        st.caption("Previous sample/paste text is cleared when you switch to Upload.")
        upload = st.file_uploader(
            "PDF or image (JPG/PNG)",
            type=["pdf", "jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"],
            help="Images use RapidOCR. PDF text layers use pdfplumber.",
            key="report_file_uploader",
        )
        if upload is not None:
            if not is_supported(upload.name):
                st.error("Unsupported file type.")
            else:
                data = read_upload_bytes(upload)
                upload_fp = f"{upload.name}:{len(data)}"
                # New file selected → replace any leftover text
                if st.session_state.get("upload_fingerprint") != upload_fp:
                    clear_extracted_state(clear_results=True)
                    st.session_state.upload_fingerprint = upload_fp

                with st.spinner("Running OCR / extracting text (first image may take a moment)..."):
                    result = extract_text(upload.name, data)
                extraction_meta = result
                extracted_text = result.text or ""
                source_name = upload.name
                if result.method == "failed" or not extracted_text.strip():
                    st.error(
                        "Could not extract text from this image. "
                        "Try the sample PNG, a sharper photo, or paste text manually."
                    )
                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
                if result.method != "failed":
                    st.caption(f"Extraction method: `{result.method}` · pages: {result.pages}")
        else:
            st.session_state.upload_fingerprint = None

    elif mode == "Paste text":
        pasted = st.text_area(
            "Paste lab report text",
            height=220,
            placeholder="Paste CBC / metabolic panel text here...",
            key="paste_box",
        )
        if st.button("Use pasted text", type="primary"):
            extracted_text = pasted
            source_name = "pasted_text.txt"

    else:  # Try sample
        st.write("Demo reports with mixed normal and abnormal values.")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Load baseline sample", type="primary"):
                extracted_text = load_sample_text("sample_blood_report.txt")
                source_name = "sample_blood_report.txt"
        with c2:
            if st.button("Load follow-up sample"):
                extracted_text = load_sample_text("sample_blood_report_followup.txt")
                source_name = "sample_blood_report_followup.txt"
        with c3:
            sample_png = ROOT / "samples" / "sample_blood_report.png"
            if sample_png.exists() and st.button("OCR sample PNG"):
                data = sample_png.read_bytes()
                with st.spinner("Running OCR on sample PNG..."):
                    result = extract_text(sample_png.name, data)
                extraction_meta = result
                extracted_text = result.text or ""
                source_name = sample_png.name
                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
                st.caption(f"Extraction method: `{result.method}`")
        sample_png = ROOT / "samples" / "sample_blood_report.png"
        if sample_png.exists():
            st.image(str(sample_png), caption="samples/sample_blood_report.png", use_container_width=True)

    return extracted_text, source_name, extraction_meta


def render_results(parsed, analyzed, health, patient_summary, doctor_summary, sex_used, saved_id=None):
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)
    st.write("")

    if saved_id:
        st.success(f"Saved to database · report id `{saved_id}`")

    col_score, col_meta = st.columns([1, 2])
    with col_score:
        st.markdown(
            f"""
            <div class="score-box">
              <div>Educational Health Score</div>
              <div class="score-value">{health.score}<span style="font-size:1.2rem">/100</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_meta:
        st.markdown("### Overview")
        st.write(health.summary)
        c1, c2, c3 = st.columns(3)
        c1.metric("Markers found", len(analyzed))
        c2.metric("Outside typical range", sum(1 for m in analyzed if m.status != "normal"))
        c3.metric("Report date", parsed.report_date or "N/A")
        st.caption(f"Reference ranges sex context: **{sex_used or 'default'}**")
        for note in parsed.notes:
            st.caption(note)

    st.subheader("Category snapshot")
    if health.category_scores:
        cols = st.columns(min(4, max(1, len(health.category_scores))))
        for i, cat in enumerate(health.category_scores):
            with cols[i % len(cols)]:
                icon = CATEGORY_ICON.get(cat.status, "•")
                st.markdown(f"**{icon} {cat.category}**")
                st.caption(f"{cat.abnormal_count}/{cat.marker_count} flagged")

    st.subheader("Biomarker results")
    if not analyzed:
        st.info("No biomarkers extracted yet.")
        return

    df = pd.DataFrame(
        [
            {
                "Status": f"{STATUS_EMOJI.get(m.status, '')} {m.status.upper()}",
                "Test": m.name,
                "Value": m.value,
                "Unit": m.unit or "",
                "Reference": (
                    f"{m.ref_low}–{m.ref_high}" if m.ref_low is not None else (m.reported_range or "N/A")
                ),
                "Category": m.category,
            }
            for m in analyzed
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    status_counts = (
        pd.DataFrame({"status": [m.status for m in analyzed]}).value_counts().reset_index(name="count")
    )
    fig = px.pie(
        status_counts,
        names="status",
        values="count",
        color="status",
        color_discrete_map={
            "normal": "#2e7d32",
            "borderline": "#f9a825",
            "low": "#c62828",
            "high": "#ef6c00",
        },
        title="Result distribution",
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Plain-language explanations")
    for m in analyzed:
        with st.expander(f"{STATUS_EMOJI.get(m.status, '')} {m.name}: {m.value} {m.unit or ''} ({m.status})"):
            st.write(m.explanation)
            if m.raw_line:
                st.code(m.raw_line, language=None)

    st.subheader("Summaries you can share")
    s1, s2 = st.columns(2)
    with s1:
        st.text_area("Patient-friendly summary", patient_summary, height=320, key="patient_sum_view")
        st.download_button(
            "Download patient summary",
            patient_summary,
            file_name="patient_summary.txt",
            mime="text/plain",
            key="dl_patient",
        )
    with s2:
        st.text_area("Clinician-oriented summary", doctor_summary, height=320, key="doctor_sum_view")
        st.download_button(
            "Download clinician summary (.txt)",
            doctor_summary,
            file_name="clinician_summary.txt",
            mime="text/plain",
            key="dl_doctor",
        )

    marker_dicts = analyzed_to_dict_list(analyzed)
    pdf_bytes = build_doctor_pdf(
        markers=marker_dicts,
        health_score=health.score,
        report_date=parsed.report_date,
        sex=sex_used,
        filename=st.session_state.get("last_source", "report"),
        patient_summary=patient_summary,
        doctor_summary=doctor_summary,
        report_id=saved_id,
    )
    st.download_button(
        "Download clinician PDF",
        data=pdf_bytes,
        file_name=f"clinician_summary_{saved_id or 'latest'}.pdf",
        mime="application/pdf",
        type="primary",
        key="dl_doctor_pdf",
    )


def page_analyze(sex_override: str) -> None:
    extracted_text, source_name, extraction_meta = render_upload_panel()

    if extraction_meta is not None:
        st.session_state.extraction_meta = {
            "method": extraction_meta.method,
            "pages": extraction_meta.pages,
            "warnings": extraction_meta.warnings,
        }

    if extracted_text is not None:
        fingerprint = f"{source_name}:{len(extracted_text)}:{hash(extracted_text)}"
        if st.session_state.get("text_fingerprint") != fingerprint:
            st.session_state.last_text = extracted_text
            st.session_state.last_source = source_name or "report"
            st.session_state.editable_report_text = extracted_text
            st.session_state.text_fingerprint = fingerprint

    if st.session_state.get("last_text") and not st.session_state.get("editable_report_text"):
        st.session_state.editable_report_text = st.session_state.last_text

    if st.session_state.get("last_text"):
        st.subheader("2. Extracted / editable text")
        src = st.session_state.get("last_source") or "unknown"
        st.caption(f"Current source: `{src}`")
        clear_col, _ = st.columns([1, 3])
        with clear_col:
            if st.button("Clear text", key="btn_clear_extracted"):
                clear_extracted_state(clear_results=True)
                st.rerun()

        edited = st.text_area(
            "Review OCR/text before analysis (you can correct mistakes)",
            height=220,
            key="editable_report_text",
        )
        st.session_state.last_text = edited

        analyze = st.button("Analyze & save report", type="primary", disabled=not (edited or "").strip())
        if analyze:
            with st.spinner("Analyzing biomarkers and saving..."):
                parsed, analyzed, health, patient_summary, doctor_summary, markers, sex_used = run_pipeline(
                    edited,
                    sex_override,
                    filename=st.session_state.get("last_source", "report.txt"),
                )
                saved = save_report(
                    filename=st.session_state.get("last_source", "report.txt"),
                    report_date=parsed.report_date,
                    sex=sex_used,
                    health_score=health.score,
                    markers=markers,
                    raw_text=edited,
                    patient_summary=patient_summary,
                    doctor_summary=doctor_summary,
                    notes="; ".join(parsed.notes),
                )
            bump_reports_cache()
            st.session_state.current = {
                "parsed": parsed,
                "analyzed": analyzed,
                "health": health,
                "patient_summary": patient_summary,
                "doctor_summary": doctor_summary,
                "sex_used": sex_used,
                "saved_id": saved.id,
            }

    if "current" in st.session_state:
        cur = st.session_state.current
        render_results(
            cur["parsed"],
            cur["analyzed"],
            cur["health"],
            cur["patient_summary"],
            cur["doctor_summary"],
            cur["sex_used"],
            saved_id=cur.get("saved_id"),
        )

    with st.expander("Raw extraction debug", expanded=False):
        meta = st.session_state.get("extraction_meta")
        if meta:
            st.json(meta)
        st.code(st.session_state.get("last_text", "")[:4000] or "No text yet.")


def page_history() -> None:
    st.subheader("Saved report history")
    st.caption("Reports persist in SQLite across browser refreshes and restarts.")
    reports = list_reports()
    if not reports:
        st.info("No saved reports yet. Analyze a report on the Analyze tab.")
        return

    rows = [
        {
            "ID": r.id,
            "Date": r.report_date or "N/A",
            "Filename": r.filename,
            "Score": r.health_score,
            "Sex": r.sex or "—",
            "Saved at": r.created_at,
        }
        for r in reports
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    labels = {
        r.id: f"{r.report_date or 'no-date'} · {r.filename} · score {r.health_score} · `{r.id}`"
        for r in reports
    }
    selected_id = st.selectbox(
        "Open a saved report",
        options=list(labels.keys()),
        format_func=lambda i: labels[i],
        key="history_selected_id",
    )

    report = get_report(selected_id)
    if not report:
        st.error("Report not found.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Health score", f"{report.health_score}/100")
    c2.metric("Markers", len(report.markers))
    c3.metric("Report date", report.report_date or "N/A")

    if report.markers:
        mdf = pd.DataFrame(
            [
                {
                    "Status": f"{STATUS_EMOJI.get(m['status'], '')} {m['status'].upper()}",
                    "Test": m["name"],
                    "Value": m["value"],
                    "Unit": m.get("unit") or "",
                    "Category": m.get("category") or "",
                }
                for m in report.markers
            ]
        )
        st.dataframe(mdf, use_container_width=True, hide_index=True)

    s1, s2 = st.columns(2)
    with s1:
        st.text_area("Patient summary", report.patient_summary, height=220, key="hist_patient")
    with s2:
        st.text_area("Clinician summary", report.doctor_summary, height=220, key="hist_doctor")

    b1, b2, b3 = st.columns(3)
    with b1:
        payload = export_report_json(report.id)
        if payload:
            st.download_button(
                "Export JSON",
                payload,
                file_name=f"report_{report.id}.json",
                mime="application/json",
            )
    with b2:
        if report.patient_summary:
            st.download_button(
                "Export patient summary",
                report.patient_summary,
                file_name=f"patient_{report.id}.txt",
                mime="text/plain",
            )
    with b3:
        pdf_bytes = build_doctor_pdf(
            markers=report.markers,
            health_score=report.health_score,
            report_date=report.report_date,
            sex=report.sex,
            filename=report.filename,
            patient_summary=report.patient_summary,
            doctor_summary=report.doctor_summary,
            report_id=report.id,
        )
        st.download_button(
            "Export clinician PDF",
            data=pdf_bytes,
            file_name=f"clinician_summary_{report.id}.pdf",
            mime="application/pdf",
            type="primary",
            key=f"hist_pdf_{report.id}",
        )

    st.markdown("---")
    st.subheader("Delete options")
    del_tab1, del_tab2, del_tab3 = st.tabs(
        ["Delete this report", "Delete selected", "Clear all history"]
    )

    with del_tab1:
        st.write(f"Remove only the open report: `{report.id}` ({report.filename})")
        confirm_one = st.checkbox("I understand this cannot be undone", key="confirm_delete_one")
        if st.button("Delete this report", type="primary", disabled=not confirm_one, key="btn_delete_one"):
            delete_report(report.id)
            bump_reports_cache()
            st.success(f"Deleted report `{report.id}`.")
            st.rerun()

    with del_tab2:
        st.write("Choose one or more reports to delete.")
        multi = st.multiselect(
            "Reports to delete",
            options=list(labels.keys()),
            format_func=lambda i: labels[i],
            default=[],
            key="bulk_delete_ids",
        )
        confirm_multi = st.checkbox(
            f"Confirm delete of {len(multi)} report(s)",
            key="confirm_delete_multi",
            disabled=not multi,
        )
        if st.button(
            "Delete selected reports",
            type="primary",
            disabled=not (multi and confirm_multi),
            key="btn_delete_multi",
        ):
            n = delete_reports(multi)
            bump_reports_cache()
            st.success(f"Deleted {n} report(s).")
            st.rerun()

    with del_tab3:
        st.warning("This permanently deletes **all** saved reports and biomarker history.")
        confirm_all = st.checkbox("Yes, clear my entire report history", key="confirm_delete_all")
        typed = st.text_input(
            'Type DELETE ALL to confirm',
            key="typed_delete_all",
            disabled=not confirm_all,
        )
        ready = confirm_all and typed.strip().upper() == "DELETE ALL"
        if st.button("Clear all history", type="primary", disabled=not ready, key="btn_delete_all"):
            n = delete_all_reports()
            bump_reports_cache()
            st.success(f"Cleared {n} report(s) from the database.")
            st.rerun()


def page_compare() -> None:
    st.subheader("Compare two reports")
    st.caption("Pick a previous and a current report to see biomarker changes over time.")
    reports = list_reports()
    if len(reports) < 2:
        st.info("Save at least 2 reports (e.g. baseline + follow-up samples) to compare.")
        return

    labels = {
        r.id: f"{r.report_date or 'no-date'} · {r.filename} · score {r.health_score}"
        for r in reports
    }
    ids = list(labels.keys())
    # Default: oldest-ish as previous (last in DESC list) and newest as current
    col_a, col_b = st.columns(2)
    with col_a:
        prev_id = st.selectbox(
            "Previous report",
            options=ids,
            index=min(1, len(ids) - 1),
            format_func=lambda i: labels[i],
            key="compare_prev",
        )
    with col_b:
        curr_id = st.selectbox(
            "Current report",
            options=ids,
            index=0,
            format_func=lambda i: labels[i],
            key="compare_curr",
        )

    if prev_id == curr_id:
        st.warning("Select two different reports.")
        return

    prev = get_report(prev_id)
    curr = get_report(curr_id)
    if not prev or not curr:
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Previous score", prev.health_score)
    m2.metric("Current score", curr.health_score, delta=curr.health_score - prev.health_score)
    m3.metric(
        "Date span",
        f"{prev.report_date or 'N/A'} → {curr.report_date or 'N/A'}",
    )

    rows = compare_reports(prev_id, curr_id)
    cdf = pd.DataFrame(rows)
    if cdf.empty:
        st.info("No overlapping markers to compare.")
        return

    cdf["Change"] = cdf["direction"].map(DIRECTION_LABEL)
    show = cdf[
        [
            "name",
            "previous_value",
            "previous_status",
            "current_value",
            "current_status",
            "delta",
            "Change",
            "unit",
            "category",
        ]
    ].rename(
        columns={
            "name": "Test",
            "previous_value": "Previous",
            "previous_status": "Prev status",
            "current_value": "Current",
            "current_status": "Curr status",
            "delta": "Delta",
            "unit": "Unit",
            "category": "Category",
        }
    )
    st.dataframe(show, use_container_width=True, hide_index=True)

    changed = cdf[cdf["direction"].isin(["up", "down"]) & cdf["delta"].notna()].copy()
    if not changed.empty:
        fig = px.bar(
            changed,
            x="name",
            y="delta",
            color="direction",
            color_discrete_map={"up": "#ef6c00", "down": "#2e7d32"},
            title="Value change (current − previous)",
            labels={"name": "Test", "delta": "Delta"},
        )
        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    improved = cdf[
        (cdf["previous_status"].isin(["low", "high", "borderline"]))
        & (cdf["current_status"] == "normal")
    ]
    worsened = cdf[
        (cdf["previous_status"].isin(["normal", "borderline"]))
        & (cdf["current_status"].isin(["low", "high"]))
    ]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Moved into typical range")
        if improved.empty:
            st.caption("None")
        else:
            st.write(", ".join(improved["name"].tolist()))
    with c2:
        st.markdown("#### Newly outside typical range")
        if worsened.empty:
            st.caption("None")
        else:
            st.write(", ".join(worsened["name"].tolist()))

    st.caption(DISCLAIMER)


def page_trends() -> None:
    st.subheader("Biomarker trends (saved reports)")
    rows = trend_rows()
    if not rows:
        st.info("No saved biomarker data yet.")
        return

    tdf = pd.DataFrame(rows)
    report_count = tdf["report_id"].nunique()
    if report_count < 2:
        st.info("Save at least 2 reports to visualize trends over time.")
        st.dataframe(tdf[["date", "name", "value", "unit", "status", "filename"]], use_container_width=True, hide_index=True)
        return

    names = sorted(tdf["name"].unique())
    default = [n for n in ["Fasting Glucose", "HbA1c", "Hemoglobin", "LDL Cholesterol", "Vitamin D"] if n in names]
    selected = st.multiselect("Biomarkers to plot", names, default=default or names[:3])
    if not selected:
        return

    plot_df = tdf[tdf["name"].isin(selected)]
    fig = px.line(
        plot_df,
        x="date",
        y="value",
        color="name",
        markers=True,
        title="Biomarker trends across saved reports",
        hover_data=["filename", "status", "unit", "health_score"],
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    score_df = (
        tdf[["report_id", "date", "filename", "health_score"]]
        .drop_duplicates()
        .sort_values("date")
    )
    fig2 = px.line(
        score_df,
        x="date",
        y="health_score",
        markers=True,
        title="Educational health score over time",
        hover_data=["filename"],
    )
    apply_plotly_theme(fig2)
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(
        plot_df[["date", "name", "value", "unit", "status", "filename"]],
        use_container_width=True,
        hide_index=True,
    )


def render_risk_cards(results) -> None:
    cols = st.columns(4)
    dark = st.session_state.get("theme_mode", "light") == "dark"
    for i, r in enumerate(results):
        pct = int(round(r.probability * 100))
        badge_bg = {
            "Lower": "#064e3b" if dark else "#ecfdf5",
            "Moderate": "#78350f" if dark else "#fffbeb",
            "Higher": "#7f1d1d" if dark else "#fef2f2",
        }.get(r.band, "#1e293b" if dark else "#f8fafc")
        badge_fg = "#a7f3d0" if (dark and r.band == "Lower") else (
            "#fde68a" if (dark and r.band == "Moderate") else (
                "#fecaca" if (dark and r.band == "Higher") else r.color
            )
        )
        auc = r.metrics.get("roc_auc")
        auc_txt = f"{auc:.3f}" if isinstance(auc, (int, float)) else "—"
        with cols[i % len(cols)]:
            st.markdown(
                f"""
                <div class="risk-card">
                  <div class="risk-card-label">{r.label.replace(' risk', '')}</div>
                  <div class="risk-card-pct" style="color:{r.color}">{pct}%</div>
                  <span class="risk-badge" style="background:{badge_bg};color:{badge_fg}">
                    {r.band} band
                  </span>
                  <div class="risk-meter"><span style="width:{pct}%;background:{r.color}"></span></div>
                  <div class="risk-meta">{r.model_type.replace('_', ' ').title()} · AUC {auc_txt}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_risk_gauges(results) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    dark = st.session_state.get("theme_mode", "light") == "dark"
    text_color = "#e2e8f0" if dark else "#0f172a"
    muted = "#94a3b8" if dark else "#64748b"
    title_color = "#cbd5e1" if dark else "#334155"
    gauge_bg = "#0f172a" if dark else "#f8fafc"
    steps = (
        [
            {"range": [0, 33], "color": "#064e3b"},
            {"range": [33, 66], "color": "#78350f"},
            {"range": [66, 100], "color": "#7f1d1d"},
        ]
        if dark
        else [
            {"range": [0, 33], "color": "#ecfdf5"},
            {"range": [33, 66], "color": "#fffbeb"},
            {"range": [66, 100], "color": "#fef2f2"},
        ]
    )

    fig = make_subplots(
        rows=1,
        cols=max(1, len(results)),
        specs=[[{"type": "indicator"}] * max(1, len(results))],
        subplot_titles=[r.label.replace(" risk", "") for r in results],
    )
    for i, r in enumerate(results, start=1):
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=round(r.probability * 100, 1),
                number={"suffix": "%", "font": {"size": 28, "color": text_color, "family": "Source Sans 3"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": muted},
                    "bar": {"color": r.color, "thickness": 0.25},
                    "bgcolor": gauge_bg,
                    "borderwidth": 0,
                    "steps": steps,
                    "threshold": {
                        "line": {"color": text_color, "width": 2},
                        "thickness": 0.75,
                        "value": round(r.probability * 100, 1),
                    },
                },
                domain={"x": [0, 1], "y": [0, 1]},
            ),
            row=1,
            col=i,
        )
    fig.update_layout(
        height=230,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans 3", color=muted, size=11),
    )
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=12, color=title_color, family="Source Sans 3")
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def page_risk() -> None:
    st.markdown(
        """
        <div class="risk-hero">
          <h3>Risk estimation dashboard</h3>
          <p>Educational model outputs from lab biomarkers and optional patient context.
          Not a diagnosis or clinical decision support system.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="disclaimer">{DISCLAIMER} '
        "Models were trained on synthetic educational cohorts for portfolio demonstration — "
        "not validated clinical models.</div>",
        unsafe_allow_html=True,
    )

    if not models_available():
        st.error(
            "Risk models not found. From the project folder run:\n\n"
            "`python scripts/train_risk_models.py`"
        )
        return

    reports = list_reports()
    source = st.radio(
        "Feature source",
        options=["Latest analysis", "Saved report", "Manual biomarkers only"],
        horizontal=True,
        key="risk_source",
    )

    markers: list[dict] = []
    default_sex = st.session_state.get("sex_override")
    if default_sex == "auto":
        default_sex = None

    if source == "Latest analysis":
        cur = st.session_state.get("current")
        if not cur:
            st.info("Analyze a report first on the Analyze tab, or pick a saved report.")
        else:
            markers = analyzed_to_dict_list(cur["analyzed"])
            default_sex = cur.get("sex_used") or default_sex
            st.caption(f"Using latest analysis · {len(markers)} markers · saved id `{cur.get('saved_id', 'n/a')}`")
    elif source == "Saved report":
        if not reports:
            st.info("No saved reports yet.")
        else:
            labels = {
                r.id: f"{r.report_date or 'no-date'} · {r.filename} · score {r.health_score}"
                for r in reports
            }
            rid = st.selectbox(
                "Report",
                options=list(labels.keys()),
                format_func=lambda i: labels[i],
                key="risk_report_id",
            )
            report = get_report(rid)
            if report:
                markers = report.markers
                default_sex = report.sex or default_sex
                st.caption(f"Loaded {len(markers)} markers from `{report.id}`")
    else:
        st.caption("Enter values manually. Unchecked fields use training medians (imputed).")

    st.markdown('<div class="risk-section-title">Patient context</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=45, step=1)
    with c2:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.5, step=0.1)
    with c3:
        sbp = st.number_input("Systolic BP", min_value=70, max_value=220, value=120, step=1)
    with c4:
        sex_opt = st.selectbox(
            "Sex",
            options=["unknown", "female", "male"],
            index=0 if not default_sex else (1 if default_sex == "female" else 2),
        )
        sex = None if sex_opt == "unknown" else sex_opt

    with st.expander("Biomarker values (edit / fill)", expanded=(source == "Manual biomarkers only")):
        marker_map = {m["key"]: m["value"] for m in markers}
        edited_markers: list[dict] = []
        keys = sorted({k for spec in RISK_FEATURE_SPECS.values() for k in spec["biomarker_keys"]})
        cols = st.columns(3)
        for i, key in enumerate(keys):
            with cols[i % 3]:
                default = float(marker_map[key]) if key in marker_map else 0.0
                has = key in marker_map
                use = st.checkbox(
                    f"Use {key}",
                    value=has or source == "Manual biomarkers only",
                    key=f"use_{key}",
                )
                val = st.number_input(
                    key,
                    value=default if has else 0.0,
                    step=0.1,
                    format="%.2f",
                    key=f"val_{key}",
                    disabled=not use,
                )
                if use:
                    edited_markers.append(
                        {
                            "key": key,
                            "name": key,
                            "value": float(val),
                            "unit": "",
                            "category": "",
                            "status": "normal",
                        }
                    )
        if edited_markers:
            markers = edited_markers

    if st.button("Estimate risks", type="primary"):
        if not markers and source != "Manual biomarkers only":
            st.warning("No biomarkers available. Analyze a report or enter values manually.")
        else:
            results = predict_all(
                markers,
                age=float(age),
                bmi=float(bmi),
                systolic_bp=float(sbp),
                sex=sex,
            )
            st.session_state.risk_results = results

    results = st.session_state.get("risk_results")
    if not results:
        st.info("Click **Estimate risks** to generate the dashboard.")
        return

    st.markdown('<div class="risk-section-title">Risk summary</div>', unsafe_allow_html=True)
    render_risk_cards(results)
    st.write("")
    render_risk_gauges(results)

    overview = pd.DataFrame(
        [
            {
                "Condition": r.label.replace(" risk", ""),
                "Probability %": round(r.probability * 100, 1),
                "Band": r.band,
                "Model": r.model_type.replace("_", " ").title(),
                "Holdout AUC": r.metrics.get("roc_auc"),
                "Holdout F1": r.metrics.get("f1"),
                "Imputed / missing": ", ".join(r.missing_biomarkers) or "None",
            }
            for r in results
        ]
    )

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="risk-section-title">Probability comparison</div>', unsafe_allow_html=True)
        fig = px.bar(
            overview,
            x="Condition",
            y="Probability %",
            color="Band",
            color_discrete_map={
                "Lower": "#059669",
                "Moderate": "#d97706",
                "Higher": "#dc2626",
            },
            text="Probability %",
            category_orders={"Band": ["Lower", "Moderate", "Higher"]},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis=dict(range=[0, 110], ticksuffix="%"), showlegend=True, height=340)
        _risk_plotly_layout(fig, "")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="risk-section-title">Profile radar</div>', unsafe_allow_html=True)
        import plotly.graph_objects as go

        radar = go.Figure()
        radar.add_trace(
            go.Scatterpolar(
                r=[row["Probability %"] for _, row in overview.iterrows()]
                + [overview.iloc[0]["Probability %"]],
                theta=list(overview["Condition"]) + [overview.iloc[0]["Condition"]],
                fill="toself",
                fillcolor="rgba(13, 148, 136, 0.18)",
                line=dict(color="#0f766e", width=2),
                name="Estimated risk %",
            )
        )
        radar.update_layout(
            height=340,
            margin=dict(l=40, r=40, t=30, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor="#111827" if st.session_state.get("theme_mode") == "dark" else "#ffffff",
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticksuffix="%",
                    gridcolor="#334155" if st.session_state.get("theme_mode") == "dark" else "#e2e8f0",
                ),
                angularaxis=dict(
                    gridcolor="#334155" if st.session_state.get("theme_mode") == "dark" else "#e2e8f0"
                ),
            ),
            showlegend=False,
            font=dict(
                family="Source Sans 3",
                color="#e2e8f0" if st.session_state.get("theme_mode") == "dark" else "#334155",
                size=12,
            ),
        )
        st.plotly_chart(radar, use_container_width=True)

    st.markdown('<div class="risk-section-title">Model output table</div>', unsafe_allow_html=True)
    st.dataframe(
        overview,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Probability %": st.column_config.ProgressColumn(
                "Probability %",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
            "Holdout AUC": st.column_config.NumberColumn(format="%.3f"),
            "Holdout F1": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    st.markdown('<div class="risk-section-title">Explainability</div>', unsafe_allow_html=True)
    for r in results:
        title = f"{r.label} · {r.probability:.0%} · {r.band} band"
        with st.expander(title, expanded=(r.band == "Higher")):
            st.markdown(f'<div class="risk-insight">{r.explanation}</div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Model", r.model_type.replace("_", " ").title())
            m2.metric("Holdout AUC", f"{r.metrics.get('roc_auc', 0):.3f}")
            m3.metric("Holdout F1", f"{r.metrics.get('f1', 0):.3f}")

            cdf = pd.DataFrame(
                [
                    {
                        "Feature": c.name.replace("_", " ").title(),
                        "Value used": c.value,
                        "Importance": round(c.importance, 4),
                        "Imputed": "Yes" if c.imputed else "No",
                        "Interpretation": c.note,
                    }
                    for c in r.contributions
                ]
            )
            chart_df = cdf.sort_values("Importance", ascending=True).tail(8)
            fig_i = px.bar(
                chart_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=["#ccfbf1", "#0f766e"],
            )
            fig_i.update_layout(coloraxis_showscale=False, height=280, margin=dict(l=10, r=10, t=10, b=10))
            _risk_plotly_layout(fig_i, "Feature influence ranking")
            st.plotly_chart(fig_i, use_container_width=True)
            st.dataframe(cdf, use_container_width=True, hide_index=True)


def _load_chat_report_bundle():
    """Resolve markers/summary for chat from latest analysis or saved report."""
    reports = list_reports()
    source = st.radio(
        "Chat about",
        options=["Latest analysis", "Saved report"],
        horizontal=True,
        key="chat_source",
    )
    markers: list[dict] = []
    patient_summary = ""
    report_date = None
    sex = None
    health_score = None
    filename = ""

    if source == "Latest analysis":
        cur = st.session_state.get("current")
        if not cur:
            st.info("Analyze a report on the Analyze tab first, or choose a saved report.")
            return None
        markers = analyzed_to_dict_list(cur["analyzed"])
        patient_summary = cur.get("patient_summary") or ""
        report_date = getattr(cur.get("parsed"), "report_date", None)
        sex = cur.get("sex_used")
        health_score = getattr(cur.get("health"), "score", None)
        filename = st.session_state.get("last_source", "latest_analysis")
        st.caption(f"Using latest analysis · {len(markers)} markers")
    else:
        if not reports:
            st.info("No saved reports yet.")
            return None
        labels = {
            r.id: f"{r.report_date or 'no-date'} · {r.filename} · score {r.health_score}"
            for r in reports
        }
        rid = st.selectbox(
            "Saved report",
            options=list(labels.keys()),
            format_func=lambda i: labels[i],
            key="chat_report_id",
        )
        report = get_report(rid)
        if not report:
            return None
        markers = report.markers
        patient_summary = report.patient_summary
        report_date = report.report_date
        sex = report.sex
        health_score = report.health_score
        filename = report.filename
        st.caption(f"Using saved report `{report.id}` · {len(markers)} markers")

    return {
        "markers": markers,
        "patient_summary": patient_summary,
        "report_date": report_date,
        "sex": sex,
        "health_score": health_score,
        "filename": filename,
    }


def page_chat() -> None:
    st.markdown(
        """
        <div class="risk-hero">
          <h3>Chat with your report</h3>
          <p>Ask plain-language questions about extracted lab values.
          Free options: local Ollama, or Groq / Gemini free API keys.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="disclaimer">{DISCLAIMER} '
        "Chat answers are educational only and may be incomplete.</div>",
        unsafe_allow_html=True,
    )

    providers = detect_providers()
    available = [p for p in providers if p.available]

    with st.expander("LLM provider status", expanded=not available):
        for p in providers:
            if p.available:
                st.success(f"**{p.name}** — {p.detail} · model `{p.model}`")
            else:
                st.warning(f"**{p.name}** — {p.detail}")
        st.markdown(
            """
**Free setup**
1. **Ollama (recommended, fully free/local):** install from [ollama.com](https://ollama.com), then run `ollama pull llama3.2`
2. **Groq free tier:** create a key at [console.groq.com](https://console.groq.com) and set `GROQ_API_KEY`
3. **Gemini free tier:** create a key in Google AI Studio and set `GEMINI_API_KEY`

You can also put keys in `.streamlit/secrets.toml`.
            """
        )

    bundle = _load_chat_report_bundle()
    if not bundle:
        return

    mode_options = [p.name for p in available] + ["Offline helper (no API key)"]
    provider_name = st.selectbox("Provider", options=mode_options, key="chat_provider")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Clear chat", key="clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()
    with c2:
        st.caption("Suggested: “What looks abnormal?” · “Explain my hemoglobin” · “Questions for my doctor?”")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about your lab report...")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        context = build_report_context(
            markers=bundle["markers"],
            patient_summary=bundle["patient_summary"],
            report_date=bundle["report_date"],
            sex=bundle["sex"],
            health_score=bundle["health_score"],
            filename=bundle["filename"],
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if provider_name.startswith("Offline"):
                        answer = offline_answer(prompt, bundle["markers"])
                    else:
                        history = [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.chat_messages[:-1]
                        ]
                        answer = ask_about_report(
                            prompt,
                            provider_name=provider_name,
                            report_context=context,
                            history=history,
                        )
                except Exception as exc:  # noqa: BLE001
                    answer = (
                        f"LLM request failed: `{exc}`\n\n"
                        "Falling back to offline helper:\n\n"
                        + offline_answer(prompt, bundle["markers"])
                    )
            st.markdown(answer)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})


def page_home() -> None:
    ws = cached_workspace_status(st.session_state.get("_reports_cache_v", 0))
    n_reports = ws["report_count"]
    llm_ready = ws["llm_ready"]
    risk_ready = ws["risk_ready"]

    st.markdown(
        """
        <div class="home-hero">
          <div class="home-tag">Educational health analytics</div>
          <h1 class="home-brand">MediParse</h1>
          <p class="home-lead">
            Turn lab PDFs and scans into clear biomarker insights, trends, and clinician-ready summaries —
            with careful educational framing, not diagnoses.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Start here")
    cta1, cta2, cta3, cta4 = st.columns(4)
    with cta1:
        if st.button("Analyze a report", type="primary", use_container_width=True, key="home_go_analyze"):
            go_to("Analyze")
    with cta2:
        if st.button("View history", type="secondary", use_container_width=True, key="home_go_history"):
            go_to("History")
    with cta3:
        if st.button("Estimate risk", type="secondary", use_container_width=True, key="home_go_risk"):
            go_to("Risk")
    with cta4:
        if st.button("Chat about labs", type="secondary", use_container_width=True, key="home_go_chat"):
            go_to("Chat")

    st.write("")
    st.markdown('<div class="home-section-title">How it works</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            """
            <div class="home-step" style="animation-delay:0.05s">
              <div class="home-step-num">01</div>
              <h4>Ingest the report</h4>
              <p>Upload a PDF/image or paste lab text. OCR extracts readable values from scans.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            """
            <div class="home-step" style="animation-delay:0.12s">
              <div class="home-step-num">02</div>
              <h4>Interpret biomarkers</h4>
              <p>Match reference ranges, flag low/borderline/high results, and explain them plainly.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            """
            <div class="home-step" style="animation-delay:0.2s">
              <div class="home-step-num">03</div>
              <h4>Share with care</h4>
              <p>Track history, estimate educational risk bands, chat about findings, and export a clinician PDF.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    more1, more2 = st.columns(2)
    with more1:
        if st.button("Open Compare", type="secondary", use_container_width=True, key="home_go_compare"):
            go_to("Compare")
    with more2:
        if st.button("Open Trends", type="secondary", use_container_width=True, key="home_go_trends"):
            go_to("Trends")

    st.write("")
    st.markdown('<div class="home-section-title">Workspace status</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="home-stat">
              <div class="home-stat-value">{n_reports}</div>
              <div class="home-stat-label">Saved reports</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="home-stat" style="animation-delay:0.08s">
              <div class="home-stat-value">{"On" if risk_ready else "Off"}</div>
              <div class="home-stat-label">Risk models</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="home-stat" style="animation-delay:0.14s">
              <div class="home-stat-value">{"LLM" if llm_ready else "Basic"}</div>
              <div class="home-stat-label">Chat mode</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="home-stat" style="animation-delay:0.2s">
              <div class="home-stat-value">20+</div>
              <div class="home-stat-label">Biomarkers</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="home-note">
          <strong>Quick start:</strong> click <em>Analyze a report</em> →
          choose <em>Try sample</em> → load the baseline report →
          <em>Analyze &amp; save</em>. Then explore History, Compare, Risk, and Chat.
          This product is for learning and portfolio demonstration only.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    init_state()
    sex_override = render_navbar()
    inject_styles()

    page = st.session_state.get("active_page", "Home")
    if page == "Home":
        page_home()
    elif page == "Analyze":
        page_analyze(sex_override)
    elif page == "History":
        page_history()
    elif page == "Compare":
        page_compare()
    elif page == "Trends":
        page_trends()
    elif page == "Risk":
        page_risk()
    elif page == "Chat":
        page_chat()
    else:
        page_home()


if __name__ == "__main__":
    main()
