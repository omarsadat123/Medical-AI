"""Shared analysis pipeline for Streamlit + FastAPI."""

from __future__ import annotations

from typing import Any, Optional

from src.analysis.explanations import build_doctor_summary, build_patient_summary
from src.analysis.health_score import compute_health_score
from src.analysis.status import analyze_biomarkers
from src.parsing.biomarkers import parse_report_text
from src.utils.history import analyzed_to_dict_list


def analyze_text(
    text: str,
    *,
    sex_override: Optional[str] = None,
    filename: str = "report.txt",
) -> dict[str, Any]:
    """
    Run full text → parse → status → score → summaries pipeline.

    sex_override: 'male' | 'female' | None (auto)
    """
    parsed = parse_report_text(text)
    sex = sex_override
    if sex in {None, "", "auto"}:
        sex = parsed.patient_sex

    analyzed = analyze_biomarkers(parsed.biomarkers, sex=sex)
    health = compute_health_score(analyzed)
    patient_summary = build_patient_summary(analyzed, health, parsed.report_date)
    doctor_summary = build_doctor_summary(analyzed, health, parsed.report_date)
    markers = analyzed_to_dict_list(analyzed)

    return {
        "filename": filename,
        "report_date": parsed.report_date,
        "sex": sex,
        "notes": parsed.notes,
        "health_score": health.score,
        "health_summary": health.summary,
        "category_scores": [
            {
                "category": c.category,
                "status": c.status,
                "marker_count": c.marker_count,
                "abnormal_count": c.abnormal_count,
            }
            for c in health.category_scores
        ],
        "markers": markers,
        "patient_summary": patient_summary,
        "doctor_summary": doctor_summary,
        "analyzed_objects": analyzed,
        "health_object": health,
        "parsed_object": parsed,
    }
