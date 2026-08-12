"""Build patient-friendly and doctor-oriented summaries."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from src.analysis.health_score import HealthScoreResult
from src.analysis.status import AnalyzedBiomarker

DISCLAIMER = (
    "This tool is for educational purposes only and is not a medical diagnosis, "
    "treatment plan, or substitute for professional medical advice. Always consult "
    "a qualified healthcare provider about your results."
)


def build_patient_summary(
    markers: list[AnalyzedBiomarker],
    health: HealthScoreResult,
    report_date: Optional[str] = None,
) -> str:
    date_line = f"Report date: {report_date}" if report_date else "Report date: not detected"
    lines = [
        "Medical Report Summary (Educational)",
        date_line,
        f"Educational health score: {health.score}/100",
        "",
        health.summary,
        "",
    ]

    abnormal = [m for m in markers if m.status != "normal"]
    normal = [m for m in markers if m.status == "normal"]

    if abnormal:
        lines.append("Items to discuss with your clinician:")
        for m in abnormal:
            ref = _ref_text(m)
            lines.append(
                f"- {m.name}: {m.value} {m.unit or ''} ({m.status.upper()}) | Typical range: {ref}"
            )
            lines.append(f"  {m.explanation}")
        lines.append("")

    if normal:
        lines.append(f"Within typical range ({len(normal)}): " + ", ".join(m.name for m in normal))
        lines.append("")

    lines.append(DISCLAIMER)
    return "\n".join(lines)


def build_doctor_summary(
    markers: list[AnalyzedBiomarker],
    health: HealthScoreResult,
    report_date: Optional[str] = None,
) -> str:
    lines = [
        "Clinician-Oriented Lab Summary (Auto-generated)",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Source report date: {report_date or 'N/A'}",
        f"Educational composite score: {health.score}/100",
        "",
        "Abnormal / borderline markers:",
    ]
    flagged = [m for m in markers if m.status != "normal"]
    if not flagged:
        lines.append("- None detected from extracted fields.")
    else:
        for m in flagged:
            ref = _ref_text(m)
            lines.append(
                f"- {m.name}: {m.value} {m.unit or ''} [{m.status}] (ref {ref})"
            )

    lines.extend(
        [
            "",
            "All extracted markers:",
        ]
    )
    for m in markers:
        ref = _ref_text(m)
        lines.append(f"- {m.name}: {m.value} {m.unit or ''} [{m.status}] (ref {ref})")

    lines.extend(["", DISCLAIMER])
    return "\n".join(lines)


def _ref_text(m: AnalyzedBiomarker) -> str:
    if m.ref_low is not None and m.ref_high is not None:
        return f"{m.ref_low}–{m.ref_high} {m.unit or ''}".strip()
    if m.reported_range:
        return m.reported_range
    return "N/A"
