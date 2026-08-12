"""Generate clinician-oriented PDF summaries."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.analysis.explanations import DISCLAIMER


STATUS_COLORS = {
    "normal": colors.HexColor("#059669"),
    "borderline": colors.HexColor("#d97706"),
    "low": colors.HexColor("#dc2626"),
    "high": colors.HexColor("#dc2626"),
}


def build_doctor_pdf(
    *,
    markers: list[dict[str, Any]],
    health_score: Optional[int] = None,
    report_date: Optional[str] = None,
    sex: Optional[str] = None,
    filename: str = "",
    patient_summary: str = "",
    doctor_summary: str = "",
    report_id: Optional[str] = None,
) -> bytes:
    """Return PDF bytes for a clinician-oriented lab summary."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="Clinician Lab Summary",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#64748b"),
        alignment=TA_CENTER,
        spaceAfter=14,
    )
    h2 = ParagraphStyle(
        "H2Custom",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#0f766e"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "BodyCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155"),
        alignment=TA_LEFT,
    )
    small = ParagraphStyle(
        "SmallCustom",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#64748b"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#475569"),
        backColor=colors.HexColor("#f8fafc"),
        borderPadding=6,
    )

    story = []
    story.append(Paragraph("Clinician-Oriented Lab Summary", title_style))
    story.append(
        Paragraph(
            "Auto-generated educational summary · Not a diagnostic report",
            subtitle,
        )
    )

    meta_rows = [
        ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Source file", filename or "n/a"],
        ["Report date", report_date or "n/a"],
        ["Sex context", sex or "n/a"],
        ["Report ID", report_id or "n/a"],
        ["Educational health score", f"{health_score}/100" if health_score is not None else "n/a"],
    ]
    meta = Table(meta_rows, colWidths=[1.8 * inch, 4.5 * inch])
    meta.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#334155")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
            ]
        )
    )
    story.append(meta)

    flagged = [m for m in markers if (m.get("status") or "normal") != "normal"]
    story.append(Paragraph("Flagged / borderline markers", h2))
    if not flagged:
        story.append(Paragraph("None detected from extracted fields.", body))
    else:
        story.append(_markers_table(flagged))

    story.append(Paragraph("All extracted markers", h2))
    if not markers:
        story.append(Paragraph("No biomarkers available.", body))
    else:
        story.append(_markers_table(markers))

    if doctor_summary:
        story.append(Paragraph("Narrative summary", h2))
        for para in doctor_summary.split("\n"):
            if para.strip():
                story.append(Paragraph(para.replace("&", "&amp;").replace("<", "&lt;"), body))
                story.append(Spacer(1, 2))

    if patient_summary:
        story.append(Paragraph("Patient-facing notes (for reference)", h2))
        # Keep short
        snippet = patient_summary[:1200]
        for para in snippet.split("\n"):
            if para.strip():
                story.append(Paragraph(para.replace("&", "&amp;").replace("<", "&lt;"), small))

    story.append(Spacer(1, 14))
    story.append(Paragraph(DISCLAIMER.replace("&", "&amp;"), disclaimer_style))

    doc.build(story)
    return buffer.getvalue()


def _markers_table(markers: list[dict[str, Any]]) -> Table:
    header = ["Test", "Value", "Unit", "Status", "Reference", "Category"]
    rows = [header]
    for m in markers:
        ref = "n/a"
        if m.get("ref_low") is not None and m.get("ref_high") is not None:
            ref = f"{m['ref_low']}–{m['ref_high']}"
        rows.append(
            [
                str(m.get("name") or ""),
                str(m.get("value") if m.get("value") is not None else ""),
                str(m.get("unit") or ""),
                str(m.get("status") or "").upper(),
                ref,
                str(m.get("category") or ""),
            ]
        )

    table = Table(rows, colWidths=[1.6 * inch, 0.7 * inch, 0.8 * inch, 0.9 * inch, 1.1 * inch, 1.2 * inch])
    style_commands = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (1, 1), (3, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]
    for i, m in enumerate(markers, start=1):
        status = (m.get("status") or "normal").lower()
        style_commands.append(("TEXTCOLOR", (3, i), (3, i), STATUS_COLORS.get(status, colors.black)))
        style_commands.append(("FONTNAME", (3, i), (3, i), "Helvetica-Bold"))
    table.setStyle(TableStyle(style_commands))
    return table
