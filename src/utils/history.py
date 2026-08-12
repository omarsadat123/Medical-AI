"""In-session report history for trend charts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class StoredReport:
    report_id: str
    filename: str
    report_date: Optional[str]
    markers: list[dict[str, Any]]
    health_score: int


def reports_to_trend_rows(reports: list[StoredReport]) -> list[dict[str, Any]]:
    """Flatten stored reports into rows suitable for Plotly trend charts."""
    rows: list[dict[str, Any]] = []
    for report in reports:
        date_label = report.report_date or report.filename
        for marker in report.markers:
            rows.append(
                {
                    "report_id": report.report_id,
                    "date": date_label,
                    "filename": report.filename,
                    "key": marker["key"],
                    "name": marker["name"],
                    "value": marker["value"],
                    "unit": marker.get("unit"),
                    "status": marker["status"],
                    "category": marker["category"],
                }
            )
    return rows


def analyzed_to_dict_list(markers) -> list[dict[str, Any]]:
    rows = []
    for m in markers:
        rows.append(
            {
                "key": m.key,
                "name": m.name,
                "value": m.value,
                "unit": m.unit,
                "category": m.category,
                "status": m.status,
                "ref_low": m.ref_low,
                "ref_high": m.ref_high,
                "explanation": m.explanation,
                "color": m.color,
            }
        )
    return rows


def serialize_report(report: StoredReport) -> dict[str, Any]:
    return asdict(report)
