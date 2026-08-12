"""CRUD helpers for persisted reports and biomarkers."""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from src.db.database import get_connection, init_db


@dataclass
class SavedReport:
    id: str
    filename: str
    report_date: Optional[str]
    sex: Optional[str]
    health_score: int
    raw_text: str = ""
    patient_summary: str = ""
    doctor_summary: str = ""
    notes: str = ""
    created_at: str = ""
    markers: list[dict[str, Any]] = field(default_factory=list)


@contextmanager
def _db(db_path: Path | None = None) -> Iterator[Any]:
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def ensure_db(db_path: Path | None = None) -> Path:
    return init_db(db_path)


def save_report(
    *,
    filename: str,
    report_date: Optional[str],
    sex: Optional[str],
    health_score: int,
    markers: list[dict[str, Any]],
    raw_text: str = "",
    patient_summary: str = "",
    doctor_summary: str = "",
    notes: str = "",
    report_id: Optional[str] = None,
    db_path: Path | None = None,
) -> SavedReport:
    ensure_db(db_path)
    rid = report_id or uuid.uuid4().hex[:12]
    created_at = datetime.now().isoformat(timespec="seconds")

    with _db(db_path) as conn:
        conn.execute(
            """
            INSERT INTO reports (
                id, filename, report_date, sex, health_score,
                raw_text, patient_summary, doctor_summary, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rid,
                filename,
                report_date,
                sex,
                health_score,
                raw_text,
                patient_summary,
                doctor_summary,
                notes,
                created_at,
            ),
        )
        for m in markers:
            conn.execute(
                """
                INSERT INTO biomarkers (
                    report_id, marker_key, name, value, unit, category,
                    status, ref_low, ref_high, explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid,
                    m.get("key"),
                    m.get("name"),
                    float(m.get("value")),
                    m.get("unit"),
                    m.get("category"),
                    m.get("status"),
                    m.get("ref_low"),
                    m.get("ref_high"),
                    m.get("explanation"),
                ),
            )

    saved = get_report(rid, db_path=db_path)
    assert saved is not None
    return saved


def list_reports(limit: int = 100, db_path: Path | None = None) -> list[SavedReport]:
    ensure_db(db_path)
    with _db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, filename, report_date, sex, health_score, raw_text,
                   patient_summary, doctor_summary, notes, created_at
            FROM reports
            ORDER BY COALESCE(report_date, created_at) DESC, created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_row_to_report(r, markers=[]) for r in rows]


def get_report(report_id: str, db_path: Path | None = None) -> Optional[SavedReport]:
    ensure_db(db_path)
    with _db(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, filename, report_date, sex, health_score, raw_text,
                   patient_summary, doctor_summary, notes, created_at
            FROM reports WHERE id = ?
            """,
            (report_id,),
        ).fetchone()
        if not row:
            return None
        markers = conn.execute(
            """
            SELECT marker_key AS key, name, value, unit, category, status,
                   ref_low, ref_high, explanation
            FROM biomarkers WHERE report_id = ?
            ORDER BY category, name
            """,
            (report_id,),
        ).fetchall()
        marker_dicts = [dict(m) for m in markers]
        report = _row_to_report(row, markers=marker_dicts)
    return report


def delete_report(report_id: str, db_path: Path | None = None) -> bool:
    ensure_db(db_path)
    with _db(db_path) as conn:
        conn.execute("DELETE FROM biomarkers WHERE report_id = ?", (report_id,))
        cur = conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
        return cur.rowcount > 0


def delete_reports(report_ids: list[str], db_path: Path | None = None) -> int:
    """Delete multiple reports. Returns number of reports removed."""
    if not report_ids:
        return 0
    ensure_db(db_path)
    deleted = 0
    with _db(db_path) as conn:
        for rid in report_ids:
            conn.execute("DELETE FROM biomarkers WHERE report_id = ?", (rid,))
            cur = conn.execute("DELETE FROM reports WHERE id = ?", (rid,))
            deleted += cur.rowcount
    return deleted


def delete_all_reports(db_path: Path | None = None) -> int:
    """Delete every saved report. Returns number of reports removed."""
    ensure_db(db_path)
    with _db(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM reports").fetchone()
        total = int(row["n"] if row else 0)
        conn.execute("DELETE FROM biomarkers")
        conn.execute("DELETE FROM reports")
    return total


def count_reports(db_path: Path | None = None) -> int:
    ensure_db(db_path)
    with _db(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM reports").fetchone()
    return int(row["n"] if row else 0)


def trend_rows(db_path: Path | None = None) -> list[dict[str, Any]]:
    """Flatten all saved biomarkers for trend charts."""
    ensure_db(db_path)
    with _db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                r.id AS report_id,
                COALESCE(r.report_date, substr(r.created_at, 1, 10)) AS date,
                r.filename,
                r.health_score,
                b.marker_key AS key,
                b.name,
                b.value,
                b.unit,
                b.status,
                b.category
            FROM biomarkers b
            JOIN reports r ON r.id = b.report_id
            ORDER BY date ASC, r.created_at ASC
            """
        ).fetchall()
        return [dict(r) for r in rows]


def compare_reports(
    report_id_a: str,
    report_id_b: str,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Compare two reports marker-by-marker.

    report_id_a = previous, report_id_b = current.
    """
    a = get_report(report_id_a, db_path=db_path)
    b = get_report(report_id_b, db_path=db_path)
    if not a or not b:
        return []

    map_a = {m["key"]: m for m in a.markers}
    map_b = {m["key"]: m for m in b.markers}
    keys = sorted(
        set(map_a) | set(map_b),
        key=lambda k: (map_b.get(k) or map_a.get(k) or {}).get("name", k),
    )

    rows: list[dict[str, Any]] = []
    for key in keys:
        prev = map_a.get(key)
        curr = map_b.get(key)
        name = (curr or prev or {}).get("name", key)
        unit = (curr or prev or {}).get("unit")
        category = (curr or prev or {}).get("category")
        prev_val = prev["value"] if prev else None
        curr_val = curr["value"] if curr else None
        delta = None
        direction = "unchanged"
        if prev_val is not None and curr_val is not None:
            delta = round(float(curr_val) - float(prev_val), 4)
            if delta > 0:
                direction = "up"
            elif delta < 0:
                direction = "down"
        elif prev_val is None and curr_val is not None:
            direction = "new"
        elif curr_val is None and prev_val is not None:
            direction = "missing"

        rows.append(
            {
                "key": key,
                "name": name,
                "unit": unit,
                "category": category,
                "previous_value": prev_val,
                "previous_status": prev.get("status") if prev else None,
                "current_value": curr_val,
                "current_status": curr.get("status") if curr else None,
                "delta": delta,
                "direction": direction,
            }
        )
    return rows


def export_report_json(report_id: str, db_path: Path | None = None) -> Optional[str]:
    report = get_report(report_id, db_path=db_path)
    if not report:
        return None
    payload = {
        "id": report.id,
        "filename": report.filename,
        "report_date": report.report_date,
        "sex": report.sex,
        "health_score": report.health_score,
        "created_at": report.created_at,
        "patient_summary": report.patient_summary,
        "doctor_summary": report.doctor_summary,
        "markers": report.markers,
    }
    return json.dumps(payload, indent=2)


def _row_to_report(row: Any, markers: list[dict[str, Any]]) -> SavedReport:
    return SavedReport(
        id=row["id"],
        filename=row["filename"],
        report_date=row["report_date"],
        sex=row["sex"],
        health_score=int(row["health_score"]),
        raw_text=row["raw_text"] or "",
        patient_summary=row["patient_summary"] or "",
        doctor_summary=row["doctor_summary"] or "",
        notes=row["notes"] or "",
        created_at=row["created_at"] or "",
        markers=markers,
    )
