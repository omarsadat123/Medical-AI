"""Smoke test for Phase 2 SQLite persistence + compare."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.explanations import build_doctor_summary, build_patient_summary
from src.analysis.health_score import compute_health_score
from src.analysis.status import analyze_biomarkers
from src.db.repository import (
    compare_reports,
    count_reports,
    delete_report,
    get_report,
    list_reports,
    save_report,
    trend_rows,
)
from src.parsing.biomarkers import parse_report_text
from src.utils.history import analyzed_to_dict_list


def _analyze_file(path: Path):
    text = path.read_text(encoding="utf-8")
    parsed = parse_report_text(text)
    analyzed = analyze_biomarkers(parsed.biomarkers, sex=parsed.patient_sex)
    health = compute_health_score(analyzed)
    return parsed, analyzed, health


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test_reports.db"
        baseline = ROOT / "samples" / "sample_blood_report.txt"
        followup = ROOT / "samples" / "sample_blood_report_followup.txt"

        p1, a1, h1 = _analyze_file(baseline)
        r1 = save_report(
            filename=baseline.name,
            report_date=p1.report_date,
            sex=p1.patient_sex,
            health_score=h1.score,
            markers=analyzed_to_dict_list(a1),
            raw_text=baseline.read_text(encoding="utf-8"),
            patient_summary=build_patient_summary(a1, h1, p1.report_date),
            doctor_summary=build_doctor_summary(a1, h1, p1.report_date),
            db_path=db,
        )

        p2, a2, h2 = _analyze_file(followup)
        r2 = save_report(
            filename=followup.name,
            report_date=p2.report_date,
            sex=p2.patient_sex,
            health_score=h2.score,
            markers=analyzed_to_dict_list(a2),
            raw_text=followup.read_text(encoding="utf-8"),
            patient_summary=build_patient_summary(a2, h2, p2.report_date),
            doctor_summary=build_doctor_summary(a2, h2, p2.report_date),
            db_path=db,
        )

        assert count_reports(db_path=db) == 2
        assert len(list_reports(db_path=db)) == 2
        loaded = get_report(r1.id, db_path=db)
        assert loaded is not None and len(loaded.markers) >= 10

        cmp_rows = compare_reports(r1.id, r2.id, db_path=db)
        assert len(cmp_rows) >= 10
        assert any(row["delta"] is not None for row in cmp_rows)

        trends = trend_rows(db_path=db)
        assert len(trends) >= 20
        assert {t["report_id"] for t in trends} == {r1.id, r2.id}

        assert delete_report(r1.id, db_path=db) is True
        assert count_reports(db_path=db) == 1

        print(f"baseline score={h1.score} followup score={h2.score}")
        print(f"compare rows={len(cmp_rows)} trend rows={len(trends)}")
        print("OK: Phase 2 persistence smoke test passed.")


if __name__ == "__main__":
    main()
