"""Quick sanity check for the Phase 1 analysis pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.explanations import build_patient_summary
from src.analysis.health_score import compute_health_score
from src.analysis.status import analyze_biomarkers
from src.parsing.biomarkers import parse_report_text


def main() -> None:
    text = (ROOT / "samples" / "sample_blood_report.txt").read_text(encoding="utf-8")
    parsed = parse_report_text(text)
    analyzed = analyze_biomarkers(parsed.biomarkers, sex=parsed.patient_sex)
    health = compute_health_score(analyzed)

    print(f"Date: {parsed.report_date}")
    print(f"Sex:  {parsed.patient_sex}")
    print(f"Markers: {len(analyzed)}")
    print(f"Score: {health.score}/100")
    print("-" * 60)
    for m in analyzed:
        unit = (m.unit or "").encode("ascii", "replace").decode("ascii")
        print(f"{m.status.upper():10} {m.name:22} {m.value} {unit}")
    print("-" * 60)
    summary = build_patient_summary(analyzed, health, parsed.report_date)[:500]
    print(summary.encode("ascii", "replace").decode("ascii"))
    assert len(analyzed) >= 10, "Expected at least 10 biomarkers from sample"
    assert any(m.status != "normal" for m in analyzed), "Expected some abnormal flags"
    assert 40 <= health.score <= 95, f"Unexpected score: {health.score}"
    print("\nOK: pipeline sanity check passed.")


if __name__ == "__main__":
    main()
