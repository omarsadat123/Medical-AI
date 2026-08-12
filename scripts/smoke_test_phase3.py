"""Smoke test for Phase 3 risk estimation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.health_score import compute_health_score
from src.analysis.status import analyze_biomarkers
from src.ml.predict import models_available, predict_all
from src.parsing.biomarkers import parse_report_text
from src.utils.history import analyzed_to_dict_list


def main() -> None:
    assert models_available(), "Train models first: python scripts/train_risk_models.py"
    text = (ROOT / "samples" / "sample_blood_report.txt").read_text(encoding="utf-8")
    parsed = parse_report_text(text)
    analyzed = analyze_biomarkers(parsed.biomarkers, sex=parsed.patient_sex)
    markers = analyzed_to_dict_list(analyzed)
    results = predict_all(
        markers,
        age=42,
        bmi=27.0,
        systolic_bp=128,
        sex=parsed.patient_sex,
    )
    assert len(results) == 4
    for r in results:
        assert 0.0 <= r.probability <= 1.0
        assert r.band in {"Lower", "Moderate", "Higher"}
        assert r.contributions
        print(f"{r.condition:10} {r.probability:6.1%}  {r.band:8}  model={r.model_type}")
    score = compute_health_score(analyzed).score
    print(f"health_score={score}")
    print("OK: Phase 3 risk smoke test passed.")


if __name__ == "__main__":
    main()
