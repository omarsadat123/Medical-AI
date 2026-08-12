"""Smoke test for Phase 5 PDF export + FastAPI routes."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.main import app
from src.export.pdf_report import build_doctor_pdf
from src.services.pipeline import analyze_text


def main() -> None:
    sample = (ROOT / "samples" / "sample_blood_report.txt").read_text(encoding="utf-8")
    result = analyze_text(sample, sex_override="female", filename="sample_blood_report.txt")
    pdf = build_doctor_pdf(
        markers=result["markers"],
        health_score=result["health_score"],
        report_date=result["report_date"],
        sex=result["sex"],
        filename=result["filename"],
        patient_summary=result["patient_summary"],
        doctor_summary=result["doctor_summary"],
        report_id="smoke1",
    )
    assert pdf[:4] == b"%PDF", "PDF header missing"
    out = ROOT / "exports" / "smoke_clinician_summary.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pdf)
    print(f"PDF bytes={len(pdf)} wrote={out}")

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200, health.text
    print("health", health.json()["status"])

    analyzed = client.post(
        "/analyze/text",
        json={"text": sample, "sex": "female", "filename": "api_sample.txt", "save": True},
    )
    assert analyzed.status_code == 200, analyzed.text
    payload = analyzed.json()
    assert payload["health_score"] > 0
    assert len(payload["markers"]) >= 10
    rid = payload["saved_id"]
    print(f"saved_id={rid} markers={len(payload['markers'])} score={payload['health_score']}")

    pdf_resp = client.get(f"/reports/{rid}/export.pdf")
    assert pdf_resp.status_code == 200, pdf_resp.text
    assert pdf_resp.headers["content-type"].startswith("application/pdf")
    assert pdf_resp.content[:4] == b"%PDF"
    print(f"api pdf bytes={len(pdf_resp.content)}")

    risk = client.post(
        "/risk/estimate",
        json={"markers": payload["markers"], "age": 42, "bmi": 27, "sex": "female"},
    )
    assert risk.status_code == 200, risk.text
    print(f"risk conditions={len(risk.json()['results'])}")

    chat = client.post(
        "/chat",
        json={
            "question": "What looks abnormal?",
            "markers": payload["markers"],
            "provider": "Offline helper (no API key)",
        },
    )
    assert chat.status_code == 200, chat.text
    print("chat provider", chat.json()["provider"])
    print("OK: Phase 5 PDF + API smoke test passed.")


if __name__ == "__main__":
    main()
