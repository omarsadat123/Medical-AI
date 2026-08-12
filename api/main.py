"""
FastAPI backend for Medical Report Analyzer.

Run:
  uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db.repository import (
    count_reports,
    delete_report,
    ensure_db,
    export_report_json,
    get_report,
    list_reports,
    save_report,
)
from src.export.pdf_report import build_doctor_pdf
from src.llm.chat import ask_about_report, build_report_context, offline_answer
from src.llm.providers import detect_providers
from src.ml.predict import models_available, predict_all
from src.ocr.extractor import extract_text, ocr_status
from src.services.pipeline import analyze_text

app = FastAPI(
    title="Medical Report Analyzer API",
    description="Educational lab-report analysis API (not a medical device).",
    version="0.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    sex: Optional[str] = Field(None, description="male | female | auto/null")
    filename: str = "pasted_text.txt"
    save: bool = True


class RiskRequest(BaseModel):
    markers: list[dict[str, Any]]
    age: Optional[float] = 45
    bmi: Optional[float] = 24.5
    systolic_bp: Optional[float] = 120
    sex: Optional[str] = None


class ChatRequest(BaseModel):
    question: str
    markers: list[dict[str, Any]] = Field(default_factory=list)
    patient_summary: str = ""
    report_date: Optional[str] = None
    sex: Optional[str] = None
    health_score: Optional[int] = None
    filename: str = ""
    provider: str = "Offline helper (no API key)"
    history: list[dict[str, str]] = Field(default_factory=list)


class PdfRequest(BaseModel):
    markers: list[dict[str, Any]]
    health_score: Optional[int] = None
    report_date: Optional[str] = None
    sex: Optional[str] = None
    filename: str = ""
    patient_summary: str = ""
    doctor_summary: str = ""
    report_id: Optional[str] = None


@app.on_event("startup")
def _startup() -> None:
    ensure_db()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "reports": count_reports(),
        "risk_models": models_available(),
        "ocr": ocr_status(),
        "llm": [
            {"name": p.name, "available": p.available, "model": p.model, "detail": p.detail}
            for p in detect_providers()
        ],
    }


@app.post("/analyze/text")
def analyze_from_text(body: AnalyzeRequest) -> dict[str, Any]:
    sex = None if body.sex in {None, "", "auto"} else body.sex
    result = analyze_text(body.text, sex_override=sex, filename=body.filename)
    saved_id = None
    if body.save:
        saved = save_report(
            filename=body.filename,
            report_date=result["report_date"],
            sex=result["sex"],
            health_score=result["health_score"],
            markers=result["markers"],
            raw_text=body.text,
            patient_summary=result["patient_summary"],
            doctor_summary=result["doctor_summary"],
            notes="; ".join(result["notes"]),
        )
        saved_id = saved.id
    # Drop non-JSON objects
    return {
        "saved_id": saved_id,
        "filename": result["filename"],
        "report_date": result["report_date"],
        "sex": result["sex"],
        "notes": result["notes"],
        "health_score": result["health_score"],
        "health_summary": result["health_summary"],
        "category_scores": result["category_scores"],
        "markers": result["markers"],
        "patient_summary": result["patient_summary"],
        "doctor_summary": result["doctor_summary"],
    }


@app.post("/analyze/upload")
async def analyze_upload(
    file: UploadFile = File(...),
    sex: Optional[str] = Query(None),
    save: bool = Query(True),
) -> dict[str, Any]:
    data = await file.read()
    extraction = extract_text(file.filename or "upload.bin", data)
    if not (extraction.text or "").strip():
        raise HTTPException(status_code=422, detail={"message": "No text extracted", "warnings": extraction.warnings})
    sex_norm = None if sex in {None, "", "auto"} else sex
    result = analyze_text(extraction.text, sex_override=sex_norm, filename=file.filename or "upload")
    saved_id = None
    if save:
        saved = save_report(
            filename=file.filename or "upload",
            report_date=result["report_date"],
            sex=result["sex"],
            health_score=result["health_score"],
            markers=result["markers"],
            raw_text=extraction.text,
            patient_summary=result["patient_summary"],
            doctor_summary=result["doctor_summary"],
            notes="; ".join(result["notes"]),
        )
        saved_id = saved.id
    return {
        "saved_id": saved_id,
        "extraction": {
            "method": extraction.method,
            "pages": extraction.pages,
            "warnings": extraction.warnings,
            "text_preview": extraction.text[:1500],
        },
        "filename": result["filename"],
        "report_date": result["report_date"],
        "sex": result["sex"],
        "health_score": result["health_score"],
        "health_summary": result["health_summary"],
        "markers": result["markers"],
        "patient_summary": result["patient_summary"],
        "doctor_summary": result["doctor_summary"],
    }


@app.get("/reports")
def reports(limit: int = Query(100, ge=1, le=500)) -> list[dict[str, Any]]:
    items = list_reports(limit=limit)
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
        for r in items
    ]


@app.get("/reports/{report_id}")
def report_detail(report_id: str) -> dict[str, Any]:
    report = get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return {
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


@app.delete("/reports/{report_id}")
def report_delete(report_id: str) -> dict[str, Any]:
    ok = delete_report(report_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"deleted": True, "id": report_id}


@app.get("/reports/{report_id}/export.json")
def report_export_json(report_id: str) -> Response:
    payload = export_report_json(report_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Report not found")
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="report_{report_id}.json"'},
    )


@app.get("/reports/{report_id}/export.pdf")
def report_export_pdf(report_id: str) -> Response:
    report = get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    pdf = build_doctor_pdf(
        markers=report.markers,
        health_score=report.health_score,
        report_date=report.report_date,
        sex=report.sex,
        filename=report.filename,
        patient_summary=report.patient_summary,
        doctor_summary=report.doctor_summary,
        report_id=report.id,
    )
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="clinician_summary_{report_id}.pdf"'},
    )


@app.post("/export/pdf")
def export_pdf(body: PdfRequest) -> Response:
    pdf = build_doctor_pdf(
        markers=body.markers,
        health_score=body.health_score,
        report_date=body.report_date,
        sex=body.sex,
        filename=body.filename,
        patient_summary=body.patient_summary,
        doctor_summary=body.doctor_summary,
        report_id=body.report_id,
    )
    name = body.report_id or "report"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="clinician_summary_{name}.pdf"'},
    )


@app.post("/risk/estimate")
def risk_estimate(body: RiskRequest) -> dict[str, Any]:
    if not models_available():
        raise HTTPException(status_code=503, detail="Risk models not trained. Run scripts/train_risk_models.py")
    results = predict_all(
        body.markers,
        age=body.age,
        bmi=body.bmi,
        systolic_bp=body.systolic_bp,
        sex=body.sex,
    )
    return {
        "results": [
            {
                "condition": r.condition,
                "label": r.label,
                "probability": r.probability,
                "band": r.band,
                "model_type": r.model_type,
                "metrics": r.metrics,
                "missing_biomarkers": r.missing_biomarkers,
                "explanation": r.explanation,
                "contributions": [
                    {
                        "name": c.name,
                        "value": c.value,
                        "importance": c.importance,
                        "imputed": c.imputed,
                        "note": c.note,
                    }
                    for c in r.contributions
                ],
            }
            for r in results
        ]
    }


@app.post("/chat")
def chat(body: ChatRequest) -> dict[str, str]:
    if body.provider.startswith("Offline") or not body.provider:
        answer = offline_answer(body.question, body.markers)
        return {"provider": "offline", "answer": answer}

    context = build_report_context(
        markers=body.markers,
        patient_summary=body.patient_summary,
        report_date=body.report_date,
        sex=body.sex,
        health_score=body.health_score,
        filename=body.filename,
    )
    try:
        answer = ask_about_report(
            body.question,
            provider_name=body.provider,
            report_context=context,
            history=body.history,
        )
        return {"provider": body.provider, "answer": answer}
    except Exception as exc:  # noqa: BLE001
        fallback = offline_answer(body.question, body.markers)
        return {
            "provider": "offline_fallback",
            "answer": f"LLM failed ({exc}). Offline helper:\n\n{fallback}",
        }
