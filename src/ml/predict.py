"""Load trained models and produce educational risk estimates + explanations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

from src.ml.features import (
    RISK_FEATURE_SPECS,
    build_feature_vector,
    markers_to_map,
)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

RISK_BANDS = (
    (0.33, "Lower", "#059669"),
    (0.66, "Moderate", "#d97706"),
    (1.01, "Higher", "#dc2626"),
)


@dataclass
class FeatureContribution:
    name: str
    value: float
    importance: float
    imputed: bool
    note: str


@dataclass
class RiskResult:
    condition: str
    label: str
    probability: float
    band: str
    color: str
    model_type: str
    metrics: dict[str, float]
    contributions: list[FeatureContribution] = field(default_factory=list)
    missing_biomarkers: list[str] = field(default_factory=list)
    explanation: str = ""
    disclaimer: str = ""


def models_available() -> bool:
    return _models_available_cached()


@lru_cache(maxsize=1)
def _models_available_cached() -> bool:
    return all((MODELS_DIR / f"{c}_model.joblib").exists() for c in RISK_FEATURE_SPECS)


@lru_cache(maxsize=1)
def _load_all() -> dict[str, tuple[Any, dict]]:
    loaded = {}
    for condition in RISK_FEATURE_SPECS:
        model_path = MODELS_DIR / f"{condition}_model.joblib"
        meta_path = MODELS_DIR / f"{condition}_meta.json"
        if not model_path.exists() or not meta_path.exists():
            continue
        model = joblib.load(model_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        loaded[condition] = (model, meta)
    return loaded


def reload_models() -> None:
    _load_all.cache_clear()
    _models_available_cached.cache_clear()


def risk_band(probability: float) -> tuple[str, str]:
    for cutoff, name, color in RISK_BANDS:
        if probability < cutoff:
            return name, color
    return "Higher", "#c62828"


def _contribution_note(condition: str, feature: str, value: float) -> str:
    """Short educational note about how a value typically relates to risk."""
    rules = {
        ("diabetes", "glucose_fasting"): (
            "higher" if value >= 100 else "near typical"
        ),
        ("diabetes", "hba1c"): "higher" if value >= 5.7 else "near typical",
        ("diabetes", "bmi"): "higher" if value >= 25 else "near typical",
        ("heart", "ldl"): "higher" if value >= 130 else "near typical",
        ("heart", "hdl"): "lower protective HDL" if value < 40 else "near typical",
        ("heart", "total_cholesterol"): "higher" if value >= 200 else "near typical",
        ("heart", "systolic_bp"): "higher" if value >= 130 else "near typical",
        ("kidney", "creatinine"): "higher" if value >= 1.2 else "near typical",
        ("kidney", "egfr"): "lower filtration" if value < 90 else "near typical",
        ("kidney", "bun"): "higher" if value >= 20 else "near typical",
        ("anemia", "hemoglobin"): "lower" if value < 12.5 else "near typical",
        ("anemia", "hematocrit"): "lower" if value < 36 else "near typical",
        ("anemia", "iron"): "lower" if value < 60 else "near typical",
        ("anemia", "ferritin"): "lower" if value < 30 else "near typical",
    }
    tip = rules.get((condition, feature))
    if tip:
        return f"Value looks {tip} relative to common educational thresholds."
    return "Included in the model feature set."


def predict_condition(
    condition: str,
    markers: list[dict[str, Any]],
    *,
    age: Optional[float] = None,
    bmi: Optional[float] = None,
    systolic_bp: Optional[float] = None,
    sex: Optional[str] = None,
) -> RiskResult:
    loaded = _load_all()
    if condition not in loaded:
        raise FileNotFoundError(
            f"Model for '{condition}' not found. Run scripts/train_risk_models.py first."
        )

    model, meta = loaded[condition]
    marker_map = markers_to_map(markers)
    names, values, imputed = build_feature_vector(
        condition,
        marker_map,
        age=age,
        bmi=bmi,
        systolic_bp=systolic_bp,
        sex=sex,
        medians=meta.get("medians"),
    )
    x = np.array([values], dtype=float)
    proba = float(model.predict_proba(x)[0, 1])
    band, color = risk_band(proba)

    importances = meta.get("feature_importances", {})
    contributions = []
    for name, value in zip(names, values, strict=True):
        contributions.append(
            FeatureContribution(
                name=name,
                value=round(float(value), 3),
                importance=float(importances.get(name, 0.0)),
                imputed=bool(imputed.get(name, False)),
                note=_contribution_note(condition, name, float(value)),
            )
        )
    contributions.sort(key=lambda c: c.importance, reverse=True)

    missing = [
        k
        for k in RISK_FEATURE_SPECS[condition]["biomarker_keys"]
        if k not in marker_map
    ]

    top = [c for c in contributions[:3]]
    top_txt = ", ".join(f"{c.name}={c.value}" for c in top)
    explanation = (
        f"Estimated {meta['label'].lower()} probability is {proba:.0%} ({band.lower()} band). "
        f"Most influential inputs in this model: {top_txt}. "
    )
    if missing:
        explanation += (
            f"Missing from the report and imputed for scoring: {', '.join(missing)}. "
        )
    explanation += "This is an educational estimate, not a diagnosis."

    return RiskResult(
        condition=condition,
        label=meta["label"],
        probability=proba,
        band=band,
        color=color,
        model_type=meta.get("model_type", "unknown"),
        metrics=meta.get("metrics", {}).get("selected", {}),
        contributions=contributions,
        missing_biomarkers=missing,
        explanation=explanation,
        disclaimer=meta.get(
            "disclaimer",
            "Educational risk estimate only — not a diagnosis.",
        ),
    )


def predict_all(
    markers: list[dict[str, Any]],
    *,
    age: Optional[float] = None,
    bmi: Optional[float] = None,
    systolic_bp: Optional[float] = None,
    sex: Optional[str] = None,
) -> list[RiskResult]:
    results = []
    for condition in RISK_FEATURE_SPECS:
        try:
            results.append(
                predict_condition(
                    condition,
                    markers,
                    age=age,
                    bmi=bmi,
                    systolic_bp=systolic_bp,
                    sex=sex,
                )
            )
        except FileNotFoundError:
            continue
    return results
