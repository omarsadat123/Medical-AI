"""Feature schemas and extraction for risk models."""

from __future__ import annotations

from typing import Any, Optional

# Features expected by each risk model (aligned with extracted biomarker keys + extras)
RISK_FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "diabetes": {
        "label": "Diabetes risk",
        "description": "Educational estimate related to glycemic markers.",
        "features": [
            "glucose_fasting",
            "hba1c",
            "glucose_random",
            "bmi",
            "age",
            "triglycerides",
        ],
        "biomarker_keys": ["glucose_fasting", "hba1c", "glucose_random", "triglycerides"],
        "extra_inputs": ["age", "bmi"],
    },
    "heart": {
        "label": "Heart disease risk",
        "description": "Educational estimate related to lipid and metabolic markers.",
        "features": [
            "total_cholesterol",
            "ldl",
            "hdl",
            "triglycerides",
            "glucose_fasting",
            "age",
            "systolic_bp",
            "bmi",
        ],
        "biomarker_keys": [
            "total_cholesterol",
            "ldl",
            "hdl",
            "triglycerides",
            "glucose_fasting",
        ],
        "extra_inputs": ["age", "systolic_bp", "bmi"],
    },
    "kidney": {
        "label": "Kidney disease risk",
        "description": "Educational estimate related to kidney function markers.",
        "features": ["creatinine", "bun", "egfr", "age", "glucose_fasting", "hemoglobin"],
        "biomarker_keys": ["creatinine", "bun", "egfr", "glucose_fasting", "hemoglobin"],
        "extra_inputs": ["age"],
    },
    "anemia": {
        "label": "Anemia risk",
        "description": "Educational estimate related to iron and blood count markers.",
        "features": [
            "hemoglobin",
            "hematocrit",
            "rbc",
            "iron",
            "ferritin",
            "sex_female",
            "age",
        ],
        "biomarker_keys": ["hemoglobin", "hematocrit", "rbc", "iron", "ferritin"],
        "extra_inputs": ["age", "sex"],
    },
}


DEFAULT_MEDIANS: dict[str, float] = {
    "glucose_fasting": 92.0,
    "hba1c": 5.4,
    "glucose_random": 110.0,
    "bmi": 24.5,
    "age": 45.0,
    "triglycerides": 120.0,
    "total_cholesterol": 180.0,
    "ldl": 100.0,
    "hdl": 55.0,
    "systolic_bp": 120.0,
    "creatinine": 0.9,
    "bun": 14.0,
    "egfr": 95.0,
    "hemoglobin": 13.5,
    "hematocrit": 40.0,
    "rbc": 4.5,
    "iron": 90.0,
    "ferritin": 80.0,
    "sex_female": 0.0,
}


def markers_to_map(markers: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in markers:
        key = m.get("key")
        val = m.get("value")
        if key is not None and val is not None:
            try:
                out[str(key)] = float(val)
            except (TypeError, ValueError):
                continue
    return out


def build_feature_vector(
    condition: str,
    marker_map: dict[str, float],
    *,
    age: Optional[float] = None,
    bmi: Optional[float] = None,
    systolic_bp: Optional[float] = None,
    sex: Optional[str] = None,
    medians: Optional[dict[str, float]] = None,
) -> tuple[list[str], list[float], dict[str, bool]]:
    """
    Build ordered feature names/values for a condition.

    Returns (feature_names, values, was_imputed).
    """
    spec = RISK_FEATURE_SPECS[condition]
    fill = {**DEFAULT_MEDIANS, **(medians or {})}
    values: list[float] = []
    imputed: dict[str, bool] = {}

    extras = {
        "age": age,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "sex_female": 1.0 if (sex or "").lower() in {"female", "f"} else (
            0.0 if (sex or "").lower() in {"male", "m"} else None
        ),
    }

    for name in spec["features"]:
        raw = marker_map.get(name, extras.get(name))
        if raw is None:
            values.append(float(fill[name]))
            imputed[name] = True
        else:
            values.append(float(raw))
            imputed[name] = False

    return list(spec["features"]), values, imputed
