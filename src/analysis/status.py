"""Determine normal / borderline / abnormal status for biomarkers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.parsing.biomarkers import ExtractedBiomarker
from src.parsing.reference_ranges import get_range_for_biomarker, load_reference_ranges


@dataclass
class AnalyzedBiomarker:
    key: str
    name: str
    value: float
    unit: Optional[str]
    category: str
    status: str  # normal | borderline | low | high
    ref_low: Optional[float]
    ref_high: Optional[float]
    reported_range: Optional[str]
    explanation: str
    color: str
    raw_line: str = ""


STATUS_COLORS = {
    "normal": "#2e7d32",
    "borderline": "#f9a825",
    "low": "#c62828",
    "high": "#c62828",
}


def analyze_biomarkers(
    biomarkers: list[ExtractedBiomarker],
    sex: Optional[str] = None,
) -> list[AnalyzedBiomarker]:
    catalog = load_reference_ranges()
    results: list[AnalyzedBiomarker] = []

    for marker in biomarkers:
        meta = catalog.get(marker.key, {})
        ref = get_range_for_biomarker(marker.key, sex=sex) or {}
        ref_low = ref.get("low")
        ref_high = ref.get("high")
        borderline_high = ref.get("borderline_high")
        borderline_low = ref.get("borderline_low")
        invert = bool(meta.get("invert_status"))

        status = _classify(
            value=marker.value,
            low=ref_low,
            high=ref_high,
            borderline_high=borderline_high,
            borderline_low=borderline_low,
            invert=invert,
        )
        explanation = _pick_explanation(meta.get("explanation", {}), status)

        results.append(
            AnalyzedBiomarker(
                key=marker.key,
                name=marker.name,
                value=marker.value,
                unit=marker.unit,
                category=marker.category,
                status=status,
                ref_low=ref_low,
                ref_high=ref_high,
                reported_range=marker.reported_range,
                explanation=explanation,
                color=STATUS_COLORS.get(status, "#455a64"),
                raw_line=marker.raw_line,
            )
        )

    return results


def _classify(
    value: float,
    low: Optional[float],
    high: Optional[float],
    borderline_high: Optional[float] = None,
    borderline_low: Optional[float] = None,
    invert: bool = False,
) -> str:
    if low is None or high is None:
        return "normal"

    if invert:
        # For HDL / eGFR: lower is concerning
        if value < low:
            return "low"
        return "normal" if value <= high else "high"

    if borderline_low is not None and value < low:
        if value >= borderline_low:
            return "borderline"
        return "low"

    if value < low:
        return "low"

    if borderline_high is not None and value > high:
        if value <= borderline_high:
            return "borderline"
        return "high"

    if value > high:
        return "high"

    return "normal"


def _pick_explanation(explanations: dict, status: str) -> str:
    if status in explanations:
        return explanations[status]
    if status in {"low", "high"} and "high" in explanations:
        return explanations.get(status, explanations.get("normal", ""))
    return explanations.get("normal", "Discuss this result with your healthcare provider.")
