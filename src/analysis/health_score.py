"""Compute a simple educational health score from analyzed biomarkers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from src.analysis.status import AnalyzedBiomarker


# Per-marker contribution to the educational composite score
STATUS_POINTS = {
    "normal": 100,
    "borderline": 70,
    "low": 40,
    "high": 40,
}


@dataclass
class CategoryScore:
    category: str
    status: str  # ok | warn | alert
    marker_count: int
    abnormal_count: int


@dataclass
class HealthScoreResult:
    score: int
    category_scores: list[CategoryScore]
    summary: str


def compute_health_score(markers: Iterable[AnalyzedBiomarker]) -> HealthScoreResult:
    markers = list(markers)
    if not markers:
        return HealthScoreResult(
            score=0,
            category_scores=[],
            summary="No biomarkers available to score.",
        )

    points = [STATUS_POINTS.get(m.status, 50) for m in markers]
    score = int(round(sum(points) / len(points)))

    by_category: dict[str, list[AnalyzedBiomarker]] = defaultdict(list)
    for m in markers:
        by_category[m.category].append(m)

    category_scores: list[CategoryScore] = []
    for category, items in sorted(by_category.items()):
        abnormal = [i for i in items if i.status != "normal"]
        if any(i.status in {"low", "high"} for i in abnormal):
            cat_status = "alert"
        elif abnormal:
            cat_status = "warn"
        else:
            cat_status = "ok"
        category_scores.append(
            CategoryScore(
                category=category,
                status=cat_status,
                marker_count=len(items),
                abnormal_count=len(abnormal),
            )
        )

    abnormal_total = sum(1 for m in markers if m.status != "normal")
    if abnormal_total == 0:
        summary = "All extracted markers are within typical reference ranges."
    elif score >= 80:
        summary = "Most markers look typical, with a few items worth reviewing with a clinician."
    elif score >= 60:
        summary = "Several markers are outside typical ranges. Share this summary with your healthcare provider."
    else:
        summary = "Multiple markers are outside typical ranges. Please review these results with a qualified clinician."

    return HealthScoreResult(score=score, category_scores=category_scores, summary=summary)
