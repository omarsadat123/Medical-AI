"""Extract biomarker values from medical report text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.parsing.reference_ranges import build_alias_index, load_reference_ranges


@dataclass
class ExtractedBiomarker:
    key: str
    name: str
    value: float
    unit: Optional[str] = None
    reported_range: Optional[str] = None
    raw_line: str = ""
    category: str = "General"


@dataclass
class ParseResult:
    biomarkers: list[ExtractedBiomarker] = field(default_factory=list)
    report_date: Optional[str] = None
    patient_sex: Optional[str] = None
    notes: list[str] = field(default_factory=list)


UNIT_RE = (
    r"g/dL|mg/dL|ng/mL|pg/mL|mIU/L|U/L|IU/L|%|"
    r"K/μL|K/uL|x10\^3/μL|×10³/μL|x103/μL|million/μL|"
    r"mL/min(?:/1\.73m[²2])?|μg/dL|ug/dL"
)

# Value after a known test name: prefers whitespace/colon separators
AFTER_NAME_VALUE_RE = re.compile(
    rf"(?:[:=\-]|\s)+(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>{UNIT_RE})?\b"
    rf"(?:\s*\(?\s*(?P<range>\d+(?:\.\d+)?\s*[-–to]+\s*\d+(?:\.\d+)?)\s*\)?)?",
    re.IGNORECASE,
)

# Standalone numeric value line (OCR often puts values on their own line)
STANDALONE_VALUE_RE = re.compile(
    rf"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>{UNIT_RE})?\s*"
    rf"(?P<range>\d+(?:\.\d+)?\s*[-–to]+\s*\d+(?:\.\d+)?)?\s*$",
    re.IGNORECASE,
)

DATE_PATTERNS = [
    re.compile(
        r"(?:report\s*date|collected|collection\s*date|date)\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        re.I,
    ),
    re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"),
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
]

SEX_RE = re.compile(r"\b(?:sex|gender)\s*[:\-]?\s*(male|female|m|f)\b", re.I)

SKIP_LINE_HINTS = (
    "reference range",
    "referencerange",
    "test name",
    "investigation",
    "page ",
    "lab no",
    "complete blood",
    "metabolic panel",
    "result",
    "unit",
)


def parse_report_text(text: str) -> ParseResult:
    alias_index = build_alias_index()
    # Also index compacted aliases (no spaces) for OCR glued words like BloodUreaNitrogen
    compact_index = {_compact(alias): key for alias, key in alias_index.items() if len(_compact(alias)) >= 3}
    catalog = load_reference_ranges()
    found: dict[str, ExtractedBiomarker] = {}
    notes: list[str] = []

    cleaned = _normalize_ocr_text(text)
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]

    # Pass 1: same-line extraction
    for line in lines:
        lower = line.lower()
        if any(h in lower for h in SKIP_LINE_HINTS) and not re.search(r"\d", line):
            continue
        match = _extract_from_line(line, alias_index, compact_index)
        if match:
            _store(found, catalog, match, line)

    # Pass 2: multi-line OCR layout  (Name\nValue\nUnit\nRange)
    i = 0
    while i < len(lines):
        line = lines[i]
        lower = line.lower()
        if any(h in lower for h in SKIP_LINE_HINTS) and not re.search(r"\d", line):
            i += 1
            continue

        key = _match_name_only(line, alias_index, compact_index)
        if key and i + 1 < len(lines):
            value_match = STANDALONE_VALUE_RE.match(lines[i + 1])
            # Sometimes unit is stuck on same value line; sometimes next lines
            if not value_match:
                # Try "11.2 g/dL" already covered; try bare number with unit on next line
                bare = re.match(r"^\s*(\d+(?:\.\d+)?)\s*$", lines[i + 1])
                if bare:
                    unit = None
                    reported_range = None
                    if i + 2 < len(lines):
                        unit_m = re.match(rf"^\s*({UNIT_RE})\s*$", lines[i + 2], re.I)
                        if unit_m:
                            unit = unit_m.group(1)
                        range_m = re.search(
                            r"(\d+(?:\.\d+)?\s*[-–to]+\s*\d+(?:\.\d+)?)",
                            " ".join(lines[i + 2 : i + 5]),
                        )
                        if range_m:
                            reported_range = range_m.group(1)
                    if key == "glucose_random" and re.search(r"fast", line, re.I):
                        key = "glucose_fasting"
                    _store(
                        found,
                        catalog,
                        (key, float(bare.group(1)), unit, reported_range),
                        f"{line} | {lines[i + 1]}",
                    )
                    i += 2
                    continue
            else:
                if key == "glucose_random" and re.search(r"fast", line, re.I):
                    key = "glucose_fasting"
                unit = value_match.group("unit")
                reported_range = value_match.group("range")
                # Peek following lines for unit/range if missing
                if unit is None and i + 2 < len(lines):
                    unit_m = re.match(rf"^\s*({UNIT_RE})\b", lines[i + 2], re.I)
                    if unit_m:
                        unit = unit_m.group(1)
                if reported_range is None:
                    range_m = re.search(
                        r"(\d+(?:\.\d+)?\s*[-–to]+\s*\d+(?:\.\d+)?)",
                        " ".join(lines[i + 1 : i + 5]),
                    )
                    if range_m:
                        reported_range = range_m.group(1)
                _store(
                    found,
                    catalog,
                    (key, float(value_match.group("value")), unit, reported_range),
                    f"{line} | {lines[i + 1]}",
                )
                i += 2
                continue
        i += 1

    report_date = _extract_date(cleaned)
    patient_sex = _extract_sex(cleaned)

    biomarkers = sorted(found.values(), key=lambda b: (b.category, b.name))
    if not biomarkers:
        notes.append("No known biomarkers were detected. Try pasting clearer text or a better scan.")
    else:
        notes.append(f"Extracted {len(biomarkers)} biomarker(s).")

    return ParseResult(
        biomarkers=biomarkers,
        report_date=report_date,
        patient_sex=patient_sex,
        notes=notes,
    )


def _store(found, catalog, match, raw_line: str) -> None:
    key, value, unit, reported_range = match
    meta = catalog[key]
    candidate = ExtractedBiomarker(
        key=key,
        name=meta["name"],
        value=value,
        unit=unit or meta.get("unit"),
        reported_range=reported_range,
        raw_line=raw_line,
        category=meta.get("category", "General"),
    )
    existing = found.get(key)
    if existing is None or (not existing.unit and candidate.unit):
        found[key] = candidate


def _normalize_ocr_text(text: str) -> str:
    """Fix common OCR quirks: glued CamelCase, missing spaces around colons."""
    # Split glued words like BloodUreaNitrogen, but do NOT split HbA1c / B12-style tokens
    text = re.sub(r"([a-z])([A-Z][a-z]+)", r"\1 \2", text)
    text = re.sub(r":\s*", ": ", text)
    # Common OCR unit mangling
    text = text.replace("7p/3u", "mg/dL").replace("7p/Bu", "mg/dL").replace("7p/8u", "mg/dL")
    text = text.replace("x103/", "x10^3/").replace("×103/", "×10³/")
    return text


def _compact(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _match_name_only(
    line: str,
    alias_index: dict[str, str],
    compact_index: dict[str, str],
) -> Optional[str]:
    """Return biomarker key if the line is essentially just a test name."""
    # Skip lines that already look like they contain a lab value
    if re.search(r"\d", line) and AFTER_NAME_VALUE_RE.search(line):
        return None

    normalized = " ".join(line.lower().replace("-", " ").replace("_", " ").split())
    compact = _compact(line)

    for alias, key in alias_index.items():
        if len(alias) <= 1:
            continue
        if normalized == alias or normalized.startswith(alias + " ") or normalized.endswith(" " + alias):
            return key
        # Exact-ish containment for short header lines
        if alias == normalized:
            return key

    if compact in compact_index:
        return compact_index[compact]

    # Fuzzy: compact alias contained in compact line for short header-only lines
    if len(compact) >= 3 and len(compact) <= 40:
        for calias, key in sorted(compact_index.items(), key=lambda kv: len(kv[0]), reverse=True):
            if calias == compact or (len(calias) >= 5 and calias in compact):
                return key
    return None


def _extract_from_line(
    line: str,
    alias_index: dict[str, str],
    compact_index: dict[str, str],
) -> Optional[tuple[str, float, Optional[str], Optional[str]]]:
    """Find the longest known alias in the line, then parse the value after it."""
    normalized = " ".join(line.lower().replace("-", " ").replace("_", " ").split())
    compact = _compact(line)

    # Whole line is just a test name (e.g. Vitamin B12 / HbA1c) → multiline pass
    if compact in compact_index:
        return None

    for alias, key in alias_index.items():
        if len(alias) <= 1:
            continue
        # Short aliases (hb, fe, tg) are easy false positives inside other names
        if len(alias) <= 3 and normalized != alias and not re.search(
            rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])\s*[:=\-]?\s*\d",
            normalized,
        ):
            continue

        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", re.I)
        m = pattern.search(normalized)
        if not m:
            continue

        # Only accept values that appear AFTER the matched alias
        after = normalized[m.end() :]
        value_match = AFTER_NAME_VALUE_RE.match(after) or AFTER_NAME_VALUE_RE.search(after)
        if not value_match:
            continue

        if key == "glucose_random" and re.search(r"fast", line, re.I):
            key = "glucose_fasting"

        return key, float(value_match.group("value")), value_match.group("unit"), value_match.group("range")

    # Compact fallback for glued OCR with value: hemoglobin11.2
    for calias, key in sorted(compact_index.items(), key=lambda kv: len(kv[0]), reverse=True):
        if not compact.startswith(calias):
            continue
        rest = compact[len(calias) :]
        num = re.match(r"(\d+(?:\.\d+)?)", rest)
        if not num:
            continue
        if key == "glucose_random" and "fast" in normalized:
            key = "glucose_fasting"
        return key, float(num.group(1)), None, None

    return None


def _extract_date(text: str) -> Optional[str]:
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            raw = match.group(1)
            for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%Y-%m-%d", "%d/%m/%y", "%d-%m-%y"):
                try:
                    return datetime.strptime(raw, fmt).date().isoformat()
                except ValueError:
                    continue
            return raw
    return None


def _extract_sex(text: str) -> Optional[str]:
    match = SEX_RE.search(text)
    if not match:
        return None
    token = match.group(1).lower()
    if token in {"m", "male"}:
        return "male"
    if token in {"f", "female"}:
        return "female"
    return None
